# -*- coding: utf-8 -*-
"""Cascaded multimodal bridge for nnUNetv2 (density-first, text-second).

Architecture overview (matches the paper Figure 2):

    image (B, 1, H, W)   --> nnUNet PlainConvUNet encoder --> [s0, ..., bn_image]

    density (B, 1, H, W) --> DensityEncoder --> dens_feat (B, C, h, w)
        # Step 1 of cascaded fusion: density refines the bottleneck query.
        I_tilde = bn_image + alpha_d * CrossAttn(Q=bn_image, K=V=dens_feat)

    text (B, D)          --> TextEncoder    --> text_feat (B, C, h, w)
        # The legacy CLS branch is global semantic modulation. A single CLS
        # vector is spatially broadcast, so identical K/V entries cannot
        # provide text-derived localisation regardless of the query. Spatial
        # word/phrase interaction requires `use_text_token_inject`, which
        # preserves non-identical BERT token states as K/V.
        bn_fused = I_tilde + alpha_t * CrossAttn(Q=I_tilde, K=V=text_feat)

    decoder([s0, ..., bn_fused])

    optional alignment loss (returned alongside the segmentation output):
        align = (1 - cos(pool(bn_image), pool(dens_feat)))
              + (1 - cos(pool(bn_image), pool(text_feat)))

The trainer is responsible for adding `align_weight * align` to the
final loss. When `density` or `text` is missing (graceful fallback),
that stage is skipped (delta = 0) and the alignment term skips it.
Specifically:
    * density-only:  bn_fused = bn_image + alpha_d * delta_d
    * text-only:     bn_fused = bn_image + alpha_t * delta_t
                                  (single-stage image-text fusion)
    * both missing:  bn_fused = bn_image (identical to baseline)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from nnunetv2.training.nnUNetTrainer.density_bridge_architecture import (
    ConvBlock,
    CrossAttentionFusion,
    DensityEncoder,
    SpatialAttention,
    _largest_divisor_at_most,
    _resolve_bottleneck_channels,
)
from nnunetv2.training.nnUNetTrainer.density_cache_utils import load_density
from nnunetv2.training.nnUNetTrainer.text_cache_utils import load_text
from nnunetv2.training.nnUNetTrainer.text_token_cache_utils import load_text_tokens


# ---------------------------------------------------------------------------
# Shared helpers for modality cache loading.
# Used by both `_resolve_density_tensor` and `_resolve_text_tensor` to share
# normalisation logic (problems found during V5 code review, 2026-05-28):
#   * case_id type may be str / Path / list / tuple / np.ndarray-of-objects
#   * str(Path("a/b")) differs across OSes -> cache-key splitting
#   * empty list -> torch.stack(...) crashes with cryptic message
#   * cache-miss returning None silently is hard to debug at training time
# ---------------------------------------------------------------------------

# A set of (modality, case_id) pairs we have already warned about. Bounded
# implicitly by the dataset size so it cannot grow unbounded. We never want
# to spam the log: one warning per (modality, missing-case_id) pair is
# enough to make the failure visible without flooding stdout.
_CACHE_MISS_WARNED: "set[Tuple[str, str]]" = set()


def _normalize_case_id_value(cid) -> str:
    """Canonicalise a single case-id token to its cache-key form.

    Handles:
      * `pathlib.Path` -> POSIX-form string (forward slashes on Windows too)
      * arbitrary objects -> their `str()` with surrounding whitespace stripped
      * empty / whitespace-only -> ValueError (caller treats it as missing)

    This is the *only* place that produces the key passed to `load_density`
    / `load_text`, so cache hits cannot fragment by accident (one of the
    risks called out in the V5 code review).
    """
    if isinstance(cid, Path):
        s = cid.as_posix()
    else:
        s = str(cid)
    s = s.strip()
    if not s:
        raise ValueError("case_id must not be empty / whitespace")
    # If the upstream accidentally passed a path-like ('a\\b'), normalise
    # backslashes -> forward slashes so the cache key is identical on
    # Windows and Linux. This is a no-op for our nnUNet case names
    # (e.g. 'PCAT_QaTaCOV19_train_001') which contain neither slash.
    if "\\" in s:
        s = s.replace("\\", "/")
    return s


def _normalize_case_ids(case_id) -> Optional[List[str]]:
    """Turn whatever `case_id` shape the caller passed into a `List[str]`
    in canonical form. Returns None when no usable id is present.

    Recognised inputs:
      * None                                  -> None
      * str / Path                            -> [str_form]
      * list / tuple / generic Iterable[str]  -> [str_form, ...]
      * empty container                       -> None  (was: stack-crash)
      * 0-d tensor / numpy scalar             -> [str_form]
    """
    if case_id is None:
        return None
    if isinstance(case_id, (str, Path)):
        try:
            return [_normalize_case_id_value(case_id)]
        except ValueError:
            return None
    if isinstance(case_id, (list, tuple)):
        if len(case_id) == 0:
            return None
        out: List[str] = []
        for cid in case_id:
            try:
                out.append(_normalize_case_id_value(cid))
            except ValueError:
                # Single bad entry -> abort the whole resolve so we don't
                # build a mis-aligned batch (B' != B). Returning None makes
                # forward() take the "modality absent" path consistently.
                return None
        return out
    # Tensor / ndarray-of-objects fallback.
    if hasattr(case_id, "tolist"):
        try:
            as_list = case_id.tolist()
        except Exception:
            as_list = [case_id]
        return _normalize_case_ids(as_list)
    try:
        return [_normalize_case_id_value(case_id)]
    except ValueError:
        return None


def unwrap_network(net):
    """Peel `DistributedDataParallel` (`.module`) and `torch.compile`
    (`._orig_mod`) wrappers off of `self.network` so trainers can access
    custom attributes (`fusion`, `density_hag`, `set_modality_drop`, ...)
    in any of the supported runtime layouts:

      * `net`                            (eager, single GPU)
      * `compile(net)`                   (eager + torch.compile)
      * `DDP(net)`                       (multi-GPU, no compile)
      * `DDP(compile(net))`              (multi-GPU + compile)

    Without this helper, every direct `self.network.fusion.<attr>` access
    in the trainer breaks under DDP with `AttributeError: 'DistributedDataParallel'
    object has no attribute 'fusion'`. Worse, V2's `_accumulate_far`
    catches `AttributeError` and returns silently, so FAR statistics are
    permanently zero under DDP with no log trace -- the worst class of
    failure.

    Idempotent: calling on an already-unwrapped module returns it as-is.
    """
    inner = net
    # DDP unwrap (PyTorch DDP exposes `.module`).
    inner = getattr(inner, "module", inner)
    # torch.compile unwrap (`._orig_mod` is the compiled-module's source).
    inner = getattr(inner, "_orig_mod", inner)
    return inner


def _warn_cache_miss_once(modality: str, cid: str, exc: BaseException) -> None:
    """Emit *one* warning per (modality, case_id) cache miss. After the
    first occurrence we go silent so subsequent epochs do not spam the log.

    Why this exists: V3 / V3.1 / V4 training logs were silently degrading
    when the density cache root was mis-configured (FAR_d hovering near 0
    with no other signal). Now any cache root mismatch surfaces *exactly
    once* per missing case, while still not breaking training (we return
    None and forward() falls back to the no-modality path).
    """
    key = (modality, cid)
    if key in _CACHE_MISS_WARNED:
        return
    _CACHE_MISS_WARNED.add(key)
    warnings.warn(
        f"[bridge-cache] {modality} cache miss for case_id={cid!r}: "
        f"{type(exc).__name__}: {exc}. Treating this sample's {modality} "
        f"branch as MISSING for the rest of training. Check the cache "
        f"root env vars (PCAT_DENSITY_CACHE_ROOT / nnUNet_raw / "
        f"PCAT_PROJECT_ROOT) and re-run the corresponding build script "
        f"if this is unexpected.",
        RuntimeWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Text encoder: cached (B, D) embedding -> (B, C_bn, H_bn, W_bn) spatial map
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Project a cached text embedding to a spatial bottleneck feature map.

    Steps:
        (B, D)
        --> MLP (D -> hidden -> bottleneck_ch)
        --> reshape to (B, C, 1, 1)
        --> broadcast / expand to (B, C, H_bn, W_bn) at fuse time

    A broadcast global embedding has no spatial localisation. Because every
    key/value entry is identical, this branch can only provide a learned
    global semantic correction; the image query cannot recover token-level
    location from it. Use `TextTokenCrossAttention` when spatially varying
    word/phrase interaction is required.
    """

    def __init__(self, in_dim: int, bottleneck_ch: int, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.bottleneck_ch = int(bottleneck_ch)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.bottleneck_ch),
        )

    def forward(self, text_emb: torch.Tensor,
                spatial_size: Tuple[int, int]) -> torch.Tensor:
        if text_emb.ndim != 2:
            raise ValueError(
                f"TextEncoder expects (B, D), got {tuple(text_emb.shape)}")
        if text_emb.shape[1] != self.in_dim:
            raise ValueError(
                f"TextEncoder expects D={self.in_dim}, got {text_emb.shape[1]}")
        feat = self.proj(text_emb)                       # (B, C)
        feat = feat[:, :, None, None]                    # (B, C, 1, 1)
        h, w = int(spatial_size[0]), int(spatial_size[1])
        feat = feat.expand(-1, -1, h, w).contiguous()    # (B, C, h, w)
        return feat


# ---------------------------------------------------------------------------
# Multi-scale word-level text injection at decoder stages
# ---------------------------------------------------------------------------


class TextTokenCrossAttention(nn.Module):
    """Word-level text cross-attention injected at a single decoder stage.

        out = x + alpha * MHCA(Q = flatten(x), K = V = proj(text_tokens))

    Motivation
    ----------
    The bottleneck text branch broadcasts a single [CLS] vector and is
    therefore spatially uniform. The QaTa-COV19 reports, however, encode
    coarse localisation per word ("upper", "left", "lower right lung"). A
    data-level audit of the test set showed that 100% of the complete-miss
    (Dice=0) cases were single-region *unilateral* lesions whose reports
    explicitly named the region, while unilateral mean Dice (0.58) trailed
    bilateral (0.83) by 25 points. Injecting the FULL token sequence at an
    intermediate decoder resolution (14x14 / 28x28) lets words be attended
    to spatially, routing the region prior back to where the lesion is --
    a resolution where small lesions still survive (~1-2 px) but the [CLS]
    bottleneck (7x7) has already lost them.

    Identity-start design (mirrors `CrossAttentionFusion`)
    ------------------------------------------------------
    * `attn.out_proj` is zero-initialised so the module is the IDENTITY map
      at step 0 -> wrapping a trained baseline does not perturb it.
    * `alpha` is a plain learnable scalar (default 0.3, excluded from weight
      decay by the trainer) giving a live gradient path so the injection
      escapes the origin instead of starving. This is the SAME recipe the
      base trainer's bottleneck text branch used to reach FAR_t ~ 0.79, so
      it is known to activate on this data.
    """

    def __init__(self, feat_ch: int, text_dim: int = 768, num_heads: int = 4,
                 alpha_init: float = 0.3, dropout: float = 0.0):
        super().__init__()
        feat_ch = int(feat_ch)
        if feat_ch % num_heads != 0:
            num_heads = max(1, _largest_divisor_at_most(feat_ch, num_heads))
        self.feat_ch = feat_ch
        self.num_heads = int(num_heads)
        self.text_dim = int(text_dim)

        self.kv_proj = nn.Linear(self.text_dim, feat_ch, bias=True)
        self.norm_q = nn.LayerNorm(feat_ch)
        self.norm_kv = nn.LayerNorm(feat_ch)
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_ch, num_heads=self.num_heads,
            dropout=float(dropout), batch_first=True,
        )
        # Identity-start: zero the output projection so attn contributes 0
        # at step 0. Gradient still flows into in_proj/kv_proj (their inputs
        # are non-zero), so out_proj quickly leaves the origin; alpha then
        # receives a real gradient. Exactly the chain that makes the
        # bottleneck text branch activate.
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Diagnostic norms (no grad), read by the trainer for a per-stage
        # text Fusion Activation Ratio: ||alpha*delta|| / ||x||.
        self._last_input_norm: Optional[torch.Tensor] = None
        self._last_delta_norm: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, tokens: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            x:      (B, C, H, W) decoder-stage feature (query source).
            tokens: (B, L, text_dim) per-token hidden states (key/value).
            key_padding_mask: (B, L) bool, True == padding position to ignore.
        """
        self._last_input_norm = None
        self._last_delta_norm = None

        B, C, H, W = x.shape
        tokens = tokens.to(x.dtype)
        q = x.flatten(2).transpose(1, 2)                 # (B, H*W, C)
        kv = self.kv_proj(tokens)                        # (B, L, C)
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(q_n, kv_n, kv_n,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)            # (B, H*W, C)

        with torch.no_grad():
            self._last_input_norm = q.detach().flatten(1).norm(dim=1)
            self._last_delta_norm = (self.alpha * out).detach().flatten(1).norm(dim=1)

        fused = q + self.alpha * out
        return fused.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Gated dual-path fusion at the bottleneck
# ---------------------------------------------------------------------------


class DualPathBottleneckFusion(nn.Module):
    """Cascaded sequential multimodal fusion (density-first, text-second).

    Implements an image-anchored CASCADED QUERY REFINEMENT scheme:

        Step 1 (density refines the query):
            I_tilde = I_1 + alpha_d * MHCA(Q=I_1,       K=V=dens_feat)

        Step 2 (text attends over the density-refined query):
            I_prime = I_tilde + alpha_t * MHCA(Q=I_tilde, K=V=text_feat)

    Interpretation:
      * Density attention is spatial because `dens_feat` varies by position.
      * The broadcast CLS step is global semantic modulation, not text-derived
        localisation. Feeding it a density-refined query preserves the cascade
        ordering but does not change that mathematical limitation.
      * Token-level decoder injection is a separate, testable variant and must
        not be conflated with this CLS-only branch in manuscript claims.

    Identity-start design (inherited from CrossAttentionFusion):
      * `out_proj` is zero-initialised, so each `alpha * out` is 0 at step 0
        and the wrapped model is initially identical to the official nnUNet
        baseline.
      * `alpha` is a plain learnable scalar (default 0.3). Empirically,
        alpha=1.0 caused over-injection on easy/large-lesion cases: per-case
        analysis on 1429 validation samples showed multimodal fusion dropped
        Dice by ~0.06 on baseline-already-good cases while gaining +0.18 on
        baseline-hard cases. A smaller alpha_init biases fusion toward being
        a *correction* rather than a *dominant signal*, letting it rescue
        hard cases without destroying easy ones. The previous `tanh(alpha)`
        gate caused alpha to stay near zero for too long under a small
        alignment loss and a polynomial-decaying learning rate; the unwrapped
        scalar combined with zero-init out_proj breaks that deadlock.
    """

    def __init__(self, channels: int, num_heads: int = 4,
                 alpha_init_d: float = 0.3, alpha_init_t: float = 0.3,
                 out_proj_init: str = 'zero',
                 gate_mode_d: str = 'scalar', alpha_max_d: float = 1.0,
                 out_proj_init_d: Optional[str] = None):
        super().__init__()
        # Density path can use the input-conditioned gate (gate_mode_d='cond'):
        # the reliability probe shows the optimal density weight varies with
        # lesion size (~1.0 small -> ~0.6 large), so a per-sample/per-region
        # gate beats the single global scalar. Text path stays a scalar gate
        # (text is a weak per-slice cue; a global correction is appropriate).
        #
        # `out_proj_init_d` overrides the density path's out_proj init ONLY
        # (falls back to the shared `out_proj_init` when None). This matters
        # for the conditioned gate: with out_proj_init='zero' the residual is
        # identically 0 at init, so the gate receives no gradient and starves;
        # the density path therefore needs 'xavier' while the text path can
        # keep the legacy 'zero' identity-start. Decoupling the two inits lets
        # us switch ONLY the density path to the conditioned gate without
        # perturbing the (validated) text-path identity-start.
        self.ca_density = CrossAttentionFusion(
            channels=channels, num_heads=num_heads, alpha_init=alpha_init_d,
            out_proj_init=(out_proj_init_d or out_proj_init),
            gate_mode=gate_mode_d, alpha_max=alpha_max_d,
        )
        self.ca_text = CrossAttentionFusion(
            channels=channels, num_heads=num_heads, alpha_init=alpha_init_t,
            out_proj_init=out_proj_init,
        )

    def forward(self, bn_image: torch.Tensor,
                dens_feat: Optional[torch.Tensor],
                text_feat: Optional[torch.Tensor]) -> torch.Tensor:
        # CRITICAL (FAR freshness): clear stale diagnostic state at the
        # start of EVERY forward. Otherwise, if `dens_feat is None` for
        # this batch (e.g. V5 Modality Dropout fires), `self.ca_density`
        # is NOT called and `_last_input_norm` / `_last_delta_norm` still
        # hold the values from the previous batch -- and V2's
        # `_accumulate_far` reads them anyway, contaminating FAR_d with
        # stale data. Same risk for text. Reset to None so the trainer's
        # `if X is not None` guard works correctly per-batch.
        self.ca_density._last_input_norm = None
        self.ca_density._last_delta_norm = None
        self.ca_text._last_input_norm = None
        self.ca_text._last_delta_norm = None

        out = bn_image
        if dens_feat is not None:
            # Cascaded step 1: density refines the bottleneck query.
            #   I_tilde = I_1 + alpha_d * MHCA(Q=I_1, K=V=dens_feat)
            out = self.ca_density(out, dens_feat)
        if text_feat is not None:
            # Cascaded step 2: text attends with the density-refined query.
            #   I_prime = I_tilde + alpha_t * MHCA(Q=I_tilde, K=V=text_feat)
            # If density is missing, out == bn_image and this gracefully degrades
            # to a single-stage image-text fusion.
            out = self.ca_text(out, text_feat)
        return out


# ---------------------------------------------------------------------------
# Alignment loss helper
# ---------------------------------------------------------------------------


def _global_pool(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


def _fg_pool(feat: torch.Tensor, fg_mask_low: torch.Tensor,
             fallback_to_global: bool = True) -> torch.Tensor:
    """Foreground-mask-pooled average over a (B, C, H, W) feature map.

    Args:
        feat:        (B, C, H, W) feature map.
        fg_mask_low: (B, 1, H, W) binary mask at the SAME spatial size as
                     `feat` — caller is responsible for downsampling the
                     full-res GT mask to the bottleneck resolution.
        fallback_to_global: if a sample has zero foreground pixels at this
                     resolution (small lesion lost by downsampling), use
                     global average pool for that sample instead.

    Returns:
        (B, C) pooled vectors — one per sample.
    """
    B, C, H, W = feat.shape
    m = fg_mask_low.to(dtype=feat.dtype)
    if m.shape != (B, 1, H, W):
        m = F.interpolate(m, size=(H, W), mode="nearest")
    num = (feat * m).flatten(2).sum(-1)               # (B, C)
    den = m.flatten(2).sum(-1).clamp_min(1e-6)        # (B, 1)
    pooled = num / den                                # (B, C)
    if fallback_to_global:
        empty = (m.flatten(2).sum(-1) < 1.0).squeeze(-1)   # (B,)
        if empty.any():
            gp = _global_pool(feat)                        # (B, C)
            pooled = torch.where(empty.unsqueeze(-1), gp, pooled)
    return pooled


def cross_modal_alignment_loss(bn_image: torch.Tensor,
                                dens_feat: Optional[torch.Tensor],
                                text_feat: Optional[torch.Tensor]) -> torch.Tensor:
    """1 − cos(image, density) + 1 − cos(image, text).

    LEGACY: global-avg-pooled, both sides differentiable. Kept for backward
    compatibility with the original `nnUNetTrainerTextDensityBridge`. New
    trainers should call `cross_modal_alignment_loss_fg` instead.
    """
    if dens_feat is None and text_feat is None:
        return bn_image.new_zeros(())
    img_vec = _global_pool(bn_image)
    img_vec = F.normalize(img_vec, dim=-1)
    total = bn_image.new_zeros(())
    if dens_feat is not None:
        d_vec = F.normalize(_global_pool(dens_feat), dim=-1)
        total = total + (1.0 - (img_vec * d_vec).sum(dim=-1).mean())
    if text_feat is not None:
        t_vec = F.normalize(_global_pool(text_feat), dim=-1)
        total = total + (1.0 - (img_vec * t_vec).sum(dim=-1).mean())
    return total


def cross_modal_alignment_loss_fg(
    bn_image: torch.Tensor,
    dens_feat: Optional[torch.Tensor],
    text_feat: Optional[torch.Tensor],
    fg_mask_low: Optional[torch.Tensor],
    detach_image: bool = True,
    text_weight: float = 0.0,
) -> torch.Tensor:
    """Segmentation-aware alignment loss (TeViA-style).

    Three differences from the legacy version:
      1. `bn_image` is detached → text/density learn to align to image,
         not the other way around. This prevents the alignment loss from
         distorting the (already learning hard) image encoder.
      2. Pooling is foreground-mask-aware: we average the bottleneck
         features ONLY over the projected lesion region, so the cosine
         alignment encodes "what the lesion looks like" rather than "what
         the average pixel looks like". This is the real fix for the 27
         mis-localised dead cases.
      3. `text_weight` defaults to 0.0 because the text branch broadcasts
         a single (B, C, 1, 1) vector over the entire bottleneck — its
         pooled feature is identical to the broadcast vector itself, so
         the cosine alignment is trivial and provides no learning signal.
         Set >0 only if the text branch evolves into a spatially-varying
         design.

    Args:
        bn_image:    (B, C, H, W) image-branch bottleneck (BEFORE fusion).
        dens_feat:   (B, C, H, W) density-branch bottleneck or None.
        text_feat:   (B, C, H, W) text-branch broadcast or None.
        fg_mask_low: (B, 1, H, W) downsampled GT mask. If None, falls back
                     to global avg pool (= behaves like the legacy loss
                     except for the detach).
        detach_image: detach the image vector from the autograd graph
                     before computing the cosine. Default True.
        text_weight: per-branch scalar weight for the text alignment term.
                     Default 0.0 (text term silenced — see rationale above).

    Returns:
        Scalar loss tensor (already averaged over the batch).
    """
    if dens_feat is None and text_feat is None:
        return bn_image.new_zeros(())

    if fg_mask_low is not None:
        img_vec = _fg_pool(bn_image, fg_mask_low)
    else:
        img_vec = _global_pool(bn_image)
    if detach_image:
        img_vec = img_vec.detach()
    img_vec = F.normalize(img_vec, dim=-1)

    total = bn_image.new_zeros(())
    if dens_feat is not None:
        if fg_mask_low is not None:
            d_vec = _fg_pool(dens_feat, fg_mask_low)
        else:
            d_vec = _global_pool(dens_feat)
        d_vec = F.normalize(d_vec, dim=-1)
        total = total + (1.0 - (img_vec * d_vec).sum(dim=-1).mean())

    if text_feat is not None and text_weight > 0:
        if fg_mask_low is not None:
            t_vec = _fg_pool(text_feat, fg_mask_low)
        else:
            t_vec = _global_pool(text_feat)
        t_vec = F.normalize(t_vec, dim=-1)
        total = total + text_weight * (1.0 - (img_vec * t_vec).sum(dim=-1).mean())

    return total


# ---------------------------------------------------------------------------
# Density-as-Attention Gate (HAG) — V3 / M-DRIS
# ---------------------------------------------------------------------------


class DensityAttentionGate(nn.Module):
    """Full-resolution density-as-attention gate (HAG).

    Replaces the failed bottleneck-additive-residual injection with a
    *multiplicative* spatial calibration at the prediction-space level,
    so the density prior cannot be silently suppressed by a strong image
    backbone (the "residual starvation" failure mode documented in the
    V2 diagnostic).

    Module
    ------
    A lightweight dilated CNN at full image resolution:
        Conv(1 -> 16) -> IN -> GELU
        Conv(16 -> 32, dilation=2) -> IN -> GELU
        Conv(32 -> 32, dilation=4) -> IN -> GELU
        Conv(32 -> 1, 1x1)  -> sigmoid
    No down-sampling: small lesions (< 1024 px) are preserved, unlike the
    legacy 7x7 bottleneck path. Dilation expands the receptive field
    cheaply (effective rf ~ 21 px) without spatial loss.

    Initialisation
    --------------
    The final 1x1 conv is initialised with very small weight (std=0.01) and
    zero bias, so the pre-sigmoid logit is near zero -> attention output
    starts at ~0.5 -> centred-modulation `(1 + gamma * (attn - 0.5))` is
    approximately the identity at step 0. This avoids the "shock at init"
    that hurts a strong baseline.

    Modulation
    ----------
    Trainer applies the centred modulation:
        logits = logits * (1 + gamma * (attention - 0.5))
    Range with default `gamma_init=1.0`: [0.5, 1.5]. This is bidirectional
    (can suppress non-lesion regions, not just amplify lesion regions),
    which matches the radiological prior of "exclude clearly normal lung".

    Why multiplicative beats additive
    ---------------------------------
    Additive residual (the V2 design): `bn = bn + alpha * delta`. When the
    image branch is strong, the optimiser learns `alpha -> 0`, `delta -> 0`
    (residual starvation). Multiplicative gating: `logits *= (1 + ...)`.
    The optimiser cannot drive the modulation to identity unless
    `gamma * (attn - 0.5) -> 0` *everywhere*, which is harder; in practice
    the density signal survives because at least some pixels carry useful
    information.
    """

    def __init__(self, in_channels: int = 1, hidden: int = 16,
                 gamma_init: float = 1.0,
                 use_softplus_gamma: bool = False,
                 freeze_gamma: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(hidden * 2, affine=True),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.InstanceNorm2d(hidden * 2, affine=True),
            nn.GELU(),
            nn.Conv2d(hidden * 2, 1, kernel_size=1, bias=True),
        )
        # Final conv: small weight + zero bias -> pre-sigmoid logit ~ 0
        # -> sigmoid output ~ 0.5 -> centred-modulation is identity at init.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

        # gamma is a learnable scalar; excluded from weight decay by the V2
        # trainer's `configure_optimizers` (since p.ndim <= 1 -> no decay).
        #
        # `use_softplus_gamma=False` (V3 default): `gamma` is unconstrained.
        # The optimiser is free to learn gamma<0, which empirically happens
        # because suppressing non-lesion regions is an easier path to lower
        # CE/Dice loss than enhancing lesion regions on this imbalanced
        # dataset. The downside is that the resulting model violates the
        # narrative "density highlights lesion" and fights the aux BCE.
        #
        # `use_softplus_gamma=True` (V3.1): effective gamma is
        #     gamma_eff = softplus(self.gamma) = log(1 + exp(self.gamma)) > 0
        # so the modulation is strictly bounded to "enhance high-attn,
        # suppress low-attn", matching both the paper narrative and the
        # aux BCE supervision direction. To start fine-tuning from a V3
        # checkpoint that may have a negative `gamma`, the trainer should
        # explicitly overwrite `self.gamma.data` post-load so that the
        # effective gamma starts at a sane value (e.g. softplus(-1.05) ~ 0.3).
        self.use_softplus_gamma = bool(use_softplus_gamma)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # `freeze_gamma=True` (V4 / Transient HAG): gamma is taken out of
        # the optimisation loop entirely. The V4 trainer sets gamma value
        # directly from a fixed schedule at the start of each epoch. The
        # optimiser cannot relax HAG out of the model, because gamma is
        # no longer a learnable parameter -- it is a deterministic
        # function of training progress. Combined with `use_softplus_gamma`
        # = False (V4 default), the trainer-set value is used as-is.
        if bool(freeze_gamma):
            self.gamma.requires_grad_(False)
        self.freeze_gamma = bool(freeze_gamma)

    @torch.no_grad()
    def set_gamma(self, value: float) -> None:
        """Imperatively set the gamma value. Only meaningful when
        `freeze_gamma=True` (V4 mode); otherwise the optimiser will
        overwrite this on the next step."""
        self.gamma.data.fill_(float(value))

    @property
    def gamma_effective(self) -> torch.Tensor:
        """Returns the gamma value actually used by `modulate`. For V3 this
        is just `self.gamma`; for V3.1 it is `softplus(self.gamma) > 0`."""
        if self.use_softplus_gamma:
            return F.softplus(self.gamma)
        return self.gamma

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """Returns (B, 1, H, W) sigmoid attention map in [0, 1]."""
        return torch.sigmoid(self.net(density))

    def modulate(self, logits: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """Centred multiplicative modulation. `attn` is auto-resized to match
        `logits` spatial size (bilinear) so this works for every deep-
        supervision head. Uses `gamma_effective` so the V3.1 softplus
        constraint is applied transparently."""
        if attn.shape[-2:] != logits.shape[-2:]:
            attn = F.interpolate(attn, size=logits.shape[-2:],
                                 mode='bilinear', align_corners=False)
        return logits * (1.0 + self.gamma_effective * (attn - 0.5))


def cross_modal_alignment_loss_projected(
    bn_image: torch.Tensor,
    dens_feat: Optional[torch.Tensor],
    text_feat: Optional[torch.Tensor],
    fg_mask_low: Optional[torch.Tensor],
    proj_image: torch.nn.Module,
    proj_density: torch.nn.Module,
    proj_text: torch.nn.Module,
    text_weight: float = 1.0,
) -> torch.Tensor:
    """V2 alignment loss: stop-grad image + projection heads + cosine.

    This is the canonical SSL alignment design (BYOL / SimSiam / DINO):

        z_I  = P_I(pool(bn_image))        -> detached  (image-anchored)
        z_D  = P_D(pool(dens_feat))       -> trainable
        z_T  = P_T(pool(text_feat))       -> trainable

        L = (1 - cos(sg(z_I), z_D)) + w_t * (1 - cos(sg(z_I), z_T))

    Why we changed each piece vs the legacy `cross_modal_alignment_loss_fg`:

      * **Projection heads** P_I/D/T (1-layer Linear, dim=`projection_dim`).
        The 512-channel bottleneck is a *segmentation* feature space; the
        text branch is a *report-level* CLS embedding. Aligning them in the
        raw bottleneck space forces "feature equality" which is too strong
        (Q2 of our pipeline diagnostic showed cos(img, text) saturating at
        0.93 already, leaving no useful gradient surface). Projecting both
        to a smaller (e.g. 128-d) shared space lets the loss enforce
        "directional consistency" without coercing equality.

      * **Stop-gradient on image** (always-on in V2; the kwarg is implicit
        because `proj_image(v_I).detach()` is built into this loss). The
        image encoder is already strong; we want text/density to align *to*
        the image, not deform the image. Same trick that makes BYOL work
        without a momentum encoder.

      * **fg-pool with global fallback** (inherited from `_fg_pool`). On
        the PCAT 7x7 bottleneck most lesions survive as 0-14 pixels --
        when 0, we fall back to global pool so the loss is still defined.

      * **No L2-normalisation before cosine** is needed: F.cosine_similarity
        does it internally; this means the projection heads' raw output
        magnitudes carry no signal -- only their directions matter, which
        is exactly the regulariser semantics we want.

    Args:
        bn_image:    (B, C, H, W) image bottleneck (before fusion).
        dens_feat:   (B, C, H, W) density bottleneck.
        text_feat:   (B, C, H, W) text bottleneck (broadcast in spatial dims).
        fg_mask_low: (B, 1, H, W) downsampled GT mask, or None for global pool.
        proj_image:  Linear (C -> projection_dim). Applied with .detach().
        proj_density: Linear (C -> projection_dim).
        proj_text:   Linear (C -> projection_dim).
        text_weight: per-branch scalar for the text term (default 1.0).

    Returns:
        Scalar (already averaged over the batch).
    """
    if dens_feat is None and text_feat is None:
        return bn_image.new_zeros(())

    def _pool(x):
        if fg_mask_low is not None:
            return _fg_pool(x, fg_mask_low)
        return _global_pool(x)

    v_I = _pool(bn_image)
    z_I = proj_image(v_I).detach()         # stop-grad anchor

    total = bn_image.new_zeros(())
    if dens_feat is not None:
        v_D = _pool(dens_feat)
        z_D = proj_density(v_D)
        total = total + (1.0 - F.cosine_similarity(z_I, z_D, dim=-1).mean())

    if text_feat is not None and text_weight > 0.0:
        v_T = _pool(text_feat)
        z_T = proj_text(v_T)
        total = total + text_weight * (
            1.0 - F.cosine_similarity(z_I, z_T, dim=-1).mean()
        )

    return total


def cross_modal_alignment_loss_seghead(
    seg_feat: torch.Tensor,
    dens_feat: Optional[torch.Tensor],
    text_feat: Optional[torch.Tensor],
    fg_mask_full: torch.Tensor,
    fg_mask_low: Optional[torch.Tensor],
    proj_seg: torch.nn.Module,
    proj_density: torch.nn.Module,
    proj_text: torch.nn.Module,
    vis_proto: Optional[torch.Tensor] = None,
    proto_weight: float = 0.0,
    text_weight: float = 1.0,
):
    """T1b alignment loss: TeViA-style segmentation-HEAD visual anchor
    (+ optional momentum visual prototype).

    What changes vs the V2 loss (`cross_modal_alignment_loss_projected`):

      * **Anchor moves from the 7x7 encoder bottleneck to the full-res
        segmentation-head feature.** `seg_feat` is the output of the LAST
        decoder stage (channels = features_per_stage[0]); foreground-pooled
        over the GT mask it is exactly TeViA's
            vis_feat = os1.detach() * (gt == 1)
        i.e. a lesion-specific, high-resolution foreground representation.
        The bottleneck loses essentially all small lesions to 32x
        downsampling (7x7); the seg head keeps them, which is precisely the
        regime where the user is currently weakest.

      * **Momentum visual prototype** (`vis_proto`, an EMA over ALL training
        data, alpha=0.99 -- see `update_vis_prototype`). The per-instance
        foreground anchor is blended with the global prototype so that
        text / density align to a STABLE global lesion semantic instead of
        a noisy per-batch target. In TeViA the prototype is fed into the
        decoder attention; the user deferred that (T2), so here it lives on
        the LOSS side as an anchor enhancement (no decoder change).

    Image-anchored (stop-grad on the visual side): density / text are pulled
    toward the frozen visual foreground, never the reverse -- the same
    BYOL/SimSiam principle already used by the V2 loss.

    Args:
        seg_feat:     (B, C_seg, Hf, Wf) full-res decoder feature.
        dens_feat:    (B, C_bn, h, w) density bottleneck feature, or None.
        text_feat:    (B, C_bn, h, w) text bottleneck feature, or None.
        fg_mask_full: (B, 1, Hf, Wf) GT foreground at `seg_feat` resolution.
        fg_mask_low:  (B, 1, h, w) GT foreground at bottleneck resolution,
                      or None for global pooling of dens/text.
        proj_seg:     Linear (C_seg -> D). Applied then `.detach()`.
        proj_density: Linear (C_bn  -> D).
        proj_text:    Linear (C_bn  -> D).
        vis_proto:    (C_seg,) EMA prototype, or None.
        proto_weight: enhancement strength of the prototype on the anchor.
        text_weight:  per-branch scalar for the text term.

    Returns:
        (loss_scalar, batch_mean_fg_vec)  where `batch_mean_fg_vec` is the
        detached (C_seg,) batch-mean foreground vector to feed the EMA
        update, or None when no auxiliary modality was active this step.
    """
    if dens_feat is None and text_feat is None:
        return seg_feat.new_zeros(()), None

    # Full-res foreground-pooled visual anchor (per-sample). _fg_pool falls
    # back to global pool for samples whose lesion vanished at this res.
    v_vis = _fg_pool(seg_feat, fg_mask_full)              # (B, C_seg)
    batch_mean = v_vis.detach().float().mean(0)           # (C_seg,) for EMA

    anchor = v_vis
    if vis_proto is not None and proto_weight > 0.0:
        p = vis_proto.detach().to(v_vis.dtype).unsqueeze(0)          # (1, C_seg)
        anchor = F.normalize(v_vis, dim=-1) + proto_weight * F.normalize(p, dim=-1)

    z_I = proj_seg(anchor).detach()                       # stop-grad anchor

    def _pool_low(x):
        if fg_mask_low is not None:
            return _fg_pool(x, fg_mask_low)
        return _global_pool(x)

    total = seg_feat.new_zeros(())
    if dens_feat is not None:
        z_D = proj_density(_pool_low(dens_feat))
        total = total + (1.0 - F.cosine_similarity(z_I, z_D, dim=-1).mean())
    if text_feat is not None and text_weight > 0.0:
        z_T = proj_text(_pool_low(text_feat))
        total = total + text_weight * (
            1.0 - F.cosine_similarity(z_I, z_T, dim=-1).mean()
        )
    return total, batch_mean


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------


class TextDensityBridgePlainConvUNet(PlainConvUNet):
    """nnUNetv2 PlainConvUNet wrapped with an image-anchored dual-path bridge.

    Both density and text branches:
      * have independent encoders,
      * fuse with the image bottleneck via gated cross-attention,
      * combine additively (no three-way interaction),
      * gates initialize to 0 so training starts from the official baseline,
      * yield an alignment loss handle through `last_alignment_loss`.
    """

    def __init__(self, *args,
                 density_in_channels: int = 1,
                 density_base_ch: int = 32,
                 text_in_dim: int = 768,
                 text_hidden: int = 256,
                 text_dropout: float = 0.1,
                 num_heads: int = 4,
                 alpha_init_d: float = 0.3,
                 alpha_init_t: float = 0.3,
                 align_loss_weight: float = 0.5,
                 out_proj_init: str = 'zero',
                 gate_mode_d: str = 'scalar',
                 alpha_max_d: float = 1.0,
                 out_proj_init_d: Optional[str] = None,
                 projection_dim: int = 0,
                 use_density_hag: bool = False,
                 hag_hidden: int = 16,
                 hag_gamma_init: float = 1.0,
                 hag_use_softplus_gamma: bool = False,
                 hag_freeze_gamma: bool = False,
                 use_seg_head_align: bool = False,
                 use_text_token_inject: bool = False,
                 text_token_dim: int = 768,
                 text_inject_stages: Sequence[int] = (0, 1),
                 text_inject_heads: int = 4,
                 text_inject_alpha: float = 0.3,
                 allow_legacy_density_cache: bool = False,
                 require_aligned_density: bool = True,
                 require_text_cache: bool = False,
                 require_text_tokens: bool = False,
                 **kwargs):
        # The raw nnUNet tensor contains image channels followed by the
        # density modality. PlainConvUNet must be constructed for image
        # channels only; forward() splits density before calling its encoder.
        # This makes nnUNet apply identical preprocessing, augmentation,
        # cropping and sliding-window coordinates to image and density.
        density_in_channels = int(density_in_channels)
        args = list(args)
        if args:
            total_input_channels = int(args[0])
            image_in_channels = (total_input_channels - density_in_channels
                                 if total_input_channels > density_in_channels
                                 else total_input_channels)
            args[0] = image_in_channels
        elif "input_channels" in kwargs:
            total_input_channels = int(kwargs["input_channels"])
            image_in_channels = (total_input_channels - density_in_channels
                                 if total_input_channels > density_in_channels
                                 else total_input_channels)
            kwargs["input_channels"] = image_in_channels
        else:
            raise TypeError("TextDensityBridgePlainConvUNet requires input_channels")

        super().__init__(*args, **kwargs)

        bottleneck_ch = _resolve_bottleneck_channels(self.encoder, kwargs)

        self.total_input_channels = total_input_channels
        self.image_in_channels = image_in_channels
        self.density_in_channels = density_in_channels
        self.allow_legacy_density_cache = bool(allow_legacy_density_cache)
        self.require_aligned_density = bool(require_aligned_density)
        self.require_text_cache = bool(require_text_cache)
        self.require_text_tokens = bool(require_text_tokens)
        self.text_in_dim = int(text_in_dim)
        self.align_loss_weight = float(align_loss_weight)

        self.density_encoder = DensityEncoder(
            bottleneck_ch=bottleneck_ch,
            in_channels=self.density_in_channels,
            base_ch=density_base_ch,
        )
        self.text_encoder = TextEncoder(
            in_dim=self.text_in_dim,
            bottleneck_ch=bottleneck_ch,
            hidden=int(text_hidden),
            dropout=float(text_dropout),
        )
        self.fusion = DualPathBottleneckFusion(
            channels=bottleneck_ch, num_heads=num_heads,
            alpha_init_d=alpha_init_d, alpha_init_t=alpha_init_t,
            out_proj_init=out_proj_init,
            gate_mode_d=gate_mode_d, alpha_max_d=alpha_max_d,
            out_proj_init_d=out_proj_init_d,
        )

        # ---- (V2) projection heads for SSL-style alignment ----------------
        # Mapping pooled per-modality vectors to a shared `projection_dim`
        # before cosine alignment. Solves the "semantic-level mismatch":
        # the image bottleneck encodes spatial 512-d segmentation features,
        # while the text branch is a global CLS embedding; comparing them
        # directly in 512-d space forces "feature equality" rather than
        # "shared semantic direction". A 1-layer Linear (BYOL/SimSiam-lite)
        # is enough -- deeper MLPs tend to learn a trivial collapse.
        #
        # `projection_dim = 0` -> legacy behaviour (no heads, trainers that
        # rely on the legacy `cross_modal_alignment_loss[_fg]` keep working).
        self.projection_dim = int(projection_dim)
        if self.projection_dim > 0:
            self.proj_image = nn.Linear(bottleneck_ch, self.projection_dim, bias=True)
            self.proj_density = nn.Linear(bottleneck_ch, self.projection_dim, bias=True)
            self.proj_text = nn.Linear(bottleneck_ch, self.projection_dim, bias=True)
        else:
            self.proj_image = self.proj_density = self.proj_text = None

        # ---- (T1b) segmentation-HEAD alignment + momentum visual prototype.
        # TeViA anchors the text<->vision alignment on the FULL-RESOLUTION
        # foreground feature from the segmentation head (its `os1`), NOT the
        # 7x7 bottleneck. We tap the last decoder stage's output via a
        # forward hook and project it with a dedicated head; we also keep a
        # momentum (EMA) visual prototype aggregated over all training data
        # (TeViA `feat*_global`, alpha=0.99).
        #
        # Gated behind `use_seg_head_align` so that EVERY pre-existing trainer
        # (V2/V3/V31/V5/Boost) builds the IDENTICAL module it built before --
        # no new parameters / buffers -> their old checkpoints still load with
        # strict=True. Only the V6 trainer passes use_seg_head_align=True.
        self._use_seg_head_align: bool = bool(use_seg_head_align)
        self._capture_seg_head_enabled: bool = False
        self._last_seg_head_feat: Optional[torch.Tensor] = None
        self._vis_proto_alpha: float = 0.99
        self.proj_seg: Optional[nn.Module] = None
        if self._use_seg_head_align and self.projection_dim > 0:
            seg_head_ch = int(self.encoder.output_channels[0])
            self.seg_head_ch = seg_head_ch
            self.proj_seg = nn.Linear(seg_head_ch, self.projection_dim, bias=True)
            # Momentum visual prototype. Buffers persist across checkpoints,
            # so a resumed run keeps its accumulated global lesion semantics.
            self.register_buffer("_vis_proto", torch.zeros(seg_head_ch))
            self.register_buffer("_vis_proto_init", torch.zeros(1))
            # Hook the LAST decoder stage (full-res feature before seg conv).
            try:
                self.decoder.stages[-1].register_forward_hook(self._seg_head_hook)
            except (AttributeError, IndexError):
                pass

        # ---- (V7 / DRIS multi-scale) word-level text token injection ------
        # Inject the FULL report token sequence into selected decoder stages
        # via cross-attention, so per-word localisation cues ("upper left
        # lung", ...) can be routed to the right region at a resolution
        # where small / unilateral lesions still survive (14x14 / 28x28) --
        # unlike the 7x7 bottleneck where they vanish.
        #
        # `text_inject_stages` are DECODER stage indices (0 = first decoder
        # stage = lowest resolution above the bottleneck). For the standard
        # 6-stage 2D U-Net on 224x224:
        #     stage 0 -> 14x14, stage 1 -> 28x28, stage 2 -> 56x56, ...
        # The per-stage channel count is encoder.output_channels[-(j+2)].
        #
        # Gated behind `use_text_token_inject` so EVERY pre-existing trainer
        # builds the identical module set it built before -> their old
        # checkpoints still load with strict=True. Only V7 passes True.
        self.use_text_token_inject: bool = bool(use_text_token_inject)
        self.text_token_dim: int = int(text_token_dim)
        self.text_inject_stages: Tuple[int, ...] = tuple(
            int(s) for s in text_inject_stages)
        self._cur_text_tokens: Optional[torch.Tensor] = None
        self._cur_text_mask: Optional[torch.Tensor] = None
        self.text_inject: Optional[nn.ModuleDict] = None
        if self.use_text_token_inject:
            self.text_inject = nn.ModuleDict()
            n_dec = len(self.decoder.stages)
            oc = list(self.encoder.output_channels)
            for j in self.text_inject_stages:
                if j < 0 or j >= n_dec:
                    warnings.warn(
                        f"[text-inject] decoder stage index {j} out of range "
                        f"[0, {n_dec}); skipping.", RuntimeWarning, stacklevel=2)
                    continue
                feat_ch = int(oc[-(j + 2)])
                mod = TextTokenCrossAttention(
                    feat_ch=feat_ch, text_dim=self.text_token_dim,
                    num_heads=int(text_inject_heads),
                    alpha_init=float(text_inject_alpha),
                )
                self.text_inject[str(j)] = mod
                # Replace the stage output with the text-modulated feature.
                self.decoder.stages[j].register_forward_hook(
                    self._make_text_inject_hook(mod))

        # ---- (V3 / M-DRIS) Density-as-Attention Gate ----------------------
        # Multiplicative prediction-space spatial calibration. Replaces the
        # functionally-dead additive bottleneck residual (V2 diagnostic
        # showed FAR_d declining from 0.025 to 0.016 within 20 epochs).
        # When `use_density_hag=False` the network is identical to V2;
        # when True, the trainer is expected to call `pop_last_attention()`
        # in train_step to feed an auxiliary BCE supervision against the
        # downsampled lesion mask (prevents HAG from collapsing to a
        # trivial "lung-mask" attention).
        self.use_density_hag = bool(use_density_hag)
        if self.use_density_hag:
            self.density_hag = DensityAttentionGate(
                in_channels=self.density_in_channels,
                hidden=int(hag_hidden),
                gamma_init=float(hag_gamma_init),
                use_softplus_gamma=bool(hag_use_softplus_gamma),
                freeze_gamma=bool(hag_freeze_gamma),
            )
        else:
            self.density_hag = None
        # Last forward's HAG attention (full-res, before any resize). Exposed
        # so the trainer can compute the auxiliary BCE supervision against
        # the downsampled ground-truth lesion mask. We cache BOTH the
        # pre-sigmoid logits (for autocast-safe BCEWithLogits in train_step)
        # AND the sigmoid output (for diagnostics / centred modulation).
        self._last_hag_logits: Optional[torch.Tensor] = None
        self._last_hag_attn: Optional[torch.Tensor] = None

        # Last forward's auxiliary loss (filled in forward, read by trainer).
        # We attach it to the module as a Python attribute (NOT a buffer) so
        # PyTorch checkpointing does not try to serialise the live graph.
        self._last_align_loss: Optional[torch.Tensor] = None

        # Last forward's bottleneck features, exposed for advanced trainers
        # (e.g. the *Boost* variant) that need to compute a foreground-pooled
        # alignment loss externally. None outside of training and / or when
        # the corresponding modality is missing for this batch.
        self._last_bn_image: Optional[torch.Tensor] = None
        self._last_dens_feat: Optional[torch.Tensor] = None
        self._last_text_feat: Optional[torch.Tensor] = None

        # ---- Modality dropout hook (consumed by `forward`) ----------------
        # The trainer (e.g. V5 = Modality Dropout) sets these to True via
        # `set_modality_drop(...)` before each train_step's forward call.
        # When True, `forward` skips the corresponding `_resolve_*_tensor`
        # fallback so the missing modality is REALLY missing (no lru_cache
        # auto-load). They are automatically reset to False at the end of
        # every forward so they never leak across steps.
        self._force_drop_density: bool = False
        self._force_drop_text: bool = False

        # ---- Sticky inference case_id (consumed by `forward`) -------------
        # Background. During the FINAL validation pass after training
        # (`nnUNetTrainer.perform_actual_validation`), the prediction
        # call goes through `nnUNetPredictor` whose internal forward is
        # `self.network(x)` -- positional argument only, no `case_id`.
        # That means the bridge network's `_resolve_*_tensor` would have
        # `case_id=None` and IMMEDIATELY return None, silently turning
        # the multimodal model into an image-only one for the entire
        # final validation. Result: the reported "Mean Validation Dice"
        # never benefited from multimodal training. (Verified: V5 with
        # Modality Dropout achieved FAR_d=0.26 and Pseudo Dice 0.8606
        # but final Mean Val Dice was the same 0.789 as image-only V31.)
        #
        # Fix: trainer sets this sticky attribute at the START of each
        # case in the validation loop (via `set_inference_case_id`), so
        # the subsequent `predictor.predict(...)` calls -- which each
        # internally call `self.network(x)` many times for sliding-window
        # tiles + mirror TTA -- can fall back to this case_id when no
        # explicit one was passed. The trainer clears it at the end of
        # the loop. See `nnUNetTrainerTextDensityBridge.perform_actual_validation`
        # for the injection point.
        self._inference_case_id: Optional[Union[str, List[str]]] = None

    # -- modality dropout API (consumed by trainer) --------------------------

    def set_inference_case_id(self,
                              case_id: Union[str, List[str], None]) -> None:
        """Install a sticky case_id used by `forward()` when the caller
        does NOT pass `case_id` explicitly (the nnUNetPredictor path).

        Call with the case identifier at the start of each case in the
        inference / validation loop, and with `None` to clear at the
        end. See `_inference_case_id` docstring for full rationale.

        Idempotent: writes a simple attribute.
        """
        self._inference_case_id = case_id

    def set_modality_drop(self, drop_density: bool = False,
                          drop_text: bool = False) -> None:
        """Tell the next forward() call to skip the density / text branches
        entirely (no encoder, no fusion residual, no HAG modulation).

        IMPORTANT: setting `density=None` / `text=None` in the trainer is
        NOT sufficient by itself, because `_resolve_density_tensor` and
        `_resolve_text_tensor` fall back to loading from the case_id-keyed
        lru_cache when the tensor is None. This setter sets a one-shot
        flag (auto-cleared at end of forward) that disables the fallback.

        Reference: Lau et al., "A Unified Representation Network for
        Segmentation with Missing Modalities", arXiv:1908.06683 (2019).
        """
        self._force_drop_density = bool(drop_density)
        self._force_drop_text = bool(drop_text)

    # -- modality fetchers ---------------------------------------------------

    def _resolve_density_tensor(
            self, density: Optional[torch.Tensor],
            case_id: Union[str, Path,
                           Sequence[Union[str, Path]], None],
            device: torch.device,
            dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Resolve the density input for one forward pass.

        Priority order (explicit > implicit, by design):
          1. If `density` is provided, use it after a channel-count
             normalisation. The `case_id`-based cache fallback is NOT
             consulted in this branch. This is intentional: when a
             trainer (e.g. nnUNetTrainerTextDensityBridge) injects
             `data[:, 1:2]` as the density channel, we must honour that
             choice instead of silently overriding with the cache. If
             you want the cache lookup, pass `density=None`.
          2. Otherwise, look up the density cache by `case_id` (one
             tensor per item). All inputs are normalised via
             `_normalize_case_ids` so `Path`s, mixed slashes, and empty
             collections never produce wrong cache keys or stack crashes.
          3. On any of (case_id missing, cache miss, shape mismatch),
             return None. The single forward() call already knows how
             to skip a missing modality. We emit a warning EXACTLY once
             per (modality, case_id) pair so a silent mis-config still
             surfaces (V3-V4 training logs used to silently hover at
             FAR_d~=0 with no signal -- now you get a RuntimeWarning).

        Output: tensor of shape (B, self.density_in_channels, h, w),
        on `device`, with `dtype`.
        """
        if density is not None:
            density = density.to(device=device, dtype=dtype)
            if density.ndim != 4:
                raise ValueError(
                    f"density must be [B,C,H,W], got {tuple(density.shape)}")
            if density.shape[1] != self.density_in_channels:
                bch = density.shape[1]
                if bch > self.density_in_channels:
                    density = density[:, : self.density_in_channels]
                else:
                    pad = self.density_in_channels - bch
                    density = torch.cat(
                        [density,
                         density.new_zeros((density.shape[0], pad,
                                            density.shape[2], density.shape[3]))],
                        dim=1,
                    )
            return density

        case_ids = _normalize_case_ids(case_id)
        if case_ids is None:
            return None

        if not self.allow_legacy_density_cache:
            raise RuntimeError(
                "Density was not supplied as an aligned nnUNet input modality. "
                "Case-level cache fallback is disabled because cached full images "
                "cannot share nnUNet augmentation or sliding-window coordinates. "
                "Export density as *_0001 and re-run planning/preprocessing."
            )

        per_item: List[torch.Tensor] = []
        ref_shape: Optional[Tuple[int, ...]] = None
        for cid in case_ids:
            try:
                arr = load_density(cid)
            except (FileNotFoundError, ValueError, OSError) as exc:
                _warn_cache_miss_once("density", cid, exc)
                return None
            t = torch.from_numpy(arr)
            if ref_shape is None:
                ref_shape = tuple(t.shape)
            elif tuple(t.shape) != ref_shape:
                # Batch members must share spatial shape for torch.stack.
                # This becomes a real risk on multi-spacing 3D datasets
                # (e.g. MosMedDataPlus) where different cases have
                # different H/W. Surface it ONCE instead of crashing
                # mid-epoch with a cryptic stack-size error.
                _warn_cache_miss_once(
                    "density",
                    cid,
                    RuntimeError(
                        f"shape mismatch within batch: "
                        f"ref={ref_shape}, this={tuple(t.shape)}"),
                )
                return None
            per_item.append(t)

        if not per_item:
            # Defensive: _normalize_case_ids already returns None for
            # empty inputs, but if some future caller bypasses it we
            # still avoid the `stack expects a non-empty TensorList`
            # crash.
            return None

        batch = torch.stack(per_item, dim=0).to(device=device, dtype=dtype)
        return batch

    def _resolve_text_tensor(
            self, text: Optional[torch.Tensor],
            case_id: Union[str, Path,
                           Sequence[Union[str, Path]], None],
            device: torch.device,
            dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Resolve the text embedding input for one forward pass.

        Same explicit-over-implicit contract as `_resolve_density_tensor`:
        a non-None `text` argument wins outright; only `text is None`
        triggers the case-id-keyed `load_text` cache lookup. See that
        method's docstring for the rationale and the warning policy.

        Output: tensor of shape (B, self.text_in_dim), on `device`, with
        `dtype`. Returns None on cache miss or normalisation failure.
        """
        if text is not None:
            text = text.to(device=device, dtype=dtype)
            if text.ndim != 2:
                raise ValueError(
                    f"text must be (B, D), got {tuple(text.shape)}")
            if text.shape[1] != self.text_in_dim:
                raise ValueError(
                    f"text dim mismatch: model expects {self.text_in_dim}, "
                    f"got {text.shape[1]}"
                )
            return text

        case_ids = _normalize_case_ids(case_id)
        if case_ids is None:
            if self.require_text_cache:
                raise RuntimeError(
                    "DTIS requires case IDs to resolve the case-aligned text "
                    "cache, but this forward call provided none.")
            return None

        per_item: List[torch.Tensor] = []
        for cid in case_ids:
            try:
                arr = load_text(cid, expected_dim=self.text_in_dim)
            except (FileNotFoundError, ValueError, OSError) as exc:
                if self.require_text_cache:
                    raise RuntimeError(
                        f"Required DTIS text cache is unavailable for "
                        f"case_id={cid!r}") from exc
                _warn_cache_miss_once("text", cid, exc)
                return None
            t = torch.from_numpy(arr)
            if t.ndim != 1 or t.shape[0] != self.text_in_dim:
                # text embeddings are always (D,) per case; mismatched
                # dims means the cache was built for a different LM.
                _warn_cache_miss_once(
                    "text",
                    cid,
                    RuntimeError(
                        f"text embedding shape mismatch: "
                        f"expected ({self.text_in_dim},), got "
                        f"{tuple(t.shape)}"),
                )
                return None
            per_item.append(t)

        if not per_item:
            return None

        batch = torch.stack(per_item, dim=0).to(device=device, dtype=dtype)
        return batch

    # -- text token sequence (multi-scale injection) -------------------------

    def _make_text_inject_hook(self, inject_module: "TextTokenCrossAttention"):
        """Build a forward hook that replaces a decoder stage's output with
        its text-token-modulated version. The hook reads the per-forward
        token stash (`self._cur_text_tokens` / `_cur_text_mask`) set in
        `forward()` right before `self.decoder(skips)`. When no tokens are
        available (cache miss, dropped modality, or token injection disabled)
        it returns None -> the stage output is left unchanged (baseline)."""

        def hook(module, inp, out):
            if not self.use_text_token_inject:
                return None
            toks = self._cur_text_tokens
            if toks is None:
                return None
            if isinstance(out, (list, tuple)):
                # Decoder StackedConvBlocks yield a single tensor; bail out
                # defensively if some future variant changes that.
                return None
            return inject_module(out, toks, self._cur_text_mask)

        return hook

    def _resolve_text_tokens(
            self, case_id: Union[str, Path,
                                 Sequence[Union[str, Path]], None],
            device: torch.device,
            dtype: torch.dtype):
        """Load the per-token report hidden states for multi-scale injection.

        Returns `(tokens, key_padding_mask)` where
            tokens:           (B, Lmax, text_token_dim) float, on `device`.
            key_padding_mask: (B, Lmax) bool, True == padding (ignored by MHA).
        Variable per-case lengths are right-padded to the batch max. Returns
        `(None, None)` on cache miss / normalisation failure / when token
        injection is disabled, so `forward` and the hooks degrade to baseline.
        """
        if not self.use_text_token_inject:
            return None, None

        case_ids = _normalize_case_ids(case_id)
        if case_ids is None:
            if self.require_text_tokens:
                raise RuntimeError(
                    "Token-level DTIS requires case IDs to resolve token "
                    "sequences, but this forward call provided none.")
            return None, None

        seqs: List[torch.Tensor] = []
        for cid in case_ids:
            try:
                arr = load_text_tokens(cid, expected_dim=self.text_token_dim)
            except (FileNotFoundError, ValueError, OSError) as exc:
                if self.require_text_tokens:
                    raise RuntimeError(
                        f"Required DTIS token cache is unavailable for "
                        f"case_id={cid!r}") from exc
                _warn_cache_miss_once("text_tokens", cid, exc)
                return None, None
            t = torch.from_numpy(arr)          # (L, D)
            if t.ndim != 2 or t.shape[1] != self.text_token_dim:
                _warn_cache_miss_once(
                    "text_tokens", cid,
                    RuntimeError(
                        f"token sequence shape mismatch: expected (L, "
                        f"{self.text_token_dim}), got {tuple(t.shape)}"))
                return None, None
            seqs.append(t)

        if not seqs:
            return None, None

        B = len(seqs)
        Lmax = max(int(s.shape[0]) for s in seqs)
        D = self.text_token_dim
        tokens = torch.zeros((B, Lmax, D), dtype=torch.float32)
        # True == pad. Start all-pad, then UN-mask the real tokens. Every
        # sample has L >= 1 (guaranteed by the loader), so no row is fully
        # masked -> MHA softmax never sees an all -inf row (no NaNs).
        mask = torch.ones((B, Lmax), dtype=torch.bool)
        for i, s in enumerate(seqs):
            L = int(s.shape[0])
            tokens[i, :L] = s
            mask[i, :L] = False

        tokens = tokens.to(device=device, dtype=dtype)
        mask = mask.to(device=device)
        return tokens, mask

    # -- forward -------------------------------------------------------------

    def forward(self, x: Optional[torch.Tensor] = None,
                density: Optional[torch.Tensor] = None,
                text: Optional[torch.Tensor] = None,
                case_id: Union[str, Path,
                               Sequence[Union[str, Path]], None] = None,
                **kwargs):
        if x is None:
            x = kwargs.get("image", None)
        if x is None:
            raise TypeError(
                "TextDensityBridgePlainConvUNet.forward() requires 'x' / 'image'")

        if x.ndim != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got {tuple(x.shape)}")
        aligned_channels = self.image_in_channels + self.density_in_channels
        if x.shape[1] == aligned_channels:
            aligned_density = x[:, self.image_in_channels:aligned_channels]
            x = x[:, :self.image_in_channels]
            if density is None:
                density = aligned_density
        elif x.shape[1] != self.image_in_channels:
            raise ValueError(
                f"Expected {self.image_in_channels} image channels or "
                f"{aligned_channels} aligned image+density channels, got "
                f"{x.shape[1]}"
            )
        if density is not None and density.shape[0] != x.shape[0]:
            raise ValueError("Image and density batch sizes must match")
        if density is not None and density.shape[2:] != x.shape[2:]:
            raise ValueError(
                "Image and density must already be spatially aligned before the "
                f"bridge; got image={tuple(x.shape)}, density={tuple(density.shape)}"
            )

        # Inference-path fallback (see `_inference_case_id` docstring):
        # nnUNetPredictor's internal call is `self.network(x)`, so case_id
        # arrives as None during the final validation pass. If the trainer
        # has stickied a case_id on us, use it. We broadcast a scalar
        # sticky id to the batch dimension because sliding-window + mirror
        # TTA stack several patches of the SAME case into one forward call.
        if case_id is None and self._inference_case_id is not None:
            sticky = self._inference_case_id
            B = int(x.shape[0])
            if isinstance(sticky, (str, Path)):
                case_id = [sticky] * B
            elif isinstance(sticky, (list, tuple)):
                if len(sticky) == 1 and B > 1:
                    case_id = [sticky[0]] * B
                else:
                    # Caller is responsible for matching B if they supplied
                    # a multi-element sticky; we keep it as-is.
                    case_id = list(sticky)
            else:
                case_id = sticky

        skips = self.encoder(x)
        skips = list(skips)
        bn_image = skips[-1]

        device = bn_image.device
        dtype = bn_image.dtype

        # Consume the one-shot modality-dropout flags set by the trainer
        # (e.g. V5). We clear them IMMEDIATELY so a forgotten reset in the
        # trainer never leaks dropout into the next forward call (in
        # particular, never into validation).
        drop_d = bool(self._force_drop_density)
        drop_t = bool(self._force_drop_text)
        self._force_drop_density = False
        self._force_drop_text = False

        if drop_d:
            density_t = None
        else:
            density_t = self._resolve_density_tensor(density, case_id, device, dtype)
            if density_t is None and self.require_aligned_density:
                raise RuntimeError(
                    "Aligned density modality is required but missing. Export it "
                    "as nnUNet channel 1 (*_0001) and re-run preprocessing."
                )
        if drop_t:
            text_t = None
        else:
            text_t = self._resolve_text_tensor(text, case_id, device, dtype)

        dens_feat: Optional[torch.Tensor] = None
        if density_t is not None:
            dens_feat = self.density_encoder(density_t)
            if dens_feat.shape[2:] != bn_image.shape[2:]:
                dens_feat = F.interpolate(dens_feat, size=bn_image.shape[2:],
                                           mode="bilinear", align_corners=False)

        text_feat: Optional[torch.Tensor] = None
        if text_t is not None:
            text_feat = self.text_encoder(text_t, spatial_size=bn_image.shape[2:])

        bn_fused = self.fusion(bn_image, dens_feat, text_feat)
        skips[-1] = bn_fused

        # ---- (V7) stash the report token sequence for decoder-stage hooks --
        # The hooks registered on selected decoder stages read these two
        # attributes during `self.decoder(skips)` below. Token injection is
        # tied to the text modality, so it is suppressed under text dropout
        # (`drop_t`). Cleared right after the decoder so nothing leaks across
        # forward calls.
        if self.use_text_token_inject and not drop_t:
            self._cur_text_tokens, self._cur_text_mask = self._resolve_text_tokens(
                case_id, device, dtype)
        else:
            self._cur_text_tokens, self._cur_text_mask = None, None

        # Alignment loss is computed BEFORE the decoder so it lives on
        # the cleanest features (post-encoder, pre-fusion mixing).
        if self.training:
            self._last_align_loss = cross_modal_alignment_loss(
                bn_image, dens_feat, text_feat
            )
            # Expose raw bottleneck features for advanced trainers (e.g. the
            # Boost variant computes a foreground-mask-pooled alignment loss
            # externally, using these tensors + a downsampled target mask).
            self._last_bn_image = bn_image
            self._last_dens_feat = dens_feat
            self._last_text_feat = text_feat
        else:
            self._last_align_loss = None
            self._last_bn_image = None
            self._last_dens_feat = None
            self._last_text_feat = None

        out = self.decoder(skips)

        # Clear the token stash immediately so it can never leak into a
        # later forward (e.g. a subsequent validation tile) where the
        # case_id / dropout state differs.
        self._cur_text_tokens = None
        self._cur_text_mask = None

        # ---- (V3) HAG: density-as-attention multiplicative modulation ----
        # Applied to ALL deep-supervision heads if `out` is a list, otherwise
        # to the single tensor. Each scale gets its own bilinear-resized
        # copy of the full-res attention map. The trainer reads
        # `self._last_hag_attn` (full-res) for auxiliary BCE supervision.
        if self.density_hag is not None and density_t is not None:
            # Split sigmoid so we can cache the pre-sigmoid logits as well
            # (needed for autocast-safe BCEWithLogits in the V3 trainer).
            hag_logits = self.density_hag.net(density_t)    # (B, 1, H_full, W_full)
            attn = torch.sigmoid(hag_logits)
            if self.training:
                self._last_hag_logits = hag_logits
                self._last_hag_attn = attn
            else:
                self._last_hag_logits = None
                self._last_hag_attn = None
            if isinstance(out, list):
                out = [self.density_hag.modulate(o, attn) for o in out]
            else:
                out = self.density_hag.modulate(out, attn)
        else:
            self._last_hag_logits = None
            self._last_hag_attn = None

        return out

    # -- public access for trainer -------------------------------------------

    def pop_alignment_loss(self) -> Optional[torch.Tensor]:
        """Return and clear the auxiliary alignment loss from the last forward.

        Returns None when not in training mode or when both auxiliary
        modalities were missing.
        """
        loss = self._last_align_loss
        self._last_align_loss = None
        return loss

    def pop_bottleneck_features(self):
        """Return and clear the raw bottleneck feature tensors from the last
        forward: (bn_image, dens_feat, text_feat). Any of them may be None.

        Used by trainers that compute a foreground-pooled alignment loss
        externally, where they need access to both the pre-fusion image
        bottleneck and the per-modality encoded features at the same
        spatial resolution as the GT (downsampled to match).
        """
        bn = self._last_bn_image
        d = self._last_dens_feat
        t = self._last_text_feat
        self._last_bn_image = None
        self._last_dens_feat = None
        self._last_text_feat = None
        return bn, d, t

    # -- (T1b) segmentation-head feature + momentum visual prototype ----------

    def _seg_head_hook(self, module, inp, out):
        """Forward hook on the last decoder stage. Captures the full-res
        segmentation-head feature for the V6 seg-head alignment loss. Only
        stores during training WITH grad enabled (so the final validation /
        predictor path -- which runs under no_grad -- never holds it)."""
        if (self._capture_seg_head_enabled and self.training
                and torch.is_grad_enabled()):
            self._last_seg_head_feat = out
        else:
            self._last_seg_head_feat = None

    def enable_seg_head_capture(self, enabled: bool = True) -> None:
        """Turn seg-head feature capture on/off (V6 trainer toggles this)."""
        self._capture_seg_head_enabled = bool(enabled)

    def pop_seg_head_feat(self) -> Optional[torch.Tensor]:
        """Return and clear the last forward's full-res seg-head feature."""
        f = self._last_seg_head_feat
        self._last_seg_head_feat = None
        return f

    @torch.no_grad()
    def update_vis_prototype(self, vec: torch.Tensor) -> None:
        """EMA update of the momentum visual prototype (TeViA feat*_global).

        First call copies `vec` (initialisation); subsequent calls do
        `proto <- alpha*proto + (1-alpha)*vec`. No-op if seg-head alignment
        was not enabled at construction time.
        """
        if getattr(self, "proj_seg", None) is None or not hasattr(self, "_vis_proto"):
            return
        v = vec.detach().float().to(self._vis_proto.device)
        if float(self._vis_proto_init.item()) < 0.5:
            self._vis_proto.copy_(v)
            self._vis_proto_init.fill_(1.0)
        else:
            a = float(self._vis_proto_alpha)
            self._vis_proto.mul_(a).add_((1.0 - a) * v)

    def get_vis_prototype(self) -> Optional[torch.Tensor]:
        """Return the EMA visual prototype once initialised, else None."""
        if getattr(self, "proj_seg", None) is None or not hasattr(self, "_vis_proto"):
            return None
        if float(self._vis_proto_init.item()) < 0.5:
            return None
        return self._vis_proto

    def pop_hag_attention(self) -> Optional[torch.Tensor]:
        """Return and clear the full-resolution HAG sigmoid attention map
        from the last forward. None outside training / when HAG disabled."""
        attn = self._last_hag_attn
        self._last_hag_attn = None
        return attn

    def pop_hag_logits(self) -> Optional[torch.Tensor]:
        """Return and clear the full-resolution HAG *pre-sigmoid* logits
        from the last forward. Use these with `binary_cross_entropy_with_logits`
        (autocast-safe) instead of `binary_cross_entropy(sigmoid(...))`
        (NOT autocast-safe). None outside training / when HAG disabled."""
        logits = self._last_hag_logits
        self._last_hag_logits = None
        return logits

    # -- (V7) per-stage text-injection diagnostics --------------------------

    def text_inject_far(self) -> "dict[str, float]":
        """Per-decoder-stage text Fusion Activation Ratio for the LAST forward.

        FAR_stage = mean_b( ||alpha * attn_out||_b / ||x||_b ). Quantifies
        how strongly each injected text cross-attention contributes relative
        to the decoder feature it modulates. Returns an empty dict when text
        injection is disabled or no tokens were active this forward.
        """
        out: "dict[str, float]" = {}
        if not self.use_text_token_inject or self.text_inject is None:
            return out
        for key, mod in self.text_inject.items():
            inp = getattr(mod, "_last_input_norm", None)
            dlt = getattr(mod, "_last_delta_norm", None)
            if inp is None or dlt is None:
                continue
            ratio = (dlt / inp.clamp_min(1e-6)).mean().item()
            out[f"stage{key}"] = float(ratio)
        return out
