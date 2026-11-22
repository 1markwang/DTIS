# -*- coding: utf-8 -*-
"""nnUNetv2 architecture wrapper that injects density at the decoder input.

This keeps the official encoder/decoder modules from dynamic_network_architectures
intact and only replaces the bottleneck feature that is fed into the decoder.

Density branch design (current):
- A 1-channel density input is consumed:
    channel 0 = distribution  (gated positive lift over per-lung p25 baseline,
                               zero outside the eroded inner lung mask)
- DensityEncoder: Conv -> SpatialAttention -> Conv (stride 2) -> Conv ->
  SpatialAttention -> Conv 1x1 projection to bottleneck channels.
- Fusion is a residual multi-head Cross-Attention: Q from the image bottleneck,
  K/V from the encoded density. This lets density CONTRIBUTE NEW INFORMATION
  to the bottleneck (additive), not just rescale it.
- The fusion strength is gated by a learnable scalar initialized to 0 so the
  network starts identical to the official nnUNetv2 baseline.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet

from nnunetv2.training.nnUNetTrainer.density_cache_utils import load_density


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention: pool along channel dim, then 7x7 conv."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


# ---------------------------------------------------------------------------
# Density encoder: 2 channels in, bottleneck_ch channels out
# ---------------------------------------------------------------------------


class DensityEncoder(nn.Module):
    """Conv + SpatialAttention twice, then 1x1 projection.

    Mirrors the topology shown in the project's framework diagram:
        Conv2d -> SpatialAttention -> Conv2d (stride 2) -> ... -> Conv 1x1.
    """

    def __init__(self, bottleneck_ch: int, in_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_ch)
        self.enc1 = ConvBlock(base_ch, base_ch * 2)
        self.attn1 = SpatialAttention()
        self.down1 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.enc2 = ConvBlock(base_ch * 2, base_ch * 4)
        self.attn2 = SpatialAttention()
        self.proj = nn.Sequential(
            nn.Conv2d(base_ch * 4, bottleneck_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(bottleneck_ch, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.enc1(x)
        x = self.attn1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.attn2(x)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Input-conditioned fusion gate (replaces the single global scalar alpha)
# ---------------------------------------------------------------------------


class ConditionedGate(nn.Module):
    """Input-conditioned replacement for the single global scalar fusion gate.

    Motivation (empirical, from this project's density-vs-CLIP reliability probe
    on CT (MosMed) + CXR (QaTa), slice-level bootstrap, n=300):
      * The low-level density/opacity cue is a robustly reliable lesion signal
        at EVERY lesion size (bootstrap AUC lower-bound >= 0.58 CT / 0.61 CXR),
        while the high-level CLIP cue is unreliable per slice (AUC <= 0.5 on
        small lesions, CI entirely below random). => density deserves a high,
        often dominant fusion weight; text/CLIP should not drive dense fusion.
      * The empirically optimal density weight w* is NOT constant: ~1.0 for
        small / focal lesions, dropping to ~0.6 for large consolidations where
        the image-only path is already strong. A SINGLE global scalar alpha is
        therefore provably suboptimal -- it must compromise between regimes
        (the documented 'alpha=1.0 over-injects on large lesions, -0.075 Dice'
        failure is exactly this).

    This module predicts the gate per sample (or per spatial location at
    bottleneck resolution) from the pooled image bottleneck + density features,
    bounded to (0, alpha_max).

    Weak-/identity-start: the final layer's weight is zero-initialised and its
    bias is set so every sample emits exactly `alpha_init` at step 0, exactly
    reproducing the scalar-alpha baseline; the gate then specialises during
    training. MUST be paired with `out_proj_init != 'zero'` in
    CrossAttentionFusion: if the residual is identically zero at step 0 then
    BOTH out_proj and this gate starve (no gradient escapes the origin).
    """

    def __init__(self, channels: int, alpha_max: float = 1.0,
                 alpha_init: float = 0.05, spatial: bool = False,
                 hidden: Optional[int] = None):
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.spatial = bool(spatial)
        hidden = int(hidden) if hidden else max(8, channels // 4)
        if self.spatial:
            self.net = nn.Sequential(
                nn.Conv2d(2 * channels, hidden, kernel_size=1), nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(2 * channels, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )
        self._last = self.net[-1]
        nn.init.zeros_(self._last.weight)
        a = float(min(max(alpha_init / max(self.alpha_max, 1e-6), 1e-4), 1.0 - 1e-4))
        nn.init.constant_(self._last.bias, math.log(a / (1.0 - a)))   # sigmoid(bias)=a

    def forward(self, bottleneck: torch.Tensor, density_feat: torch.Tensor) -> torch.Tensor:
        if self.spatial:
            g = torch.sigmoid(self.net(torch.cat([bottleneck, density_feat], dim=1)))
            return g * self.alpha_max                              # (B, 1, H, W)
        b = bottleneck.mean(dim=(2, 3))
        d = density_feat.mean(dim=(2, 3))
        g = torch.sigmoid(self.net(torch.cat([b, d], dim=1))) * self.alpha_max
        return g.view(g.shape[0], 1, 1, 1)                         # (B, 1, 1, 1)


# ---------------------------------------------------------------------------
# Cross-attention fusion: Q from image bottleneck, K/V from density feature
# ---------------------------------------------------------------------------


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention with a learnable identity-start gate.

    Forward:
        out = bottleneck + alpha * CrossAttn(Q=bottleneck, K=V=density)

    Identity-start design (fixes the 'alpha stuck at 0' failure mode):
      * `out_proj.weight` is initialised to all zeros, so the cross-attention
        output is exactly 0 at step 0. The bottleneck is therefore identical
        to the official baseline at initialisation.
      * `alpha` is a plain learnable scalar (NOT wrapped in tanh) initialised
        to a small value (default 0.3). Empirically alpha=1.0 caused over-
        injection on easy/large lesion cases (where the image-only baseline
        is already near-perfect), degrading large-lesion Dice by ~0.075.
        A smaller initial alpha keeps fusion as a *correction* rather than
        a *dominant signal*, preserving the baseline's strengths while still
        rescuing hard cases. It is exposed via the `.alpha` attribute and
        is intended to be excluded from weight decay by the trainer.
      * At the first step, gradients flow through `v_proj`/`q_proj`/`k_proj`
        (which are randomly initialised); `out_proj` then receives a real
        gradient signal and rapidly leaves zero.
    """

    def __init__(self, channels: int, num_heads: int = 4, alpha_init: float = 0.3,
                 out_proj_init: str = 'zero', gate_mode: str = 'scalar',
                 alpha_max: float = 1.0):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = max(1, _largest_divisor_at_most(channels, num_heads))
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.GroupNorm(num_groups=min(32, self.channels), num_channels=self.channels)
        self.norm_kv = nn.GroupNorm(num_groups=min(32, self.channels), num_channels=self.channels)

        self.q_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)

        # out_proj initialisation strategy. Two options:
        # 'zero'   (legacy): bn_fused == bn_image at step 0, but the residual
        #          path is starved -- d(seg_loss)/d(out_proj) is multiplicatively
        #          near zero, so out_proj never escapes the origin. Validated
        #          empirically on PCAT: after 78 epochs, ||out_proj|| = 0.20
        #          (vs Q/K/V's ~9.0), so the cross-attention residual is ~2%
        #          of the baseline signal -- effectively dead.
        # 'xavier' (V2): out_proj starts non-zero, so even a tiny alpha gives
        #          a real residual from step 0. Paired with `alpha_init=0.05`
        #          this gives weak-but-active injection (5% magnitude) that
        #          the optimiser can immediately tune up or down. Avoids the
        #          residual-starvation collapse.
        op_init = (out_proj_init or 'zero').lower()
        if op_init == 'zero':
            nn.init.zeros_(self.out_proj.weight)
        elif op_init in ('xavier', 'xavier_uniform'):
            nn.init.xavier_uniform_(self.out_proj.weight)
        elif op_init in ('kaiming', 'kaiming_normal'):
            nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity='linear')
        else:
            raise ValueError(
                f"out_proj_init must be one of 'zero' / 'xavier' / 'kaiming', got {out_proj_init!r}")
        self._out_proj_init = op_init

        # alpha is a plain learnable scalar. No tanh, no clipping. Under
        # `out_proj_init='zero'`, alpha's magnitude is irrelevant at step 0
        # (multiplied by zero anyway). Under `out_proj_init='xavier'`, alpha
        # directly controls how strongly the residual is injected at step 0
        # -- the recommended pairing is `alpha_init=0.05` (weak injection
        # the optimiser can actively grow if useful, shrink if harmful).
        # The trainer is responsible for excluding `alpha` from weight decay.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # gate_mode:
        #   'scalar'       (default, backward-compatible): effective gate = self.alpha,
        #                  a single global learnable scalar. Identical to the legacy
        #                  behaviour -- nothing changes unless a caller opts in.
        #   'cond'         per-SAMPLE gate predicted from pooled (bottleneck, density)
        #                  features. Lets the network inject MORE density on small/
        #                  focal-lesion images and LESS on large-consolidation images
        #                  (the regime where a global alpha is forced to compromise).
        #   'cond_spatial' per-LOCATION gate at bottleneck resolution (a (B,1,H,W)
        #                  map): same idea but spatially resolved.
        # In the conditioned modes `self.alpha` is retained (so trainers that read
        # `.alpha` / exclude it from weight decay keep working) but is NOT applied
        # to the residual; the conditioned gate replaces it. Conditioned modes REQUIRE
        # out_proj_init != 'zero' (else the residual is 0 at step 0 and the gate
        # starves) -- the density-bridge trainer already passes 'xavier'.
        self.gate_mode = (gate_mode or 'scalar').lower()
        if self.gate_mode not in ('scalar', 'cond', 'cond_spatial'):
            raise ValueError(
                f"gate_mode must be 'scalar' / 'cond' / 'cond_spatial', got {gate_mode!r}")
        if self.gate_mode == 'scalar':
            self.cond_gate = None
        else:
            if self._out_proj_init == 'zero':
                import warnings
                warnings.warn(
                    "ConditionedGate with out_proj_init='zero' starves at init "
                    "(residual==0 => no gradient). Use out_proj_init='xavier'.",
                    RuntimeWarning, stacklevel=2)
            self.cond_gate = ConditionedGate(
                self.channels, alpha_max=alpha_max, alpha_init=float(alpha_init),
                spatial=(self.gate_mode == 'cond_spatial'))

        # Diagnostic state -- populated in forward, read by trainer for FAR
        # logging (Fusion Activation Ratio = ||gate*delta|| / ||bn_image||).
        # See `nnUNetTrainerTextDensityBridgeV2.on_epoch_end` for usage.
        self._last_input_norm: Optional[torch.Tensor] = None
        self._last_delta_norm: Optional[torch.Tensor] = None
        self._last_gate_mean: Optional[torch.Tensor] = None

    def forward(self, bottleneck: torch.Tensor, density_feat: torch.Tensor) -> torch.Tensor:
        """Gated additive cross-attention fusion.

        Args:
            bottleneck: query feature (B, C, H, W).
            density_feat: key/value feature (B, C, H, W).

        Returns:
            `bottleneck + alpha * MHCA(Q=bottleneck, K=V=density_feat)`.
        """
        # Pre-clear diagnostic state so callers that skip this forward
        # (e.g. when their `density_feat is None` guard fires) cannot
        # observe stale FAR readings from the previous batch. Symmetric
        # with the same defence in `DualPathBottleneckFusion.forward`.
        self._last_input_norm = None
        self._last_delta_norm = None
        self._last_gate_mean = None

        B, C, H, W = bottleneck.shape
        if density_feat.shape[2:] != bottleneck.shape[2:]:
            density_feat = F.interpolate(density_feat, size=(H, W), mode="bilinear",
                                         align_corners=False)

        q = self.q_proj(self.norm_q(bottleneck))
        kv_in = self.norm_kv(density_feat)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)

        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(B, C, H, W)
        out = self.out_proj(out)

        # Effective fusion gate: legacy global scalar, or an input-conditioned
        # gate (per-sample / per-location). `gate` broadcasts against `out`.
        if self.cond_gate is None:
            gate = self.alpha
        else:
            gate = self.cond_gate(bottleneck, density_feat)
        delta = gate * out

        # Diagnostic: per-call FAR norms (no gradient). Used by V2 trainer
        # to log Fusion Activation Ratio (residual magnitude vs trunk).
        with torch.no_grad():
            self._last_input_norm = bottleneck.detach().flatten(1).norm(dim=1)
            self._last_delta_norm = delta.detach().flatten(1).norm(dim=1)
            gate_t = gate if torch.is_tensor(gate) else torch.as_tensor(float(gate))
            self._last_gate_mean = gate_t.detach().float().mean()

        return bottleneck + delta


def _largest_divisor_at_most(n: int, target: int) -> int:
    for k in range(target, 0, -1):
        if n % k == 0:
            return k
    return 1


def _resolve_bottleneck_channels(encoder, init_kwargs: dict) -> int:
    """Return the channel count of the encoder bottleneck (= deepest stage).

    Prefers the encoder's runtime `output_channels` attribute, falls back to
    the `features_per_stage` constructor argument. Always validates the value
    so we never silently pass an empty list / 0 to downstream Conv layers.
    """
    out_channels = getattr(encoder, "output_channels", None)
    if out_channels is not None and len(out_channels) > 0:
        ch = int(out_channels[-1])
    else:
        feats = init_kwargs.get("features_per_stage")
        if not feats:
            raise ValueError(
                "Cannot resolve bottleneck channel count: encoder has no "
                "`output_channels` and no `features_per_stage` was provided."
            )
        ch = int(feats[-1])
    if ch <= 0:
        raise ValueError(f"Resolved bottleneck channel count must be positive, got {ch}.")
    return ch


# ---------------------------------------------------------------------------
# Wrapped nnUNetv2 PlainConvUNet with the density bridge attached
# ---------------------------------------------------------------------------


class DensityBridgePlainConvUNet(PlainConvUNet):
    """Official nnUNetv2 2D U-Net with a density cross-attention bridge.

    The encoder and decoder are unchanged. Only the deepest skip (= bottleneck)
    is augmented by a cross-attention residual that consumes a 1-channel
    density input (lung-confined gated distribution).
    """

    def __init__(self, *args, density_base_ch: int = 32, density_in_channels: int = 1,
                 num_heads: int = 4, alpha_init: float = 0.3,
                 out_proj_init: str = 'zero', gate_mode: str = 'scalar',
                 alpha_max: float = 1.0,
                 allow_legacy_density_cache: bool = False,
                 require_aligned_density: bool = True,
                 **kwargs):
        # nnUNet must see image+density as separate input modalities so its
        # preprocessing, augmentation and sliding-window inference transform
        # them with identical spatial coordinates. PlainConvUNet, however,
        # must still receive image channels only. Adapt its constructor here
        # instead of making every trainer variant duplicate this logic.
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
            raise TypeError("DensityBridgePlainConvUNet requires input_channels")

        super().__init__(*args, **kwargs)

        bottleneck_ch = _resolve_bottleneck_channels(self.encoder, kwargs)

        self.total_input_channels = total_input_channels
        self.image_in_channels = image_in_channels
        self.density_in_channels = density_in_channels
        self.allow_legacy_density_cache = bool(allow_legacy_density_cache)
        self.require_aligned_density = bool(require_aligned_density)
        self.density_encoder = DensityEncoder(
            bottleneck_ch=bottleneck_ch,
            in_channels=self.density_in_channels,
            base_ch=density_base_ch,
        )
        # `out_proj_init` newly exposed so trainers can opt into 'xavier'
        # to avoid the residual-starvation collapse (alpha=0 + out_proj=0
        # is a chain-locked dead branch -- gradient through alpha is
        # multiplied by out=0, and gradient through out_proj is multiplied
        # by alpha=0; neither parameter ever escapes the origin). See
        # `CrossAttentionFusion.__init__` for the detailed analysis. The
        # default is kept at 'zero' for backward compatibility, but the
        # density bridge trainer now passes 'xavier' (paired with a small
        # non-zero alpha_init).
        self.fusion = CrossAttentionFusion(
            channels=bottleneck_ch, num_heads=num_heads, alpha_init=alpha_init,
            out_proj_init=out_proj_init, gate_mode=gate_mode, alpha_max=alpha_max,
        )

    def _resolve_density(self, density: Optional[torch.Tensor],
                        case_id: Union[str, List[str], None],
                        device: torch.device,
                        dtype: torch.dtype) -> Optional[torch.Tensor]:
        if density is not None:
            if density.ndim != 4:
                raise ValueError(f"Expected density [B,C,H,W], got {tuple(density.shape)}")
            density = density.to(device=device, dtype=dtype)
            if density.shape[1] != self.density_in_channels:
                # Truncate extra channels (e.g. legacy 2-channel cache loaded
                # through some external path); pad with zeros if fewer.
                bch = density.shape[1]
                if bch > self.density_in_channels:
                    density = density[:, : self.density_in_channels]
                else:
                    pad = self.density_in_channels - bch
                    density = torch.cat([density,
                                         density.new_zeros((density.shape[0],
                                                            pad,
                                                            density.shape[2],
                                                            density.shape[3]))],
                                        dim=1)
            return density

        if case_id is None:
            return None

        if not self.allow_legacy_density_cache:
            raise RuntimeError(
                "Density was not supplied as an aligned nnUNet input modality. "
                "Case-level cache fallback is disabled because cached full images "
                "cannot share nnUNet random augmentation or sliding-window patch "
                "coordinates. Export density as *_0001, re-run planning/"
                "preprocessing, and pass the two-channel tensor to the bridge."
            )

        # case_id can be a single string or a list of strings (one per item).
        if isinstance(case_id, (list, tuple)):
            case_ids = [str(c) for c in case_id]
        else:
            case_ids = [str(case_id)]

        per_item = []
        for cid in case_ids:
            arr = load_density(cid)  # (1, H, W); loader raises on legacy formats
            if arr.ndim != 3 or arr.shape[0] != self.density_in_channels:
                raise ValueError(
                    f"Density cache for {cid!r} has shape {arr.shape}, "
                    f"expected ({self.density_in_channels}, H, W). "
                    f"Re-run build_density_cache.py --overwrite."
                )
            per_item.append(torch.from_numpy(arr))
        batch = torch.stack(per_item, dim=0).to(device=device, dtype=dtype)
        return batch

    def _fuse(self, bottleneck: torch.Tensor, density: Optional[torch.Tensor],
              case_id: Union[str, List[str], None]) -> torch.Tensor:
        density_t = self._resolve_density(density, case_id, bottleneck.device, bottleneck.dtype)
        if density_t is None:
            return bottleneck

        density_feat = self.density_encoder(density_t)
        if density_feat.shape[2:] != bottleneck.shape[2:]:
            density_feat = F.interpolate(density_feat, size=bottleneck.shape[2:],
                                         mode="bilinear", align_corners=False)
        return self.fusion(bottleneck, density_feat)

    def forward(self, x: Optional[torch.Tensor] = None, density: Optional[torch.Tensor] = None,
                case_id: Union[str, List[str], None] = None, **kwargs):
        if x is None:
            x = kwargs.get("image", None)
        if x is None:
            raise TypeError("DensityBridgePlainConvUNet.forward() requires 'x' or 'image' input")
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
        if density is None and self.require_aligned_density:
            raise RuntimeError(
                "Aligned density modality is required but missing. Export it as "
                "nnUNet channel 1 (*_0001) and re-run preprocessing."
            )
        if density is not None and density.shape[0] != x.shape[0]:
            raise ValueError("Image and density batch sizes must match")
        if density is not None and density.shape[2:] != x.shape[2:]:
            raise ValueError(
                "Image and density must already be spatially aligned before the "
                f"bridge; got image={tuple(x.shape)}, density={tuple(density.shape)}"
            )

        skips = self.encoder(x)
        skips = list(skips)
        skips[-1] = self._fuse(skips[-1], density, case_id)
        return self.decoder(skips)
