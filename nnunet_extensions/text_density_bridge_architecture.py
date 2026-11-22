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

_CACHE_MISS_WARNED: "set[Tuple[str, str]]" = set()

def _normalize_case_id_value(cid) -> str:
    if isinstance(cid, Path):
        s = cid.as_posix()
    else:
        s = str(cid)
    s = s.strip()
    if not s:
        raise ValueError("case_id must not be empty / whitespace")
    if "\\" in s:
        s = s.replace("\\", "/")
    return s


def _normalize_case_ids(case_id) -> Optional[List[str]]:
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
                return None
        return out

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
    inner = net
    # DDP unwrap (PyTorch DDP exposes `.module`).
    inner = getattr(inner, "module", inner)
    # torch.compile unwrap (`._orig_mod` is the compiled-module's source).
    inner = getattr(inner, "_orig_mod", inner)
    return inner


def _warn_cache_miss_once(modality: str, cid: str, exc: BaseException) -> None:
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

class TextEncoder(nn.Module):
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

class TextTokenCrossAttention(nn.Module):
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
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self._last_input_norm: Optional[torch.Tensor] = None
        self._last_delta_norm: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, tokens: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

class DualPathBottleneckFusion(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4,
                 alpha_init_d: float = 0.3, alpha_init_t: float = 0.3,
                 out_proj_init: str = 'zero',
                 gate_mode_d: str = 'scalar', alpha_max_d: float = 1.0,
                 out_proj_init_d: Optional[str] = None):
        super().__init__()
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
        self.ca_density._last_input_norm = None
        self.ca_density._last_delta_norm = None
        self.ca_text._last_input_norm = None
        self.ca_text._last_delta_norm = None

        out = bn_image
        if dens_feat is not None:
            out = self.ca_density(out, dens_feat)
        if text_feat is not None:
            out = self.ca_text(out, text_feat)
        return out

def _global_pool(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


def _fg_pool(feat: torch.Tensor, fg_mask_low: torch.Tensor,
             fallback_to_global: bool = True) -> torch.Tensor:
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

class DensityAttentionGate(nn.Module):
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
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)
        self.use_softplus_gamma = bool(use_softplus_gamma)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        if bool(freeze_gamma):
            self.gamma.requires_grad_(False)
        self.freeze_gamma = bool(freeze_gamma)

    @torch.no_grad()
    def set_gamma(self, value: float) -> None:
        self.gamma.data.fill_(float(value))

    @property
    def gamma_effective(self) -> torch.Tensor:
        if self.use_softplus_gamma:
            return F.softplus(self.gamma)
        return self.gamma

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(density))

    def modulate(self, logits: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
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

    if dens_feat is None and text_feat is None:
        return bn_image.new_zeros(())

    def _pool(x):
        if fg_mask_low is not None:
            return _fg_pool(x, fg_mask_low)
        return _global_pool(x)

    v_I = _pool(bn_image)
    z_I = proj_image(v_I).detach()         

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
    if dens_feat is None and text_feat is None:
        return seg_feat.new_zeros(()), None

    v_vis = _fg_pool(seg_feat, fg_mask_full)           
    batch_mean = v_vis.detach().float().mean(0)           

    anchor = v_vis
    if vis_proto is not None and proto_weight > 0.0:
        p = vis_proto.detach().to(v_vis.dtype).unsqueeze(0)         
        anchor = F.normalize(v_vis, dim=-1) + proto_weight * F.normalize(p, dim=-1)

    z_I = proj_seg(anchor).detach()                   

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

class TextDensityBridgePlainConvUNet(PlainConvUNet):
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

        self.projection_dim = int(projection_dim)
        if self.projection_dim > 0:
            self.proj_image = nn.Linear(bottleneck_ch, self.projection_dim, bias=True)
            self.proj_density = nn.Linear(bottleneck_ch, self.projection_dim, bias=True)
            self.proj_text = nn.Linear(bottleneck_ch, self.projection_dim, bias=True)
        else:
            self.proj_image = self.proj_density = self.proj_text = None
        
        self._use_seg_head_align: bool = bool(use_seg_head_align)
        self._capture_seg_head_enabled: bool = False
        self._last_seg_head_feat: Optional[torch.Tensor] = None
        self._vis_proto_alpha: float = 0.99
        self.proj_seg: Optional[nn.Module] = None
        if self._use_seg_head_align and self.projection_dim > 0:
            seg_head_ch = int(self.encoder.output_channels[0])
            self.seg_head_ch = seg_head_ch
            self.proj_seg = nn.Linear(seg_head_ch, self.projection_dim, bias=True)
            self.register_buffer("_vis_proto", torch.zeros(seg_head_ch))
            self.register_buffer("_vis_proto_init", torch.zeros(1))
            try:
                self.decoder.stages[-1].register_forward_hook(self._seg_head_hook)
            except (AttributeError, IndexError):
                pass

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
            
        self._last_hag_logits: Optional[torch.Tensor] = None
        self._last_hag_attn: Optional[torch.Tensor] = None
        self._last_align_loss: Optional[torch.Tensor] = None
        self._last_bn_image: Optional[torch.Tensor] = None
        self._last_dens_feat: Optional[torch.Tensor] = None
        self._last_text_feat: Optional[torch.Tensor] = None
                     
        self._force_drop_density: bool = False
        self._force_drop_text: bool = False
        self._inference_case_id: Optional[Union[str, List[str]]] = None


    def set_inference_case_id(self,
                              case_id: Union[str, List[str], None]) -> None:
        self._inference_case_id = case_id

    def set_modality_drop(self, drop_density: bool = False,
                          drop_text: bool = False) -> None:
        self._force_drop_density = bool(drop_density)
        self._force_drop_text = bool(drop_text)


    def _resolve_density_tensor(
            self, density: Optional[torch.Tensor],
            case_id: Union[str, Path,
                           Sequence[Union[str, Path]], None],
            device: torch.device,
            dtype: torch.dtype) -> Optional[torch.Tensor]:
       
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
            return None

        batch = torch.stack(per_item, dim=0).to(device=device, dtype=dtype)
        return batch

    def _resolve_text_tensor(
            self, text: Optional[torch.Tensor],
            case_id: Union[str, Path,
                           Sequence[Union[str, Path]], None],
            device: torch.device,
            dtype: torch.dtype) -> Optional[torch.Tensor]:

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

    def _make_text_inject_hook(self, inject_module: "TextTokenCrossAttention"):
        def hook(module, inp, out):
            if not self.use_text_token_inject:
                return None
            toks = self._cur_text_tokens
            if toks is None:
                return None
            if isinstance(out, (list, tuple)):
                return None
            return inject_module(out, toks, self._cur_text_mask)

        return hook

    def _resolve_text_tokens(
            self, case_id: Union[str, Path,
                                 Sequence[Union[str, Path]], None],
            device: torch.device,
            dtype: torch.dtype):
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
        mask = torch.ones((B, Lmax), dtype=torch.bool)
        for i, s in enumerate(seqs):
            L = int(s.shape[0])
            tokens[i, :L] = s
            mask[i, :L] = False

        tokens = tokens.to(device=device, dtype=dtype)
        mask = mask.to(device=device)
        return tokens, mask


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
        if case_id is None and self._inference_case_id is not None:
            sticky = self._inference_case_id
            B = int(x.shape[0])
            if isinstance(sticky, (str, Path)):
                case_id = [sticky] * B
            elif isinstance(sticky, (list, tuple)):
                if len(sticky) == 1 and B > 1:
                    case_id = [sticky[0]] * B
                else:
                    case_id = list(sticky)
            else:
                case_id = sticky

        skips = self.encoder(x)
        skips = list(skips)
        bn_image = skips[-1]

        device = bn_image.device
        dtype = bn_image.dtype

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

        if self.use_text_token_inject and not drop_t:
            self._cur_text_tokens, self._cur_text_mask = self._resolve_text_tokens(
                case_id, device, dtype)
        else:
            self._cur_text_tokens, self._cur_text_mask = None, None
        if self.training:
            self._last_align_loss = cross_modal_alignment_loss(
                bn_image, dens_feat, text_feat
            )
            self._last_bn_image = bn_image
            self._last_dens_feat = dens_feat
            self._last_text_feat = text_feat
        else:
            self._last_align_loss = None
            self._last_bn_image = None
            self._last_dens_feat = None
            self._last_text_feat = None

        out = self.decoder(skips)

        self._cur_text_tokens = None
        self._cur_text_mask = None

        if self.density_hag is not None and density_t is not None:
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


    def pop_alignment_loss(self) -> Optional[torch.Tensor]:
        loss = self._last_align_loss
        self._last_align_loss = None
        return loss

    def pop_bottleneck_features(self):
        bn = self._last_bn_image
        d = self._last_dens_feat
        t = self._last_text_feat
        self._last_bn_image = None
        self._last_dens_feat = None
        self._last_text_feat = None
        return bn, d, t

 
    def _seg_head_hook(self, module, inp, out):
        if (self._capture_seg_head_enabled and self.training
                and torch.is_grad_enabled()):
            self._last_seg_head_feat = out
        else:
            self._last_seg_head_feat = None

    def enable_seg_head_capture(self, enabled: bool = True) -> None:
        self._capture_seg_head_enabled = bool(enabled)

    def pop_seg_head_feat(self) -> Optional[torch.Tensor]:
        f = self._last_seg_head_feat
        self._last_seg_head_feat = None
        return f

    @torch.no_grad()
    def update_vis_prototype(self, vec: torch.Tensor) -> None:
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
        if getattr(self, "proj_seg", None) is None or not hasattr(self, "_vis_proto"):
            return None
        if float(self._vis_proto_init.item()) < 0.5:
            return None
        return self._vis_proto

    def pop_hag_attention(self) -> Optional[torch.Tensor]:
        attn = self._last_hag_attn
        self._last_hag_attn = None
        return attn

    def pop_hag_logits(self) -> Optional[torch.Tensor]:
        logits = self._last_hag_logits
        self._last_hag_logits = None
        return logits


    def text_inject_far(self) -> "dict[str, float]":
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
