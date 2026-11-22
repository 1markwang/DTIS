from __future__ import annotations
import math
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.nnUNetTrainer.density_cache_utils import load_density

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

class DensityEncoder(nn.Module):
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

class ConditionedGate(nn.Module):
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

class CrossAttentionFusion(nn.Module):
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
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
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
        self._last_input_norm: Optional[torch.Tensor] = None
        self._last_delta_norm: Optional[torch.Tensor] = None
        self._last_gate_mean: Optional[torch.Tensor] = None

    def forward(self, bottleneck: torch.Tensor, density_feat: torch.Tensor) -> torch.Tensor:   
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

class DensityBridgePlainConvUNet(PlainConvUNet):
    def __init__(self, *args, density_base_ch: int = 32, density_in_channels: int = 1,
                 num_heads: int = 4, alpha_init: float = 0.3,
                 out_proj_init: str = 'zero', gate_mode: str = 'scalar',
                 alpha_max: float = 1.0,
                 allow_legacy_density_cache: bool = False,
                 require_aligned_density: bool = True,
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
