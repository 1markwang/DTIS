from __future__ import annotations
from pydoc import locate
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTextDensityBridgeBoost import (
    nnUNetTrainerTextDensityBridgeBoost,
)
from nnunetv2.training.nnUNetTrainer.text_density_bridge_architecture import (
    TextDensityBridgePlainConvUNet,
    cross_modal_alignment_loss_projected,
    unwrap_network,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


class nnUNetTrainerTextDensityBridgeV2(nnUNetTrainerTextDensityBridgeBoost): 
    out_proj_init: str = 'xavier'
    alpha_init_d: float = 0.05
    alpha_init_t: float = 0.05
    projection_dim: int = 128
    network_align_weight: float = 0.0

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        _ = plans_manager
        _ = num_input_channels

        arch = configuration_manager.network_arch_init_kwargs
        conv_op = locate(arch["conv_op"]) if isinstance(arch["conv_op"], str) else arch["conv_op"]
        norm_op = locate(arch["norm_op"]) if isinstance(arch.get("norm_op"), str) else arch.get("norm_op")
        dropout_op = locate(arch["dropout_op"]) if isinstance(arch.get("dropout_op"), str) else arch.get("dropout_op")
        nonlin = locate(arch["nonlin"]) if isinstance(arch.get("nonlin"), str) else arch.get("nonlin")
        n_conv_per_stage = arch.get("n_conv_per_stage", arch.get("n_blocks_per_stage"))

        return TextDensityBridgePlainConvUNet(
            input_channels=num_input_channels,
            n_stages=len(arch["features_per_stage"]),
            features_per_stage=arch["features_per_stage"],
            conv_op=conv_op,
            kernel_sizes=arch["kernel_sizes"],
            strides=arch["strides"],
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=arch["n_conv_per_stage_decoder"],
            conv_bias=arch.get("conv_bias", False),
            norm_op=norm_op,
            norm_op_kwargs=arch.get("norm_op_kwargs", None),
            dropout_op=dropout_op,
            dropout_op_kwargs=arch.get("dropout_op_kwargs", None),
            nonlin=nonlin,
            nonlin_kwargs=arch.get("nonlin_kwargs", None),
            deep_supervision=enable_deep_supervision,
            density_in_channels=1,
            density_base_ch=32,
            text_in_dim=768,
            text_hidden=256,
            text_dropout=0.1,
            num_heads=4,
            alpha_init_d=float(nnUNetTrainerTextDensityBridgeV2.alpha_init_d),
            alpha_init_t=float(nnUNetTrainerTextDensityBridgeV2.alpha_init_t),
            out_proj_init=str(nnUNetTrainerTextDensityBridgeV2.out_proj_init),
            projection_dim=int(nnUNetTrainerTextDensityBridgeV2.projection_dim),
            align_loss_weight=float(nnUNetTrainerTextDensityBridgeV2.network_align_weight),
        )


    def _alignment_weight(self) -> float:
        if self.align_warmup_epochs <= 0:
            return float(self.align_lambda_max)
        t = float(self.current_epoch)
        ratio = min(t / float(self.align_warmup_epochs), 1.0)
        return float(self.align_lambda_max) * ratio

    def train_step(self, batch: dict) -> dict:
        image, density, text, case_ids = self._extract_inputs(batch)
        target = batch["target"]

        image = image.to(self.device, non_blocking=True)
        if density is not None:
            density = density.to(self.device, non_blocking=True)
        if text is not None:
            text = text.to(self.device, non_blocking=True)
        target = self._move_target(target)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = (torch.autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
        with ctx:
            output = self.network(image=image, density=density, text=text,
                                   case_id=case_ids)
            l_seg = self.loss(output, target)
            net_inner = unwrap_network(self.network)
            l_align: Optional[torch.Tensor] = None
            lam = self._alignment_weight()
            if lam > 0.0 and hasattr(net_inner, "pop_bottleneck_features"):
                bn_image, dens_feat, text_feat = \
                    net_inner.pop_bottleneck_features()
            
                if hasattr(net_inner, "pop_alignment_loss"):
                    _ = net_inner.pop_alignment_loss()
                if bn_image is not None and (dens_feat is not None or text_feat is not None):
                    proj_I = getattr(net_inner, "proj_image", None)
                    proj_D = getattr(net_inner, "proj_density", None)
                    proj_T = getattr(net_inner, "proj_text", None)
                    if proj_I is None or proj_D is None or proj_T is None:
                        l_align = None
                    else:
                        hi_target = target[0] if isinstance(target, list) else target
                        if hi_target.ndim == 3:
                            hi_target = hi_target.unsqueeze(1)
                        fg_mask = (hi_target > 0).to(dtype=bn_image.dtype)
                        bn_hw = bn_image.shape[2:]
                        if fg_mask.shape[2:] != bn_hw:
                            fg_mask_low = F.interpolate(
                                fg_mask, size=bn_hw, mode="nearest")
                        else:
                            fg_mask_low = fg_mask
                        l_align = cross_modal_alignment_loss_projected(
                            bn_image, dens_feat, text_feat,
                            fg_mask_low=fg_mask_low,
                            proj_image=proj_I,
                            proj_density=proj_D,
                            proj_text=proj_T,
                            text_weight=float(self.text_align_weight),
                        )

            if l_align is not None:
                l = l_seg + lam * l_align
            else:
                l = l_seg
              
            self._accumulate_far()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def _ensure_far_buffers(self):
        if not hasattr(self, "_far_d_sum"):
            self._far_d_sum = 0.0
            self._far_t_sum = 0.0
            self._far_d_count = 0
            self._far_t_count = 0

    def _accumulate_far(self):
        self._ensure_far_buffers()
        net = unwrap_network(self.network)
        try:
            d_in = net.fusion.ca_density._last_input_norm
            d_dt = net.fusion.ca_density._last_delta_norm
            t_in = net.fusion.ca_text._last_input_norm
            t_dt = net.fusion.ca_text._last_delta_norm
        except AttributeError:
            return
          
        if d_in is not None and d_dt is not None:
            self._far_d_sum += float((d_dt / d_in.clamp_min(1e-6)).mean().item())
            self._far_d_count += 1
        if t_in is not None and t_dt is not None:
            self._far_t_sum += float((t_dt / t_in.clamp_min(1e-6)).mean().item())
            self._far_t_count += 1
          
    def on_epoch_end(self):
        self._ensure_far_buffers()
        far_d = self._far_d_sum / max(1, self._far_d_count)
        far_t = self._far_t_sum / max(1, self._far_t_count)
        net = unwrap_network(self.network)
        try:
            alpha_d = float(net.fusion.ca_density.alpha.item())
            alpha_t = float(net.fusion.ca_text.alpha.item())
            op_d = float(net.fusion.ca_density.out_proj.weight.detach().float().norm().item())
            op_t = float(net.fusion.ca_text.out_proj.weight.detach().float().norm().item())
            lam = self._alignment_weight()
            self.print_to_log_file(
                f"[V2-diag] epoch {self.current_epoch:3d}  "
                f"FAR_d={far_d:.4f} (n={self._far_d_count})  "
                f"FAR_t={far_t:.4f} (n={self._far_t_count})  "
                f"alpha_d={alpha_d:+.4f}  alpha_t={alpha_t:+.4f}  "
                f"|W_O^d|={op_d:.3f}  |W_O^t|={op_t:.3f}  "
                f"lam_align={lam:.3f}"
            )
        except AttributeError:
            pass
        self._far_d_sum = 0.0
        self._far_t_sum = 0.0
        self._far_d_count = 0
        self._far_t_count = 0

        super().on_epoch_end()


class nnUNetTrainerTextDensityBridgeV2_50epochs(nnUNetTrainerTextDensityBridgeV2):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerTextDensityBridgeV2_100epochs(nnUNetTrainerTextDensityBridgeV2):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerTextDensityBridgeV2_250epochs(nnUNetTrainerTextDensityBridgeV2):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
