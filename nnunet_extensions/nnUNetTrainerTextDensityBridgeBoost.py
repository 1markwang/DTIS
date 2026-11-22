from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTextDensityBridge import (
    nnUNetTrainerTextDensityBridge,
)
from nnunetv2.training.nnUNetTrainer.text_density_bridge_architecture import (
    cross_modal_alignment_loss_fg,
    unwrap_network,
)
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerTextDensityBridgeBoost(nnUNetTrainerTextDensityBridge):
    align_loss_weight: float = 0.3
    text_align_weight: float = 0.3

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.66

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()

        loss = DC_and_CE_loss(
            soft_dice_kwargs={
                'batch_dice': False,                 
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

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
            if hasattr(net_inner, "pop_bottleneck_features"):
                bn_image, dens_feat, text_feat = \
                    net_inner.pop_bottleneck_features()
                if hasattr(net_inner, "pop_alignment_loss"):
                    _ = net_inner.pop_alignment_loss()
                if bn_image is not None:
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
                    l_align = cross_modal_alignment_loss_fg(
                        bn_image, dens_feat, text_feat,
                        fg_mask_low=fg_mask_low,
                        detach_image=True,
                        text_weight=float(self.text_align_weight),
                    )
            else:
                if hasattr(net_inner, "pop_alignment_loss"):
                    l_align = net_inner.pop_alignment_loss()

            if l_align is not None:
                l = l_seg + float(self.align_loss_weight) * l_align
            else:
                l = l_seg

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

class nnUNetTrainerTextDensityBridgeBoost_50epochs(
        nnUNetTrainerTextDensityBridgeBoost):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerTextDensityBridgeBoost_100epochs(
        nnUNetTrainerTextDensityBridgeBoost):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerTextDensityBridgeBoost_250epochs(
        nnUNetTrainerTextDensityBridgeBoost):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
