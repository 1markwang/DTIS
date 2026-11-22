from __future__ import annotations
from os.path import join
from pydoc import locate
from typing import List, Optional
import torch
import torch.nn as nn
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.text_density_bridge_architecture import (
    TextDensityBridgePlainConvUNet,
    unwrap_network,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

DEFAULT_TEXT_DIM = 768
DEFAULT_ALIGN_WEIGHT = 0.5


class nnUNetTrainerTextDensityBridge(nnUNetTrainer):
    text_in_dim: int = DEFAULT_TEXT_DIM
    align_loss_weight: float = DEFAULT_ALIGN_WEIGHT
    ckpt_save_interval: int = 10
    
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        _ = plans_manager
        _ = configuration_manager
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
            text_in_dim=int(nnUNetTrainerTextDensityBridge.text_in_dim),
            text_hidden=256,
            text_dropout=0.1,
            num_heads=4,
            alpha_init_d=0.3,
            alpha_init_t=0.3,
            align_loss_weight=float(nnUNetTrainerTextDensityBridge.align_loss_weight),
        )

  
    def configure_optimizers(self):
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        no_decay_names: List[str] = []
        for name, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".alpha") or p.ndim <= 1:
                no_decay_params.append(p)
                no_decay_names.append(name)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params,    "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.SGD(
            param_groups,
            lr=self.initial_lr,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        self.print_to_log_file(
            f"[bridge] excluded {len(no_decay_params)} param tensors from weight decay "
            f"(alpha and 1-D); decay group has {len(decay_params)} tensors."
        )
        for n in no_decay_names:
            if n.endswith(".alpha"):
                self.print_to_log_file(f"[bridge] no-decay alpha param: {n}")
        return optimizer, lr_scheduler

    def on_epoch_end(self):
        epoch_just_finished = self.current_epoch
        super().on_epoch_end()

        if self.ckpt_save_interval <= 0:
            return
        if self.local_rank != 0 or self.disable_checkpointing:
            return
        if epoch_just_finished == self.num_epochs - 1:
            return
        if (epoch_just_finished + 1) % self.ckpt_save_interval == 0:
            ckpt_name = f'checkpoint_epoch{epoch_just_finished + 1:04d}.pth'
            self.save_checkpoint(join(self.output_folder, ckpt_name))
            self.print_to_log_file(
                f"[periodic-save] saved independent ckpt: {ckpt_name}")

    def _extract_inputs(self, batch: dict):
        if "image" in batch:
            image = batch["image"]
        elif "data" in batch:
            data = batch["data"]
            if data.ndim != 4 or data.shape[1] < 1:
                raise ValueError(f"Unexpected data tensor shape: {tuple(data.shape)}")
            image = data[:, :1]
        else:
            raise KeyError("batch must contain 'image' or 'data'")

        density = batch.get("density", None)
        if density is None and "data" in batch:
            data = batch["data"]
            if data.ndim == 4 and data.shape[1] >= 2:
                density = data[:, 1:2]

        text = batch.get("text", None)

        case_id = batch.get("case_id", batch.get("keys", None))
        if case_id is None:
            raise KeyError("batch must contain 'case_id' or 'keys' for the bridge")
        if isinstance(case_id, (list, tuple)):
            case_id = [str(i) for i in case_id]
        elif hasattr(case_id, "shape"):
            try:
                case_id = [str(i) for i in case_id.tolist()]
            except Exception:
                case_id = [str(case_id)]
        else:
            case_id = [str(case_id)]
        return image, density, text, case_id

    def _move_target(self, target):
        if isinstance(target, list):
            return [t.to(self.device, non_blocking=True) for t in target]
        return target.to(self.device, non_blocking=True)

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
            l_align = net_inner.pop_alignment_loss() if hasattr(
                net_inner, "pop_alignment_loss") else None
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

    def validation_step(self, batch: dict) -> dict:
        image, density, text, case_ids = self._extract_inputs(batch)
        target = batch["target"]

        image = image.to(self.device, non_blocking=True)
        if density is not None:
            density = density.to(self.device, non_blocking=True)
        if text is not None:
            text = text.to(self.device, non_blocking=True)
        target = self._move_target(target)

        ctx = (torch.autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
        with ctx:
            output = self.network(image=image, density=density, text=text,
                                   case_id=case_ids)
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float16
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard,
        }


    def perform_actual_validation(self, save_probabilities: bool = False):
        inner = unwrap_network(self.network)
        if not hasattr(inner, "set_inference_case_id"):
            return super().perform_actual_validation(save_probabilities)

        base_cls = self.dataset_class
        if base_cls is None:
            return super().perform_actual_validation(save_probabilities)

        network_ref = inner  # captured by the subclass below

        class _CaseIdAwareDataset(base_cls):
            def load_case(self, k):
                # Set BEFORE delegating, so any side-effect during
                # super().load_case (e.g. logging) cannot fire a forward
                # without the case_id. We pass the raw scalar string; the
                # architecture's forward() will broadcast to batch dim.
                network_ref.set_inference_case_id(k)
                return super().load_case(k)

        self.dataset_class = _CaseIdAwareDataset
        try:
            self.print_to_log_file(
                "[bridge-val] perform_actual_validation: injecting case_id "
                "into nnUNetPredictor path via dataset wrapper "
                "(multimodal final-validation enabled)."
            )
            return super().perform_actual_validation(save_probabilities)
        finally:
            self.dataset_class = base_cls
            network_ref.set_inference_case_id(None)

class nnUNetTrainerTextDensityBridge_50epochs(nnUNetTrainerTextDensityBridge):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerTextDensityBridge_100epochs(nnUNetTrainerTextDensityBridge):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerTextDensityBridge_250epochs(nnUNetTrainerTextDensityBridge):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
