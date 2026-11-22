# -*- coding: utf-8 -*-
"""nnUNetv2 trainer for the image-anchored dual-path (text + density) bridge.

Usage:
    nnUNetv2_train <DATASET> <CONFIG> <FOLD> -tr nnUNetTrainerTextDensityBridge
    nnUNetv2_train <DATASET> <CONFIG> <FOLD> -tr nnUNetTrainerTextDensityBridge_100epochs

Both density and text are loaded via their case-id-aligned caches at
forward time (no I/O changes to the official nnUNet data loader). When
the cache for a modality is missing, that path is silently disabled and
training continues with whichever modalities ARE available — useful when
text annotations cover only a subset of cases.

The total loss is:
    L = L_seg(output, target) + align_weight * align_loss

where align_loss is the cosine-distance alignment between pooled image
bottleneck features and the per-branch encoded features. Set
`align_weight` to 0 to disable the alignment term.
"""

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


# Default text-embedding dim. PubMedBERT / BioBERT / BERT-base are all 768.
# Override per subclass if you use a different encoder.
DEFAULT_TEXT_DIM = 768
# Increased from 0.1 -> 0.5. With alpha now starting at 1.0 (no tanh) and
# out_proj zero-initialised, the bridge branches start contributing
# immediately; a stronger alignment signal is needed to keep
# density/text bottleneck features aligned with the image features so that
# the additive contribution remains useful rather than noisy.
DEFAULT_ALIGN_WEIGHT = 0.5


class nnUNetTrainerTextDensityBridge(nnUNetTrainer):
    """Official nnUNetv2 trainer that builds the dual-path bridge network."""

    text_in_dim: int = DEFAULT_TEXT_DIM
    align_loss_weight: float = DEFAULT_ALIGN_WEIGHT
    # Save an INDEPENDENT (non-overwriting) checkpoint every this many epochs,
    # in addition to the EMA-pseudo-dice-based `checkpoint_best.pth`. Set to 0
    # to disable. Used so we can evaluate real Mean Validation Dice on every
    # mid/late-training snapshot after training and pick the truly best one
    # (EMA pseudo dice is patch-aggregated and can overestimate by 0.05-0.13).
    ckpt_save_interval: int = 10

    # --- network -----------------------------------------------------------

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

    # --- optimizer ----------------------------------------------------------

    def configure_optimizers(self):
        """Build SGD with the bridge's `alpha` scalars excluded from weight decay.

        The bridge uses learnable per-branch gates (`fusion.ca_density.alpha`,
        `fusion.ca_text.alpha`). They are single scalars and are critical
        for letting the additive cross-attention branches contribute. They
        must NOT decay toward zero; doing so was the main failure mode of
        the previous configuration (`alpha` stayed near 0 for 60+ epochs).
        """
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        no_decay_names: List[str] = []
        for name, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            # Exclude any module-level 'alpha' scalar from weight decay.
            # Also exclude 1-D buffers like norm/bias to follow common
            # PyTorch best practice (small effect on nnUNet but harmless).
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

    # --- periodic independent checkpoint ----------------------------------

    def on_epoch_end(self):
        """Override to additionally save an independent (epoch-tagged) ckpt
        every `ckpt_save_interval` epochs. The base class already handles
        `checkpoint_latest.pth` and `checkpoint_best.pth`; we just persist
        extra snapshots without overwriting them.
        """
        # Determine which epoch is about to be "completed" (super() will then
        # advance self.current_epoch). Match base class semantics that uses
        # `(current_epoch + 1) % save_every == 0` BEFORE the increment.
        epoch_just_finished = self.current_epoch
        super().on_epoch_end()

        if self.ckpt_save_interval <= 0:
            return
        if self.local_rank != 0 or self.disable_checkpointing:
            return
        # Skip the very last epoch — base class will save `checkpoint_final.pth`.
        if epoch_just_finished == self.num_epochs - 1:
            return
        if (epoch_just_finished + 1) % self.ckpt_save_interval == 0:
            ckpt_name = f'checkpoint_epoch{epoch_just_finished + 1:04d}.pth'
            self.save_checkpoint(join(self.output_folder, ckpt_name))
            self.print_to_log_file(
                f"[periodic-save] saved independent ckpt: {ckpt_name}")

    # --- batch handling ----------------------------------------------------

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

    # --- train / val steps -------------------------------------------------

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
        # NOTE: never `torch.no_grad()` in a training step -- that breaks
        # the autograd graph on non-CUDA devices. nnUNet's standard
        # `dummy_context()` is a no-op that preserves gradients.
        ctx = (torch.autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
        with ctx:
            output = self.network(image=image, density=density, text=text,
                                   case_id=case_ids)
            l_seg = self.loss(output, target)
            # DDP / torch.compile wraps `self.network` so custom methods
            # (`pop_alignment_loss`) live on the *inner* module. Unwrap.
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

    # --- final validation: inject case_id into the predictor path ---------

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Override to make `nnUNetPredictor`'s sliding-window inference
        aware of the current case_id, so the bridge network can load
        density / text from cache during final validation.

        Problem solved
        --------------
        `nnUNetPredictor._internal_maybe_mirror_and_predict` calls
        `self.network(x)` positional-only, so the bridge sees
        `case_id=None` and skips both density and text branches. The
        reported "Mean Validation Dice" then secretly reflects an
        IMAGE-ONLY model -- regardless of how strongly multimodal the
        training was. (Concrete evidence: V5 had FAR_d=0.26 and Pseudo
        Dice 0.8606 at training time but the same 0.789 final Val Dice
        as image-only V31, because the predictor never saw the density
        cache.)

        Injection strategy (chosen for minimal LoC over copy-pasting
        ~130 lines of the parent method)
        --------------------------------
        Temporarily SUBCLASS `self.dataset_class` so that its
        `load_case(k)` method sticky-sets `case_id=k` on the bridge
        network before returning the data. The parent's case loop calls
        `dataset_val.load_case(k)` exactly once per case, so this fires
        right before the first `predictor.predict_sliding_window(...)`
        call for that case -- precisely the right moment.

        Cleanup is RAII-safe: `try/finally` restores the original class
        and clears the sticky id even if the parent method raises.
        """
        # Resolve the inner module so we can call its sticky API even
        # under DDP / torch.compile wrappers.
        inner = unwrap_network(self.network)
        if not hasattr(inner, "set_inference_case_id"):
            # Architecture without sticky-case-id support -> fall back to
            # parent (image-only final val). Should not happen with our
            # `TextDensityBridgePlainConvUNet`.
            return super().perform_actual_validation(save_probabilities)

        base_cls = self.dataset_class
        if base_cls is None:
            # Trainer was never fully initialised; let the parent raise
            # the proper error message.
            return super().perform_actual_validation(save_probabilities)

        network_ref = inner  # captured by the subclass below

        class _CaseIdAwareDataset(base_cls):
            """Wrapper that sticky-sets the bridge network's inference
            case_id every time `load_case(k)` is called. Inherits
            everything else from the real dataset class so it remains
            a drop-in replacement."""

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


# --- short-schedule variants for the 3-day deadline ---------------------------


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
