# -*- coding: utf-8 -*-
"""nnUNetv2 *V2* trainer for the text + density bridge.

V2 fixes the structural failure mode diagnosed in `_diag_pipeline.py`:

  > "residual starvation" / "self-suppressed residual learning collapse"
  >
  > After 78 epochs of `nnUNetTrainerTextDensityBridgeBoost`:
  >   * density / text encoders trained (norms ~30-35)
  >   * fusion Q/K/V trained (norms ~9.0)
  >   * fusion.ca_*.out_proj.weight = 0.20 / 0.14 (zero init, never escaped)
  >   * alpha_d / alpha_t moved 0.06 / 0.02 (init 0.3)
  >   * forward-pass ablation: density+text changes < 0.025 % of pixels
  >
  > Conclusion: multimodal bridge is functionally dead. Almost all observed
  > Boost-vs-baseline gain comes from `batch_dice=False`, not multimodal.

V2 makes 3 coordinated structural changes:

  1. **Fusion-side fix** (`out_proj_init='xavier'`, `alpha_init=0.05`).
     Replaces the zero-init residual with a *weak-but-active* one. With
     alpha=0.05 the day-0 contribution is ~5 % of the trunk's magnitude --
     small enough not to destabilise the strong baseline encoder, large
     enough that d(seg_loss)/d(out_proj) is non-trivial and the optimiser
     can either grow or shrink the residual based on real learning signal.

  2. **Alignment-side fix** (projection heads + stop-grad + cosine +
     warm-up + small lambda). Replaces the legacy fg-pooled raw cosine
     with the canonical SimSiam-style design:
       - 1-layer Linear projection P_I/D/T (default `projection_dim=128`)
       - stop-gradient on the image side (image-anchored)
       - cosine similarity on projected vectors (NOT MSE)
       - linear warm-up over the first 8 epochs (lambda(t) = lambda_max * min(t/8, 1))
       - lambda_max = 0.10 (was 0.30 in Boost, 0.50 in legacy)
     The alignment loss is now a real regulariser (~1-3 % of seg_loss)
     instead of either dominating (0.50 case) or being saturated useless
     (0.30 with raw cosine -> cos ~ 0.94, gradient ~ 0).

  3. **Diagnostic logging** (FAR, alpha, out_proj norms per epoch). The
     Fusion Activation Ratio
              FAR = || alpha * delta || / || bn_image ||
     is the right metric for "did the multimodal residual actually
     contribute". Logged per epoch so we can plot it as a paper figure.

What is preserved from Boost (so V2 is a clean A->B comparison):
  * DC + CE loss with `batch_dice=False`
  * lru-cached density / text loaders
  * `_extract_inputs`, periodic 10-epoch checkpoints, SGD, PolyLR, AMP
  * nnU-Net's deep supervision pipeline
  * Cascaded fusion order (density refines query, then text attends)
  * All data augmentation defaults
"""

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
    """V2 trainer. See module docstring for the full rationale."""

    # ---- alignment hyper-parameters --------------------------------------
    # Max alignment-loss weight at the end of warm-up. Empirical sweep:
    #   0.02  -> too small (0.3 % of seg loss, basically inactive)
    #   0.05  -> minimum useful (~ 0.7 %)
    #   0.10  -> recommended  (~ 1.4 %)
    #   0.30  -> Boost default, fg cosine saturates -> gradient ~ 0
    #   0.50  -> legacy, fights segmentation loss
    align_lambda_max: float = 0.10
    # Linear warm-up length (in epochs). Alignment starts at 0 and ramps to
    # `align_lambda_max` over this many epochs. 8 is a good balance: long
    # enough that the seg branch finds a good local minimum before alignment
    # bites, short enough that alignment is fully active by mid-training.
    align_warmup_epochs: int = 8
    # Text branch weight relative to density branch in the alignment loss.
    text_align_weight: float = 1.0

    # ---- fusion-side hyper-parameters (passed to the network) -------------
    # `out_proj_init='xavier'` + `alpha_init=0.05` is the V2 fix for the
    # residual-starvation collapse documented at the top of this file.
    out_proj_init: str = 'xavier'
    alpha_init_d: float = 0.05
    alpha_init_t: float = 0.05
    projection_dim: int = 128

    # `align_loss_weight` on the network itself is ignored in V2 because
    # we compute alignment externally with our own warm-up; but we still
    # pass *something* into the architecture constructor.
    network_align_weight: float = 0.0

    # --- network ----------------------------------------------------------

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
            # multimodal branches (same as Boost)
            density_in_channels=1,
            density_base_ch=32,
            text_in_dim=768,
            text_hidden=256,
            text_dropout=0.1,
            num_heads=4,
            # V2 changes:
            alpha_init_d=float(nnUNetTrainerTextDensityBridgeV2.alpha_init_d),
            alpha_init_t=float(nnUNetTrainerTextDensityBridgeV2.alpha_init_t),
            out_proj_init=str(nnUNetTrainerTextDensityBridgeV2.out_proj_init),
            projection_dim=int(nnUNetTrainerTextDensityBridgeV2.projection_dim),
            # V2 owns alignment, so the legacy in-network alignment is silent.
            align_loss_weight=float(nnUNetTrainerTextDensityBridgeV2.network_align_weight),
        )

    # --- train step (projected alignment with warm-up) --------------------

    def _alignment_weight(self) -> float:
        """Linear warm-up: 0 -> `align_lambda_max` over `align_warmup_epochs`."""
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

            # DDP / torch.compile wraps `self.network`; unwrap once so we
            # can access bridge-specific attributes (`pop_*`, projection
            # heads). Without this, DDP single-GPU works only by accident
            # (DDP wrap is skipped when world_size==1).
            net_inner = unwrap_network(self.network)

            # V2 projected alignment loss with linear warm-up.
            l_align: Optional[torch.Tensor] = None
            lam = self._alignment_weight()
            if lam > 0.0 and hasattr(net_inner, "pop_bottleneck_features"):
                bn_image, dens_feat, text_feat = \
                    net_inner.pop_bottleneck_features()
                # Drop the legacy precomputed alignment loss (V2 overrides it).
                if hasattr(net_inner, "pop_alignment_loss"):
                    _ = net_inner.pop_alignment_loss()
                if bn_image is not None and (dens_feat is not None or text_feat is not None):
                    proj_I = getattr(net_inner, "proj_image", None)
                    proj_D = getattr(net_inner, "proj_density", None)
                    proj_T = getattr(net_inner, "proj_text", None)
                    if proj_I is None or proj_D is None or proj_T is None:
                        # Mis-configured network (no projection heads); skip
                        # alignment rather than silently fall back to a
                        # different formulation that distorts our ablation.
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

            # Accumulate per-iteration FAR (mean over batch) so on_epoch_end
            # can summarise. Done inside autocast for free; values are
            # detached scalars so AMP does not flag them.
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

    # --- FAR diagnostic logging --------------------------------------------

    def _ensure_far_buffers(self):
        if not hasattr(self, "_far_d_sum"):
            self._far_d_sum = 0.0
            self._far_t_sum = 0.0
            # Per-branch counters: under Modality Dropout (V5), density
            # and text are active in different *subsets* of batches; a
            # single shared counter would average them against each
            # other's missing-batches and produce misleadingly small
            # FAR values. See `_accumulate_far` for the per-branch logic.
            self._far_d_count = 0
            self._far_t_count = 0

    def _accumulate_far(self):
        """Pull the last-call FAR from the fusion modules and accumulate.

        FAR (Fusion Activation Ratio) = ||alpha * delta|| / ||bn_image||,
        measured per fusion branch. We deliberately track density and
        text counters SEPARATELY now: under V5 Modality Dropout some
        batches have only density, some only text, some neither -- the
        old shared `_far_count` would skew the means. Each modality's
        mean is now `sum / its-own-count`, fairly representing only the
        batches where that modality was actually active.
        """
        self._ensure_far_buffers()
        # IMPORTANT: must unwrap DDP / compile to reach `.fusion`. The
        # old code did `net.fusion.ca_density.<x>` and caught
        # `AttributeError` silently -- under DDP that means FAR is
        # permanently zero, which we mistake for "the residual is dead"
        # when in fact the metric itself is broken.
        net = unwrap_network(self.network)
        try:
            d_in = net.fusion.ca_density._last_input_norm
            d_dt = net.fusion.ca_density._last_delta_norm
            t_in = net.fusion.ca_text._last_input_norm
            t_dt = net.fusion.ca_text._last_delta_norm
        except AttributeError:
            return
        # Each branch was reset to None at the start of this batch's
        # forward (see `DualPathBottleneckFusion.forward`). They are
        # *only* non-None when the corresponding modality was actually
        # consumed in this batch. Per-branch counters keep the means
        # honest under V5 Modality Dropout.
        if d_in is not None and d_dt is not None:
            self._far_d_sum += float((d_dt / d_in.clamp_min(1e-6)).mean().item())
            self._far_d_count += 1
        if t_in is not None and t_dt is not None:
            self._far_t_sum += float((t_dt / t_in.clamp_min(1e-6)).mean().item())
            self._far_t_count += 1

    # --- on_epoch_end: extra diagnostics -----------------------------------

    def on_epoch_end(self):
        # Compute mean FAR over this epoch before the parent advances state.
        self._ensure_far_buffers()
        far_d = self._far_d_sum / max(1, self._far_d_count)
        far_t = self._far_t_sum / max(1, self._far_t_count)
        # Unwrap DDP / torch.compile to reach the inner module's
        # `fusion.ca_*` attributes (same reason as in `_accumulate_far`).
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
        # Reset FAR accumulators for next epoch.
        self._far_d_sum = 0.0
        self._far_t_sum = 0.0
        self._far_d_count = 0
        self._far_t_count = 0

        super().on_epoch_end()


# --- short-schedule variants ------------------------------------------------


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
