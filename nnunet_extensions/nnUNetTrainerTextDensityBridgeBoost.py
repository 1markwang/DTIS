# -*- coding: utf-8 -*-
"""nnUNetv2 *Boost* trainer for the text + density bridge.

This trainer keeps the EXACT same network architecture as
`nnUNetTrainerTextDensityBridge` (6-stage encoder, deep supervision, official
data augmentation, official optimizer / LR schedule). It only changes the
parts that empirically hurt small-lesion performance on the CXR-PCAT task:

  1. **`batch_dice = False`**  — switches the Dice loss from the
     batch-pooled (micro) formulation to the per-sample formulation.
     Under the project's training regime (batch_size=65, patch=224, batch_dice
     pools 3.26 M pixels into one Dice; a 300-px lesion contributes
     0.009 % → its gradient is numerically zero). Per-sample Dice gives every
     case an equal vote, so small lesions get real gradient.

  2. **DC + CE preserved (TopK rejected after empirical failure)** — an
     earlier draft used `DC_and_topk_loss` (top-10 % hardest pixels). It
     **catastrophically collapsed the model to the all-background degenerate
     solution within 3 epochs** (epoch 0 Dice 0.005 → epoch 2 Dice 0.001):
     at random init every pixel has CE ≈ 0.69, and on PCAT background
     outnumbers foreground by ~30:1, so the top-10 % pixels selected are
     almost exclusively background. The net thus receives the gradient
     "predict background everywhere". This trainer therefore keeps plain
     DC + CE — Dice's per-sample fg-recall term provides a strong, robust
     foreground gradient that pairs well with `batch_dice = False`.

  3. **`oversample_foreground_percent = 0.66`** (up from 0.33). Under the
     'patch_size == median_image_size' regime this knob has no effect via
     the patch sampler (no random cropping happens), but it is preserved
     here for any future reconfiguration that does use sub-image patches.

  4. **Foreground-pooled alignment loss** — replaces the legacy global-avg-
     pooled cosine alignment with a foreground-aware version computed
     externally in `train_step`:
       * `bn_image` is `.detach()`-ed → text / density learn to align to the
         image, not the other way around (no distortion of image encoder).
       * Pool is taken ONLY over the projected lesion region (GT mask
         downsampled to bottleneck resolution); cosine therefore encodes
         "what the lesion looks like" instead of "average pixel".
       * Text branch weight defaults to 0 (its broadcast-uniform feature
         makes the cosine alignment trivial; setting weight > 0 just
         injects free noise).

What is preserved verbatim (so the experiment is a clean ablation):

  * The network architecture (TextDensityBridgePlainConvUNet, 6-stage
    encoder, density / text branches, dual-path cross-attention fusion).
  * The cached density and text loaders.
  * `_extract_inputs`, periodic checkpoint dumps, optimizer (SGD w/o
    weight-decay on alpha / 1-D), PolyLR schedule, AMP, grad-clipping.
  * nnU-Net's deep supervision pipeline (and its multi-scale weighting).
  * All data augmentation defaults (rotation, mirroring, elastic, etc.).

Usage:
    nnUNetv2_train <DATASET> <CONFIG> <FOLD> \
        -tr nnUNetTrainerTextDensityBridgeBoost_100epochs
"""

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
    """Small-lesion-aware variant. See module docstring for full rationale."""

    # Alignment loss weight (down from 0.5 in the legacy trainer): the
    # new fg-pooled formulation produces a sharper signal, so a smaller
    # weight is sufficient to keep text/density features aligned without
    # competing too aggressively with the segmentation loss.
    align_loss_weight: float = 0.3

    # Text branch's contribution to alignment loss.
    #
    # Note on `text_align_weight > 0` despite text being broadcast-uniform:
    # the smoke test revealed that with `fusion.out_proj` zero-initialised,
    # the segmentation loss cannot propagate any gradient through the
    # text path until `out_proj` leaves zero — and the ONLY signal that
    # can move `out_proj` away from zero in the first place is the
    # alignment loss. Setting `text_align_weight = 0` makes the text
    # branch silently dead (zero gradient throughout training). A small
    # non-zero value provides a bootstrap gradient so the text path can
    # actually start contributing; the mask-pool produces a trivially
    # constant cosine numerically, but the GRADIENT it produces is still
    # valid and unsticks `out_proj`.
    text_align_weight: float = 0.3

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Raise fg-patch sampling proportion. Safe (and recommended) under
        # any sub-image patch sampler; harmless when patch_size equals the
        # full image size, since the patch sampler then degenerates to
        # full-image sampling and the knob is ignored.
        self.oversample_foreground_percent = 0.66

    # -- loss ----------------------------------------------------------------

    def _build_loss(self):
        """DC + CE with batch_dice forced to False, otherwise mirrors the
        official nnUNetTrainer `_build_loss` (incl. deep-supervision weighting).

        Why DC + CE and NOT DC + TopK(k=10)?
        ------------------------------------
        TopKLoss inside nnU-Net selects the top-k% pixels with the LARGEST
        per-pixel CE loss. On PCAT (foreground ratio ~1-3%, batch_size=65,
        patch=224, ~3.26 M pixels per batch), top-10% = ~326 k pixels. At
        random init every pixel's CE ≈ ln(2) ≈ 0.69, so TopK is dominated
        by background pixels by sheer count → gradient direction = "predict
        background everywhere". Per-sample Dice provides only weak
        counter-gradient on samples with NO predicted foreground, so the
        net collapses to the trivial all-background solution. Observed
        empirically: epoch 0 Dice = 0.005, epoch 2 Dice = 0.001.

        Plain CE does not have this pathology (every pixel weighted
        equally → background dominates by 30:1 in raw count too, but Dice
        loss's per-sample fg-recall term keeps a strong fg gradient because
        it is normalized over the small fg area). batch_dice=False keeps
        the small-lesion gradient benefit.
        """

        if self.label_manager.has_regions:
            # Region-based segmentation (e.g. multi-region labels) goes
            # through BCE in nnU-Net; fall back to the parent's choice
            # rather than silently reinterpreting the labels.
            return super()._build_loss()

        loss = DC_and_CE_loss(
            soft_dice_kwargs={
                'batch_dice': False,                  # <-- key change
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

    # -- train step (fg-pooled alignment) ------------------------------------

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
        # `dummy_context()` (not `torch.no_grad()`) is the nnUNet-standard
        # non-CUDA fallback. The training step must keep autograd alive.
        ctx = (torch.autocast(self.device.type, enabled=True)
               if self.device.type == 'cuda' else dummy_context())
        with ctx:
            output = self.network(image=image, density=density, text=text,
                                   case_id=case_ids)
            l_seg = self.loss(output, target)

            # Unwrap DDP / torch.compile so bridge-specific `pop_*` methods
            # are reachable. Without unwrap, the hasattr() guards below
            # silently skip the alignment loss under DDP.
            net_inner = unwrap_network(self.network)

            # Replace the legacy global-pooled alignment with the fg-pooled
            # version. We need to (a) grab the raw bottleneck features the
            # network stashed during forward, and (b) downsample the
            # highest-resolution target mask to bottleneck resolution.
            l_align: Optional[torch.Tensor] = None
            if hasattr(net_inner, "pop_bottleneck_features"):
                bn_image, dens_feat, text_feat = \
                    net_inner.pop_bottleneck_features()
                # Drop the legacy precomputed alignment loss (we override it).
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
                # Network without the new hook (e.g. someone instantiating
                # this trainer with the wrong architecture). Fall back to
                # the legacy alignment loss to preserve correctness.
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


# --- short-schedule variants ------------------------------------------------


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
