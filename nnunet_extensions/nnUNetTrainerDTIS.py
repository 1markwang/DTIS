"""Canonical, paper-facing DTIS trainer.

This class freezes the implementation used for the revised experiments:

* image + aligned density are nnUNet modalities 0 and 1;
* density-first, text-second cascaded bottleneck cross-attention;
* Xavier output projections with learnable gates initialised to 0.05;
* projected foreground-aware alignment, linearly warmed to lambda=0.1;
* 250 training epochs.

Historical V2--V7 trainers remain available for development, but manuscript
tables should name this trainer and must not mix their checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from pydoc import locate

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTextDensityBridgeV2 import (
    nnUNetTrainerTextDensityBridgeV2,
)
from nnunetv2.training.nnUNetTrainer.text_density_bridge_architecture import (
    TextDensityBridgePlainConvUNet,
)
from nnunetv2.training.nnUNetTrainer.text_cache_utils import text_cache_root
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


def build_dtis_network(
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
    *,
    use_text_tokens: bool = False,
) -> nn.Module:
    """Build the frozen DTIS architecture without legacy silent fallbacks."""
    del plans_manager
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
        norm_op_kwargs=arch.get("norm_op_kwargs"),
        dropout_op=dropout_op,
        dropout_op_kwargs=arch.get("dropout_op_kwargs"),
        nonlin=nonlin,
        nonlin_kwargs=arch.get("nonlin_kwargs"),
        deep_supervision=enable_deep_supervision,
        density_in_channels=1,
        density_base_ch=32,
        text_in_dim=768,
        text_hidden=256,
        text_dropout=0.1,
        num_heads=4,
        alpha_init_d=0.05,
        alpha_init_t=0.05,
        out_proj_init="xavier",
        projection_dim=128,
        align_loss_weight=0.0,
        allow_legacy_density_cache=False,
        require_aligned_density=True,
        require_text_cache=True,
        use_text_token_inject=use_text_tokens,
        text_token_dim=768,
        text_inject_stages=(0, 1),
        text_inject_heads=4,
        text_inject_alpha=0.05,
        require_text_tokens=use_text_tokens,
    )


class nnUNetTrainerDTIS(nnUNetTrainerTextDensityBridgeV2):
    """Frozen DTIS configuration for reproducible paper experiments."""

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        channels = dataset_json.get("channel_names", dataset_json.get("modality", {}))
        if len(channels) != 2:
            raise ValueError(
                "DTIS requires two spatially aligned nnUNet modalities: "
                "channel 0=image and channel 1=density. Re-export the dataset "
                "with *_0001 density files and re-run planning/preprocessing."
            )
        density_name = str(channels.get("1", channels.get(1, ""))).casefold()
        if density_name != "rgb_to_0_1":
            raise ValueError(
                "DTIS density channel must use nnUNet's rgb_to_0_1 "
                "normalizer so uint8 density PNGs map deterministically to [0,1]."
            )
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Cache routing must be derived from the nnU-Net plans rather than a
        # manually maintained shell variable. This prevents Dataset002 from
        # silently loading Dataset001 reports (their case IDs overlap).
        os.environ["NNUNET_DATASET_NAME"] = self.plans_manager.dataset_name
        self.num_epochs = 250

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return build_dtis_network(
            plans_manager, configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision,
            use_text_tokens=False)

    def on_train_start(self) -> None:
        super().on_train_start()
        self._write_dtis_manifest()

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    def _write_dtis_manifest(self) -> None:
        """Freeze run identity so manuscript numbers remain traceable."""
        trainer_file = Path(__file__).resolve()
        architecture_file = Path(
            __import__(
                "nnunetv2.training.nnUNetTrainer.text_density_bridge_architecture",
                fromlist=["__file__"],
            ).__file__).resolve()
        cache_root = text_cache_root()
        split_file = Path(self.preprocessed_dataset_folder_base) / "splits_final.json"
        splits = json.loads(split_file.read_text(encoding="utf-8"))
        split = splits[int(self.fold)]
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "paper_model_name": "DTIS-CLS",
            "trainer_class": self.__class__.__name__,
            "dataset_name": self.plans_manager.dataset_name,
            "fold": self.fold,
            "configuration": self.configuration_name,
            "num_epochs": self.num_epochs,
            "iterations_per_epoch": self.num_iterations_per_epoch,
            "validation_iterations_per_epoch": self.num_val_iterations_per_epoch,
            "channel_names": self.dataset_json.get("channel_names"),
            "data_split": {
                "file": str(split_file),
                "sha256": self._hash_file(split_file),
                "train_count": len(split["train"]),
                "validation_count": len(split["val"]),
                "overlap_count": len(set(split["train"]) & set(split["val"])),
            },
            "cache_routing": {
                "NNUNET_DATASET_NAME": os.environ.get("NNUNET_DATASET_NAME"),
                "text_cache_root": str(cache_root),
                "text_cache_file_count": len(list(cache_root.glob("*.npy"))),
            },
            "method": {
                "density_input": "aligned nnU-Net modality 1",
                "legacy_density_cache": False,
                "text_interaction": "broadcast CLS global semantic modulation",
                "alpha_density_init": self.alpha_init_d,
                "alpha_text_init": self.alpha_init_t,
                "alignment_lambda_max": self.align_lambda_max,
                "alignment_warmup_epochs": self.align_warmup_epochs,
                "projection_dim": self.projection_dim,
            },
            "implementation": {
                "trainer_file": str(trainer_file),
                "trainer_sha256": self._hash_file(trainer_file),
                "architecture_file": str(architecture_file),
                "architecture_sha256": self._hash_file(architecture_file),
                "torch": torch.__version__,
            },
        }
        output = Path(self.output_folder) / "dtis_run_manifest.json"
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        self.print_to_log_file(f"[DTIS] run manifest: {output}")
