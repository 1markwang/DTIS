# DTIS

**Density and Text Informed Medical Image Segmentation**

This repository archives the August 2026 DTIS implementation for lung lesion
segmentation. DTIS extends nnU-Net v2 with a density branch and a text branch.
At the encoder bottleneck, it first fuses anatomical density features with
image features, then uses a global text embedding to modulate the fused features.

## Repository layout

| Path | Contents |
| --- | --- |
| `nnunet_extensions/` | DTIS trainer, cascaded fusion network, and cache loaders |
| `data_tools/` | Scripts for adding existing density maps, preparing splits, and building text caches |
| `evaluation/` | Scripts for evaluating Dice and IoU from predicted masks |
| `SOURCE_MANIFEST.json` | Source provenance and SHA256 checksums |

## Training

1. Install nnU-Net v2 in a separate Python environment. Copy the Python files
   from `nnunet_extensions/` into `nnunetv2/training/nnUNetTrainer/`.
2. Prepare the nnU-Net dataset with spatially aligned image (`_0000`) and
   density (`_0001`) inputs. Set the density channel name to `rgb_to_0_1`.
   Provide one 768-dimensional CLS embedding per case.
3. Set `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` for your
   environment. After preprocessing and preparing the fold-0 split, run:

   ```bash
   nnUNetv2_train 1 2d 0 -tr nnUNetTrainerDTIS
   ```

## Version and reproducibility

The DTIS trainer and fusion architecture match the source checksums recorded
for the August 24, 2026 QaTa-COV19 run. Reproducing that run requires the original
density inputs, text embeddings, and dataset split. 
