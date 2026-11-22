# DTIS: aligned, cascaded bottleneck implementation (August 2026)

This is a curated **source-code release candidate** for the repaired DTIS implementation used in the QaTa-COV19 run started on 2026-08-24. It is distinct from the earlier case-ID density-cache implementation and from the later PSDF/Local Semantic model.

## What the model does

- nnU-Net v2 receives two spatially aligned input channels: CXR image (`_0000`) and precomputed density map (`_0001`). The image encoder consumes only channel 0; the density branch consumes channel 1.
- At the bottleneck, image-to-density cross attention is applied first. A second, image-to-text interaction uses a cached 768-dimensional CLS embedding. This CLS representation is global semantic modulation; it does not establish word-level spatial localization.
- The trainer uses 250 epochs, 250 training iterations and 50 validation iterations per epoch, a warm-started projected alignment loss, and a fixed train/validation split.

## Source layout

| Directory | Contents |
|---|---|
| `nnunet_extensions/` | DTIS trainer, cascaded architecture, parent trainers, and cache loaders |
| `data_tools/` | Historical scripts to backfill aligned density inputs, fix the split, and construct candidate CLS text caches |
| `evaluation/` | Historical evaluation script |
| `SOURCE_MANIFEST.json` | SHA256 and original source of every included file |

The exact bytes of `nnUNetTrainerDTIS.py` and `text_density_bridge_architecture.py` match the SHA256 values in the 2026-08-24 QaTa run manifest. Other files are nearby historical snapshots or unchanged source files, but were **not** individually hashed by that run manifest. See `SOURCE_MANIFEST.json` for their status.

## Input contract

The prepared nnU-Net dataset must have the same case IDs in its image, density, label, and text cache. Its `dataset.json` uses two channel names, `"0": "image"` and `"1": "rgb_to_0_1"`. The density channel is an aligned, single-channel uint8 PNG. The CLS cache stores one 768-dimensional NumPy vector per case. The QaTa split in the recorded run contained 5,716 training and 1,429 validation cases (fold 0; split SHA256 `d281609082679095d711728fc6dd66d577f23f4f68476d6558925182f8bb4d28`).

The exact August density-map generator snapshot has **not** been recovered. `data_tools/backfill_density_modality.py` only inserts previously generated density maps into the aligned nnU-Net dataset; it does not generate them. The source and weights used for the recorded CLS cache have also not been independently verified. The included text-cache scripts are historical candidates and must not be described as proven cache provenance.

## Installation and training

Use an isolated Python environment with nnU-Net v2 and its dependencies installed. Copy the files in `nnunet_extensions/` into that environment's `nnunetv2/training/nnUNetTrainer/` package. Do not copy them into a running experiment environment without first preserving its existing custom files. Set `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` to your prepared directories. Once the density channel, CLS caches, and locked split are present, the historical trainer entry point is:

```powershell
nnUNetv2_train 1 2d 0 -tr nnUNetTrainerDTIS
```

The 2026-08-24 manifest identifies PyTorch `2.12.0.dev20260408+cu128`; it does not freeze every package version. This archive is therefore a source release, **not a one-command reproduction environment**.

## Result identity and limits

The original manuscript reported **91.87% Dice**. That number has not been tied to a verifiable checkpoint and evaluation aggregation in the surviving records. The repaired August QaTa run's saved 1,429-case validation summary reports mean Dice **0.79995849** under its recorded evaluation, and is a different result. Do not present 91.87% as a measured output of this release.

The surviving August MosMed record for this trainer is a one-epoch smoke test, not a completed corresponding MosMed result. No images, labels, reports, cached embeddings, pretrained weights, or checkpoints are included here.

## GitHub publication

Review `SOURCE_MANIFEST.json` and the files before publishing. Create a new **empty** GitHub repository, then run from this directory:

```powershell
git init -b main
git add .
git status --short
git commit -m "Release August 2026 DTIS cascaded source"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Choose a license only after confirming ownership of this code and attribution requirements for the upstream nnU-Net components. If the research is still under review, choose the repository visibility accordingly.
