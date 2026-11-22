"""Convert case-aligned density caches into nnUNet channel-1 images.

The previous bridge loaded a full density map by case id after nnUNet had
already cropped/augmented the image. That breaks spatial correspondence. This
one-time migration writes ``<case>_0001.png`` next to every ``_0000.png`` so
nnUNet owns both modalities from preprocessing through sliding-window
inference.

The command is non-destructive by default: existing channel-1 images are
validated and retained. ``dataset.json`` is updated only after every channel-0
image has a matching density file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def _case_id(channel0_path: Path) -> str:
    suffix = "_0000"
    if not channel0_path.stem.endswith(suffix):
        raise ValueError(f"Not an nnUNet channel-0 file: {channel0_path}")
    return channel0_path.stem[: -len(suffix)]


def _density_u8(cache_path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    density = np.load(cache_path, allow_pickle=False)
    if density.ndim == 3 and density.shape[0] == 1:
        density = density[0]
    if density.ndim != 2:
        raise ValueError(f"Expected (1,H,W) or (H,W), got {density.shape}: {cache_path}")
    if tuple(density.shape) != tuple(expected_shape):
        raise ValueError(
            f"Geometry mismatch for {cache_path.name}: density={density.shape}, "
            f"image={expected_shape}. Refusing to resize a spatial prior."
        )
    if not np.isfinite(density).all():
        raise ValueError(f"Non-finite density values: {cache_path}")
    return np.rint(np.clip(density, 0.0, 1.0) * 255.0).astype(np.uint8)


def migrate(dataset_root: Path, cache_root: Path) -> dict:
    dataset_root = dataset_root.resolve()
    cache_root = cache_root.resolve()
    dataset_json = dataset_root / "dataset.json"
    if not dataset_json.is_file():
        raise FileNotFoundError(dataset_json)
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)

    image_dirs = [p for p in (dataset_root / "imagesTr", dataset_root / "imagesTs")
                  if p.is_dir()]
    channel0_files = [
        path
        for image_dir in image_dirs
        for path in sorted(image_dir.glob("*_0000.png"))
    ]
    if not channel0_files:
        raise RuntimeError(f"No *_0000.png files found under {dataset_root}")

    written = 0
    retained = 0
    for channel0 in channel0_files:
        case_id = _case_id(channel0)
        cache_path = cache_root / f"{case_id}.npy"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing density cache for {case_id}: {cache_path}")
        image = cv2.imread(str(channel0), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Cannot read {channel0}")
        expected_shape = tuple(image.shape[:2])
        density = _density_u8(cache_path, expected_shape)
        channel1 = channel0.with_name(f"{case_id}_0001.png")
        if channel1.exists():
            existing = cv2.imread(str(channel1), cv2.IMREAD_UNCHANGED)
            if existing is None or tuple(existing.shape[:2]) != expected_shape:
                raise ValueError(f"Invalid existing density modality: {channel1}")
            retained += 1
            continue
        if not cv2.imwrite(str(channel1), density):
            raise RuntimeError(f"Failed to write {channel1}")
        written += 1

    payload = json.loads(dataset_json.read_text(encoding="utf-8"))
    channels = payload.get("channel_names")
    if channels not in ({"0": "image"}, {"0": "image", "1": "rgb_to_0_1"}):
        raise ValueError(f"Unexpected channel_names; refusing to overwrite: {channels!r}")
    payload["channel_names"] = {"0": "image", "1": "rgb_to_0_1"}

    backup = dataset_root / "dataset.before_density_modality.json"
    if not backup.exists():
        shutil.copy2(dataset_json, backup)
    temporary = dataset_root / "dataset.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, dataset_json)

    return {
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "channel0_cases": len(channel0_files),
        "density_written": written,
        "density_retained": retained,
        "dataset_json_backup": str(backup),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.dataset_root, args.cache_root), indent=2))


if __name__ == "__main__":
    main()
