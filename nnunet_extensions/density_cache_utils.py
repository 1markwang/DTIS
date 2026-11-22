# -*- coding: utf-8 -*-
"""Utilities for loading case-aligned density cache files.

Placed inside the nnUNetv2 package so the custom trainer can import it without
relying on the project working directory being on PYTHONPATH.

Density cache root resolution order (first non-empty wins):
1. PCAT_DENSITY_CACHE_ROOT env var       (explicit absolute override)
2. nnUNet_raw / NNUNET_RAW env var       (-> <raw>/../density_cache/<dataset>)
3. PCAT_PROJECT_ROOT env var             (-> <root>/nnunetv2_data/density_cache/<dataset>)
4. Current working directory             (-> ./nnunetv2_data/density_cache/<dataset>)

The same resolver is reused by the project-side build script so the write
path and the read path never drift apart.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np


def _resolve_density_cache_root() -> Path:
    explicit = os.environ.get("PCAT_DENSITY_CACHE_ROOT", "").strip()
    if explicit:
        return Path(explicit).resolve()

    dataset_name = os.environ.get("NNUNET_DATASET_NAME", "Dataset001_PCAT").strip() or "Dataset001_PCAT"

    nnunet_raw = (os.environ.get("nnUNet_raw") or os.environ.get("NNUNET_RAW") or "").strip()
    if nnunet_raw:
        return (Path(nnunet_raw).resolve().parent / "density_cache" / dataset_name).resolve()

    project_root_env = os.environ.get("PCAT_PROJECT_ROOT", "").strip()
    project_root = Path(project_root_env).resolve() if project_root_env else Path.cwd().resolve()
    return (project_root / "nnunetv2_data" / "density_cache" / dataset_name).resolve()


def density_cache_root() -> Path:
    """Public accessor for the active density cache root."""
    return _resolve_density_cache_root()


def density_cache_path(case_id: str) -> Path:
    case_id = str(case_id).strip()
    if not case_id:
        raise ValueError("case_id must not be empty")
    return (_resolve_density_cache_root() / f"{case_id}.npy").resolve()


@functools.lru_cache(maxsize=None)
def _load_density_raw(case_id: str) -> np.ndarray:
    """Cached low-level loader.

    `functools.lru_cache(maxsize=None)` retains every (case_id -> array)
    mapping in process memory after the first load. For the PCAT dataset
    (~6 k cases, each ~50-200 KB) total RAM footprint is < 1 GB, far below
    typical workstation budgets, but it removes ~16 k disk reads per epoch
    (65 batch * 250 iter), which was the dominant per-epoch bottleneck
    once the GPU-side compute was optimised. The returned array is set
    read-only to prevent a caller from silently corrupting the cache (the
    rest of the pipeline only does `torch.from_numpy(...)` which is safe).
    """
    path = density_cache_path(case_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Density cache not found for case_id={case_id!r}: {path}\n"
            f"Active cache root: {density_cache_root()}\n"
            f"Override with PCAT_DENSITY_CACHE_ROOT, nnUNet_raw or PCAT_PROJECT_ROOT, "
            f"or run `python build_density_cache.py --overwrite` to (re)build it."
        )

    density = np.load(path)
    if density.ndim == 2:
        density = density[None, ...]
    if density.ndim != 3 or density.shape[0] != 1:
        raise ValueError(
            f"Density cache must be a (1, H, W) array, got shape {density.shape} from {path}.\n"
            f"This is most likely a legacy 2-channel cache. "
            f"Re-run `python build_density_cache.py --overwrite` to regenerate it."
        )

    density = density.astype(np.float32, copy=False)
    if density.min() < 0.0 or density.max() > 1.0:
        density = np.clip(density, 0.0, 1.0)
    # NOTE: not setting write=False; PyTorch's `torch.from_numpy` warns on
    # read-only arrays even though our downstream consumers never mutate
    # the buffer (they always `stack(...).to(device, dtype)` which copies).
    # The cached array's identity is preserved across calls regardless.
    return density


def load_density(case_id: str) -> np.ndarray:
    """Load a single-channel density cache file (in-memory cached).

    First call per `case_id`: disk read + decode. Subsequent calls: O(1)
    dict lookup. The returned array is read-only; if a caller needs to
    mutate the data it must explicitly `.copy()` it first.

    Returns
    -------
    np.ndarray of shape (1, H, W), float32, values in [0, 1].
        channel 0 = distribution (gated positive lift over per-lung p25
                    baseline, zero outside the eroded inner lung mask).

    Raises
    ------
    FileNotFoundError
        If the cache file is missing.
    ValueError
        If the cache file is not exactly a (1, H, W) array. Legacy multi-
        channel caches are NOT silently demoted: re-run
        `python build_density_cache.py --overwrite` to regenerate them in
        the current format.
    """
    return _load_density_raw(str(case_id))


def clear_density_cache() -> None:
    """Drop every (case_id -> array) entry. Useful for memory pressure or
    when the underlying cache directory is rebuilt mid-process."""
    _load_density_raw.cache_clear()
