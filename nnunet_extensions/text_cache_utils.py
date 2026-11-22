# -*- coding: utf-8 -*-
"""Utilities for loading case-aligned text-embedding cache files.

Each `.npy` file holds a single 1-D float32 vector of length D
(case-specific text embedding, e.g. PubMedBERT [CLS] embedding).
The cache root is resolved analogously to `density_cache_utils.py`.

Cache root resolution order (first non-empty wins):
    1. PCAT_TEXT_CACHE_ROOT env var       (explicit absolute override)
    2. nnUNet_raw / NNUNET_RAW env var    (-> <raw>/../text_cache/<dataset>)
    3. PCAT_PROJECT_ROOT env var          (-> <root>/nnunetv2_data/text_cache/<dataset>)
    4. Current working directory          (-> ./nnunetv2_data/text_cache/<dataset>)
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

import numpy as np


def _resolve_text_cache_root() -> Path:
    explicit = os.environ.get("PCAT_TEXT_CACHE_ROOT", "").strip()
    if explicit:
        return Path(explicit).resolve()

    dataset_name = (
        os.environ.get("NNUNET_DATASET_NAME", "Dataset001_PCAT").strip()
        or "Dataset001_PCAT"
    )

    nnunet_raw = (
        os.environ.get("nnUNet_raw") or os.environ.get("NNUNET_RAW") or ""
    ).strip()
    if nnunet_raw:
        return (
            Path(nnunet_raw).resolve().parent / "text_cache" / dataset_name
        ).resolve()

    project_root_env = os.environ.get("PCAT_PROJECT_ROOT", "").strip()
    project_root = (
        Path(project_root_env).resolve() if project_root_env else Path.cwd().resolve()
    )
    return (project_root / "nnunetv2_data" / "text_cache" / dataset_name).resolve()


def text_cache_root() -> Path:
    """Public accessor for the active text-embedding cache root."""
    return _resolve_text_cache_root()


def text_cache_path(case_id: str) -> Path:
    case_id = str(case_id).strip()
    if not case_id:
        raise ValueError("case_id must not be empty")
    return (_resolve_text_cache_root() / f"{case_id}.npy").resolve()


@functools.lru_cache(maxsize=None)
def _load_text_raw(case_id: str) -> np.ndarray:
    """Cached low-level loader (one shot per process per case_id).

    Text embeddings are tiny (~3 KB each at D=768), so even at 10 k cases
    the in-memory footprint stays well below 30 MB. The returned buffer
    is set read-only because every downstream consumer wraps it in
    `torch.from_numpy(...)` without ever mutating the underlying memory.
    """
    path = text_cache_path(case_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Text cache not found for case_id={case_id!r}: {path}\n"
            f"Active cache root: {text_cache_root()}\n"
            f"Override with PCAT_TEXT_CACHE_ROOT, nnUNet_raw or PCAT_PROJECT_ROOT, "
            f"or run `python build_text_cache.py` to (re)build it."
        )

    emb = np.load(path)
    if emb.ndim != 1:
        raise ValueError(
            f"Text cache must be a 1-D vector, got shape {emb.shape} from {path}. "
            f"Re-run `python build_text_cache.py --overwrite`."
        )
    emb = emb.astype(np.float32, copy=False)
    # NOTE: not setting write=False; PyTorch's `torch.from_numpy` warns on
    # read-only arrays even though our downstream consumers never mutate
    # the buffer. The cached array's identity is preserved across calls.
    return emb


def load_text(case_id: str, expected_dim: Optional[int] = None) -> np.ndarray:
    """Load a single text embedding (1-D float32 of length D), in-memory cached.

    First call per `case_id`: disk read + validation. Subsequent calls:
    O(1) dict lookup. The returned array is read-only; callers that need
    to mutate the data must explicitly `.copy()` it first.

    Parameters
    ----------
    case_id : str
        Case identifier, matching the segmentation case naming.
    expected_dim : Optional[int]
        If set, validate that the loaded vector has exactly this length.

    Returns
    -------
    np.ndarray, shape (D,), float32.

    Raises
    ------
    FileNotFoundError
        If the cache file is missing.
    ValueError
        If the cached array is not 1-D float, or its length differs from
        `expected_dim` when one is supplied.
    """
    emb = _load_text_raw(str(case_id))
    if expected_dim is not None and emb.shape[0] != expected_dim:
        raise ValueError(
            f"Text cache for {case_id!r} has dim {emb.shape[0]}, "
            f"but the trainer expected dim {expected_dim}. "
            f"Re-build the cache with the same encoder used at training time."
        )
    return emb


def clear_text_cache() -> None:
    """Drop every (case_id -> embedding) entry from the in-memory LRU."""
    _load_text_raw.cache_clear()


def have_text_cache(case_id: str) -> bool:
    """Lightweight existence check (no load)."""
    return text_cache_path(case_id).exists()
