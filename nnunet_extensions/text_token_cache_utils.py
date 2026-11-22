# -*- coding: utf-8 -*-
"""Utilities for loading case-aligned text *token-sequence* cache files.

Unlike `text_cache_utils.py` (which stores ONE 1-D [CLS] vector per case for
the bottleneck text branch), this module stores the FULL per-token hidden
states `(L, D)` for a case's report. These are consumed by the multi-scale
text token cross-attention injected into the decoder (V7 / DRIS multi-scale).

Rationale
---------
The QaTa-COV19 reports are templated but DO carry coarse localisation, e.g.
"... upper left lung and lower right lung". Pooling them into a single [CLS]
vector (the bottleneck branch) discards the per-word spatial cues that a
decoder-stage cross-attention can route to the correct region. Keeping the
full token sequence lets words like "upper"/"left"/"lower" be attended to
spatially at 14x14 / 28x28 decoder resolutions.

Cache root resolution order (first non-empty wins):
    1. PCAT_TEXT_TOKEN_CACHE_ROOT env var   (explicit absolute override)
    2. nnUNet_raw / NNUNET_RAW env var       (-> <raw>/../text_token_cache/<dataset>)
    3. PCAT_PROJECT_ROOT env var             (-> <root>/nnunetv2_data/text_token_cache/<dataset>)
    4. Current working directory             (-> ./nnunetv2_data/text_token_cache/<dataset>)

Each `.npy` file holds a 2-D float32 array of shape (L, D): L tokens (incl.
[CLS]/[SEP]) by D hidden dims (768 for BERT-family encoders). Empty reports
are stored as a single zero row (1, D) so cross-attention never receives an
all-masked key set (which would produce NaNs).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

import numpy as np


def _resolve_token_cache_root() -> Path:
    explicit = os.environ.get("PCAT_TEXT_TOKEN_CACHE_ROOT", "").strip()
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
            Path(nnunet_raw).resolve().parent / "text_token_cache" / dataset_name
        ).resolve()

    project_root_env = os.environ.get("PCAT_PROJECT_ROOT", "").strip()
    project_root = (
        Path(project_root_env).resolve() if project_root_env else Path.cwd().resolve()
    )
    return (project_root / "nnunetv2_data" / "text_token_cache" / dataset_name).resolve()


def text_token_cache_root() -> Path:
    """Public accessor for the active token-sequence cache root."""
    return _resolve_token_cache_root()


def text_token_cache_path(case_id: str) -> Path:
    case_id = str(case_id).strip()
    if not case_id:
        raise ValueError("case_id must not be empty")
    return (_resolve_token_cache_root() / f"{case_id}.npy").resolve()


@functools.lru_cache(maxsize=None)
def _load_text_tokens_raw(case_id: str) -> np.ndarray:
    """Cached low-level loader (one disk read per process per case_id).

    Token sequences are short (reports < ~40 tokens). At D=768 a 40-token
    case is ~120 KB; even 10 k cases stay well under ~1.2 GB worst-case, and
    in practice each data-loader worker only touches its shard.
    """
    path = text_token_cache_path(case_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Text-token cache not found for case_id={case_id!r}: {path}\n"
            f"Active cache root: {text_token_cache_root()}\n"
            f"Override with PCAT_TEXT_TOKEN_CACHE_ROOT, nnUNet_raw or "
            f"PCAT_PROJECT_ROOT, or run `python build_text_token_cache.py`."
        )

    seq = np.load(path)
    if seq.ndim != 2:
        raise ValueError(
            f"Text-token cache must be a 2-D (L, D) array, got shape "
            f"{seq.shape} from {path}. Re-run `python build_text_token_cache.py "
            f"--overwrite`."
        )
    return seq.astype(np.float32, copy=False)


def load_text_tokens(case_id: str, expected_dim: Optional[int] = None) -> np.ndarray:
    """Load a single (L, D) token-sequence array, in-memory cached.

    Parameters
    ----------
    case_id : str
        Case identifier, matching the segmentation case naming.
    expected_dim : Optional[int]
        If set, validate that the hidden dim D equals this value.

    Returns
    -------
    np.ndarray, shape (L, D), float32 with L >= 1.

    Raises
    ------
    FileNotFoundError
        If the cache file is missing.
    ValueError
        If the cached array is not 2-D, has L == 0, or its hidden dim differs
        from `expected_dim` when one is supplied.
    """
    seq = _load_text_tokens_raw(str(case_id))
    if seq.shape[0] < 1:
        raise ValueError(
            f"Text-token cache for {case_id!r} is empty (L=0); expected at "
            f"least one token. Re-build with `build_text_token_cache.py`."
        )
    if expected_dim is not None and seq.shape[1] != expected_dim:
        raise ValueError(
            f"Text-token cache for {case_id!r} has hidden dim {seq.shape[1]}, "
            f"but the trainer expected dim {expected_dim}. Re-build the cache "
            f"with the same encoder used at training time."
        )
    return seq


def clear_text_token_cache() -> None:
    """Drop every (case_id -> token-sequence) entry from the in-memory LRU."""
    _load_text_tokens_raw.cache_clear()


def have_text_token_cache(case_id: str) -> bool:
    """Lightweight existence check (no load)."""
    return text_token_cache_path(case_id).exists()
