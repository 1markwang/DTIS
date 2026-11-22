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
    _load_text_tokens_raw.cache_clear()


def have_text_token_cache(case_id: str) -> bool:
    """Lightweight existence check (no load)."""
    return text_token_cache_path(case_id).exists()
