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
    return _resolve_text_cache_root()

def text_cache_path(case_id: str) -> Path:
    case_id = str(case_id).strip()
    if not case_id:
        raise ValueError("case_id must not be empty")
    return (_resolve_text_cache_root() / f"{case_id}.npy").resolve()

@functools.lru_cache(maxsize=None)
def _load_text_raw(case_id: str) -> np.ndarray:
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
    return emb

def load_text(case_id: str, expected_dim: Optional[int] = None) -> np.ndarray:
    emb = _load_text_raw(str(case_id))
    if expected_dim is not None and emb.shape[0] != expected_dim:
        raise ValueError(
            f"Text cache for {case_id!r} has dim {emb.shape[0]}, "
            f"but the trainer expected dim {expected_dim}. "
            f"Re-build the cache with the same encoder used at training time."
        )
    return emb

def clear_text_cache() -> None:
    _load_text_raw.cache_clear()

def have_text_cache(case_id: str) -> bool:
    return text_cache_path(case_id).exists()
