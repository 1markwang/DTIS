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
    return _resolve_density_cache_root()

def density_cache_path(case_id: str) -> Path:
    case_id = str(case_id).strip()
    if not case_id:
        raise ValueError("case_id must not be empty")
    return (_resolve_density_cache_root() / f"{case_id}.npy").resolve()


@functools.lru_cache(maxsize=None)
def _load_density_raw(case_id: str) -> np.ndarray:
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
    return density

def load_density(case_id: str) -> np.ndarray:
    return _load_density_raw(str(case_id))

def clear_density_cache() -> None:
    _load_density_raw.cache_clear()
