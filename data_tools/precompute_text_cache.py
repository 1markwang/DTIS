# -*- coding: utf-8 -*-
"""Offline text embedding precomputation for the multimodal pipeline.

This script generates one .npy file per sample so training can read text
features from disk without running BERT inside DataLoader workers.

Typical usage:

python precompute_text_cache.py ^
  --xlsx F:\pcat\datasets\Covid19\Train_Folder\Train_Val_text.xlsx ^
  --out-dir F:\pcat\datasets\Covid19\Train_Folder\text_cache ^
  --bert-path F:\pcat\bert-base-uncased

It supports multiple xlsx files and optional key normalization so it can bridge
raw filenames and nnUNet-style names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from text_prior import encode_text_with_bert


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".nii.gz", ".nii", ".npy")


def normalize_sample_key(name: str) -> str:
    s = str(name or "").strip().lower().replace("\\", "/")
    if not s:
        return ""
    base = Path(s).name
    base = base.replace("mask_", "", 1)
    for ext in (".nii.gz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".nii", ".npy"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    for suf in ("_0000", "_img", "_image", "-img", "-image"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    base = base.replace("mask_", "")
    base = base.replace(" ", "").replace("-", "_")
    return base


def _read_mapping_xlsx(xlsx_path: Path) -> Dict[str, str]:
    df = pd.read_excel(xlsx_path)
    cols = [str(c).strip().lower() for c in df.columns]
    key_col = None
    text_col = None
    for cand in ("filename", "file", "image", "image_name", "name", "case_id"):
        if cand in cols:
            key_col = df.columns[cols.index(cand)]
            break
    for cand in ("text", "report", "description", "caption", "label_text"):
        if cand in cols:
            text_col = df.columns[cols.index(cand)]
            break
    if key_col is None or text_col is None:
        raise ValueError(f"Could not find key/text columns in {xlsx_path.name}")

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        key = str(row[key_col]).strip()
        text = str(row[text_col]).strip() if not pd.isna(row[text_col]) else ""
        if not key or key.lower() == "nan":
            continue
        mapping[key] = text
        if not Path(key).suffix:
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                mapping[key + ext] = text
    return mapping


def _build_lookup(mapping: Dict[str, str]) -> Dict[str, str]:
    norm = {}
    for k, v in mapping.items():
        nk = normalize_sample_key(k)
        if nk and nk not in norm:
            norm[nk] = v
    return norm


def _iter_candidates(key: str) -> Iterable[str]:
    base = Path(str(key or "")).name
    yield key
    yield base
    yield Path(base).stem
    nk = normalize_sample_key(key)
    if nk:
        yield nk


def lookup_text(key: str, mapping: Dict[str, str], norm_mapping: Dict[str, str]) -> str:
    for cand in _iter_candidates(key):
        if cand in mapping:
            return mapping[cand]
        nk = normalize_sample_key(cand)
        if nk in norm_mapping:
            return norm_mapping[nk]
    return ""


def _safe_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{path.parent.name}")
    np.save(str(tmp), arr)
    tmp_npy = tmp if tmp.suffix == ".npy" else Path(str(tmp) + ".npy")
    if tmp_npy.exists():
        tmp_npy.replace(path)
    elif tmp.exists():
        tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute offline text embeddings")
    parser.add_argument("--xlsx", nargs="+", required=True, help="One or more mapping xlsx files")
    parser.add_argument("--out-dir", required=True, help="Output directory for .npy caches")
    parser.add_argument("--bert-path", default="F:/pcat/bert-base-uncased", help="Local BERT directory")
    parser.add_argument("--text-key-col", default="", help="Optional explicit key column name")
    parser.add_argument("--text-col", default="", help="Optional explicit text column name")
    parser.add_argument("--limit", type=int, default=0, help="Optional max samples to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    parser.add_argument("--save-manifest", action="store_true", help="Write a manifest jsonl next to caches")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bert_path = Path(args.bert_path)

    all_pairs: List[Tuple[str, str]] = []
    for xlsx in args.xlsx:
        p = Path(xlsx)
        if not p.exists():
            raise FileNotFoundError(str(p))

        if args.text_key_col or args.text_col:
            df = pd.read_excel(p)
            cols = [str(c).strip().lower() for c in df.columns]

            if args.text_key_col:
                if args.text_key_col.lower() not in cols:
                    raise ValueError(f"Key column '{args.text_key_col}' not found in {p.name}")
                key_col = df.columns[cols.index(args.text_key_col.lower())]
            else:
                key_col = None
                for cand in ("filename", "file", "image", "image_name", "name", "case_id"):
                    if cand in cols:
                        key_col = df.columns[cols.index(cand)]
                        break
            if args.text_col:
                if args.text_col.lower() not in cols:
                    raise ValueError(f"Text column '{args.text_col}' not found in {p.name}")
                text_col = df.columns[cols.index(args.text_col.lower())]
            else:
                text_col = None
                for cand in ("text", "report", "description", "caption", "label_text"):
                    if cand in cols:
                        text_col = df.columns[cols.index(cand)]
                        break

            if key_col is None or text_col is None:
                raise ValueError(f"Could not infer key/text columns in {p.name}")

            for _, row in df.iterrows():
                key = str(row[key_col]).strip()
                text = str(row[text_col]).strip() if not pd.isna(row[text_col]) else ""
                if not key or key.lower() == "nan":
                    continue
                all_pairs.append((key, text))
        else:
            mapping = _read_mapping_xlsx(p)
            all_pairs.extend(mapping.items())

    if args.limit and args.limit > 0:
        all_pairs = all_pairs[: args.limit]

    seen = set()
    dedup_pairs: List[Tuple[str, str]] = []
    for key, text in all_pairs:
        nk = normalize_sample_key(key)
        if nk in seen:
            continue
        seen.add(nk)
        dedup_pairs.append((key, text))

    print(f"[Cache] total_pairs={len(all_pairs)} dedup_pairs={len(dedup_pairs)} out_dir={out_dir}")

    mapping = {k: v for k, v in dedup_pairs}
    norm_mapping = _build_lookup(mapping)
    # If a source xlsx already contains a filename field, the script can also verify
    # that each normalized key maps to a known sample key before saving.

    manifest_f = None
    if args.save_manifest:
        manifest_f = (out_dir / "manifest.jsonl").open("a", encoding="utf-8")

    hits = 0
    misses = 0
    skipped = 0

    for idx, (key, text) in enumerate(dedup_pairs, start=1):
        norm_key = normalize_sample_key(key)
        cache_path = out_dir / f"{norm_key}.npy"
        if args.skip_existing and cache_path.exists():
            skipped += 1
            continue

        try:
            emb, conf = encode_text_with_bert(text, model_path=bert_path)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb[None, :]
            _safe_save_npy(cache_path, emb)
            conf_path = out_dir / f"{norm_key}_conf.txt"
            conf_path.write_text(str(float(conf)), encoding="utf-8")
            hits += 1
            if manifest_f is not None:
                rec = {
                    "key": key,
                    "norm_key": norm_key,
                    "cache_path": str(cache_path),
                    "text_len": len(text),
                    "confidence": float(conf),
                }
                manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            misses += 1
            print(f"[Cache][WARN] failed key={key}: {e}")

        if idx % 100 == 0 or idx == len(dedup_pairs):
            print(f"[Cache] {idx}/{len(dedup_pairs)} hits={hits} misses={misses} skipped={skipped}")

    if manifest_f is not None:
        manifest_f.close()

    print(f"[Cache][DONE] hits={hits} misses={misses} skipped={skipped} out_dir={out_dir}")


if __name__ == "__main__":
    main()
