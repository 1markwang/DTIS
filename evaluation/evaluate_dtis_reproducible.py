from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    dice: float
    iou: float
    tp: int
    fp: int
    fn: int
    tn: int
    gt_foreground: int
    pred_foreground: int
    both_empty: bool


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _binary_mask(path: Path, threshold: float) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr.astype(np.float32, copy=False) > threshold


def _case_id(path: Path, suffix: str) -> str:
    stem = path.stem
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    if not stem:
        raise ValueError(f"Empty case ID after removing suffix {suffix!r}: {path}")
    return stem


def _index_pngs(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*.png")):
        cid = _case_id(path, suffix)
        if cid in result:
            raise ValueError(f"Duplicate case ID {cid!r}: {result[cid]} and {path}")
        result[cid] = path.resolve()
    if not result:
        raise ValueError(f"No PNG files found in {directory}")
    return result


def _metrics(case_id: str, pred: np.ndarray, gt: np.ndarray,
             empty_policy: str) -> CaseMetrics | None:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {case_id}: pred={pred.shape}, gt={gt.shape}")
    tp = int(np.logical_and(pred, gt).sum(dtype=np.int64))
    fp = int(np.logical_and(pred, ~gt).sum(dtype=np.int64))
    fn = int(np.logical_and(~pred, gt).sum(dtype=np.int64))
    tn = int(np.logical_and(~pred, ~gt).sum(dtype=np.int64))
    pred_fg, gt_fg = tp + fp, tp + fn
    both_empty = pred_fg == 0 and gt_fg == 0

    if both_empty:
        if empty_policy == "exclude":
            return None
        value = 1.0 if empty_policy == "perfect" else 0.0
        dice = iou = value
    else:
        dice = 2.0 * tp / float(2 * tp + fp + fn)
        iou = tp / float(tp + fp + fn)
    return CaseMetrics(case_id, dice, iou, tp, fp, fn, tn,
                       gt_fg, pred_fg, both_empty)


def _percentile_ci(values: np.ndarray, rng: np.random.Generator,
                   n_bootstrap: int, confidence: float) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    if n_bootstrap <= 0:
        return [float("nan"), float("nan")]
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, 512):
        size = min(512, n_bootstrap - start)
        idx = rng.integers(0, values.size, size=(size, values.size))
        draws[start:start + size] = values[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [float(np.quantile(draws, alpha)),
            float(np.quantile(draws, 1.0 - alpha))]


def _aggregate(rows: Sequence[CaseMetrics], seed: int, n_bootstrap: int,
               confidence: float) -> dict:
    if not rows:
        raise ValueError("No evaluable cases remain under the chosen empty policy")
    dice = np.asarray([r.dice for r in rows], dtype=np.float64)
    iou = np.asarray([r.iou for r in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    fn = sum(r.fn for r in rows)
    dice = 2.0 * tp / float(2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    miou = tp / float(tp + fp + fn) if tp + fp + fn else 1.0
    return {
        "primary_endpoint": "case_dice",
        "n_cases_evaluated": len(rows),
        "n_cases_both_empty": sum(r.both_empty for r in rows),
        "case_dice": {
            "mean": float(dice.mean()), "std": float(dice.std(ddof=1)) if len(rows) > 1 else 0.0,
            "median": float(np.median(dice)),
            "bootstrap_ci": _percentile_ci(dice, rng, n_bootstrap, confidence),
        },
        "case_miou": {
            "mean": float(iou.mean()), "std": float(iou.std(ddof=1)) if len(rows) > 1 else 0.0,
            "median": float(np.median(miou)),
            "bootstrap_ci": _percentile_ci(iou, rng, n_bootstrap, confidence),
        },
        "secondary_dice": float(dice),
        "secondary_iou": float(miou),
        "confusion_totals": {"tp": tp, "fp": fp, "fn": fn,
                             "tn": sum(r.tn for r in rows)},
    }


def _paired_comparison(first: Sequence[CaseMetrics], second: Sequence[CaseMetrics],
                       seed: int, n_bootstrap: int, confidence: float) -> dict:
    a = {r.case_id: r for r in first}
    b = {r.case_id: r for r in second}
    if set(a) != set(b):
        raise ValueError("Paired comparison requires identical evaluable case IDs")
    ids = sorted(a)
    dice_diff = np.asarray([a[i].dice - b[i].dice for i in ids])
    iou_diff = np.asarray([a[i].iou - b[i].iou for i in ids])
    rng = np.random.default_rng(seed)
    n_perm = max(10_000, n_bootstrap)
    observed = abs(float(dice_diff.mean()))
    extreme = 0
    for start in range(0, n_perm, 512):
        size = min(512, n_perm - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(ids)))
        extreme += int((np.abs((signs * dice_diff).mean(axis=1)) >= observed).sum())
    return {
        "orientation": "first_minus_second",
        "n_paired_cases": len(ids),
        "dice_difference": float(dice_diff.mean()),
        "dice_difference_bootstrap_ci": _percentile_ci(
            dice_diff, rng, n_bootstrap, confidence),
        "mIou_difference": float(iou_diff.mean()),
        "mIou_difference_bootstrap_ci": _percentile_ci(
            iou_diff, rng, n_bootstrap, confidence),
        "paired_randomization_p_dice_two_sided": float((extreme + 1) / (n_perm + 1)),
        "n_randomizations": n_perm,
    }


def evaluate(pred_dir: Path, gt_dir: Path, pred_suffix: str, gt_suffix: str,
             threshold: float, empty_policy: str) -> tuple[list[CaseMetrics], list[dict]]:
    preds = _index_pngs(pred_dir, pred_suffix)
    gts = _index_pngs(gt_dir, gt_suffix)
    missing_pred = sorted(set(gts) - set(preds))
    extra_pred = sorted(set(preds) - set(gts))
    if missing_pred or extra_pred:
        raise ValueError(
            f"Prediction/GT case sets differ: missing predictions={missing_pred[:10]} "
            f"(n={len(missing_pred)}), extra predictions={extra_pred[:10]} "
            f"(n={len(extra_pred)})")
    rows: list[CaseMetrics] = []
    manifest: list[dict] = []
    for cid in sorted(gts):
        pred_path, gt_path = preds[cid], gts[cid]
        row = _metrics(cid, _binary_mask(pred_path, threshold),
                       _binary_mask(gt_path, threshold), empty_policy)
        manifest.append({"case_id": cid, "prediction": str(pred_path),
                         "prediction_sha256": _sha256(pred_path),
                         "ground_truth": str(gt_path),
                         "ground_truth_sha256": _sha256(gt_path),
                         "included": row is not None})
        if row is not None:
            rows.append(row)
    return rows, manifest


def _write_csv(path: Path, rows: Iterable[CaseMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CaseMetrics.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--compare-pred-dir", type=Path)
    parser.add_argument("--pred-suffix", default="")
    parser.add_argument("--gt-suffix", default="")
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help=("Foreground is greyscale > threshold. Default 0 supports hard "
              "label maps encoded as either 0/1 (nnU-Net) or 0/255."),
    )
    parser.add_argument("--empty-policy", choices=("perfect", "zero", "exclude"),
                        default="perfect")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative")
    rows, manifest = evaluate(args.pred_dir, args.gt_dir, args.pred_suffix,
                              args.gt_suffix, args.threshold, args.empty_policy)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "protocol": {
            "unit": "case/patient (caller must verify one case ID per patient)",
            "threshold_greyscale": args.threshold,
            "empty_policy": args.empty_policy,
            "bootstrap_resamples": args.bootstrap,
            "confidence": args.confidence,
            "seed": args.seed,
        },
        "results": _aggregate(rows, args.seed, args.bootstrap, args.confidence),
        "manifest": manifest,
    }
    if args.compare_pred_dir is not None:
        comparison_rows, comparison_manifest = evaluate(
            args.compare_pred_dir, args.gt_dir, args.pred_suffix, args.gt_suffix,
            args.threshold, args.empty_policy)
        summary["comparison"] = _paired_comparison(
            rows, comparison_rows, args.seed, args.bootstrap, args.confidence)
        summary["comparison_manifest"] = comparison_manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    _write_csv(args.output.with_suffix(".per_case.csv"), rows)
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2))
    if "comparison" in summary:
        print(json.dumps(summary["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
