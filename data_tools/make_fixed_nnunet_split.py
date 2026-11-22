"""Create a provenance-preserving nnU-Net split from case-ID prefixes."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dataset", type=Path, required=True)
    parser.add_argument("--configuration", default="nnUNetPlans_2d")
    parser.add_argument("--train-prefix", default="train_")
    parser.add_argument("--val-prefix", default="val_")
    parser.add_argument("--expect-train", type=int, required=True)
    parser.add_argument("--expect-val", type=int, required=True)
    args = parser.parse_args()

    case_dir = args.preprocessed_dataset.resolve() / args.configuration
    ids = sorted(p.name[:-5] for p in case_dir.glob("*.b2nd")
                 if not p.name.endswith("_seg.b2nd"))
    train = [cid for cid in ids if cid.startswith(args.train_prefix)]
    val = [cid for cid in ids if cid.startswith(args.val_prefix)]
    other = sorted(set(ids) - set(train) - set(val))
    if len(train) != args.expect_train or len(val) != args.expect_val or other:
        raise RuntimeError(
            f"Refusing split: train={len(train)} expected={args.expect_train}; "
            f"val={len(val)} expected={args.expect_val}; other={other[:10]}")
    if set(train) & set(val):
        raise RuntimeError("Train/validation overlap detected")

    output = args.preprocessed_dataset.resolve() / "splits_final.json"
    if output.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output.with_name(f"splits_final.auto_backup_{stamp}.json")
        shutil.copy2(output, backup)
        print(f"backup={backup}")
    output.write_text(json.dumps([{"train": train, "val": val}], indent=2),
                      encoding="utf-8")
    print(f"output={output}")
    print(f"train={len(train)} val={len(val)} overlap=0")


if __name__ == "__main__":
    main()
