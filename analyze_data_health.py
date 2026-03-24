from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def keyset(rows: list[dict[str, str]], field: str) -> set[str]:
    return {f"{r['dataset']}|{r[field]}" for r in rows if r.get(field)}


def pct(counter: Counter, total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {k: round(v * 100.0 / total, 2) for k, v in counter.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze manifest quality and leakage risk")
    parser.add_argument("--manifests-dir", type=str, required=True)
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    splits = {}
    for split in ["train", "val", "test"]:
        path = manifests_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        splits[split] = load_rows(path)

    for split, rows in splits.items():
        cls = Counter(r["class_name"] for r in rows)
        ds = Counter(r["dataset"] for r in rows)
        vids = len({r["video_id"] for r in rows})
        print(f"\n[{split}] rows={len(rows)} videos={vids}")
        print(f" class_count={dict(cls)}")
        print(f" class_pct={pct(cls, len(rows))}")
        print(f" dataset_pct={pct(ds, len(rows))}")

    print("\n[overlap-check]")
    for field in ["video_id", "sample_id", "original_id", "identity_id", "source_id"]:
        tv = keyset(splits["train"], field) & keyset(splits["val"], field)
        tt = keyset(splits["train"], field) & keyset(splits["test"], field)
        vt = keyset(splits["val"], field) & keyset(splits["test"], field)
        print(f" {field}: train-val={len(tv)} train-test={len(tt)} val-test={len(vt)}")


if __name__ == "__main__":
    main()
