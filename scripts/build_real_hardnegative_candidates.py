#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


IN_COLUMNS = [
    "video_id",
    "shorts_url",
    "webpage_url",
    "title",
    "uploader",
    "channel_id",
    "duration_sec",
    "upload_date",
    "view_count",
    "source_type",
    "source_value",
    "collected_at_utc",
    "suggested_label",
    "note",
]

OUT_COLUMNS = IN_COLUMNS + ["heuristic_score", "decision_reason", "review_bucket"]

STRONG_REJECT = {
    " ai ",
    "#ai",
    "ai ",
    " ai",
    "sora",
    "midjourney",
    "generated",
    "aiart",
    "ai video",
    "aivideo",
    "ai asmr",
    "ai glass",
    "animated asmr",
    "animation",
}

SOFT_REJECT = {
    "fake food",
    "fake or real",
    "real or fake",
}

POSITIVE_HINTS = {
    "illusion",
    "cake",
    "hyperrealistic",
    "hyper realistic",
    "mirror glaze",
    "resin",
    "soap",
    "jelly",
    "candy",
    "glass",
    "fruit carving",
    "food art",
    "dessert",
    "asmr",
    "miniature",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter query-collected Shorts into real hard-negative review candidates."
    )
    parser.add_argument("--input-csv", required=True, help="Raw query collection CSV")
    parser.add_argument("--output-csv", required=True, help="Filtered candidate CSV")
    parser.add_argument("--excluded-csv", required=True, help="Excluded rows CSV")
    parser.add_argument(
        "--min-score",
        type=int,
        default=1,
        help="Minimum heuristic score to keep as candidate",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return f" {text.strip().lower()} "


def score_row(row: dict[str, str]) -> tuple[int, str, str]:
    text = normalize(f"{row.get('title', '')} {row.get('uploader', '')} {row.get('source_value', '')}")
    reasons: list[str] = []
    score = 0

    for token in STRONG_REJECT:
        if token in text:
            score -= 4
            reasons.append(f"strong_reject:{token.strip()}")

    for token in SOFT_REJECT:
        if token in text:
            score -= 1
            reasons.append(f"soft_reject:{token.strip()}")

    for token in POSITIVE_HINTS:
        if token in text:
            score += 1
            reasons.append(f"positive:{token}")

    query = normalize(row.get("source_value", ""))
    if "ultra realistic asmr" in query or "glass fruit" in query:
        score -= 1
        reasons.append("query_bias:often_generated")

    if "mirror glaze" in query or "fruit carving" in query or "miniature cooking" in query:
        score += 1
        reasons.append("query_bias:likely_real")

    if score >= 3:
        bucket = "priority_review"
    elif score >= 1:
        bucket = "review"
    else:
        bucket = "exclude"

    return score, ";".join(reasons) if reasons else "neutral", bucket


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k: row.get(k, "") for k in IN_COLUMNS} for row in csv.DictReader(f)]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in OUT_COLUMNS})


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input_csv))
    kept: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []

    for row in rows:
        score, reason, bucket = score_row(row)
        output_row = dict(row)
        output_row["heuristic_score"] = str(score)
        output_row["decision_reason"] = reason
        output_row["review_bucket"] = bucket

        if score >= args.min_score:
            kept.append(output_row)
        else:
            excluded.append(output_row)

    kept.sort(key=lambda r: (int(r["heuristic_score"]), r["title"]), reverse=True)
    excluded.sort(key=lambda r: (int(r["heuristic_score"]), r["title"]))

    write_rows(Path(args.output_csv), kept)
    write_rows(Path(args.excluded_csv), excluded)

    print("[INFO] real hard-negative candidate filtering finished")
    print(f"[INFO] input={args.input_csv}")
    print(f"[INFO] kept={len(kept)} excluded={len(excluded)} min_score={args.min_score}")
    print(f"[INFO] output_csv={args.output_csv}")
    print(f"[INFO] excluded_csv={args.excluded_csv}")


if __name__ == "__main__":
    main()
