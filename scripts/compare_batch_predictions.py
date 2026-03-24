#!/usr/bin/env python
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two batch inference result JSON files and summarize prediction differences."
    )
    parser.add_argument("--left-json", type=str, required=True, help="Reference batch result JSON")
    parser.add_argument("--right-json", type=str, required=True, help="Candidate batch result JSON")
    parser.add_argument("--left-name", type=str, default="left", help="Display name for left JSON")
    parser.add_argument("--right-name", type=str, default="right", help="Display name for right JSON")
    parser.add_argument("--expected-label", type=str, default="", help="Optional expected label for hit-rate summary")
    parser.add_argument("--output-json", type=str, required=True, help="Where to save comparison JSON")
    parser.add_argument("--output-md", type=str, default="", help="Optional markdown summary path")
    return parser.parse_args()


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_id_from_relative_path(relative_path: str) -> str:
    parts = relative_path.split("/", 1)
    return parts[0] if parts else "unknown"


def expected_hit_rate(results: list[dict], expected_label: str) -> float | None:
    if not expected_label:
        return None
    if not results:
        return 0.0
    hits = sum(1 for row in results if row.get("prediction") == expected_label)
    return hits / len(results)


def source_breakdown(results: list[dict]) -> dict[str, dict[str, int]]:
    per_source: dict[str, Counter] = defaultdict(Counter)
    for row in results:
        source_id = source_id_from_relative_path(str(row.get("relative_path", "")))
        per_source[source_id][str(row.get("prediction", "unknown"))] += 1
    return {source: dict(counter) for source, counter in sorted(per_source.items())}


def changed_predictions(left_results: dict[str, dict], right_results: dict[str, dict]) -> list[dict]:
    changed: list[dict] = []
    for relative_path, left_row in sorted(left_results.items()):
        right_row = right_results.get(relative_path)
        if right_row is None:
            continue
        if left_row.get("prediction") == right_row.get("prediction"):
            continue
        changed.append(
            {
                "relative_path": relative_path,
                "left_prediction": left_row.get("prediction"),
                "right_prediction": right_row.get("prediction"),
                "generated_prob": float(left_row.get("generated_prob", 0.0)),
                "frame_generated_prob": float(left_row.get("frame_generated_prob", 0.0)),
                "temporal_generated_prob": float(left_row.get("temporal_generated_prob", 0.0)),
                "right_decision_reason": right_row.get("decision_reason", ""),
            }
        )
    return changed


def to_result_map(results: list[dict]) -> dict[str, dict]:
    return {str(row.get("relative_path", "")): row for row in results}


def build_markdown(summary: dict, left_name: str, right_name: str, expected_label: str) -> str:
    left = summary["left"]
    right = summary["right"]
    lines = [
        "# Batch Prediction Comparison",
        "",
        f"- Left: `{left_name}`",
        f"- Right: `{right_name}`",
        f"- Videos: `{summary['num_videos']}`",
    ]
    if expected_label:
        lines.append(f"- Expected label: `{expected_label}`")
    lines.extend(
        [
            "",
            "## Summary",
            f"- `{left_name}` prediction counts: `{left['prediction_counts']}`",
            f"- `{right_name}` prediction counts: `{right['prediction_counts']}`",
        ]
    )
    if expected_label:
        lines.extend(
            [
                f"- `{left_name}` expected-label hit rate: `{left['expected_label_rate']:.4f}`",
                f"- `{right_name}` expected-label hit rate: `{right['expected_label_rate']:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Source Breakdown",
            f"- `{left_name}`: `{left['source_breakdown']}`",
            f"- `{right_name}`: `{right['source_breakdown']}`",
            "",
            "## Changed Predictions",
            f"- Count: `{len(summary['changed_predictions'])}`",
        ]
    )
    for row in summary["changed_predictions"][:20]:
        lines.append(
            f"- `{row['relative_path']}`: `{left_name}={row['left_prediction']}`, "
            f"`{right_name}={row['right_prediction']}`, "
            f"`generated_prob={row['generated_prob']:.4f}`, "
            f"`temporal_generated_prob={row['temporal_generated_prob']:.4f}`, "
            f"`reason={row['right_decision_reason']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    left_json = Path(args.left_json)
    right_json = Path(args.right_json)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    left_payload = load_results(left_json)
    right_payload = load_results(right_json)
    left_results = left_payload.get("results", [])
    right_results = right_payload.get("results", [])

    left_map = to_result_map(left_results)
    right_map = to_result_map(right_results)

    common_paths = sorted(set(left_map) & set(right_map))
    left_common = [left_map[path] for path in common_paths]
    right_common = [right_map[path] for path in common_paths]

    summary = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "expected_label": args.expected_label,
        "num_videos": len(common_paths),
        "left": {
            "prediction_counts": dict(Counter(str(row.get("prediction", "unknown")) for row in left_common)),
            "expected_label_rate": expected_hit_rate(left_common, args.expected_label),
            "source_breakdown": source_breakdown(left_common),
        },
        "right": {
            "prediction_counts": dict(Counter(str(row.get("prediction", "unknown")) for row in right_common)),
            "expected_label_rate": expected_hit_rate(right_common, args.expected_label),
            "source_breakdown": source_breakdown(right_common),
        },
        "changed_predictions": changed_predictions(left_map, right_map),
    }

    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] saved comparison json: {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(
            build_markdown(summary, args.left_name, args.right_name, args.expected_label),
            encoding="utf-8",
        )
        print(f"[INFO] saved comparison markdown: {output_md}")


if __name__ == "__main__":
    main()
