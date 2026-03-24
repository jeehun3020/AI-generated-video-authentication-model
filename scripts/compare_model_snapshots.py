#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two experiment snapshots and render a markdown report.")
    parser.add_argument("--before-val", required=True)
    parser.add_argument("--before-test", required=True)
    parser.add_argument("--before-hardneg", required=True)
    parser.add_argument("--before-hardneg-query", required=True)
    parser.add_argument("--after-val", required=True)
    parser.add_argument("--after-test", required=True)
    parser.add_argument("--after-hardneg", required=True)
    parser.add_argument("--after-hardneg-query", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_delta(after: float, before: float) -> str:
    delta = after - before
    return f"{after:.4f} ({delta:+.4f})"


def main() -> None:
    args = parse_args()
    b_val = load_json(args.before_val)
    b_test = load_json(args.before_test)
    b_hn = load_json(args.before_hardneg)
    b_hnq = load_json(args.before_hardneg_query)
    a_val = load_json(args.after_val)
    a_test = load_json(args.after_test)
    a_hn = load_json(args.after_hardneg)
    a_hnq = load_json(args.after_hardneg_query)

    lines = [
        "# Model Snapshot Comparison",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
        f"| Val video F1 | {b_val['video']['f1']:.4f} | {a_val['video']['f1']:.4f} | {(a_val['video']['f1'] - b_val['video']['f1']):+.4f} |",
        f"| Val video AUC | {b_val['video']['auc']:.4f} | {a_val['video']['auc']:.4f} | {(a_val['video']['auc'] - b_val['video']['auc']):+.4f} |",
        f"| Test video F1 | {b_test['video']['f1']:.4f} | {a_test['video']['f1']:.4f} | {(a_test['video']['f1'] - b_test['video']['f1']):+.4f} |",
        f"| Test video AUC | {b_test['video']['auc']:.4f} | {a_test['video']['auc']:.4f} | {(a_test['video']['auc'] - b_test['video']['auc']):+.4f} |",
        f"| Original hard-negative real hit rate | {b_hn['real_hit_rate']:.4f} | {a_hn['real_hit_rate']:.4f} | {(a_hn['real_hit_rate'] - b_hn['real_hit_rate']):+.4f} |",
        f"| Query hard-negative real hit rate | {b_hnq['real_hit_rate']:.4f} | {a_hnq['real_hit_rate']:.4f} | {(a_hnq['real_hit_rate'] - b_hnq['real_hit_rate']):+.4f} |",
        "",
        "## Notes",
        "",
        f"- Original hard-negative after retraining: `{a_hn['prediction_counts']['real']}/{a_hn['num_videos']}` real.",
        f"- Query hard-negative after retraining: `{a_hnq['prediction_counts']['real']}/{a_hnq['num_videos']}` real.",
        "- The original hard-negative set now contains seen sources, so its post-train improvement should not be treated as unseen generalization.",
        "- The query hard-negative set is the more useful proxy for unseen hard-negative behavior here.",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] saved comparison report: {args.output_md}")


if __name__ == "__main__":
    main()
