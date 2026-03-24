#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


CSV_COLUMNS = [
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


@dataclass
class CollectStats:
    query_count: int = 0
    channel_count: int = 0
    seen_entries: int = 0
    kept_entries: int = 0
    skipped_duplicates: int = 0
    skipped_not_short: int = 0
    skipped_duration: int = 0
    skipped_missing_id: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect YouTube Shorts URLs from search queries and channel shorts tabs."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/youtube/shorts_urls.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Search query (repeatable). Example: --query \"ai cooking\"",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default="",
        help="Text file with one search query per line",
    )
    parser.add_argument(
        "--channel",
        action="append",
        default=[],
        help="Channel URL or @handle URL (repeatable). Example: --channel https://www.youtube.com/@channelname",
    )
    parser.add_argument(
        "--channels-file",
        type=str,
        default="",
        help="Text file with one channel URL per line",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=120,
        help="Maximum results per query",
    )
    parser.add_argument(
        "--max-per-channel",
        type=int,
        default=200,
        help="Maximum shorts per channel",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=70,
        help="Max duration in seconds for shorts filtering",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=3,
        help="Min duration in seconds (skip near-empty clips)",
    )
    parser.add_argument(
        "--allow-non-shorts-url",
        action="store_true",
        help="If set, keep entries even when URL doesn't look like /shorts/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite output CSV instead of appending/merging",
    )
    parser.add_argument(
        "--suggested-label",
        type=str,
        default="",
        help="Optional suggested label to write in CSV (e.g., real/generated/deepfake)",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=0.0,
        help="yt-dlp sleep interval between requests",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show yt-dlp logs",
    )
    return parser.parse_args()


def read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def parse_video_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path_parts = [p for p in parsed.path.split("/") if p]

    if "youtu.be" in host and path_parts:
        return path_parts[0]

    if "youtube.com" in host:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            return path_parts[1]

    return ""


def normalize_channel_shorts_url(channel: str) -> str:
    url = channel.strip()
    if not url:
        return url

    if url.startswith("@"):
        return f"https://www.youtube.com/{url}/shorts"

    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://www.youtube.com/{url.lstrip('/')}"

    url = url.rstrip("/")
    parsed = urlparse(url)
    if "youtube.com" not in parsed.netloc.lower():
        raise ValueError(f"Not a YouTube channel URL: {channel}")

    if parsed.path.endswith("/shorts"):
        return url
    return f"{url}/shorts"


def entry_to_row(
    entry: dict[str, Any],
    source_type: str,
    source_value: str,
    max_duration: int,
    min_duration: int,
    require_shorts_url: bool,
    suggested_label: str,
    shorts_context: bool = False,
) -> dict[str, str] | None:
    video_id = str(entry.get("id") or "").strip()
    if not video_id:
        video_id = parse_video_id_from_url(str(entry.get("url") or ""))
    if not video_id:
        video_id = parse_video_id_from_url(str(entry.get("webpage_url") or ""))
    if not video_id:
        return None

    webpage_url = str(entry.get("webpage_url") or "").strip()
    if not webpage_url:
        webpage_url = f"https://www.youtube.com/watch?v={video_id}"

    shorts_url = f"https://www.youtube.com/shorts/{video_id}"
    duration_val = entry.get("duration")
    duration = int(duration_val) if isinstance(duration_val, (int, float)) else -1

    if duration >= 0:
        if duration > max_duration or duration < min_duration:
            return {}

    looks_like_short = shorts_context or "/shorts/" in webpage_url or (duration >= 0 and duration <= max_duration)
    if require_shorts_url and not looks_like_short:
        return {"_skip_reason": "not_short"}

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = {
        "video_id": video_id,
        "shorts_url": shorts_url,
        "webpage_url": webpage_url,
        "title": str(entry.get("title") or ""),
        "uploader": str(entry.get("uploader") or entry.get("channel") or ""),
        "channel_id": str(entry.get("channel_id") or ""),
        "duration_sec": str(duration) if duration >= 0 else "",
        "upload_date": str(entry.get("upload_date") or ""),
        "view_count": str(entry.get("view_count") or ""),
        "source_type": source_type,
        "source_value": source_value,
        "collected_at_utc": now_utc,
        "suggested_label": suggested_label,
        "note": "",
    }
    return row


def load_existing_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    if not path.exists():
        return [], set()

    rows: list[dict[str, str]] = []
    video_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {k: row.get(k, "") for k in CSV_COLUMNS}
            rows.append(normalized)
            vid = normalized.get("video_id", "").strip()
            if vid:
                video_ids.add(vid)
    return rows, video_ids


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def _extract_entries(info: dict[str, Any]) -> list[dict[str, Any]]:
    entries = info.get("entries")
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return [info] if isinstance(info, dict) else []


def build_ydl_opts(max_results: int, sleep_interval: float, verbose: bool) -> dict[str, Any]:
    return {
        "skip_download": True,
        "extract_flat": True,
        "quiet": not verbose,
        "no_warnings": not verbose,
        "ignoreerrors": True,
        "playlistend": max_results,
        "sleep_interval": max(0.0, sleep_interval),
    }


def collect_from_queries(
    ydl: Any,
    queries: list[str],
    max_per_query: int,
    max_duration: int,
    min_duration: int,
    require_shorts_url: bool,
    suggested_label: str,
    existing_video_ids: set[str],
    stats: CollectStats,
) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []

    for query in queries:
        stats.query_count += 1
        search_query = query if "shorts" in query.lower() else f"{query} shorts"
        search_url = f"ytsearch{max_per_query}:{search_query}"
        info = ydl.extract_info(search_url, download=False)
        for entry in _extract_entries(info or {}):
            stats.seen_entries += 1
            row = entry_to_row(
                entry=entry,
                source_type="query",
                source_value=query,
                max_duration=max_duration,
                min_duration=min_duration,
                require_shorts_url=require_shorts_url,
                suggested_label=suggested_label,
            )
            if row is None:
                stats.skipped_missing_id += 1
                continue
            if row == {}:
                stats.skipped_duration += 1
                continue
            if row.get("_skip_reason") == "not_short":
                stats.skipped_not_short += 1
                continue

            vid = row["video_id"]
            if vid in existing_video_ids:
                stats.skipped_duplicates += 1
                continue

            existing_video_ids.add(vid)
            stats.kept_entries += 1
            collected.append(row)
    return collected


def collect_from_channels(
    ydl: Any,
    channels: list[str],
    max_per_channel: int,
    max_duration: int,
    min_duration: int,
    require_shorts_url: bool,
    suggested_label: str,
    existing_video_ids: set[str],
    stats: CollectStats,
) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []

    for channel in channels:
        stats.channel_count += 1
        channel_shorts_url = normalize_channel_shorts_url(channel)
        info = ydl.extract_info(channel_shorts_url, download=False)
        entries = _extract_entries(info or {})[:max_per_channel]
        for entry in entries:
            stats.seen_entries += 1
            row = entry_to_row(
                entry=entry,
                source_type="channel",
                source_value=channel_shorts_url,
                max_duration=max_duration,
                min_duration=min_duration,
                require_shorts_url=require_shorts_url,
                suggested_label=suggested_label,
                shorts_context=True,
            )
            if row is None:
                stats.skipped_missing_id += 1
                continue
            if row == {}:
                stats.skipped_duration += 1
                continue
            if row.get("_skip_reason") == "not_short":
                stats.skipped_not_short += 1
                continue

            vid = row["video_id"]
            if vid in existing_video_ids:
                stats.skipped_duplicates += 1
                continue

            existing_video_ids.add(vid)
            stats.kept_entries += 1
            collected.append(row)
    return collected


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)

    queries = [q.strip() for q in args.query if q.strip()]
    channels = [c.strip() for c in args.channel if c.strip()]

    if args.queries_file:
        queries.extend(read_text_lines(Path(args.queries_file)))
    if args.channels_file:
        channels.extend(read_text_lines(Path(args.channels_file)))

    # Preserve order while removing duplicates.
    queries = list(dict.fromkeys(queries))
    channels = list(dict.fromkeys(channels))

    if not queries and not channels:
        raise SystemExit("[ERROR] Provide at least one --query/--queries-file or --channel/--channels-file.")

    existing_rows: list[dict[str, str]] = []
    existing_ids: set[str] = set()
    if output_csv.exists() and not args.overwrite:
        existing_rows, existing_ids = load_existing_rows(output_csv)

    try:
        from yt_dlp import YoutubeDL  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("[ERROR] yt-dlp is required. Install with: pip install yt-dlp") from exc

    stats = CollectStats()
    ydl_opts = build_ydl_opts(
        max_results=max(args.max_per_query, args.max_per_channel),
        sleep_interval=args.sleep_interval,
        verbose=args.verbose,
    )

    with YoutubeDL(ydl_opts) as ydl:
        new_rows: list[dict[str, str]] = []
        if queries:
            new_rows.extend(
                collect_from_queries(
                    ydl=ydl,
                    queries=queries,
                    max_per_query=max(1, args.max_per_query),
                    max_duration=max(1, args.max_duration),
                    min_duration=max(0, args.min_duration),
                    require_shorts_url=not args.allow_non_shorts_url,
                    suggested_label=args.suggested_label.strip(),
                    existing_video_ids=existing_ids,
                    stats=stats,
                )
            )
        if channels:
            new_rows.extend(
                collect_from_channels(
                    ydl=ydl,
                    channels=channels,
                    max_per_channel=max(1, args.max_per_channel),
                    max_duration=max(1, args.max_duration),
                    min_duration=max(0, args.min_duration),
                    require_shorts_url=not args.allow_non_shorts_url,
                    suggested_label=args.suggested_label.strip(),
                    existing_video_ids=existing_ids,
                    stats=stats,
                )
            )

    merged_rows = new_rows if args.overwrite else (existing_rows + new_rows)
    write_rows(output_csv, merged_rows)

    print("[INFO] shorts collection finished")
    print(f"[INFO] output_csv={output_csv}")
    print(f"[INFO] total_rows={len(merged_rows)} (new={len(new_rows)}, existing={len(existing_rows)})")
    print(
        "[INFO] stats: "
        f"queries={stats.query_count}, channels={stats.channel_count}, seen={stats.seen_entries}, "
        f"kept={stats.kept_entries}, dup={stats.skipped_duplicates}, "
        f"not_short={stats.skipped_not_short}, duration_skip={stats.skipped_duration}, "
        f"missing_id={stats.skipped_missing_id}"
    )


if __name__ == "__main__":
    main()
