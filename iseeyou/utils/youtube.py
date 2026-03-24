from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PLACEHOLDER_TOKENS = {
    "video_id",
    "<video_id>",
    "{video_id}",
    "your_video_id",
    "shorts/video_id",
    "watch?v=video_id",
}


def _extract_video_id(url: str) -> str:
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


def extract_video_id(url: str) -> str:
    return _extract_video_id(url)


def validate_youtube_url(url: str) -> str:
    normalized = (url or "").strip()
    if not normalized:
        raise ValueError("YouTube URL is empty.")

    lower = normalized.lower()
    if any(token in lower for token in PLACEHOLDER_TOKENS):
        raise ValueError(
            "Placeholder URL detected (`VIDEO_ID`). "
            "Replace it with a real YouTube video id."
        )

    parsed = urlparse(normalized)
    host = parsed.netloc.lower()
    if "youtube.com" not in host and "youtu.be" not in host:
        raise ValueError("Invalid YouTube URL. Only youtube.com or youtu.be links are supported.")

    video_id = _extract_video_id(normalized)
    if not video_id:
        raise ValueError("Could not parse YouTube video id from URL.")

    if len(video_id) != 11:
        raise ValueError(
            f"Invalid YouTube video id length: `{video_id}`. "
            "Expected 11 characters."
        )

    return normalized


def resolve_downloaded_video_path(
    info: dict[str, Any],
    ydl: Any,
    download_dir: Path,
    video_extensions: set[str],
) -> Path | None:
    candidate = info
    entries = candidate.get("entries")
    if isinstance(entries, list) and entries:
        candidate = entries[0]

    requested_downloads = candidate.get("requested_downloads")
    if isinstance(requested_downloads, list):
        for item in requested_downloads:
            if not isinstance(item, dict):
                continue
            fp = item.get("filepath")
            if fp:
                path = Path(fp)
                if path.exists() and path.suffix.lower() in video_extensions:
                    return path

    for key in ("filepath", "_filename"):
        fp = candidate.get(key)
        if fp:
            path = Path(fp)
            if path.exists() and path.suffix.lower() in video_extensions:
                return path

    try:
        prepared = Path(ydl.prepare_filename(candidate))
        if prepared.exists() and prepared.suffix.lower() in video_extensions:
            return prepared
    except Exception:
        # TODO: add debug logger if needed.
        pass

    video_id = candidate.get("id")
    if isinstance(video_id, str) and video_id:
        id_matches = [
            p
            for p in download_dir.glob(f"{video_id}*")
            if p.is_file() and p.suffix.lower() in video_extensions
        ]
        if id_matches:
            return max(id_matches, key=lambda p: p.stat().st_mtime)

    return None


def find_downloaded_video_by_url(
    url: str,
    download_dir: Path,
    video_extensions: set[str],
) -> Path | None:
    video_id = _extract_video_id(url)
    if not video_id:
        return None

    matches = [
        p
        for p in download_dir.glob(f"{video_id}*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]
    if not matches:
        return None

    return max(matches, key=lambda p: p.stat().st_mtime)
