from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class RawSample:
    dataset: str
    path: Path
    rel_path: str
    media_type: str  # video | image
    class_name: str
    video_id: str
    identity_id: str
    source_id: str
    original_id: str

    @property
    def sample_id(self) -> str:
        raw = f"{self.dataset}::{self.rel_path}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]



def _scan_files(root: Path, exts: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    return sorted(files)



def _identity_from_parts(parts: tuple[str, ...]) -> str:
    for part in parts:
        if part.startswith("id") and len(part) >= 3:
            return part
    return parts[0] if parts else ""



def parse_ucf101(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    class_name = cfg.get("class_name", "real")
    samples: list[RawSample] = []

    for path in _scan_files(root, VIDEO_EXTS):
        rel = path.relative_to(root).as_posix()
        stem = path.stem
        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id="",
                source_id=stem,
                original_id=stem,
            )
        )

    return samples



def parse_voxceleb2(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    class_name = cfg.get("class_name", "real")
    samples: list[RawSample] = []

    for path in _scan_files(root, VIDEO_EXTS):
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        identity = _identity_from_parts(rel_path.parts)

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=identity,
                source_id=rel_path.parent.as_posix(),
                original_id=path.stem,
            )
        )

    return samples



def parse_stylegan(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    class_name = cfg.get("class_name", "generated")
    samples: list[RawSample] = []

    for path in _scan_files(root, IMAGE_EXTS):
        rel = path.relative_to(root).as_posix()
        stem = path.stem
        identity = stem.split("_")[0]

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="image",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=identity,
                source_id=identity,
                original_id=stem,
            )
        )

    return samples



def parse_faceforensicspp(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    include_original_sequences = cfg.get("include_original_sequences", False)
    deepfake_dirs = {
        "deepfakedetection",
        "deepfakes",
        "faceswap",
        "face2face",
        "faceshifter",
        "neuraltextures",
    }
    samples: list[RawSample] = []

    for path in _scan_files(root, VIDEO_EXTS):
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        rel_lower = rel.lower()
        top_dir_lower = rel_path.parts[0].lower() if rel_path.parts else ""

        if "manipulated_sequences" in rel_lower or top_dir_lower in deepfake_dirs:
            class_name = cfg.get("class_name", "deepfake")
        elif include_original_sequences and (
            "original_sequences" in rel_lower or top_dir_lower == "original"
        ):
            class_name = "real"
        else:
            continue

        stem = path.stem
        match = re.search(r"(\d{2,3})_(\d{2,3})", stem)
        if match:
            a, b = match.group(1), match.group(2)
            original_id = "_".join(sorted([a, b]))
            identity_id = f"{a}_{b}"
        else:
            original_id = stem
            identity_id = stem

        parts = rel_path.parts
        method = "unknown"
        if "manipulated_sequences" in parts:
            idx = parts.index("manipulated_sequences")
            if idx + 1 < len(parts):
                method = parts[idx + 1]
        elif len(parts) > 0:
            method = parts[0]

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=identity_id,
                source_id=f"{method}:{original_id}",
                original_id=original_id,
            )
        )

    return samples



def parse_celebdf(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    include_real = cfg.get("include_real", False)
    samples: list[RawSample] = []

    for path in _scan_files(root, VIDEO_EXTS | IMAGE_EXTS):
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        rel_lower = rel.lower()
        media_type = "video" if path.suffix.lower() in VIDEO_EXTS else "image"

        if (
            "synthesis" in rel_lower
            or "/fake/" in rel_lower
            or rel_lower.startswith("fake/")
            or rel_lower.startswith("train/fake/")
            or rel_lower.startswith("val/fake/")
            or rel_lower.startswith("test/fake/")
        ):
            class_name = cfg.get("class_name", "deepfake")
        elif include_real and (
            "/real/" in rel_lower
            or rel_lower.startswith("real/")
            or rel_lower.startswith("train/real/")
            or rel_lower.startswith("val/real/")
            or rel_lower.startswith("test/real/")
        ):
            class_name = "real"
        else:
            continue

        stem = path.stem
        parts = stem.split("_")
        id_tokens = [p for p in parts if p.startswith("id")]
        if len(id_tokens) >= 2:
            pair = sorted(id_tokens[:2])
            identity_id = f"{pair[0]}_{pair[1]}"
        elif len(id_tokens) == 1:
            identity_id = id_tokens[0]
        else:
            identity_id = parts[0] if parts else stem

        # For Celeb_V2 image-heavy splits, grouping by per-frame stem can leak identity.
        # Use identity-derived original_id to keep same identity pair in one split.
        original_id = identity_id if media_type == "image" else stem

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type=media_type,
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=identity_id,
                source_id=rel_path.parent.as_posix(),
                original_id=original_id,
            )
        )

    return samples



def parse_generic(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    media_type = cfg.get("media_type", "video")
    class_name = cfg.get("class_name", "real")

    exts = VIDEO_EXTS if media_type == "video" else IMAGE_EXTS
    samples: list[RawSample] = []

    for path in _scan_files(root, exts):
        rel = path.relative_to(root).as_posix()
        stem = path.stem
        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type=media_type,
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id="",
                source_id=stem,
                original_id=stem,
            )
        )

    return samples


def parse_youtube_shorts(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    class_name = cfg.get("class_name", "real")
    samples: list[RawSample] = []

    for path in _scan_files(root, VIDEO_EXTS):
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        parts = rel_path.parts
        channel_slug = parts[0] if parts else "unknown_channel"
        stem = path.stem

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=channel_slug,
                source_id=channel_slug,
                original_id=stem,
            )
        )

    return samples


PARSERS = {
    "ucf101": parse_ucf101,
    "voxceleb2": parse_voxceleb2,
    "stylegan": parse_stylegan,
    "faceforensicspp": parse_faceforensicspp,
    "celebdf": parse_celebdf,
    "generic": parse_generic,
    "youtube_shorts": parse_youtube_shorts,
}



def collect_samples_from_config(datasets_cfg: dict) -> list[RawSample]:
    all_samples: list[RawSample] = []

    for dataset_name, cfg in datasets_cfg.items():
        if not cfg.get("enabled", True):
            continue

        root = Path(cfg["root"]).expanduser()
        if not root.exists():
            print(f"[WARN] dataset root not found, skipping: {dataset_name} -> {root}")
            continue

        parser_name = cfg.get("parser", dataset_name).lower()
        parser_fn = PARSERS.get(parser_name)
        if parser_fn is None:
            print(f"[WARN] unknown parser={parser_name}, falling back to generic")
            parser_fn = parse_generic

        dataset_samples = parser_fn(dataset_name, root, cfg)
        max_samples = int(cfg.get("max_samples", 0) or 0)
        if max_samples > 0 and len(dataset_samples) > max_samples:
            # TODO: support randomized but reproducible sampling with config seed.
            dataset_samples = dataset_samples[:max_samples]
        print(f"[INFO] {dataset_name}: {len(dataset_samples)} samples")
        all_samples.extend(dataset_samples)

    return all_samples
