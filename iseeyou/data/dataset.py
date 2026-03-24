from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from iseeyou.constants import LabelMapper, TaskSpec

from .manifest import read_manifest


class FaceFrameDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        task_spec: TaskSpec,
        transform=None,
    ):
        self.rows = read_manifest(manifest_path)
        self.transform = transform
        self.label_mapper = LabelMapper(task_spec)
        self.labels = [self.label_mapper.to_index(row["class_name"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]

        image_path = row["frame_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label_mapper.to_index(row["class_name"])

        return {
            "image": image,
            "label": label,
            "video_id": row["video_id"],
            "frame_path": image_path,
            "class_name": row["class_name"],
        }

    def get_labels(self) -> list[int]:
        return self.labels
