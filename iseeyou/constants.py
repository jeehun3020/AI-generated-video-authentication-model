from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CLASSES = ["real", "generated", "deepfake"]


@dataclass
class TaskSpec:
    mode: str
    classes: list[str]
    positive_classes: list[str]

    @property
    def num_classes(self) -> int:
        if self.mode == "binary":
            return 2
        return len(self.classes)


class LabelMapper:
    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.class_to_idx = {name: idx for idx, name in enumerate(task_spec.classes)}

    def to_index(self, class_name: str) -> int:
        if self.task_spec.mode == "binary":
            return int(class_name in self.task_spec.positive_classes)

        if class_name not in self.class_to_idx:
            raise KeyError(f"Unknown class name: {class_name}")
        return self.class_to_idx[class_name]

    def index_to_name(self, index: int) -> str:
        if self.task_spec.mode == "binary":
            if index == 0:
                return "real"
            if len(self.task_spec.positive_classes) == 1:
                return self.task_spec.positive_classes[0]
            return "fake"

        return self.task_spec.classes[index]


def build_task_spec(cfg_task: dict) -> TaskSpec:
    mode = cfg_task.get("mode", "multiclass")
    classes = cfg_task.get("classes", DEFAULT_CLASSES)
    positive_classes = cfg_task.get("positive_classes", ["generated", "deepfake"])

    if mode not in {"multiclass", "binary"}:
        raise ValueError(f"Unsupported task mode: {mode}")

    return TaskSpec(mode=mode, classes=classes, positive_classes=positive_classes)
