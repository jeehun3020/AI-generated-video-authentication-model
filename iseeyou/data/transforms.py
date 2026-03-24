from __future__ import annotations

from torchvision import transforms

from .frequency import convert_representation, validate_representation


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RepresentationTransform:
    def __init__(self, input_representation: str):
        self.input_representation = validate_representation(input_representation)

    def __call__(self, image):
        return convert_representation(image, self.input_representation)


def build_train_transform(
    image_size: int,
    aug_cfg: dict | None = None,
    input_representation: str = "rgb",
):
    aug_cfg = aug_cfg or {}
    input_representation = validate_representation(input_representation)
    hflip_p = float(aug_cfg.get("hflip_p", 0.5))
    color_jitter = bool(aug_cfg.get("color_jitter", True))
    color_jitter_strength = float(aug_cfg.get("color_jitter_strength", 0.1))
    random_erasing = bool(aug_cfg.get("random_erasing", True))
    random_erasing_p = float(aug_cfg.get("random_erasing_p", 0.25))

    transform_steps = [
        transforms.Resize((image_size, image_size)),
        RepresentationTransform(input_representation),
        transforms.RandomHorizontalFlip(p=hflip_p),
    ]

    if color_jitter and input_representation in {"rgb", "rgb_fft"}:
        transform_steps.append(
            transforms.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=min(0.5 * color_jitter_strength, 0.05),
            )
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if random_erasing:
        transform_steps.append(transforms.RandomErasing(p=random_erasing_p, scale=(0.02, 0.15)))

    return transforms.Compose(
        transform_steps
    )


def build_eval_transform(image_size: int, input_representation: str = "rgb"):
    input_representation = validate_representation(input_representation)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            RepresentationTransform(input_representation),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
