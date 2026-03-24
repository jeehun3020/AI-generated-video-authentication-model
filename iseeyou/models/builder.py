from __future__ import annotations

import timm
import torch.nn as nn


class FrameClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x):
        return self.model(x)


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    # TODO: expose layer-freezing option for transfer learning.
    return FrameClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
