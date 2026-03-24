from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # TODO: set deterministic=True only when reproducibility is more important than speed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
