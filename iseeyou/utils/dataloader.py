from __future__ import annotations

import platform


def resolve_num_workers(requested: int) -> int:
    workers = max(0, int(requested))
    if platform.system() == "Darwin" and workers > 0:
        print(
            "[WARN] num_workers>0 is not reliable on this macOS environment. "
            "Falling back to num_workers=0."
        )
        return 0
    return workers
