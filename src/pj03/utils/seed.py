"""Deterministic seeding for python, numpy and torch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``.

    CPU-safe: importing torch is optional and only attempted if the package
    is present, so this module works in a minimal environment.

    Args:
        seed: The integer seed to apply everywhere.
        deterministic_torch: If True, ask cuDNN/TF32 for deterministic kernels
            (no-op on CPU-only installs).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
