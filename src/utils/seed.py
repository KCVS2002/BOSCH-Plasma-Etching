"""Global seeding for reproducibility.

Call `set_seed(seed)` at the top of every script that involves randomness
(data splitting, model init, dropout, etc.). Record the seed in the
experiment's config.yaml so runs are reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
