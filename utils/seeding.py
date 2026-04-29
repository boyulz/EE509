"""Centralized seeding for reproducibility.

Call set_seed() at the start of every training run.
This seeds Python's random, NumPy, and PyTorch (CPU + CUDA),
and disables nondeterministic CuDNN kernels.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs and force deterministic CuDNN behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN. Slower, but reproducibility > throughput here.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False