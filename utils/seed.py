"""Global random seed utilities for reproducible experiments."""
import random

import numpy as np

_DEFAULT_SEED = 42


def set_seed(seed: int = _DEFAULT_SEED) -> None:
    """Set seed for Python random and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
