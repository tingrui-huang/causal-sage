"""
Reproducibility utilities.

Centralizes all randomness control (Python / NumPy / PyTorch) so that
entrypoints can make experiments repeatable with a single call.
"""

from __future__ import annotations

from typing import Optional


def set_global_seed(
    seed: int,
    *,
    deterministic_torch: bool = False,
    set_pythonhashseed: bool = True,
) -> None:
    """
    Set RNG seeds for common libraries.

    Notes:
    - Setting PYTHONHASHSEED is best-effort. For full effect it should be set
      before the Python process starts; we still set it here for transparency.
    - deterministic_torch may reduce performance and may raise errors if
      non-deterministic ops are used.
    """
    if seed is None:
        raise ValueError("seed must be an int (got None)")

    # Python stdlib
    import os
    import random

    random.seed(seed)

    if set_pythonhashseed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # NumPy not installed or other import issue: ignore
        pass

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            # cuDNN settings (safe even on CPU-only builds)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

            # Force deterministic algorithms where supported
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        # PyTorch not installed or other import issue: ignore
        pass


