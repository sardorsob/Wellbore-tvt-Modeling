"""Random seed management for reproducibility."""

import random
import numpy as np


def set_global_seeds(seed: int = 42) -> None:
    """Set seeds for Python random, NumPy, and optionally JAX."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_seed_dict(seed: int = 42) -> dict:
    """Return seed dict suitable for logging in seeds.json."""
    return {
        "global_seed": seed,
        "numpy_seed": seed,
        "python_random_seed": seed,
        "jax_key": [0, seed],  # jax.random.PRNGKey(seed)
        "torch_seed": seed,
    }


def jax_key(seed: int = 42):
    """Return a JAX PRNGKey. Import is deferred so JAX is optional."""
    import jax
    return jax.random.PRNGKey(seed)
