"""Bayesian Neural Network via Flax (JAX) + NumpPyro SVI.

Architecture
------------
- MLP with configurable depth/width
- Weight priors: Normal(0, prior_scale)
- Likelihood: Normal(mu, noise_std)
- Inference: SVI with AutoDiagonalNormal guide (mean-field VI)
- Prediction: posterior predictive mean and std

For larger datasets or longer training, SVI converges much faster than NUTS.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide

from src.utils.logging_utils import get_logger
from src.utils.seeds import jax_key

logger = get_logger(__name__)


# ── Flax MLP definition ───────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple multi-layer perceptron."""
    hidden_dims: Sequence[int]
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim)(x).squeeze(-1)


# ── NumpPyro BNN model ────────────────────────────────────────────────────────

def bnn_model(
    X: jnp.ndarray,
    y: jnp.ndarray | None = None,
    hidden_dims: tuple[int, ...] = (64, 64),
    prior_scale: float = 1.0,
) -> None:
    """NumpPyro BNN model.

    Places Normal priors over all Flax MLP parameters using numpyro.module.
    """
    n_features = X.shape[1]

    # Use numpyro's functional NN interface to sample weights
    mlp = MLP(hidden_dims=hidden_dims)
    params = numpyro.module("mlp", mlp, input_shape=(1, n_features),
                            prior=dist.Normal(0, prior_scale))
    mu = mlp.apply(params, X)

    noise = numpyro.sample("noise", dist.HalfNormal(0.5))
    numpyro.sample("obs", dist.Normal(mu, noise), obs=y)


class BayesianNNModel:
    """Bayesian Neural Network (Flax + NumpPyro SVI).

    Parameters
    ----------
    config : dict with keys:
        hidden_dims       : list[int] (default [128, 64, 32])
        prior_scale       : float (default 1.0)
        lr                : learning rate (default 0.005)
        num_steps         : SVI iterations (default 5000)
        num_pred_samples  : posterior samples for prediction (default 300)
        seed              : int (default 42)
        progress_bar      : bool (default False)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._guide = None
        self._svi_result = None
        self._n_features: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianNNModel":
        seed = self.config.get("seed", 42)
        hidden_dims = tuple(self.config.get("hidden_dims", [128, 64, 32]))
        prior_scale = self.config.get("prior_scale", 1.0)
        lr = self.config.get("lr", 0.005)
        num_steps = self.config.get("num_steps", 5000)
        progress = self.config.get("progress_bar", False)

        self._n_features = X.shape[1]
        X_jax = jnp.array(X, dtype=jnp.float32)
        y_jax = jnp.array(y, dtype=jnp.float32)

        key = jax_key(seed)

        model_fn = lambda X, y=None: bnn_model(X, y, hidden_dims, prior_scale)
        self._guide = autoguide.AutoDiagonalNormal(model_fn)
        optimizer = numpyro.optim.Adam(step_size=lr)
        svi = SVI(model_fn, self._guide, optimizer, loss=Trace_ELBO())
        self._svi_result = svi.run(key, num_steps, X_jax, y_jax,
                                   progress_bar=progress)

        final_loss = -float(self._svi_result.losses[-1])
        logger.info("BNN SVI finished: %d steps, final ELBO %.4f", num_steps, final_loss)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (posterior_mean, posterior_std) across predictive samples."""
        assert self._guide is not None, "Model not fitted."
        seed = self.config.get("seed", 42)
        hidden_dims = tuple(self.config.get("hidden_dims", [128, 64, 32]))
        prior_scale = self.config.get("prior_scale", 1.0)
        num_samples = self.config.get("num_pred_samples", 300)

        key = jax_key(seed + 999)
        X_jax = jnp.array(X, dtype=jnp.float32)

        model_fn = lambda X, y=None: bnn_model(X, y, hidden_dims, prior_scale)
        posterior = self._guide.sample_posterior(
            key, self._svi_result.params, sample_shape=(num_samples,)
        )
        predictive = Predictive(model_fn, posterior_samples=posterior, num_samples=num_samples)
        pred = predictive(key, X_jax)

        samples = np.array(pred["obs"])  # (num_samples, n_test)
        return samples.mean(axis=0), samples.std(axis=0)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "svi_result": self._svi_result,
            "n_features": self._n_features,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("BNN saved to %s", path)

    def load(self, path: str | Path) -> "BayesianNNModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.config = payload["config"]
        self._svi_result = payload["svi_result"]
        self._n_features = payload["n_features"]
        # Rebuild guide
        hidden_dims = tuple(self.config.get("hidden_dims", [128, 64, 32]))
        prior_scale = self.config.get("prior_scale", 1.0)
        model_fn = lambda X, y=None: bnn_model(X, y, hidden_dims, prior_scale)
        self._guide = autoguide.AutoDiagonalNormal(model_fn)
        return self
