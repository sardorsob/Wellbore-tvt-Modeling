"""Bayesian Gaussian Process model via NumpPyro (JAX backend).

Architecture
------------
- Kernel: Matérn 5/2 (smooth but non-analytic — good for geological sequences)
- Inference: NUTS (exact MCMC) for small datasets; SVI (ADVI) for speed
- Prediction: posterior predictive mean ± std

Usage
-----
    model = BayesianGPModel(config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_mean, y_std = model.predict_with_uncertainty(X_test)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.einstein import SteinVI  # noqa — optional import guard
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, autoguide

from src.utils.logging_utils import get_logger
from src.utils.seeds import jax_key

logger = get_logger(__name__)

numpyro.set_host_device_count(1)


def _rbf_kernel(x1: jnp.ndarray, x2: jnp.ndarray, length_scale: float, amplitude: float) -> jnp.ndarray:
    """Squared-exponential (RBF) kernel."""
    diff = x1[:, None, :] - x2[None, :, :]  # (n, m, d)
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return amplitude ** 2 * jnp.exp(-0.5 * sq_dist / length_scale ** 2)


def _matern52_kernel(x1: jnp.ndarray, x2: jnp.ndarray, length_scale: float, amplitude: float) -> jnp.ndarray:
    """Matérn 5/2 kernel — C² smooth, better for geological sequences than RBF."""
    diff = x1[:, None, :] - x2[None, :, :]
    r = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)
    r_scaled = jnp.sqrt(5.0) * r / length_scale
    return amplitude ** 2 * (1.0 + r_scaled + r_scaled ** 2 / 3.0) * jnp.exp(-r_scaled)


def gp_model(X: jnp.ndarray, y: jnp.ndarray | None = None, kernel: str = "matern52") -> None:
    """NumpPyro GP model over input features X.

    For full datasets, use sparse GP (inducing-point approximation) to control
    the O(n³) cost.  This implementation is exact — suitable for ≤2000 points.
    """
    n = X.shape[0]

    # Priors
    amplitude = numpyro.sample("amplitude", dist.LogNormal(0.0, 1.0))
    length_scale = numpyro.sample("length_scale", dist.LogNormal(0.0, 1.0))
    noise = numpyro.sample("noise", dist.LogNormal(-2.0, 1.0))

    # Kernel matrix
    if kernel == "matern52":
        K = _matern52_kernel(X, X, length_scale, amplitude)
    else:
        K = _rbf_kernel(X, X, length_scale, amplitude)

    K = K + (noise ** 2 + 1e-6) * jnp.eye(n)

    # Likelihood
    numpyro.sample("obs", dist.MultivariateNormal(jnp.zeros(n), K), obs=y)


class BayesianGPModel:
    """Gaussian Process model with JAX/NumpPyro backend.

    Parameters
    ----------
    config : dict with keys:
        inference      : "nuts" | "svi"  (default "svi")
        num_samples    : MCMC samples or SVI steps (default 1000)
        warmup_steps   : NUTS warmup (default 500)
        kernel         : "matern52" | "rbf" (default "matern52")
        seed           : RNG seed (default 42)
        max_train_rows : subsample training set if too large (default 2000)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._mcmc = None
        self._svi_result = None
        self._guide = None
        self._X_train = None
        self._y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianGPModel":
        seed = self.config.get("seed", 42)
        inference = self.config.get("inference", "svi")
        kernel = self.config.get("kernel", "matern52")
        max_rows = self.config.get("max_train_rows", 2000)

        # Subsample for tractability
        if len(X) > max_rows:
            logger.warning("Subsampling training set from %d to %d rows for GP.", len(X), max_rows)
            idx = np.random.choice(len(X), max_rows, replace=False)
            X, y = X[idx], y[idx]

        self._X_train = jnp.array(X, dtype=jnp.float32)
        self._y_train = jnp.array(y, dtype=jnp.float32)

        key = jax_key(seed)

        if inference == "nuts":
            self._fit_nuts(key, kernel)
        else:
            self._fit_svi(key, kernel)

        return self

    def _fit_nuts(self, key, kernel: str) -> None:
        num_samples = self.config.get("num_samples", 500)
        warmup = self.config.get("warmup_steps", 300)

        nuts_kernel = NUTS(gp_model)
        self._mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=num_samples)
        self._mcmc.run(key, self._X_train, self._y_train, kernel)
        logger.info("GP NUTS finished: %d samples", num_samples)

    def _fit_svi(self, key, kernel: str) -> None:
        num_steps = self.config.get("num_samples", 2000)
        self._guide = autoguide.AutoDiagonalNormal(gp_model)
        optimizer = numpyro.optim.Adam(step_size=self.config.get("lr", 0.01))
        svi = SVI(gp_model, self._guide, optimizer, loss=Trace_ELBO())
        self._svi_result = svi.run(key, num_steps, self._X_train, self._y_train, kernel,
                                   progress_bar=self.config.get("progress_bar", False))
        logger.info("GP SVI finished: %d steps, final ELBO %.2f",
                    num_steps, -self._svi_result.losses[-1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (posterior_mean, posterior_std)."""
        key = jax_key(self.config.get("seed", 42) + 1)
        X_jax = jnp.array(X, dtype=jnp.float32)
        kernel = self.config.get("kernel", "matern52")
        num_pred_samples = self.config.get("num_pred_samples", 200)

        if self._mcmc is not None:
            posterior = self._mcmc.get_samples()
        else:
            posterior = self._guide.sample_posterior(
                key, self._svi_result.params, sample_shape=(num_pred_samples,)
            )

        predictive = Predictive(gp_model, posterior_samples=posterior, num_samples=num_pred_samples)
        pred = predictive(key, X_jax, kernel=kernel)
        samples = np.array(pred["obs"])  # (num_samples, n_test)

        return samples.mean(axis=0), samples.std(axis=0)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "X_train": np.array(self._X_train) if self._X_train is not None else None,
            "y_train": np.array(self._y_train) if self._y_train is not None else None,
            "svi_result": self._svi_result,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("GP model saved to %s", path)

    def load(self, path: str | Path) -> "BayesianGPModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.config = payload["config"]
        self._X_train = jnp.array(payload["X_train"]) if payload["X_train"] is not None else None
        self._y_train = jnp.array(payload["y_train"]) if payload["y_train"] is not None else None
        self._svi_result = payload["svi_result"]
        return self
