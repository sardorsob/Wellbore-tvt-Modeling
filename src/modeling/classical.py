"""Classical ML models: XGBoost and LightGBM wrappers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class XGBoostModel:
    """XGBoost regression wrapper with a consistent fit/predict/save/load interface."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model: BaseEstimator | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> "XGBoostModel":
        """Fit XGBRegressor.

        Parameters
        ----------
        X : feature matrix (rows × features)
        y : TVT target array
        eval_set : optional (X_val, y_val) tuple for early stopping
        """
        import xgboost as xgb

        params = {
            "n_estimators": self.config.get("n_estimators", 1000),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 6),
            "min_child_weight": self.config.get("min_child_weight", 5),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "random_state": self.config.get("seed", 42),
            "n_jobs": -1,
            "objective": "reg:squarederror",
        }

        callbacks = []
        if eval_set is not None:
            early_stop = self.config.get("early_stopping_rounds", 50)
            callbacks.append(xgb.callback.EarlyStopping(rounds=early_stop, save_best=True))

        self.model = xgb.XGBRegressor(**params)
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = self.config.get("verbose", 100)
        self.model.fit(X, y, callbacks=callbacks, **fit_kwargs)
        logger.info("XGBoost fitted — best iteration: %s", getattr(self.model, "best_iteration", "N/A"))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted."
        return self.model.predict(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str | Path) -> "XGBoostModel":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self


class LGBMModel:
    """LightGBM regression wrapper."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model: BaseEstimator | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> "LGBMModel":
        import lightgbm as lgb

        params = {
            "n_estimators": self.config.get("n_estimators", 1000),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "num_leaves": self.config.get("num_leaves", 63),
            "min_child_samples": self.config.get("min_child_samples", 20),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "random_state": self.config.get("seed", 42),
            "n_jobs": -1,
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
        }

        callbacks = []
        if eval_set is not None:
            early_stop = self.config.get("early_stopping_rounds", 50)
            callbacks.append(lgb.early_stopping(early_stop, verbose=False))
            callbacks.append(lgb.log_evaluation(period=self.config.get("verbose", 100)))

        self.model = lgb.LGBMRegressor(**params)
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
        self.model.fit(X, y, callbacks=callbacks if callbacks else None, **fit_kwargs)
        logger.info("LightGBM fitted")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted."
        return self.model.predict(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("LightGBM model saved to %s", path)

    def load(self, path: str | Path) -> "LGBMModel":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self
