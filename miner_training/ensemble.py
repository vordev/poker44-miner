"""Picklable recency-aware ensemble for Poker44 bot detection.

A probability-averaging ensemble of gradient-boosted + randomized-tree models.
Stored `estimators` are plain sklearn / LightGBM objects (picklable), so a fitted
`RecencyEnsemble` pickles cleanly. This class must be importable at serve time to
unpickle a saved model -- `serve.py` imports it for exactly that reason.

Recency weighting is applied at *fit* time via `sample_weight` (see
`recency_weights`): recent release dates count more than old ones, because the
live bots evolve and stale generations should not dominate. The fitted model
carries no weighting state -- inference is a plain average.

LightGBM is used when installed (best measured result); otherwise the ensemble
falls back to an sklearn-only configuration so the toolkit still runs. Whatever
is used at train time must also be importable at serve time to unpickle.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from typing import List, Optional, Sequence

import numpy as np

# Cosmetic: LightGBM warns when scored with a bare ndarray after fitting; harmless.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


def build_estimators() -> List:
    """Configured (unfitted) base estimators; uses LightGBM if importable."""
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )

    estimators: List = []
    if _lightgbm_available():
        from lightgbm import LGBMClassifier

        estimators += [
            LGBMClassifier(
                n_estimators=350, learning_rate=0.02, num_leaves=12, min_child_samples=30,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=5.0, reg_alpha=1.0, random_state=s, verbose=-1, n_jobs=-1,
            )
            for s in range(6)
        ]
    estimators += [
        RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=8, n_jobs=-1, random_state=s)
        for s in range(2)
    ]
    estimators += [
        ExtraTreesClassifier(n_estimators=500, max_depth=10, min_samples_leaf=6, n_jobs=-1, random_state=s)
        for s in range(2)
    ]
    estimators += [
        HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.04, max_leaf_nodes=12,
            min_samples_leaf=30, l2_regularization=3.0, random_state=0,
        )
    ]
    return estimators


class RecencyEnsemble:
    """Probability-averaging ensemble; supports per-sample weights at fit time."""

    def __init__(self, estimators: Optional[List] = None) -> None:
        self.estimators = estimators if estimators is not None else build_estimators()

    def fit(self, X, y, sample_weight=None) -> "RecencyEnsemble":
        for est in self.estimators:
            try:
                est.fit(X, y, sample_weight=sample_weight)
            except TypeError:  # estimator without sample_weight support
                est.fit(X, y)
        return self

    def predict_proba(self, X):
        proba = np.mean([est.predict_proba(X)[:, 1] for est in self.estimators], axis=0)
        return np.column_stack([1.0 - proba, proba])

    @property
    def feature_importances_(self):
        parts = []
        for est in self.estimators:
            fi = getattr(est, "feature_importances_", None)
            if fi is not None:
                fi = np.asarray(fi, dtype=float)
                total = fi.sum()
                if total > 0:
                    parts.append(fi / total)
        if not parts:
            raise AttributeError("no estimator exposes feature_importances_")
        return np.mean(parts, axis=0)


def recency_weights(dates: Sequence[str], ref_date: str, half_life_days: float = 10.0) -> np.ndarray:
    """0.5 ** (age_in_days / half_life) for each row's release date, relative to ref_date."""
    ref = datetime.strptime(ref_date, "%Y-%m-%d")
    ages = np.array(
        [max(0.0, (ref - datetime.strptime(str(d), "%Y-%m-%d")).days) for d in dates],
        dtype=float,
    )
    return np.power(0.5, ages / float(half_life_days))
