"""Composable model wrappers used to persist the exact final stacking models.

The notebooks train base learners first and then fit a meta-learner on validation
predictions. Saving only the meta-learner is not enough to reproduce predictions;
these wrappers keep the base models, the meta model, and the ordered base keys
together as one artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


@dataclass
class StackingClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Classifier wrapper exposing sklearn-like ``predict_proba``.

    Inherits from ``BaseEstimator + ClassifierMixin`` for sklearn
    Pipeline / cross_val_score / CalibratedClassifierCV compatibility.
    The ``fit`` method is a no-op — base models and meta-learner are
    pre-trained in the modeling notebook.
    """

    base_models: Mapping[str, object]
    meta: object
    base_keys: Sequence[str]

    def _meta_features(self, X) -> np.ndarray:
        return np.column_stack([
            self.base_models[key].predict_proba(X)[:, 1]
            for key in self.base_keys
        ])

    def predict_proba(self, X) -> np.ndarray:
        return self.meta.predict_proba(self._meta_features(X))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def fit(self, X, y=None):
        """No-op fit — base models and meta-learner are pre-trained."""
        return self


@dataclass
class StackingRegressorWrapper(BaseEstimator, RegressorMixin):
    """Regressor wrapper exposing sklearn-like ``predict``.

    Inherits from ``BaseEstimator + RegressorMixin`` for sklearn
    Pipeline / cross_val_score compatibility.
    """

    base_models: Mapping[str, object]
    meta: object
    base_keys: Sequence[str]

    def _meta_features(self, X) -> np.ndarray:
        return np.column_stack([
            self.base_models[key].predict(X)
            for key in self.base_keys
        ])

    def predict(self, X) -> np.ndarray:
        return self.meta.predict(self._meta_features(X))

    def fit(self, X, y=None):
        """No-op fit — base models and meta-learner are pre-trained."""
        return self
