"""
Temporal target encoding for M-estimate with expanding window.

Provides ``m_estimate_encoding`` (stateless function) and ``TemporalTargetEncoder``
(sklearn-compatible transformer) for target encoding with temporal integrity.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def m_estimate_encoding(
    df_train: pd.DataFrame,
    df_encode: pd.DataFrame,
    entity_col: str,
    target_col: str,
    time_col: str = "sale_year",
    m: float = 10.0,
    min_count: int = 5,
) -> np.ndarray:
    """
    M-estimate target encoding with temporal expanding window.

    For each row in ``df_encode``, computes the M-estimate of ``target_col``
    conditioned on ``entity_col``, using only data in ``df_train`` with
    ``time_col < row[time_col]``.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data used to compute the encoding (the "history").
    df_encode : pd.DataFrame
        Data to encode (must contain ``entity_col`` and ``time_col``).
    entity_col : str
        Entity column to group by (e.g. ``"sire_entity"``).
    target_col : str
        Target column whose mean is computed.
    time_col : str
        Temporal column used for the expanding window.
    m : float
        Regularisation constant. Higher = stronger shrinkage toward global mean.
    min_count : int
        Minimum entity occurrences required before using entity-specific mean
        (currently implemented via M-estimate shrinkage; unused in formula).

    Returns
    -------
    np.ndarray
        Encoded values of shape ``(len(df_encode),)``.
    """
    global_mean = float(df_train[target_col].mean())
    result = np.full(len(df_encode), global_mean, dtype=float)

    for idx in range(len(df_encode)):
        row = df_encode.iloc[idx]
        entity_val = row[entity_col]
        year_val = row[time_col]

        prior_mask = (df_train[entity_col] == entity_val) & (
            df_train[time_col] < year_val
        )
        prior_data = df_train.loc[prior_mask, target_col].dropna()
        n = len(prior_data)

        if n == 0:
            result[idx] = global_mean
        else:
            entity_mean = float(prior_data.mean())
            result[idx] = (n * entity_mean + m * global_mean) / (n + m)

    return result


class TemporalTargetEncoder(BaseEstimator, TransformerMixin):
    """M-estimate target encoding with temporal expanding window.

    Sklearn-compatible transformer.  ``fit`` stores the global mean and the
    training data; ``transform`` computes M-estimate encodings per row using
    only prior data.

    Parameters
    ----------
    time_col : str
        Column identifying the temporal order (default ``"sale_year"``).
    target_col : str
        Target column for the M-estimate.
    entity_cols : Sequence[str]
        Entity columns to encode.
    m : float
        Regularisation constant (default 10).
    min_count : int
        Minimum occurrences (default 5; applied via M-estimate).
    encoding_base_mask : callable or None
        Optional function ``callable(df) -> pd.Series(bool)`` that selects which
        rows of the training set form the encoding base.  If ``None``, all rows
        with a non-NaN target are used.
    """

    def __init__(
        self,
        time_col: str = "sale_year",
        target_col: str = "log_price_gns",
        entity_cols: Optional[Sequence[str]] = None,
        m: float = 10.0,
        min_count: int = 5,
        encoding_base_mask: Optional[callable] = None,
    ):
        self.time_col = time_col
        self.target_col = target_col
        self.entity_cols = list(entity_cols) if entity_cols else []
        self.m = m
        self.min_count = min_count
        self.encoding_base_mask = encoding_base_mask
        self.global_mean_: float = 0.0
        self.df_train_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TemporalTargetEncoder":
        """Store the training data and compute the global mean.

        Parameters
        ----------
        X : pd.DataFrame
            Training data — must contain ``time_col`` and all ``entity_cols``.
        y : ignored
            Not used, present for sklearn Pipeline compatibility.
        """
        self.df_train_ = X.copy()

        # Determine encoding base rows
        if self.encoding_base_mask is not None:
            base = X[self.encoding_base_mask(X)]
        else:
            base = X[X[self.target_col].notna()]

        if len(base) == 0:
            self.global_mean_ = 0.0
        else:
            self.global_mean_ = float(base[self.target_col].mean())

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Compute M-estimate encodings for ``X``.

        Returns a 2-D array with one column per entity column.
        """
        if self.df_train_ is None:
            raise RuntimeError("TemporalTargetEncoder has not been fitted yet.")

        result = np.zeros((len(X), len(self.entity_cols)), dtype=float)

        for col_idx, entity_col in enumerate(self.entity_cols):
            global_mean = self.global_mean_
            col_vals = np.full(len(X), global_mean, dtype=float)

            for row_idx in range(len(X)):
                row = X.iloc[row_idx]
                entity_val = row[entity_col]
                year_val = row[self.time_col]

                prior_mask = (
                    self.df_train_[entity_col] == entity_val
                ) & (self.df_train_[self.time_col] < year_val)
                prior_data = self.df_train_.loc[
                    prior_mask, self.target_col
                ].dropna()
                n = len(prior_data)

                if n == 0:
                    col_vals[row_idx] = global_mean
                else:
                    entity_mean = float(prior_data.mean())
                    col_vals[row_idx] = (
                        n * entity_mean + self.m * global_mean
                    ) / (n + self.m)

            result[:, col_idx] = col_vals

        return result

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return feature names for the encoded columns."""
        return [f"{col}_enc" for col in self.entity_cols]
