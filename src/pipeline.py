"""Splitting, preprocessing and sequence assembly.

This is where the leakage discipline lives. Two rules are enforced here
rather than left to the notebook:

1. Splits are made BY RECORDING, never by window. Windows overlap by 50%, so
   two windows from one clip share half their samples; putting them on
   opposite sides of a split lets the model see its own test data.
2. Feature pruning, standardisation and PCA are fitted on the training split
   only, then applied unchanged to held-out data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import MAX_FEATURE_CORR, N_COMPONENTS, RANDOM_SEED
from features import prune_features


def split_recordings(table: pd.DataFrame, n_test_per_activity: int = 3,
                     seed: int = RANDOM_SEED) -> tuple[list[str], list[str]]:
    """Hold out whole recordings, stratified by activity.

    Returns (train_recordings, test_recordings).
    """
    rng = np.random.default_rng(seed)
    train, test = [], []
    for activity, grp in table.groupby("activity", sort=True):
        recs = sorted(grp["recording"].unique())
        if n_test_per_activity >= len(recs):
            raise ValueError(f"{activity}: cannot hold out "
                             f"{n_test_per_activity} of {len(recs)}")
        picked = {str(r) for r in
                  rng.choice(recs, size=n_test_per_activity, replace=False)}
        test.extend(sorted(picked))
        train.extend([str(r) for r in recs if r not in picked])
    return sorted(train), sorted(test)


class Preprocessor:
    """Prune -> standardise -> PCA, fitted once on the training split."""

    def __init__(self, n_components: int = N_COMPONENTS,
                 max_corr: float = MAX_FEATURE_CORR,
                 seed: int = RANDOM_SEED):
        self.n_components = n_components
        self.max_corr = max_corr
        self.seed = seed
        self.columns_: list[str] | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None

    def fit(self, train_table: pd.DataFrame, verbose: bool = False):
        self.columns_ = prune_features(train_table, max_corr=self.max_corr,
                                       verbose=verbose)
        X = train_table[self.columns_].to_numpy(float)
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = PCA(n_components=self.n_components,
                        random_state=self.seed).fit(self.scaler_.transform(X))
        if verbose:
            ev = self.pca_.explained_variance_ratio_
            print(f"PCA {len(self.columns_)} -> {self.n_components} dims, "
                  f"{ev.sum():.1%} of variance "
                  f"({', '.join(f'{v:.1%}' for v in ev)})")
        return self

    def transform(self, table: pd.DataFrame) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        missing = [c for c in self.columns_ if c not in table.columns]
        if missing:
            raise ValueError(f"held-out data is missing features: {missing[:5]}")
        X = table[self.columns_].to_numpy(float)
        return self.pca_.transform(self.scaler_.transform(X))

    def fit_transform(self, train_table: pd.DataFrame, verbose: bool = False):
        return self.fit(train_table, verbose=verbose).transform(train_table)


def to_sequences(table: pd.DataFrame, Z: np.ndarray
                 ) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Group projected observations into per-recording sequences.

    Each recording becomes one sequence of consecutive windows, ordered by
    window_index. Returns (sequences, label_sequences, recording_names).
    """
    if len(table) != len(Z):
        raise ValueError("table and projected matrix lengths disagree")

    work = table.reset_index(drop=True).copy()
    work["_row"] = np.arange(len(work))

    sequences, labels, names = [], [], []
    for rec, grp in work.groupby("recording", sort=True):
        grp = grp.sort_values("window_index")
        rows = grp["_row"].to_numpy()
        sequences.append(Z[rows])
        labels.append(grp["state"].to_numpy(int))
        names.append(rec)
    return sequences, labels, names


def subset(table: pd.DataFrame, recordings: list[str]) -> pd.DataFrame:
    """Rows belonging to the given recordings, order preserved."""
    return table[table["recording"].isin(set(recordings))].copy()
