"""Stratified sampling helpers used for the support/anchor split and the
ID calibration rounds."""

import numpy as np


def stratified_bootstrap(X, y, n_samples, rng):
    """Bootstrap n_samples from (X, y) ensuring at least 2 samples per class.

    Returns (X_sampled, y_sampled, indices). Sampling is with replacement
    beyond the per-class minimum, matching the paper's calibration rounds.
    """
    classes = np.unique(y)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, min(2, len(c_idx)), replace=False)
        indices.extend(chosen.tolist())
    remaining = n_samples - len(indices)
    if remaining > 0:
        extra = rng.choice(len(y), remaining, replace=True)
        indices.extend(extra.tolist())
    idx = np.array(indices[:n_samples])
    return X[idx], y[idx], idx


def stratified_sample_fixed(X, y, n_samples, rng):
    """Sample n_samples WITHOUT replacement for fixed sets (support/anchor).

    Returns (X_sampled, y_sampled, indices).
    If n_samples == -1 or >= len(y), returns everything (shuffled).
    """
    if n_samples == -1 or n_samples >= len(y):
        idx = rng.permutation(len(y))
        return X[idx], y[idx], idx

    classes = np.unique(y)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, min(2, len(c_idx)), replace=False)
        indices.extend(chosen.tolist())

    remaining = n_samples - len(indices)
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(y)), indices)
        extra = rng.choice(available, min(remaining, len(available)), replace=False)
        indices.extend(extra.tolist())

    idx = np.array(indices[:n_samples])
    rng.shuffle(idx)
    return X[idx], y[idx], idx
