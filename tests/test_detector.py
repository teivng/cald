"""Detector semantics with a tiny synthetic backbone — no checkpoint needed."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from cald import CALD, CALDResult
from cald.adapters.base import CALDBackbone


class ToyBackbone(CALDBackbone):
    """Logistic head on raw features; the 'decoder' is a 2-layer MLP.

    Mimics the structural properties CALD relies on: confident on ID,
    less confident off-distribution.
    """

    def __init__(self, dim=4, seed=0):
        torch.manual_seed(seed)
        self.decoder = nn.Sequential(nn.Linear(dim, 16), nn.GELU(), nn.Linear(16, 2))
        self._pristine = {k: v.clone() for k, v in self.decoder.state_dict().items()}

    def fit(self, X_support, y_support):
        self.classes_ = np.unique(y_support)
        # cheap supervised fit of the toy decoder so it is confident on ID
        X = torch.tensor(X_support, dtype=torch.float32)
        y = torch.tensor(np.searchsorted(self.classes_, y_support), dtype=torch.long)
        opt = torch.optim.Adam(self.decoder.parameters(), lr=1e-2)
        for _ in range(200):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(self.decoder(X), y)
            loss.backward()
            opt.step()
        self._pristine = {k: v.clone() for k, v in self.decoder.state_dict().items()}
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.decoder(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def forward_logits(self, X_anchor, X_eval):
        X = torch.tensor(np.vstack([X_anchor, X_eval]), dtype=torch.float32)
        return self.decoder(X)

    def decoder_parameters(self):
        return list(self.decoder.parameters())

    def reset_decoder(self):
        self.decoder.load_state_dict(self._pristine)


def _data(n=800, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, dim))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    return X, y


def test_fire_low_semantics_and_result_shape():
    X, y = _data()
    det = CALD(ToyBackbone(), lr=1e-2, n_steps=10, n_calibration=20,
               support_size=400, anchor_size=64, eval_size=64, seed=42)
    det.fit(X, y)

    assert det.threshold_ is not None
    assert len(det.id_scores_) == 20
    # the threshold is the LOW quantile of ID scores — fire-low, never flipped
    assert det.threshold_ <= float(np.median(det.id_scores_))

    rng = np.random.default_rng(1)
    X_id = X[rng.choice(len(X), 64, replace=False)]
    r = det.detect(X_id)
    assert isinstance(r, CALDResult)
    # paper eq. (verdict): one-sided, strict — λ* < Quantile_α(record) => OOD
    assert r.fired == (r.statistic < r.threshold)
    assert len(r.step_traj) == 10 and len(r.post_traj) == 10


def test_id_false_positive_rate_is_controlled():
    X, y = _data()
    det = CALD(ToyBackbone(), lr=1e-2, n_steps=10, n_calibration=40,
               support_size=400, anchor_size=64, eval_size=64, seed=42)
    det.fit(X, y)
    rng = np.random.default_rng(2)
    fires = [det.detect(X[rng.choice(len(X), 64, replace=False)]).fired
             for _ in range(20)]
    assert np.mean(fires) <= 0.25  # ID batches mostly stay quiet


def test_support_anchor_disjoint_guard():
    X, y = _data(n=100)
    det = CALD(ToyBackbone(), support_size=90, anchor_size=64)
    with pytest.raises(ValueError):
        det.fit(X, y)


def test_unfit_raises():
    det = CALD(ToyBackbone())
    with pytest.raises(RuntimeError):
        det.detect(np.zeros((8, 4)))
