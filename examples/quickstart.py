"""CALD quickstart on synthetic data — CPU, under a minute.

A 2-class Gaussian problem; the shifted batch moves the class-conditional
means so the backbone's decision rule genuinely degrades (a harmful shift).
"""

import numpy as np

from cald import CALD, get_adapter


def make_data(n, rng, shift=0.0):
    y = rng.integers(0, 2, n)
    mu = np.where(y[:, None] == 1, 1.0, -1.0) * np.array([1.0, 0.8, 0.0, 0.0])
    X = mu + rng.normal(0, 1.0, (n, 4))
    X[:, 2:] = rng.normal(0, 1.0, (n, 2))      # nuisance features
    X[:, 0] += shift                            # covariate shift on the signal axis
    return X, y


def main():
    rng = np.random.default_rng(0)
    X_id, y_id = make_data(1200, rng)

    backbone = get_adapter("nanotabiclv2", device="cpu")
    det = CALD(backbone, lr=1e-3, n_steps=20, fpr=0.05, n_calibration=30,
               support_size=512, anchor_size=96, eval_size=64, seed=42)
    print("calibrating on ID data (30 rounds)...")
    det.fit(X_id, y_id)
    print(f"threshold (5% quantile of ID λ): {det.threshold_:.4f}")

    X_test_id, _ = make_data(64, rng)
    X_test_ood, _ = make_data(64, rng, shift=3.0)

    r_id = det.detect(X_test_id)
    r_ood = det.detect(X_test_ood)
    print(f"ID batch : λ={r_id.statistic:.4f}  fired={r_id.fired}")
    print(f"OOD batch: λ={r_ood.statistic:.4f}  fired={r_ood.fired}")
    print("(CALD fires when λ is LOW — the shifted batch should fire.)")


if __name__ == "__main__":
    main()
