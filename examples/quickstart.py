"""CALD quickstart on synthetic data (seconds on GPU, a few minutes on CPU).

Two eval batches make the mechanism visible:

1. an in-distribution batch                        -> stays quiet;
2. a harmful shift — the support set's feature-label
   signal is destroyed, the backbone turns uncertain,
   the disagreement gradient weakens, and the anchor
   loss survives finetuning                        -> fires (λ LOW).

The direction (fire when the post-finetune anchor loss is LOW) is part of the
method and is fixed. For the complementary case — a real covariate shift the
backbone handles confidently, where CALD correctly stays quiet — run
``examples/real_datasets.py`` (adult income, folktables).
"""

import numpy as np

from cald import CALD, get_adapter


def make_id(n, rng):
    """Two classes separated along the first two features; two nuisance dims."""
    y = rng.integers(0, 2, n)
    mu = np.where(y[:, None] == 1, 1.0, -1.0) * np.array([1.0, 0.8, 0.0, 0.0])
    X = mu + rng.normal(0, 1.0, (n, 4))
    X[:, 2:] = rng.normal(0, 1.0, (n, 2))
    return X, y


def main():
    rng = np.random.default_rng(0)
    X_id, y_id = make_id(1200, rng)

    backbone = get_adapter("nanotabiclv2")
    det = CALD(backbone, lr=1e-2, n_steps=50, alpha=0.05, n_calibration=30,
               support_size=512, anchor_size=96, eval_size=64, seed=42)
    print("calibrating on ID data (30 rounds)...")
    det.fit(X_id, y_id)
    print(f"threshold (5% quantile of ID λ): {det.threshold_:.4f}\n")

    X_eval_id, _ = make_id(64, rng)
    # harmful: discriminative axes collapse, nuisance axes inflate — the
    # support-set feature-label relation no longer applies
    X_harmful = np.hstack([rng.normal(0, 0.3, (64, 2)), rng.normal(0, 3.0, (64, 2))])

    for name, X in [("ID batch     ", X_eval_id), ("harmful shift", X_harmful)]:
        r = det.detect(X)
        print(f"{name}: λ={r.statistic:.4f}  fired={r.fired}")

    print("\nCALD fires when λ is LOW: the harmful batch fires, the ID batch stays quiet.")


if __name__ == "__main__":
    main()
