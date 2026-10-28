"""CALD on real tabular datasets with the paper's ID/OOD splits.

Datasets (install extras first: pip install -e ".[examples]"):
- covertype  (sklearn): elevation 2000-2500 m (ID) vs > 3000 m (OOD) — a strong,
  harmful shift; CALD should fire on OOD batches and stay quiet on ID.
- adult_income (ucimlrepo #2): Male (ID) vs Female (OOD) workers — a benign
  shift for strong backbones; CALD should mostly stay quiet.
- folktables (ACS Income 2018): California (ID) vs Southern states (OOD) —
  mild shift, mostly quiet.

Usage: python examples/real_datasets.py [--datasets covertype adult_income folktables]
       [--device cuda] [--rounds 5]
"""

import argparse

import numpy as np

from cald import CALD, get_adapter


def load_covertype(rng):
    from sklearn.datasets import fetch_covtype

    d = fetch_covtype()
    X, y = d.data, d.target.astype(int)
    elev = X[:, 0]
    id_mask = (elev >= 2000) & (elev <= 2500)
    ood_mask = elev > 3000
    X_id_all, y_id_all = X[id_mask], y[id_mask]
    idx = rng.permutation(len(y_id_all))
    n_fit = min(6000, len(idx) // 2)
    fit_idx, pool_idx = idx[:n_fit], idx[n_fit:]
    return ((X_id_all[fit_idx], y_id_all[fit_idx]),
            (X_id_all[pool_idx], y_id_all[pool_idx]),
            (X[ood_mask], y[ood_mask]))


def load_adult_income(rng):
    from sklearn.preprocessing import LabelEncoder
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=2)
    X = ds.data.features.copy()
    y = (ds.data.targets.iloc[:, 0].str.strip()
         .str.replace(".", "", regex=False) == ">50K").astype(int).values
    sex = X["sex"].str.strip().values
    X = X.drop(columns=["sex", "fnlwgt"], errors="ignore")
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)

    male, female = np.where(sex == "Male")[0], np.where(sex == "Female")[0]
    rng.shuffle(male)
    n_fit = len(male) // 2
    return ((X[male[:n_fit]], y[male[:n_fit]]),
            (X[male[n_fit:]], y[male[n_fit:]]),
            (X[female], y[female]))


def load_folktables(rng):
    from folktables import ACSDataSource, ACSIncome

    src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    Xca, yca, _ = ACSIncome.df_to_numpy(src.get_data(states=["CA"], download=True))
    Xo, yo = [], []
    for st in ["TX", "MS", "LA"]:
        Xs, ys, _ = ACSIncome.df_to_numpy(src.get_data(states=[st], download=True))
        Xo.append(Xs), yo.append(ys)
    X_ood, y_ood = np.vstack(Xo), np.concatenate(yo).astype(int)
    yca = yca.astype(int)
    idx = rng.permutation(len(yca))
    n_fit = min(8000, len(idx) // 2)
    return ((Xca[idx[:n_fit]], yca[idx[:n_fit]]),
            (Xca[idx[n_fit:]], yca[idx[n_fit:]]),
            (X_ood, y_ood))


LOADERS = {"covertype": load_covertype, "adult_income": load_adult_income,
           "folktables": load_folktables}


def eval_pool(det, X_pool, n_rounds, eval_size, rng):
    stats, fires = [], []
    for _ in range(n_rounds):
        batch = X_pool[rng.choice(len(X_pool), eval_size, replace=False)]
        r = det.detect(batch)
        stats.append(r.statistic), fires.append(r.fired)
    return float(np.mean(stats)), float(np.mean(fires))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS),
                    choices=list(LOADERS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--rounds", type=int, default=5, help="eval batches per split")
    args = ap.parse_args()

    backbone = get_adapter("nanotabiclv2", device=args.device)
    print(f"{'dataset':14s} {'thr':>8s} {'λ(ID)':>8s} {'fire(ID)':>8s} "
          f"{'λ(OOD)':>8s} {'fire(OOD)':>9s}")
    for name in args.datasets:
        rng = np.random.default_rng(42)
        (X_fit, y_fit), (X_id_pool, _), (X_ood_pool, _) = LOADERS[name](rng)
        det = CALD(backbone, lr=1e-3, n_steps=50, alpha=0.05, n_calibration=50,
                   support_size=2000, anchor_size=128, eval_size=128, seed=42)
        det.fit(X_fit, y_fit)
        lam_id, fire_id = eval_pool(det, X_id_pool, args.rounds, det.eval_size, rng)
        lam_ood, fire_ood = eval_pool(det, X_ood_pool, args.rounds, det.eval_size, rng)
        print(f"{name:14s} {det.threshold_:8.4f} {lam_id:8.4f} {fire_id:8.2f} "
              f"{lam_ood:8.4f} {fire_ood:9.2f}")


if __name__ == "__main__":
    main()
