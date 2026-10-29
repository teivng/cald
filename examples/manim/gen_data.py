"""Stage A for the CALD manim animation: run the real CALD episodes once and
dump everything the scene needs (grids, boundaries, trajectories, calibration
record) into assets/.

Same task, seeds and hyperparameters as examples/visualize_cald.ipynb, so the
numbers match (threshold ~0.699, λ_ID ~0.71, λ_OOD ~0.074).

Run:  python gen_data.py        (GPU recommended; ~5 min)
"""

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from cald import CALD, get_adapter
from cald.losses import disagreement_loss
from cald.sampling import stratified_bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ---------------------------------------------------------------- the task
rng = np.random.default_rng(0)


def make_id(n, rng):
    y = rng.integers(0, 2, n)
    mu = np.where(y[:, None] == 1, 1.0, -1.0) * np.array([1.4, 1.0])
    return mu + rng.normal(0, 0.9, (n, 2)), y


X_id, y_id = make_id(1200, rng)
X_eval_id, _ = make_id(64, rng)                 # deployment batch, still ID
# harmful shift: the SAME batch, rotated clockwise — the paper's
# controlled 2-D experiment (rotation -> harmful); validated λ: 45° = 0.50,
# 90° = 0.38, vs K=50 threshold 0.689
ROT_DEG = 90
_t = np.deg2rad(ROT_DEG)
_R = np.array([[np.cos(_t), np.sin(_t)], [-np.sin(_t), np.cos(_t)]])
X_eval_shift = X_eval_id @ _R.T

backbone = get_adapter("nanotabiclv2")
det = CALD(backbone, lr=1e-2, n_steps=50, alpha=0.05, n_calibration=50,
           support_size=512, anchor_size=96, eval_size=64, seed=42)

# replicate CALD.fit's split + conditioning exactly (same seeds, same order),
# but keep the calibration loop out here so each round can be instrumented
from cald.sampling import stratified_sample_fixed
rng_sup = np.random.default_rng(det.seed + 1000)
rng_anc = np.random.default_rng(det.seed + 2000)
rng_eval = np.random.default_rng(det.seed)
X_sup, y_sup, sup_idx = stratified_sample_fixed(X_id, y_id, det.support_size,
                                                rng_sup)
mask = np.ones(len(y_id), dtype=bool); mask[sup_idx] = False
X_pool, y_pool = X_id[mask], y_id[mask]
X_anc, y_anc, anc_idx = stratified_sample_fixed(X_pool, y_pool,
                                                det.anchor_size, rng_anc)
mask2 = np.ones(len(y_pool), dtype=bool); mask2[anc_idx] = False
X_cal, y_cal = X_pool[mask2], y_pool[mask2]

det.backbone.fit(X_sup, y_sup)
det._X_anchor = X_anc
det._y_anchor_enc = torch.tensor(det.backbone.encode_labels(y_anc),
                                 dtype=torch.long)

RES = 80
g = np.linspace(-4, 4, RES)
GX, GY = np.meshgrid(g, g)
X_grid = np.stack([GX.ravel(), GY.ravel()], axis=1)


def run_episode(X_eval, capture_steps):
    """One CALD episode; returns (grids, traj, pseudo). grids has 51 fields
    when capture_steps else just [initial, final]."""
    bb = det.backbone
    bb.reset_decoder()
    n_anchor = len(det._X_anchor)
    pseudo = bb.predict_proba(np.asarray(X_eval)).argmax(axis=1)
    pseudo_t = torch.as_tensor(pseudo, dtype=torch.long)
    opt = torch.optim.SGD(bb.decoder_parameters(), lr=det.lr)
    grids, traj, y_anc_t = [bb.predict_proba(X_grid)[:, 1]], [], None
    for step in range(det.n_steps):
        opt.zero_grad()
        logits = bb.forward_logits(det._X_anchor, X_eval)
        if y_anc_t is None:
            y_anc_t = det._y_anchor_enc.to(logits.device)
            pseudo_t = pseudo_t.to(logits.device)
        loss_anc = F.cross_entropy(logits[:n_anchor], y_anc_t)
        loss_dis = disagreement_loss(logits[n_anchor:], pseudo_t, bb.n_classes)
        (loss_anc + loss_dis).backward()
        opt.step()
        with torch.no_grad():
            post = bb.forward_logits(det._X_anchor, X_eval)
            traj.append(float(F.cross_entropy(post[:n_anchor], y_anc_t).item()))
        if capture_steps or step == det.n_steps - 1:
            grids.append(bb.predict_proba(X_grid)[:, 1])
    bb.reset_decoder()
    return np.stack(grids), np.array(traj), pseudo


# --------------------------------------------------- calibration, instrumented
N_ANIM = 10              # rounds animated with the live loss curve
cal_lams, cal_final_grids, cal_eval_batches, cal_eval_ys = [], [], [], []
cal_full = {}            # first N_ANIM rounds keep every step for sequence 3
for k in range(det.n_calibration):
    X_ev, y_ev, _ = stratified_bootstrap(X_cal, y_cal, det.eval_size, rng_eval)
    full = k < N_ANIM
    grids, traj, _ = run_episode(X_ev, capture_steps=full)
    cal_lams.append(traj[-1])
    cal_final_grids.append(grids[-1])
    cal_eval_batches.append(X_ev)
    cal_eval_ys.append(y_ev)
    if full:
        cal_full[k] = (grids, traj)
    print(f"calibration round {k}: λ = {traj[-1]:.4f}")

threshold = float(np.percentile(cal_lams, det.alpha * 100))
print(f"threshold = {threshold:.4f}")

# ------------------------------------------------------------- deployment
dep_id_grids, dep_id_traj, dep_id_pseudo = run_episode(X_eval_id, True)
dep_sh_grids, dep_sh_traj, dep_sh_pseudo = run_episode(X_eval_shift, True)
print(f"deployment λ: ID = {dep_id_traj[-1]:.4f}  shifted = {dep_sh_traj[-1]:.4f}")

# ------------------------------------------------- heatmap PNGs + contours
BG = "#1e1e2e"           # catppuccin mocha base: uncertain regions fade to bg
C0, C1 = "#f38ba8", "#89b4fa"
cmap = LinearSegmentedColormap.from_list("mocha_dark", [C0, BG, C1])


BG_RGB = np.array([0x1e, 0x1e, 0x2e], dtype=np.float32)


def save_heat(field, name):
    rgba = (cmap(field.reshape(RES, RES)) * 255).astype(np.float32)
    rgba[..., :3] = 0.72 * rgba[..., :3] + 0.28 * BG_RGB   # keep points legible
    img = Image.fromarray(rgba.astype(np.uint8)).resize((480, 480),
                                                        Image.BICUBIC)
    img.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(ASSETS, name))
    # transpose: row 0 of the grid is y=-4, PNGs count rows from the top


def contours(field):
    cs = plt.contour(g, g, field.reshape(RES, RES), levels=[0.5])
    segs = [s.copy() for s in cs.allsegs[0]]
    plt.close()
    return segs


heat_index, all_contours = {}, {}


def record(key, grids):
    for i, f in enumerate(grids):
        name = f"{key}_{i:02d}.png"
        save_heat(f, name)
        heat_index[f"{key}_{i}"] = name
        all_contours[f"{key}_{i}"] = contours(f)


for k in range(N_ANIM):
    record(f"cal{k}", cal_full[k][0])     # 51 fields each
record("calfin", np.stack(cal_final_grids))   # K final fields
record("depid", dep_id_grids)             # 51 fields
record("depsh", dep_sh_grids)             # 51 fields

with open(os.path.join(ASSETS, "contours.pkl"), "wb") as f:
    pickle.dump(all_contours, f)

np.savez(
    os.path.join(ASSETS, "data.npz"),
    X_sup=det.backbone._X_sup.cpu().numpy(),
    y_sup=det.backbone._y_sup[0].cpu().numpy(),
    X_anc=det._X_anchor,
    y_anc=det._y_anchor_enc.numpy(),
    X_dep_id=X_eval_id, X_ood=X_eval_shift, rot_deg=ROT_DEG,
    cal_lams=np.array(cal_lams),
    cal_trajs=np.stack([cal_full[k][1] for k in range(N_ANIM)]),
    cal_evals=np.stack(cal_eval_batches[:N_ANIM]),
    cal_evals_y=np.stack(cal_eval_ys[:N_ANIM]),
    dep_id_traj=dep_id_traj, dep_sh_traj=dep_sh_traj,
    threshold=threshold,
)
print("assets written to", ASSETS)
