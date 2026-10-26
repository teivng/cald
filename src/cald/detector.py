"""The CALD detector: one backbone-agnostic loop, calibrate-then-detect.

CALD (Calibrated Anchor-Loss Disagreement) finetunes only the backbone's
decoder to *disagree* with its own pseudolabels on an unlabelled eval batch
while still *agreeing* with a held-out labelled ID anchor batch. The statistic
λ is the post-finetune anchor cross-entropy.

THE DIRECTION IS FIXED, BY THEORY: CALD fires when λ is LOW. On a shifted eval
batch the backbone is uncertain, the disagreement gradient is weak, and the
anchor loss survives finetuning (stays low). On an in-distribution eval batch
the backbone is confident, the disagreement gradient is strong and conflicts
with the anchor, and the anchor loss rises. There is deliberately no option to
flip, auto-calibrate, or two-side this comparison: the detector follows the
mechanism, it does not fit an arbitrary discriminator.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .adapters.base import CALDBackbone
from .losses import disagreement_loss
from .sampling import stratified_bootstrap, stratified_sample_fixed
from .seeding import seed_everything


@dataclass
class CALDResult:
    """Outcome of one CALD detection call."""

    statistic: float          # λ: post-finetune anchor cross-entropy
    threshold: float          # fpr-quantile of the ID calibration statistics
    fired: bool               # statistic <= threshold  (fire-low, by theory)
    step_traj: list[float] = field(repr=False, default_factory=list)
    post_traj: list[float] = field(repr=False, default_factory=list)


class CALD:
    """Calibrate on in-distribution data, then detect shift in eval batches.

    Args:
        backbone: a ``CALDBackbone`` adapter (see ``cald.adapters``).
        lr: SGD learning rate for the decoder finetuning (paper default 1e-3).
        n_steps: finetuning steps per detection call (paper default 50).
        fpr: target false-positive rate; the threshold is the ``fpr``-quantile
            of the ID calibration statistics (default 0.05).
        n_calibration: number of bootstrapped ID calibration rounds (default 50;
            with fewer than ~20 the low quantile is poorly resolved).
        support_size: labelled rows used as the in-context support set
            (-1 = all available after the anchor is reserved).
        anchor_size: held-out labelled ID rows used as the anchor batch.
        eval_size: rows per calibration eval batch; detection batches of a
            different size are allowed but calibrate/detect sizes should match
            for the FPR guarantee to be meaningful.
        seed: RNG seed for the splits, the calibration bootstraps and torch.
    """

    def __init__(self, backbone: CALDBackbone, lr: float = 1e-3, n_steps: int = 50,
                 fpr: float = 0.05, n_calibration: int = 50,
                 support_size: int = 2000, anchor_size: int = 128,
                 eval_size: int = 128, seed: int = 42):
        self.backbone = backbone
        self.lr = lr
        self.n_steps = n_steps
        self.fpr = fpr
        self.n_calibration = n_calibration
        self.support_size = support_size
        self.anchor_size = anchor_size
        self.eval_size = eval_size
        self.seed = seed

        self.threshold_: float | None = None
        self.id_scores_: list[float] | None = None
        self._X_anchor: np.ndarray | None = None
        self._y_anchor_enc: torch.Tensor | None = None
        self._rng_eval = None

    # ------------------------------------------------------------------ fit
    def fit(self, X_id: np.ndarray, y_id: np.ndarray) -> "CALD":
        """Split ID data into support / anchor / calibration pool, condition
        the backbone, and calibrate the fire-low threshold."""
        X_id, y_id = np.asarray(X_id), np.asarray(y_id)
        seed_everything(self.seed)
        rng_sup = np.random.default_rng(self.seed + 1000)
        rng_anc = np.random.default_rng(self.seed + 2000)
        self._rng_eval = np.random.default_rng(self.seed)

        n_sup = self.support_size
        if n_sup == -1:
            n_sup = len(y_id) - self.anchor_size
        if n_sup <= 0 or n_sup + self.anchor_size > len(y_id):
            raise ValueError(
                f"support_size={self.support_size} + anchor_size={self.anchor_size} "
                f"exceeds the {len(y_id)} ID rows available; pass support_size=-1 "
                f"to use everything that the anchor does not need")

        X_sup, y_sup, sup_idx = stratified_sample_fixed(X_id, y_id, n_sup, rng_sup)

        # anchor is disjoint from support — no leakage
        mask = np.ones(len(y_id), dtype=bool)
        mask[sup_idx] = False
        X_pool, y_pool = X_id[mask], y_id[mask]
        if len(y_pool) < self.anchor_size:
            raise ValueError(
                f"only {len(y_pool)} non-support rows left for the anchor "
                f"(need {self.anchor_size}); reduce support_size or anchor_size")
        X_anc, y_anc, anc_idx = stratified_sample_fixed(
            X_pool, y_pool, self.anchor_size, rng_anc)

        # calibration pool: ID rows not in support; the anchor rows are
        # excluded too when enough data remains
        mask2 = np.ones(len(y_pool), dtype=bool)
        mask2[anc_idx] = False
        X_cal, y_cal = X_pool[mask2], y_pool[mask2]
        if len(y_cal) < self.eval_size:
            X_cal, y_cal = X_pool, y_pool  # small-data fallback: reuse anchor pool

        self.backbone.fit(X_sup, y_sup)
        self._X_anchor = X_anc
        self._y_anchor_enc = torch.tensor(
            np.searchsorted(self.backbone.classes_, y_anc), dtype=torch.long)

        self.id_scores_ = []
        for _ in range(self.n_calibration):
            X_ev, _, _ = stratified_bootstrap(X_cal, y_cal, self.eval_size, self._rng_eval)
            self.id_scores_.append(self.score(X_ev)[0])
        self.threshold_ = float(np.percentile(self.id_scores_, self.fpr * 100))
        return self

    # ---------------------------------------------------------------- score
    def score(self, X_eval: np.ndarray) -> tuple[float, dict]:
        """Run one CALD finetuning episode; return (λ, trajectories).

        Lower λ means more evidence of shift (see the module docstring).
        """
        if self._X_anchor is None:
            raise RuntimeError("call fit(X_id, y_id) before score/detect")
        bb = self.backbone
        bb.reset_decoder()

        n_anchor = len(self._X_anchor)
        n_classes = bb.n_classes
        y_anc_t = None  # placed on the logits' device at first use

        pseudo = bb.predict_proba(np.asarray(X_eval)).argmax(axis=1)
        pseudo_t = torch.as_tensor(pseudo, dtype=torch.long)

        optimizer = torch.optim.SGD(bb.decoder_parameters(), lr=self.lr)
        step_traj, post_traj = [], []

        for _ in range(self.n_steps):
            optimizer.zero_grad()
            logits = bb.forward_logits(self._X_anchor, X_eval)
            if y_anc_t is None:
                y_anc_t = self._y_anchor_enc.to(logits.device)
                pseudo_t = pseudo_t.to(logits.device)

            loss_anc = F.cross_entropy(logits[:n_anchor], y_anc_t)
            loss_dis = disagreement_loss(logits[n_anchor:], pseudo_t, n_classes)
            (loss_anc + loss_dis).backward()
            step_traj.append(float(loss_anc.item()))
            optimizer.step()

            with torch.no_grad():
                post = bb.forward_logits(self._X_anchor, X_eval)
                post_traj.append(float(
                    F.cross_entropy(post[:n_anchor], y_anc_t).item()))

        bb.reset_decoder()
        return post_traj[-1], {"step_traj": step_traj, "post_traj": post_traj}

    # --------------------------------------------------------------- detect
    def detect(self, X_eval: np.ndarray) -> CALDResult:
        """Score an eval batch and compare against the calibrated threshold.

        Fires iff λ <= threshold. The comparison direction is fixed (fire-low).
        """
        if self.threshold_ is None:
            raise RuntimeError("call fit(X_id, y_id) before detect")
        lam, traj = self.score(X_eval)
        return CALDResult(
            statistic=lam,
            threshold=self.threshold_,
            fired=bool(lam <= self.threshold_),
            step_traj=traj["step_traj"],
            post_traj=traj["post_traj"],
        )
