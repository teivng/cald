"""The TabICLv2 adapter: upstream nanoTabICL architecture + official weights,
speaking the CALD attachment contract."""

import copy

import numpy as np
import torch

from ..base import CALDBackbone
from .weights import load_pretrained


class NanoTabICLv2Adapter(CALDBackbone):
    """CALD backbone wrapping nanoTabICL with the official TabICLv2 weights.

    Notes:
        - Features must be numeric (float) arrays without NaNs; the model
          z-scores each feature internally from the support rows.
        - The decoder CALD finetunes is ``out_mlp``, the FULL 2-layer MLP
          Linear(512 -> 1024) - GELU - Linear(1024 -> 10); the pre-decoder
          LayerNorm stays frozen, matching the paper's TabICL integration.
        - At most 10 classes (the checkpoint's ``max_classes``).
    """

    def __init__(self, device: str | None = None, checkpoint_path: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_pretrained(device=self.device, checkpoint_path=checkpoint_path)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.out_mlp.parameters():
            p.requires_grad = True
        self._pristine_decoder = copy.deepcopy(self.model.out_mlp.state_dict())
        self._X_sup = None
        self._y_sup = None

    # ------------------------------------------------------------ contract
    def fit(self, X_support, y_support):
        self.classes_ = np.unique(y_support)
        if len(self.classes_) > self.model.out_mlp[-1].out_features:
            raise ValueError(
                f"{len(self.classes_)} classes exceed the checkpoint's maximum "
                f"of {self.model.out_mlp[-1].out_features}")
        y_enc = np.searchsorted(self.classes_, y_support)
        self._X_sup = torch.tensor(np.asarray(X_support), dtype=torch.float32,
                                   device=self.device)
        self._y_sup = torch.tensor(y_enc, dtype=torch.long,
                                   device=self.device).unsqueeze(0)
        return self

    def predict_proba(self, X):
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            seq = torch.cat([self._X_sup, X_t], dim=0).unsqueeze(0)
            logits = self.model(seq, self._y_sup)[0, :, :self.n_classes]
            return torch.softmax(logits, dim=1).cpu().numpy()

    def forward_logits(self, X_anchor, X_eval):
        X_anc = torch.tensor(np.asarray(X_anchor), dtype=torch.float32, device=self.device)
        X_ev = torch.tensor(np.asarray(X_eval), dtype=torch.float32, device=self.device)
        seq = torch.cat([self._X_sup, X_anc, X_ev], dim=0).unsqueeze(0)
        return self.model(seq, self._y_sup)[0, :, :self.n_classes]

    def decoder_parameters(self):
        return list(self.model.out_mlp.parameters())

    def reset_decoder(self):
        self.model.out_mlp.load_state_dict(self._pristine_decoder)
