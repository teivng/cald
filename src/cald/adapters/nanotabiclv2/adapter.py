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

    Args:
        device: torch device (default: cuda if available).
        checkpoint_path: local official checkpoint instead of the Hugging Face
            download.
        model: an already-built nano model (e.g. from
            ``cald.adapters.nanotabiclv2.weights.load_pretrained``, or a test
            double); when given, no weights are downloaded and
            ``checkpoint_path`` is ignored.
    """

    def __init__(self, device: str | None = None, checkpoint_path: str | None = None,
                 model: torch.nn.Module | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model.to(self.device) if model is not None
                      else load_pretrained(device=self.device,
                                           checkpoint_path=checkpoint_path))
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.out_mlp.parameters():
            p.requires_grad = True
        self._pristine_decoder = copy.deepcopy(self.model.out_mlp.state_dict())
        self._X_sup = None
        self._y_sup = None

    def _features(self, X) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D feature array, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError(
                "features contain NaN or inf; impute or drop them first "
                "(the model z-scores raw numeric features internally)")
        return torch.from_numpy(X).to(self.device)

    # ------------------------------------------------------------ contract
    def fit(self, X_support, y_support):
        X_sup = self._features(X_support)  # validate before mutating any state
        self.classes_ = np.unique(y_support)
        if len(self.classes_) > self.model.out_mlp[-1].out_features:
            raise ValueError(
                f"{len(self.classes_)} classes exceed the checkpoint's maximum "
                f"of {self.model.out_mlp[-1].out_features}")
        self._X_sup = X_sup
        self._y_sup = torch.tensor(self.encode_labels(y_support), dtype=torch.long,
                                   device=self.device).unsqueeze(0)
        return self

    def predict_proba(self, X):
        X_t = self._features(X)
        with torch.no_grad():
            seq = torch.cat([self._X_sup, X_t], dim=0).unsqueeze(0)
            logits = self.model(seq, self._y_sup)[0, :, :self.n_classes]
            return torch.softmax(logits, dim=1).cpu().numpy()

    def forward_logits(self, X_anchor, X_eval):
        X_anc = self._features(X_anchor)
        X_ev = self._features(X_eval)
        seq = torch.cat([self._X_sup, X_anc, X_ev], dim=0).unsqueeze(0)
        return self.model(seq, self._y_sup)[0, :, :self.n_classes]

    def decoder_parameters(self):
        return list(self.model.out_mlp.parameters())

    def reset_decoder(self):
        self.model.out_mlp.load_state_dict(self._pristine_decoder)
