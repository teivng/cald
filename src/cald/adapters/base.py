"""The CALD attachment contract.

A backbone adapter makes an in-context tabular model speak CALD. Exactly four
backbone-specific capabilities are needed; everything else (pseudolabels,
losses, the finetuning loop, calibration, thresholding) lives once in
``cald.detector`` and is shared by every backbone.

Adapters must ensure that only the decoder parameters returned by
``decoder_parameters()`` have ``requires_grad=True`` (freeze everything else
in ``fit``), and that the backbone is deterministic in the forward pass
(disable dropout and other stochastic layers).
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class CALDBackbone(ABC):
    """Contract between a tabular in-context backbone and the CALD detector.

    After ``fit``, the adapter must expose ``classes_`` (sorted unique original
    labels, as produced by ``np.unique``). ``encode_labels`` is the single
    source of truth for the label encoding; all logits/probabilities are in
    that encoded class order.

    A backbone instance belongs to one detector at a time: ``fit`` repoints
    the in-context support that every later call uses, so sharing one adapter
    between two live ``CALD`` detectors silently corrupts both. Reuse an
    adapter only sequentially (fit -> detect..., then fit again).
    """

    classes_: np.ndarray

    def _check_fitted(self) -> None:
        if getattr(self, "classes_", None) is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted — call "
                "fit(X_support, y_support) first")

    @property
    def n_classes(self) -> int:
        self._check_fitted()
        return len(self.classes_)

    def encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Map original labels to the 0..C-1 encoding used by the logits.

        The default is positional lookup in the sorted ``classes_`` (the same
        scheme as scikit-learn's LabelEncoder). Adapters that encode labels
        differently must override this so the detector stays consistent.
        """
        self._check_fitted()
        return np.searchsorted(self.classes_, np.asarray(y))

    @abstractmethod
    def fit(self, X_support: np.ndarray, y_support: np.ndarray) -> "CALDBackbone":
        """Condition the backbone on the labelled support set (in-context).

        Must freeze every parameter except the decoder subset and cache the
        pristine decoder weights for ``reset_decoder``.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, n_classes) class probabilities in encoded class order, no grad."""

    @abstractmethod
    def forward_logits(self, X_anchor: np.ndarray, X_eval: np.ndarray) -> torch.Tensor:
        """Differentiable logits for [anchor; eval] rows, in that order.

        Returns a (n_anchor + n_eval, n_classes) tensor with gradients flowing
        to the decoder parameters.
        """

    @abstractmethod
    def decoder_parameters(self) -> list[torch.nn.Parameter]:
        """The decoder parameters CALD finetunes — the FULL decoder MLP.

        CALD finetunes the entire decoder head (both linear layers of the MLP),
        not just the last layer; last-layer-only finetuning is an ablation in
        the paper, not the method.
        """

    @abstractmethod
    def reset_decoder(self) -> None:
        """Restore the decoder to its pristine pretrained weights."""
