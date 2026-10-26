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
    labels, as produced by ``np.unique``); the detector encodes labels with
    ``np.searchsorted(classes_, y)`` and all logits/probabilities are in this
    encoded class order.
    """

    classes_: np.ndarray

    @property
    def n_classes(self) -> int:
        return len(self.classes_)

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
