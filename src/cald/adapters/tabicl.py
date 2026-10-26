"""TabICL v1.1 adapter — PLACEHOLDER.

The paper's TabICL integration (tabicl == 0.1.x) finetunes the full decoder
MLP ``clf.model_.icl_predictor.decoder`` and forwards through
``clf.model_._train_forward`` on tensors produced by ``clf.X_encoder_`` and
``clf.ensemble_generator_``. These are private APIs, so the adapter must pin
the tabicl version. Not shipped yet. For the TabICLv2 architecture, use
``NanoTabICLv2Adapter`` instead.
"""

from .base import CALDBackbone


class TabICLAdapter(CALDBackbone):
    """Not implemented yet — see the module docstring and the paper's research
    repository for the reference integration."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TabICLAdapter is a placeholder. Use NanoTabICLv2Adapter, or see "
            "the paper repository for the TabICL v1.1 reference integration.")

    def fit(self, X_support, y_support):  # pragma: no cover
        raise NotImplementedError

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError

    def forward_logits(self, X_anchor, X_eval):  # pragma: no cover
        raise NotImplementedError

    def decoder_parameters(self):  # pragma: no cover
        raise NotImplementedError

    def reset_decoder(self):  # pragma: no cover
        raise NotImplementedError
