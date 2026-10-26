"""TabPFN adapter — PLACEHOLDER.

The paper's TabPFN integration (tabpfn < 7) finetunes the full decoder MLP
``model.models_[0].decoder_dict["standard"]`` and forwards through the fitted
estimator's preprocessed tensors (``executor_.ensemble_members[0]``,
``transform_X_test``). Porting it onto the ``CALDBackbone`` contract is
mechanical but requires pinning tabpfn (these are private attributes), and
tabpfn >= 7 adds a Prior Labs model-license acceptance step. Not shipped yet.
"""

from .base import CALDBackbone


class TabPFNAdapter(CALDBackbone):
    """Not implemented yet — see the module docstring and the paper's research
    repository for the reference integration."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TabPFNAdapter is a placeholder. Use NanoTabICLv2Adapter, or see "
            "the paper repository for the TabPFN reference integration.")

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
