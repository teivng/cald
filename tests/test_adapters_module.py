"""The adapters module itself: registry, contract defaults, guards."""

import numpy as np
import pytest

from cald.adapters import CALDBackbone, get_adapter, register_adapter


class _Dummy(CALDBackbone):
    def fit(self, X_support, y_support):
        self.classes_ = np.unique(y_support)
        return self

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError

    def forward_logits(self, X_anchor, X_eval):  # pragma: no cover
        raise NotImplementedError

    def decoder_parameters(self):  # pragma: no cover
        return []

    def reset_decoder(self):  # pragma: no cover
        pass


def test_register_and_get():
    register_adapter("dummy-test", _Dummy)
    bb = get_adapter("dummy-test")
    assert isinstance(bb, _Dummy)


def test_register_rejects_builtin_names_and_non_backbones():
    with pytest.raises(ValueError):
        register_adapter("nanotabiclv2", _Dummy)
    with pytest.raises(TypeError):
        register_adapter("not-a-backbone", object)


def test_unknown_adapter():
    with pytest.raises(ValueError, match="unknown adapter"):
        get_adapter("nope")


def test_placeholders_raise_cleanly():
    for name in ("tabpfn", "tabicl"):
        with pytest.raises(NotImplementedError):
            get_adapter(name)


def test_unfitted_guards():
    bb = _Dummy()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = bb.n_classes
    with pytest.raises(RuntimeError, match="not fitted"):
        bb.encode_labels([0, 1])


def test_encode_labels_default():
    bb = _Dummy().fit(np.zeros((4, 2)), np.array(["b", "a", "b", "c"]))
    assert list(bb.classes_) == ["a", "b", "c"]
    assert bb.encode_labels(np.array(["c", "a", "b"])).tolist() == [2, 0, 1]
    assert bb.n_classes == 3


def test_nano_rejects_nan_features(nano_adapter):
    X = np.zeros((40, 4))
    y = np.array([0, 1] * 20)
    X_bad = X.copy()
    X_bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        nano_adapter.fit(X_bad, y)
