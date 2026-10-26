"""Contract tests for the live nano/TabICLv2 adapter (needs the checkpoint)."""

import numpy as np
import torch


def _toy(n=120, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, dim))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_grads_reach_only_decoder(nano_adapter):
    X, y = _toy()
    nano_adapter.fit(X[:80], y[:80])
    logits = nano_adapter.forward_logits(X[80:100], X[100:])
    assert logits.requires_grad
    logits.sum().backward()
    for name, p in nano_adapter.model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            assert name.startswith("out_mlp."), f"gradient leaked to {name}"
    nano_adapter.model.zero_grad(set_to_none=True)


def test_decoder_is_full_two_layer_mlp(nano_adapter):
    params = nano_adapter.decoder_parameters()
    assert len(params) == 4  # two Linears, each weight + bias
    shapes = sorted(tuple(p.shape) for p in params)
    assert (1024, 512) in shapes and (10, 1024) in shapes


def test_reset_restores_bit_identical(nano_adapter):
    before = {k: v.clone() for k, v in nano_adapter.model.out_mlp.state_dict().items()}
    for p in nano_adapter.decoder_parameters():
        p.data.add_(0.123)
    nano_adapter.reset_decoder()
    after = nano_adapter.model.out_mlp.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_forward_logits_order_and_shape(nano_adapter):
    X, y = _toy()
    nano_adapter.fit(X[:80], y[:80])
    out = nano_adapter.forward_logits(X[80:96], X[96:120])
    assert out.shape == (16 + 24, nano_adapter.n_classes)
    proba = nano_adapter.predict_proba(X[96:120])
    assert proba.shape == (24, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
