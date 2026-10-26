"""Golden-logits regression: the submodule + patches + checkpoint remap must
keep reproducing the logits verified against the official tabicl package
(max |Δlogit| <= 8.3e-5 at port time; the golden file was generated from the
verified port and matched bit-identically)."""

from pathlib import Path

import numpy as np
import torch

GOLDEN = Path(__file__).parent / "golden" / "nano_logits_breast_cancer.npz"


def test_golden_logits(nano_adapter):
    g = np.load(GOLDEN)
    X = torch.tensor(g["X"], dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(g["y_train"], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = nano_adapter.model(X, y)[0].numpy()
    # 5e-4 absorbs fp32 SDPA kernel drift across torch versions (~3e-5 seen
    # between 2.11 and 2.14); a porting error (wrong grouping offsets, wrong
    # key mapping) produces O(1) differences.
    assert np.abs(logits - g["logits"]).max() <= 5e-4
    acc = (logits[:, :2].argmax(1) == g["y_test"]).mean()
    assert acc >= 0.95
