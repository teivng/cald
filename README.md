# CALD

**CALD** (Calibrated Anchor-Loss Disagreement) detects harmful distribution
shift for in-context tabular foundation models — attachable to any backbone
that exposes a differentiable forward pass and a decoder head.

From **"When to Trust a PFN: Detecting Harmful Shift in Tabular Foundation
Models"** (NeurIPS 2026).

The idea is short: take a labelled support set, a held-out labelled **anchor**
batch from the same distribution, and an unlabelled **eval** batch of unknown
origin. Freeze the backbone, finetune only its decoder for a handful of steps,
pushing it to *disagree* with its own pseudolabels on the eval batch while
still *agreeing* with the anchor. Whether the anchor loss survives the
disagreement gradient is a sharp signal for whether the eval batch is in- or
out-of-distribution.

```
              X_support (labelled, in-context)
                       │
                       ▼
                ┌─────────────┐
   X_anchor ───►│  frozen     │──► CE(·, y_anchor)        anchor loss
   (labelled)   │  backbone   │
                │  + decoder  │     finetune to minimize both
                │  (trained)  │
   X_eval ─────►│             │──► DCE(·, ŷ_eval)         disagreement loss
   (unlabelled) └─────────────┘
                       │
                       ▼
              anchor-loss trajectory
              final value λ LOW  ⇔  eval is shifted
```

## The direction is part of the method

CALD **fires when the post-finetune anchor loss is LOW** — on shifted data the
backbone is uncertain, the disagreement gradient is weak, and the anchor loss
survives. This package deliberately offers **no option to flip, auto-calibrate,
or two-side** that comparison: the detector follows the mechanism; it is not an
arbitrary discriminator. One honest consequence: "shifted" is relative to the
backbone. A shift that barely degrades a strong backbone is near-distribution
*for it*, and CALD will treat it accordingly.

## Install

```bash
git clone --recurse-submodules <this-repo>
cd cald
pip install -e .            # or: uv pip install -e .
pip install -e ".[examples]"   # for examples/real_datasets.py
```

The nanoTabICL source is a pinned git submodule (`third_party/nanotabicl`,
BSD-3) — if you cloned without `--recurse-submodules`, run
`git submodule update --init`. The official TabICLv2 weights (110 MB) are
downloaded from Hugging Face (`jingang/TabICL`) on first use.

## Quickstart

```python
import numpy as np
from cald import CALD, get_adapter

backbone = get_adapter("nanotabiclv2")           # TabICLv2 via nanoTabICL
det = CALD(backbone, lr=1e-3, n_steps=50, fpr=0.05, n_calibration=50,
           support_size=2000, anchor_size=128, eval_size=128, seed=42)

det.fit(X_id, y_id)        # labelled in-distribution data: builds the
                           # support/anchor split and calibrates the threshold
result = det.detect(X_new) # an unlabelled batch of unknown origin
print(result.fired, result.statistic, result.threshold)
```

`CALD.fit` splits the ID data into a support set (the backbone's in-context
prompt), a disjoint anchor batch, and a calibration pool; it then runs
`n_calibration` bootstrapped ID rounds and sets the threshold at the `fpr`
quantile of their statistics. `detect` runs one finetuning episode on a fresh
copy of the decoder and compares λ against that threshold.

## Backbones

Adapters live in `cald.adapters`; the attachment contract is
`cald.adapters.base.CALDBackbone` — five methods: `fit`, `predict_proba`,
`forward_logits`, `decoder_parameters`, `reset_decoder`. CALD finetunes the
**full decoder MLP** (both layers), never just the last linear.

| Adapter | Status | Notes |
|---|---|---|
| `nanotabiclv2` | ✅ functional | TabICLv2 architecture via the [nanoTabICL](https://github.com/soda-inria/nanotabicl) submodule; official checkpoint loaded by on-the-fly key translation (`cald.adapters.nanotabiclv2.weights`), verified against the official package (max logit diff ≤ 8.3e-5, 100% argmax agreement). |
| `tabpfn` | 🚧 placeholder | The paper's TabPFN integration (tabpfn < 7, private decoder attributes). |
| `tabicl` | 🚧 placeholder | The paper's TabICL v1.1 integration (tabicl 0.1.x, private forward). |

To attach CALD to your own in-context model, implement `CALDBackbone` — the
nano adapter (`cald/adapters/nanotabiclv2/adapter.py`, ~90 lines) is the
template — and optionally expose it by name:

```python
from cald.adapters import register_adapter
register_adapter("mymodel", MyModelAdapter)
```

## Examples

```bash
python examples/quickstart.py       # synthetic shifts (harmful fires, confident
                                    # shift stays quiet); seconds on GPU,
                                    # a few minutes on CPU
python examples/real_datasets.py    # covertype / adult income / folktables
```

## Tests

```bash
pip install -e ".[test]"
pytest tests/            # detector semantics run without weights;
                         # adapter/parity tests download the checkpoint
                         # (or set CALD_TEST_CKPT=/path/to/tabicl-classifier-v2-*.ckpt)
```

## Paper settings

`lr=1e-3`, `n_steps=50`, `anchor_size=128`, `eval_size=128`, calibration over
50 rounds, threshold at the 5% quantile (TPR@5%FPR). The full experimental
harness (15 datasets, 16 baselines, harm labelling via a Hoeffding bound) lives
in the paper's research repository.

## Citation

```bibtex
@inproceedings{cald2026,
  title     = {When to Trust a {PFN}: Detecting Harmful Shift in Tabular Foundation Models},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## License

MIT. The vendored nanoTabICL source (`third_party/nanotabicl`) is BSD-3-Clause,
© its authors. TabICLv2 weights are downloaded from Hugging Face under their
own terms and are not redistributed here.
