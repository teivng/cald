# CALD

**CALD** (Calibrated Anchor-Loss Disagreement) detects harmful distribution
shift for in-context tabular foundation models — attachable to any backbone
that exposes a differentiable forward pass and a decoder head.

From **"When to Trust a PFN: Detecting Harmful Shift in Tabular Foundation
Models"** (NeurIPS 2026).

CALD partitions the queries into two roles beside the in-context **support**
set: a held-out *labelled* ID **anchor** batch, and an *unlabelled* **eval**
batch whose distributional origin is unknown. The decoder is briefly finetuned
under a joint objective — a cross-entropy agreement term on the anchor against
its true labels, plus a disagreement cross-entropy (DCE) term on the eval
batch rewarding low probability on the model's own frozen pseudolabels. The
post-finetune anchor loss **λ := L_anc(θ_T)** is systematically *lower* when
the eval batch is OOD than when it is ID, and CALD turns this signal into a
calibrated one-sided test.

![CALD in three steps: setup (anchor inside the ID manifold, eval batch of unknown origin), mechanism (joint agree/disagree finetuning; the anchor-loss trajectory separates ID from OOD), decision (one-sided test of the deployment statistic against the ID calibration record)](docs/figure1.png)

## Method

Given hyperparameters T (finetune steps), K (calibration rounds), η (learning
rate) and α (test level), CALD proceeds in two phases:

1. **Calibration phase.** For each round k = 1…K, draw an ID eval batch,
   freeze its pseudolabels assigned by the pretrained model, reset θ ← θ₀, and
   run T gradient steps on the joint loss, forward-passing the sequence
   (support, anchor, eval) at each step. Record the post-finetune anchor loss
   λ⁽ᵏ⁾ := L_anc(θ_T). The K values form an empirical null record under
   known-ID conditions.
2. **Deployment phase.** On a batch of unknown origin, freeze its
   pseudolabels, reset θ ← θ₀, run the same T-step finetune, and record λ\*.
   The verdict is a one-sided test against the calibration record:

   λ\* < Quantile_α(λ⁽¹⁾, …, λ⁽ᴷ⁾)  ⟹  declare OOD.

The support set and anchor are held fixed across both phases; only the eval
batch varies. Under the null (the deployment batch drawn from the same
distribution as the calibration batches, hyperparameters fixed before
deployment), the false-alarm rate is at most α + 1/(K+1).

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
det = CALD(backbone, lr=1e-3, n_steps=50, alpha=0.05, n_calibration=50,
           support_size=2000, anchor_size=128, eval_size=128, seed=42)

det.fit(X_id, y_id)        # calibration phase: support/anchor split +
                           # K calibration rounds -> threshold
result = det.detect(X_new) # deployment phase on a batch of unknown origin
print(result.fired, result.statistic, result.threshold)
```

`CALD.fit` splits the ID data into a support set (the backbone's in-context
prompt), a disjoint anchor batch, and a calibration pool, then runs the
calibration phase (`n_calibration` = K rounds, eval batches bootstrapped from
the calibration pool) and sets the threshold at the empirical α-quantile of
the λ record. `detect` runs the deployment phase on a fresh copy of the
decoder (θ ← θ₀ each call).

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

The paper's certified deployment mode is η = `lr=1e-3` frozen before
deployment, with `n_steps=50` (T), `n_calibration=50` (K), `alpha=0.05`,
`anchor_size=128`, `eval_size=128` — these are the package defaults. Like the
paper's experiments, the decoder finetuned is the backbone's full decoder MLP
(last-layer-only finetuning appears in the paper only as an ablation). The
full experimental harness (15 datasets, 16 baselines, harm labelling via a
Hoeffding bound) lives in the paper's research repository.

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
