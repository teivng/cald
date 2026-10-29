# CALD

**CALD** (Calibrated Anchor-Loss Disagreement) detects harmful distribution
shift for in-context tabular foundation models — attachable to any backbone
that exposes a differentiable forward pass and a decoder head.

From **"When to Trust a PFN: Detecting Harmful Shift in Tabular Foundation
Models"** (NeurIPS 2026).

CALD asks the model to argue against itself: briefly finetune its decoder to
*disagree* with its own predictions on a new unlabelled batch while *staying
right* on a small trusted set of labelled examples (the anchor). On familiar
data the model defends its predictions and the fight drags the anchor loss up;
on shifted data it was never confident to begin with, disagreement is cheap,
and the anchor loss stays low — so a low post-finetune anchor loss, compared
against its range on known in-distribution batches, is the alarm.

![CALD, animated: a PFN's in-context decision boundary; calibration rounds finetune the decoder to disagree on bootstrapped ID eval batches while agreeing on the anchor, recording the post-finetune anchor loss; at deployment a benign batch lands inside the calibration record while a harmfully shifted batch falls below the threshold and fires.](docs/cald_story.gif)

*The method in 90 seconds, on a 2-D task with the real TabICLv2 backbone —
every number in the animation comes from actual CALD runs
([`examples/manim/`](examples/manim/) regenerates it; full-quality video:
[`docs/cald_story.mp4`](docs/cald_story.mp4)).*

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
calibration phase (`n_calibration` $`= K`$ rounds, eval batches bootstrapped
from the calibration pool) and sets the threshold at the empirical
$`\alpha`$-quantile of the $`\lambda`$ record. `detect` runs the deployment phase
on a fresh copy of the decoder ($`\theta \leftarrow \theta_0`$ each call).

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

## Hyperparameters

The paper's certified deployment mode, and this package's defaults:

| Parameter | Paper symbol | Default |
|---|---|---|
| `lr` | $`\eta`$ | `1e-3` (frozen before deployment) |
| `n_steps` | $`T`$ | 50 |
| `n_calibration` | $`K`$ | 50 |
| `alpha` | $`\alpha`$ | 0.05 |
| `anchor_size` | — | 128 |
| `eval_size` | — | 128 |

Like the paper's experiments, the decoder finetuned is the backbone's full
decoder MLP (last-layer-only finetuning appears in the paper only as an
ablation). The full experimental harness (15 datasets, 16 baselines, harm
labelling via a Hoeffding bound) lives in the paper's research repository.

**Hardware.** CALD is light: only the decoder MLP trains (about 0.5M
parameters for TabICLv2), so a detection call is $`T`$ forward/backward passes,
not model training. The paper's experiments ran on a single NVIDIA A6000
(48 GB), with peak GPU memory around 10 GB on the largest dataset; the
TabICLv2/nanoTabICL experiments ran on a single RTX PRO 6000. On plain tabular
data (tens of features, support of a few thousand rows) a detection call takes
seconds on any modern GPU, and CPU-only works for small problems — the
quickstart runs on a laptop.

## Method

This section follows Section 2 of the paper.

![Figure 1 from the paper. CALD in three steps: setup (a labelled anchor inside the ID manifold, an eval batch of unknown origin), mechanism (joint agree/disagree finetuning; the anchor-loss trajectory separates ID from OOD), decision (a one-sided test of the deployment statistic against the ID calibration record).](docs/figure1.png)

Denote the ID distribution $`p_0`$ and let
$`\mathcal{D}^{\mathrm{sup}} = (X_{\mathrm{sup}}, y_{\mathrm{sup}}) \sim p_0^n`$
be a size-$`n`$ ID support set. The queries are partitioned into two disjoint
roles:

- **Anchor** $`(X_{\mathrm{anc}}, y_{\mathrm{anc}}) \sim p_0`$ with
  $`X_{\mathrm{anc}} \cap X_{\mathrm{sup}} = \emptyset`$: a held-out set of
  *labelled* ID samples, serving as a reference ID signal.
- **Eval** $`X_{\mathrm{eval}} \sim p_1 \overset{?}{=} p_0`$: a batch of
  *unlabelled* query points whose distributional origin is unknown.

Let $`q_\theta`$ denote the PFN conditioned on $`\mathcal{D}^{\mathrm{sup}}`$, and
write $`\hat{y}_{\mathrm{eval}}`$ for the pseudolabels
$`\hat{y} = \arg\max_y q_{\theta}(y \mid x, \mathcal{D}^{\mathrm{sup}})`$ it
assigns to $`X_{\mathrm{eval}}`$. CALD computes the cross-entropy loss rewarding
correct predictions on the anchor,

$$
\mathcal{L}_{\mathrm{anc}}(\theta) = \frac{1}{|X_{\mathrm{anc}}|}
\sum_{(x,y)} -\log q_\theta(y \mid x, \mathcal{D}^{\mathrm{sup}}),
$$

and the Disagreement Cross-Entropy (DCE) loss on the eval set, rewarding the
model for placing low probability on its own initial pseudolabels,

$$
\mathcal{L}_{\mathrm{eval}}(\theta) = \frac{1}{|X_{\mathrm{eval}}|}
\sum_{(x,\hat{y})} \left[ \frac{1}{C-1} \sum_{c \neq \hat{y}}
-\log q_\theta(c \mid x) \right].
$$

Gradient descent on the joint objective
$`\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{anc}}(\theta) + \mathcal{L}_{\mathrm{eval}}(\theta)`$
forces $`q_\theta`$ to disagree on the eval set while remaining performant on ID
samples. The *post-finetune anchor loss*
$`\lambda := \mathcal{L}_{\mathrm{anc}}(\theta_T)`$ is systematically **lower**
when $`X_{\mathrm{eval}}`$ is OOD than when it is ID; CALD turns this
one-dimensional signal into a calibrated one-sided test.

Given hyperparameters $`T`$ (finetune steps), $`K`$ (calibration rounds), $`\eta`$
(learning rate), and $`\alpha`$ (test level), CALD proceeds in two phases:

1. **Calibration phase.** For each round $`k = 1, \ldots, K`$, draw an ID eval
   batch $`X_{\mathrm{eval}}^{(k)} \sim p_0`$, freeze its pseudolabels
   $`\hat{y}_{\mathrm{eval}}^{(k)}`$ assigned by $`q_{\theta_0}`$, and reset
   $`\theta \leftarrow \theta_0`$. At each of $`T`$ gradient steps, the PFN
   forward-passes the joint sequence
   $`(X_{\mathrm{sup}}, y_{\mathrm{sup}}, X_{\mathrm{anc}}, X_{\mathrm{eval}}^{(k)})`$:
   the anchor logits supply $`\mathcal{L}_{\mathrm{anc}}(\theta)`$ against the
   true labels, the eval logits supply $`\mathcal{L}_{\mathrm{eval}}(\theta)`$
   against the frozen pseudolabels, and the joint loss is descended via
   $`\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)`$ with
   only the decoder trainable. Record
   $`\lambda^{(k)} := \mathcal{L}_{\mathrm{anc}}(\theta_T)`$. After $`K`$ rounds,
   the record $`\{\lambda^{(k)}\}_{k=1}^{K}`$ is an empirical null under
   known-ID conditions.
2. **Deployment phase.** On a deployment batch $`X_{\mathrm{eval}}^{*}`$ of
   unknown origin, freeze its pseudolabels, reset $`\theta \leftarrow \theta_0`$,
   and run the same $`T`$-step finetune, recording the deployment statistic
   $`\lambda^{*} := \mathcal{L}_{\mathrm{anc}}(\theta_T)`$. The verdict is a
   one-sided test against the calibration record:

$$
\lambda^{*} < \mathrm{Quantile}_\alpha\left(\lambda^{(1)}, \ldots, \lambda^{(K)}\right)
\;\Longrightarrow\; \text{declare OOD}.
$$

The support set and anchor are held fixed across both phases; only the eval
batch varies. Under the null (the deployment batch exchangeable with the
calibration batches, hyperparameters fixed before deployment), the false-alarm
rate is at most $`\alpha + \frac{1}{K+1}`$.

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
