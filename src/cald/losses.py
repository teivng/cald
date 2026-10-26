"""CALD losses.

L_anc is plain cross-entropy on the anchor batch (torch.nn.functional applies).
L_dis, the disagreement loss, pushes the decoder away from its own frozen
pseudolabels on the eval batch:

- binary: cross-entropy against the flipped pseudolabel;
- multiclass: cross-entropy against the uniform distribution over the
  C-1 non-pseudolabel classes ("zero-hot"), which for C=2 reduces to the
  flipped-label form.
"""

import torch
import torch.nn.functional as F


def disagreement_loss(logits_eval: torch.Tensor, pseudo_y: torch.Tensor,
                      n_classes: int) -> torch.Tensor:
    """Mean disagreement loss over the eval batch.

    Args:
        logits_eval: (n_eval, n_classes) differentiable logits.
        pseudo_y: (n_eval,) long tensor of pretrained-model pseudolabels.
        n_classes: number of classes C.
    """
    if n_classes == 2:
        return F.cross_entropy(logits_eval, 1 - pseudo_y)
    zero_hot = 1.0 - F.one_hot(pseudo_y, num_classes=n_classes).float()
    return (-(logits_eval * zero_hot).sum(1) / (n_classes - 1)
            + torch.logsumexp(logits_eval, dim=1)).mean()
