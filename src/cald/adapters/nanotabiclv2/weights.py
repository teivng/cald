"""Load the OFFICIAL TabICLv2 checkpoint onto the (patched) nano module.

The released checkpoint (Hugging Face ``jingang/TabICL``,
``tabicl-classifier-v2-20260212.ckpt``) stores its state dict under the
official package's module names (``col_embedder`` / ``row_interactor`` /
``icl_predictor``). This module translates every key to nano's flat naming,
asserts the translation is a complete bijection, and loads strictly. Nothing
is redistributed: the checkpoint is downloaded from Hugging Face on first use
(or taken from ``checkpoint_path``).

Key facts baked into the translation (verified against official tabicl 2.2.0):
- per-block parameters map through one shared inner table (attention packed
  in_proj/out_proj, MLP linear1/2 -> mlp.0/.2, norms -> ln_attn/ln_mlp, SSMax);
- inducing vectors and CLS tokens need leading singleton dims;
- RoPE frequencies are recomputed by nano; we check they match the checkpoint
  exactly instead of loading them.
"""

import re

import torch

from .patches import get_model_class

HF_REPO = "jingang/TabICL"
HF_FILE = "tabicl-classifier-v2-20260212.ckpt"

# Inner-parameter mapping shared by every transformer block:
# official TransformerBlock -> nano TransformerBlock (an nn.MultiheadAttention)
_INNER = {
    "linear1.weight": "mlp.0.weight",
    "linear1.bias": "mlp.0.bias",
    "linear2.weight": "mlp.2.weight",
    "linear2.bias": "mlp.2.bias",
    "norm1.weight": "ln_attn.weight",
    "norm1.bias": "ln_attn.bias",
    "norm2.weight": "ln_mlp.weight",
    "norm2.bias": "ln_mlp.bias",
    "attn.in_proj_weight": "in_proj_weight",
    "attn.in_proj_bias": "in_proj_bias",
    "attn.out_proj.weight": "out_proj.weight",
    "attn.out_proj.bias": "out_proj.bias",
    "attn.ssmax_layer.base_mlp.0.weight": "ssmax_layer.base_mlp.0.weight",
    "attn.ssmax_layer.base_mlp.0.bias": "ssmax_layer.base_mlp.0.bias",
    "attn.ssmax_layer.base_mlp.2.weight": "ssmax_layer.base_mlp.2.weight",
    "attn.ssmax_layer.base_mlp.2.bias": "ssmax_layer.base_mlp.2.bias",
    "attn.ssmax_layer.query_mlp.0.weight": "ssmax_layer.query_mlp.0.weight",
    "attn.ssmax_layer.query_mlp.0.bias": "ssmax_layer.query_mlp.0.bias",
    "attn.ssmax_layer.query_mlp.2.weight": "ssmax_layer.query_mlp.2.weight",
    "attn.ssmax_layer.query_mlp.2.bias": "ssmax_layer.query_mlp.2.bias",
}

_ROPE_SENTINEL = "ROPE_FREQS"


def _translate(key: str):
    """official state_dict key -> (nano key, reshape fn), or None to skip."""
    ident = lambda t: t  # noqa: E731

    simple = {
        "col_embedder.in_linear.weight": "x_embed.weight",
        "col_embedder.in_linear.bias": "x_embed.bias",
        "col_embedder.y_encoder.weight": "y_embed_in.weight",
        "col_embedder.y_encoder.bias": "y_embed_in.bias",
        "row_interactor.out_ln.weight": "row_ln.weight",
        "row_interactor.out_ln.bias": "row_ln.bias",
        "icl_predictor.y_encoder.weight": "y_embed_icl.weight",
        "icl_predictor.y_encoder.bias": "y_embed_icl.bias",
        "icl_predictor.ln.weight": "out_ln.weight",
        "icl_predictor.ln.bias": "out_ln.bias",
    }
    if key in simple:
        return simple[key], ident
    if key == "row_interactor.cls_tokens":
        return "row_cls_tokens", lambda t: t.unsqueeze(0).unsqueeze(0)
    if key == "row_interactor.tf_row.rope.freqs":
        return _ROPE_SENTINEL, ident
    if key.startswith("icl_predictor.decoder."):
        return key.replace("icl_predictor.decoder.", "out_mlp."), ident

    m = re.match(r"col_embedder\.tf_col\.blocks\.(\d+)\.(.+)$", key)
    if m:
        i, rest = m.group(1), m.group(2)
        if rest == "ind_vectors":
            return f"col_blocks.{i}.inducing_vectors", lambda t: t.unsqueeze(0)
        m2 = re.match(r"multihead_attn([12])\.(.+)$", rest)
        if m2 and m2.group(2) in _INNER:
            tfm = {"1": "tfm1", "2": "tfm2"}[m2.group(1)]
            return f"col_blocks.{i}.{tfm}.{_INNER[m2.group(2)]}", ident
        return None

    for stage, prefix in [("row_interactor.tf_row", "row_blocks"),
                          ("icl_predictor.tf_icl", "icl_blocks")]:
        m = re.match(rf"{re.escape(stage)}\.blocks\.(\d+)\.(.+)$", key)
        if m and m.group(2) in _INNER:
            return f"{prefix}.{m.group(1)}.{_INNER[m.group(2)]}", ident

    return None


def load_pretrained(device: str = "cpu", checkpoint_path: str | None = None):
    """Build a patched NanoTabICLv2 carrying the official TabICLv2 weights.

    Returns the model (on ``device``, in eval mode). Raises if the key
    translation is not a complete bijection or the strict load fails.
    """
    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(HF_REPO, HF_FILE)

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd, cfg = ck["state_dict"], ck["config"]

    # the nano architecture implements exactly this configuration family
    assert cfg["col_feature_group"] == "same" and cfg["col_target_aware"] is True
    assert cfg["row_rope_interleaved"] is False and cfg["bias_free_ln"] is False

    cls = get_model_class()
    model = cls(
        max_classes=cfg["max_classes"], out_dim=cfg["max_classes"],
        embed_dim=cfg["embed_dim"], col_num_blocks=cfg["col_num_blocks"],
        row_num_blocks=cfg["row_num_blocks"], icl_num_blocks=cfg["icl_num_blocks"],
        col_nhead=cfg["col_nhead"], row_nhead=cfg["row_nhead"],
        icl_nhead=cfg["icl_nhead"],
        feature_group_size=cfg["col_feature_group_size"],
        n_cls_cols=cfg["row_num_cls"], n_cls_rows=cfg["col_num_inds"],
    )

    new_sd, unmapped, rope_freqs = {}, [], None
    for k, v in sd.items():
        tr = _translate(k)
        if tr is None:
            unmapped.append(k)
            continue
        nk, fn = tr
        if nk == _ROPE_SENTINEL:
            rope_freqs = v.clone()
            continue
        new_sd[nk] = fn(v)

    nano_keys = set(model.state_dict().keys())
    missing = sorted(nano_keys - set(new_sd))
    extra = sorted(set(new_sd) - nano_keys)
    if unmapped or missing or extra:
        raise RuntimeError(
            "checkpoint key translation is not a bijection — refusing to load.\n"
            f"  unmapped official keys: {unmapped[:5]}{'...' if len(unmapped) > 5 else ''}\n"
            f"  nano keys not covered: {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"  unknown mapped keys: {extra[:5]}{'...' if len(extra) > 5 else ''}")

    model.load_state_dict(new_sd, strict=True)

    if rope_freqs is not None:
        nano_inv = model.row_blocks[0].rope.inv_freq
        if not torch.allclose(rope_freqs.float(), nano_inv.float(), rtol=0, atol=1e-6):
            raise RuntimeError("RoPE frequencies differ between checkpoint and nano")

    return model.to(device).eval()
