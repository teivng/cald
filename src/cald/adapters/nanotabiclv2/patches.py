"""Two behavioural deltas applied ON TOP of the upstream nanoTabICL source
(which is never edited):

1. **Feature-grouping offsets.** Upstream nano groups features with offsets
   ``(2**i - 1)`` = (0, 1, 3); the official tabicl 2.2.0 ``ColEmbedding``
   (feature-grouping mode "same", which the v2 checkpoint was trained with)
   uses ``(idxs + 2**i) % n_cols`` = (1, 2, 4). Because the row stage applies
   RoPE over these grouped columns, the offsets change the logits — matching
   the official ones is REQUIRED for parity with the released checkpoint
   (verified: max |Δlogit| <= 8.3e-5 vs the official package).
   Applied as a ``forward`` override in a subclass.

2. **Chunked row attention.** Upstream materialises the fp32 attention matrix
   for all (batch x rows) at once; on wide tables (e.g. 768 embedding features
   -> 776 row tokens) this needs tens of GB. We chunk over the merged
   batch-rows axis, which is embarrassingly parallel, so the result is
   bit-identical (verified: max diff exactly 0.0). Applied as a method
   override on ``TableAttnBase``, which every attention block inherits.
"""

import torch

from .upstream import load_upstream

# ~2 GB fp32 attention-matrix budget per row_attn chunk
CHUNK_ELEMS = 2 ** 29

_cache: dict = {}


def _chunked_row_attn(self, q, kv=None, **kwargs):
    n_batch, n_rows, n_cols, embed_dim = q.shape
    q, kv = (None if t is None else t.flatten(0, 1) for t in [q, kv])
    nhead = self.num_heads if hasattr(self, "num_heads") else self.tfm1.num_heads
    seq = q.shape[-2] if kv is None else max(q.shape[-2], kv.shape[-2])
    chunk = max(1, int(CHUNK_ELEMS // max(1, nhead * seq * seq)))
    if q.shape[0] > chunk:
        outs = [self(q[i:i + chunk], None if kv is None else kv[i:i + chunk], **kwargs)
                for i in range(0, q.shape[0], chunk)]
        return torch.cat(outs, dim=0).reshape(n_batch, n_rows, -1, embed_dim)
    return self(q, kv, **kwargs).reshape(n_batch, n_rows, -1, embed_dim)


def get_model_class():
    """Return the patched NanoTabICLv2 class (built once, cached)."""
    if "cls" in _cache:
        return _cache["cls"]

    nano = load_upstream()
    nano.TableAttnBase.row_attn = _chunked_row_attn  # patch 2 (bit-identical)

    class NanoTabICLv2Official(nano.NanoTabICLv2):
        """Upstream NanoTabICLv2 with official-tabicl feature grouping.

        The forward is upstream's verbatim except the grouping offsets
        (patch 1 above): ``(idxs + 2**i) % n_cols`` instead of
        ``(idxs + 2**i - 1) % n_cols``.
        """

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            n_batch, n_rows, n_cols = x.shape
            n_batch, n_train = y.shape

            x = (x - x[:, :n_train].mean(dim=1, keepdim=True)) / (
                x[:, :n_train].std(dim=1, unbiased=False, keepdim=True) + 1e-8)
            idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
            # patch 1: official tabicl 2.2.0 grouping offsets (2**i), not (2**i - 1)
            x = torch.stack([x[:, :, (idxs + 2 ** i) % n_cols]
                             for i in range(self.feature_group_size)], dim=-1)
            emb = self.x_embed(x)
            emb[:, :n_train] += self.y_embed_in(y[:, :, None, None])

            for block in self.col_blocks:
                emb = block.col_attn(emb, kv_max_idx=n_train)

            emb = torch.cat([self.row_cls_tokens.expand(n_batch, n_rows, -1, -1), emb], dim=2)
            for block in self.row_blocks[:-1]:
                emb = block.row_attn(emb)
            emb = self.row_blocks[-1].row_attn(emb, q_max_idx=self.row_cls_tokens.size(-2))
            emb = self.row_ln(emb).flatten(-2, -1)

            emb[:, :n_train] += self.y_embed_icl(y[:, :, None])
            for block in self.icl_blocks[:-1]:
                emb = block(emb, kv_max_idx=n_train)
            emb = self.icl_blocks[-1](emb[:, n_train:], emb[:, :n_train])

            return self.out_mlp(self.out_ln(emb))

    _cache["cls"] = NanoTabICLv2Official
    return NanoTabICLv2Official
