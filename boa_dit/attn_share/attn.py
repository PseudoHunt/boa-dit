"""Patched DiT attention: explicit maps for statistics, or stored mean maps in place of softmax(QK^T).

Batch layout is the repo's CFG layout: rows [0, n) cond, [n, 2n) uncond (CTX.n). CTRL selects the behaviour:
  mode None     : the original timm forward (fused SDPA), untouched
  mode "stats"  : fused output as usual; additionally P = softmax(q k^T scale) is rebuilt in fp32 from the same
                  q, k and handed to CTRL.cb(block, step, P_cond, P_uncond, v) (nothing is stored here)
  mode "explicit": P computed explicitly (fp32 softmax) and used for the output (parity check vs fused)
  mode "share"  : at steps s < CTRL.k, P comes from CTRL.maps(block, step) -> (Mc, Mu) [H, N, N]; heads where
                  CTRL.heads(block, step) is False keep the computed map. Only V is projected for a fully shared
                  block; Q and K are skipped.
"""
import torch
import torch.nn.functional as F

from boa_dit.sample import CTX


class _Ctrl:
    mode = None
    cb = None
    k = 0
    maps = None      # fn(block, step) -> (Mc, Mu), fp16 [H, N, N] on the model's device
    heads = None     # fn(block, step) -> bool tensor [H] (True = replace) or None for all heads


CTRL = _Ctrl()


def _explicit(q, k):
    return torch.softmax((q.float() @ k.float().transpose(-2, -1)) * (q.shape[-1] ** -0.5), dim=-1)


def _make_forward(attn, bi):
    orig = attn.forward
    H, hd = attn.num_heads, attn.head_dim

    def forward(x, attn_mask=None, is_causal=False):
        mode, s = CTRL.mode, CTX.step
        if mode is None or s is None or (mode == "share" and s >= CTRL.k):
            return orig(x)
        if mode == "stats":
            return _stats_forward(x)
        B, N, C = x.shape
        n = CTX.n
        if mode == "share":
            mask = CTRL.heads(bi, s) if CTRL.heads is not None else None
            Mc, Mu = CTRL.maps(bi, s)
            if mask is None or bool(mask.all()):
                w, b = attn.qkv.weight[2 * C:], attn.qkv.bias[2 * C:]
                v = F.linear(x, w, b).reshape(B, N, H, hd).transpose(1, 2)            # [B, H, N, hd]
                P = torch.cat([Mc.expand(n, -1, -1, -1), Mu.expand(n, -1, -1, -1)]).to(v.dtype)
            else:
                qkv = attn.qkv(x).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                P = _explicit(q, k)
                stored = torch.cat([Mc.expand(n, -1, -1, -1), Mu.expand(n, -1, -1, -1)]).float()
                P = torch.where(mask.view(1, H, 1, 1), stored, P).to(v.dtype)
        else:  # explicit
            qkv = attn.qkv(x).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            P = _explicit(q, k).to(v.dtype)
        out = (P @ v).transpose(1, 2).reshape(B, N, C)
        return attn.proj(out)

    def _stats_forward(x):
        B, N, C = x.shape
        out = orig(x)                                                                   # forward unchanged
        qkv = attn.qkv(x).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        with torch.autocast("cuda", enabled=False):
            P = _explicit(qkv[0], qkv[1])
            CTRL.cb(bi, CTX.step, P[:CTX.n], P[CTX.n:])
        return out

    return forward


def patch(model):
    for bi, blk in enumerate(model.blocks):
        a = blk.attn
        assert all(type(m).__name__ == "Identity" for m in (a.q_norm, a.k_norm, getattr(a, "norm", a.q_norm)))
        assert getattr(a, "gate", None) is None
        blk.attn.forward = _make_forward(blk.attn, bi)
    return model
