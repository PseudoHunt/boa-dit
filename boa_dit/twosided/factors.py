"""Output-side factors for the two-sided solves (handoff §1), collected in the same per-block pass as the Grams.

Per token t of a (step, branch) slab: x = qkv input (LN1-modulated); h = fc1 pre-activation; d_t = GELU'_tanh(h_t);
z = LN2 input; g1, g2, gamma2 = the sample's gate_msa, gate_mlp, scale_mlp. Bucket means: E_b[g] where a gate enters
linearly, E_b[g^2] where it enters squared (rule.md, Conventions). Branches combined with cfg weights like H_col.

  fc1  : C_b = E_b[d d^T];  H_row(fc1) = sum_b p_b (W2^T diag(E_b[g2^2]) W2) * C_b      (Variant 1)
  attn : G_h = E[(A_h X)^T (A_h X)] per head (attention-weighted V input Gram), A_h rebuilt explicitly in fp32
  tok  : 520 sampled tokens; A_t = I + diag(gbar2_b) W2 diag(d_t) W1 L_t;  R_b = E_b[A_t^T A_t];
         H_row(W_o) = sum_b p_b diag(gbar1_b) R_b diag(gbar1_b)                              (Variant 2)
  V    : Variant 1 W_oh^T diag(sum_b p_b E_b[g1^2]) W_oh;  Variant 2 W_oh^T H_row(W_o) W_oh   (per head, 72 x 72)
All accumulation fp32 (TF32 off, as everywhere in the repo); the hooks only read.
"""
import math

import torch

D, DH, NH, HD = 1152, 4608, 16, 72
TOK_PER_BUCKET_BRANCH = 26


def gelu_tanh_grad(h):
    k = math.sqrt(2.0 / math.pi)
    u = k * (h + 0.044715 * h ** 3)
    t = torch.tanh(u)
    return 0.5 * (1 + t) + 0.5 * h * (1 - t * t) * k * (1 + 3 * 0.044715 * h * h)


def attn_fp32(Wqkv, bqkv, Wo, bo, x):
    """DiT attention output in fp32 from explicit weights (no module call, so no hooks fire). x [n, 256, D]."""
    n = x.shape[0]
    qkv = (x @ Wqkv.T + bqkv).reshape(n, 256, 3, NH, HD).permute(2, 0, 3, 1, 4)
    P = torch.softmax((qkv[0] @ qkv[1].transpose(-2, -1)) * (HD ** -0.5), dim=-1)
    return (P @ qkv[2]).transpose(1, 2).reshape(n, 256, D) @ Wo.T + bo


def attn_weights(attn):
    return tuple(t.detach().float() for t in (attn.qkv.weight, attn.qkv.bias, attn.proj.weight, attn.proj.bias))


def token_plan(l, N, S=50, B=10):
    """(step, branch) -> list of (slot, image, token); stratified over buckets, branches and images; seed = block."""
    g = torch.Generator().manual_seed(l)
    per = S // B
    plan, slot, meta = {}, 0, []
    for b in range(B):
        for br in range(2):
            imgs = torch.randperm(N, generator=g)[:TOK_PER_BUCKET_BRANCH]
            for im in imgs.tolist():
                s = b * per + int(torch.randint(per, (1,), generator=g))
                t = int(torch.randint(256, (1,), generator=g))
                plan.setdefault((s, br), []).append((slot, im, t))
                meta.append((b, br))
                slot += 1
    return plan, meta


class FactorAccumulator:
    """Hooks on one DiT block. needs: subset of {"fc1", "attn", "tok", "fc1diag", "projx"}. attn_q: optional quantized
    attention weights (attn_weights(...) of the W4 block; D2) -> e_t at the sampled tokens. gates: "bucket" (the handoff's
    bucket-mean gates in A_t and diag(g1bar) R diag(g1bar)) or "sample" (each token's own sample gates g1, g2 inside the
    expectation: H_row(W_o) = E[diag(g1) A^T A diag(g1)], A_t = I + diag(g2) W2 D W1 L). cur = (step, bucket, branch)."""

    def __init__(self, block, l, N, dev, needs, B=10, cfg=1.5, attn_q=None, gates="bucket"):
        self.blk, self.l, self.N, self.dev, self.needs, self.B = block, l, N, dev, set(needs), B
        self.gates = gates
        wc, wu = cfg ** 2, (cfg - 1) ** 2
        self.wbr = (wc / (wc + wu), wu / (wc + wu))
        self.cur, self.attn_q, self._attn_fp = None, attn_q, None
        z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float32)
        self.gs = {k: z(B, 2, D) for k in ("g1", "g1sq", "g2", "g2sq")}
        self.gn = torch.zeros(B, 2, device=dev, dtype=torch.float64)
        self.slab_gates = None
        if "fc1" in self.needs:
            self.Csum, self.Cn = z(B, 2, DH, DH), torch.zeros(B, 2, dtype=torch.float64)
        if "attn" in self.needs:
            self.Gsum, self.Gn = z(B, 2, NH, D, D), torch.zeros(B, 2, dtype=torch.float64)
        if "fc1diag" in self.needs:                                                 # E_b[d_j^2] per fc1 row
            self.Dsum, self.Dn = z(B, 2, DH), torch.zeros(B, 2, dtype=torch.float64)
        self.want_tok = "tok" in self.needs or "projx" in self.needs or attn_q is not None
        if self.want_tok:
            self.plan, self.meta = token_plan(l, N)
            n = len(self.meta)
            self.tz, self.td, self.tgam, self.te = z(n, D), z(n, DH), z(n, D), z(n, D)
            self.tg1, self.tg2, self.tx = z(n, D), z(n, D), z(n, D)
        h = [block.adaLN_modulation.register_forward_hook(self._ada)]
        if "attn" in self.needs or attn_q is not None:
            h.append(block.attn.qkv.register_forward_hook(self._qkv))
        if self.want_tok:
            h.append(block.norm2.register_forward_pre_hook(self._norm2))
        if "fc1" in self.needs or "fc1diag" in self.needs or self.want_tok:
            h.append(block.mlp.fc1.register_forward_hook(self._fc1))
        if "projx" in self.needs:
            h.append(block.attn.proj.register_forward_pre_hook(self._proj))
        self.hooks = h

    def _sel(self):
        s, _, br = self.cur
        return self.plan.get((s, br), []) if self.want_tok else []

    def _ada(self, mod, inp, out):
        with torch.autocast("cuda", enabled=False):
            o = out.float()
            _, b, br = self.cur
            g1, gam2, g2 = o[:, 2 * D:3 * D], o[:, 4 * D:5 * D], o[:, 5 * D:6 * D]
            self.gs["g1"][b, br] += g1.sum(0); self.gs["g1sq"][b, br] += g1.pow(2).sum(0)
            self.gs["g2"][b, br] += g2.sum(0); self.gs["g2sq"][b, br] += g2.pow(2).sum(0)
            self.gn[b, br] += o.shape[0]
            self.slab_gam2, self.slab_g1, self.slab_g2 = gam2, g1, g2

    def _qkv(self, mod, inp, out):
        with torch.autocast("cuda", enabled=False):
            x = inp[0].float()                                                     # [N, 256, D]
            n = x.shape[0]
            _, b, br = self.cur
            if "attn" in self.needs:
                qkv = out.float().reshape(n, 256, 3, NH, HD).permute(2, 0, 3, 1, 4)
                P = torch.softmax((qkv[0] @ qkv[1].transpose(-2, -1)) * (HD ** -0.5), dim=-1)   # [N, H, 256, 256]
                for h0 in range(0, NH, 4):                                          # 4 heads at a time (memory)
                    AX = P[:, h0:h0 + 4] @ x.unsqueeze(1)                           # [N, 4, 256, D]
                    AX = AX.transpose(0, 1).reshape(4, n * 256, D)
                    self.Gsum[b, br, h0:h0 + 4] += AX.transpose(1, 2) @ AX
                    del AX
                self.Gn[b, br] += n * 256
                del P, qkv
            if self.attn_q is not None:
                if self._attn_fp is None:
                    self._attn_fp = attn_weights(self.blk.attn)
                for slot, im, t in self._sel():
                    xi = x[im:im + 1]
                    self.te[slot] = (attn_fp32(*self.attn_q, xi) - attn_fp32(*self._attn_fp, xi))[0, t]

    def _norm2(self, mod, inp):
        z = inp[0]
        for slot, im, t in self._sel():
            self.tz[slot] = z[im, t].float()
            self.tgam[slot] = self.slab_gam2[im]
            self.tg1[slot], self.tg2[slot] = self.slab_g1[im], self.slab_g2[im]

    def _proj(self, mod, inp):
        x = inp[0]
        for slot, im, t in self._sel():
            self.tx[slot] = x[im, t].float()

    def _fc1(self, mod, inp, out):
        with torch.autocast("cuda", enabled=False):
            _, b, br = self.cur
            dmat = gelu_tanh_grad(out.float())                                    # [N, 256, DH]
            for slot, im, t in self._sel():
                self.td[slot] = dmat[im, t]
            if "fc1" in self.needs:
                df = dmat.reshape(-1, DH)
                self.Csum[b, br] += df.T @ df
                self.Cn[b, br] += df.shape[0]
            if "fc1diag" in self.needs:
                self.Dsum[b, br] += dmat.pow(2).reshape(-1, DH).sum(0)
                self.Dn[b, br] += dmat.shape[0] * dmat.shape[1]
            del dmat

    def remove(self):
        for h in self.hooks:
            h.remove()

    # ---------------------------------------------------------------- finalize
    def _bucket(self, key):
        """[B, D] cfg-branch-combined bucket mean of a gate statistic (fp64)."""
        m = self.gs[key].double() / self.gn[..., None]
        return self.wbr[0] * m[:, 0] + self.wbr[1] * m[:, 1]

    @torch.no_grad()
    def finalize(self, W1, W2, Wo):
        """W1 [DH, D], W2 [D, DH], Wo [D, D] FP weights on dev. Returns dict of fp32 factors on dev (+ fp64 extras)."""
        B, wb = self.B, self.wbr
        f = {"g1": self._bucket("g1"), "g1sq": self._bucket("g1sq"), "g2": self._bucket("g2"), "g2sq": self._bucket("g2sq")}
        f["gate_energy_ratio"] = {k: float((f[f"{k}sq"] / f[k].pow(2).clamp_min(1e-30)).median()) for k in ("g1", "g2")}
        W1d, W2d, Wod = W1.double(), W2.double(), Wo.double()
        if "fc1" in self.needs:
            H = torch.zeros(DH, DH, device=self.dev, dtype=torch.float64)
            Cm = torch.zeros_like(H)
            for b in range(B):
                Cb = wb[0] * self.Csum[b, 0].double() / self.Cn[b, 0] + wb[1] * self.Csum[b, 1].double() / self.Cn[b, 1]
                Mb = (W2d.T * f["g2sq"][b]) @ W2d
                H += (Mb * Cb) / B
                Cm += Cb / B
                del Cb, Mb
            f["H_row_fc1"], f["C_mean"] = H, Cm
            del self.Csum
        if "attn" in self.needs:
            G = torch.zeros(NH, D, D, device=self.dev, dtype=torch.float64)
            for b in range(B):
                G += (wb[0] * self.Gsum[b, 0].double() / self.Gn[b, 0] + wb[1] * self.Gsum[b, 1].double() / self.Gn[b, 1]) / B
            f["G_attn"] = G
            del self.Gsum
        Woh = Wod.view(D, NH, HD).permute(1, 0, 2)                                 # [H, D(out), HD]
        g1sq_mean = f["g1sq"].mean(0)                                              # sum_b p_b E_b[g1^2]
        f["H_row_V1"] = Woh.transpose(1, 2) @ (Woh * g1sq_mean[None, :, None])
        if "fc1diag" in self.needs:
            f["Cdiag_b"] = torch.stack([wb[0] * self.Dsum[b, 0].double() / self.Dn[b, 0] + wb[1] * self.Dsum[b, 1].double() / self.Dn[b, 1]
                                        for b in range(B)])                        # [B, DH] E_b[d_j^2]
        if "tok" in self.needs:
            Rb = torch.zeros(B, D, D, device=self.dev, dtype=torch.float64)
            Hb = torch.zeros(B, D, D, device=self.dev, dtype=torch.float64)    # per-sample gates: E_b[G1 A^T A G1]
            wsum = torch.zeros(B, device=self.dev, dtype=torch.float64)
            for sl, A in self.token_A(W1, W2, f):
                for j, s in enumerate(sl):
                    b, br = self.meta[s]
                    AtA = (A[j].T @ A[j]).double()
                    Rb[b] += wb[br] * AtA / TOK_PER_BUCKET_BRANCH
                    if self.gates == "sample":
                        g1t = self.tg1[s].double()
                        Hb[b] += wb[br] * (g1t[:, None] * AtA * g1t[None, :]) / TOK_PER_BUCKET_BRANCH
                    wsum[b] += wb[br] / TOK_PER_BUCKET_BRANCH
            Rb /= wsum[:, None, None]
            f["R_b"] = Rb
            f["R"] = Rb.mean(0)
            if self.gates == "sample":
                f["H_row_Wo"] = (Hb / wsum[:, None, None]).mean(0)
            else:
                g1 = f["g1"]
                f["H_row_Wo"] = sum((g1[b][:, None] * Rb[b] * g1[b][None, :]) for b in range(B)) / B
            f["H_row_V2"] = Woh.transpose(1, 2) @ f["H_row_Wo"] @ Woh
        if self.attn_q is not None:
            f["e_tok"], f["tok_bucket"] = self.te.double(), [m[0] for m in self.meta]
        return f

    @torch.no_grad()
    def token_A(self, W1, W2, f, chunk=8):
        """Yields (token slots, A [T, D, D] fp32) for the sampled tokens: A_t = I + diag(g2) W2 diag(d_t) W1 L_t, g2 the
        bucket mean E_b[g2] (gates="bucket") or the token's own sample gate (gates="sample")."""
        eye = torch.eye(D, device=self.dev, dtype=torch.float32)
        W1f, W2f = W1.float(), W2.float()
        idx = list(range(len(self.meta)))
        for c0 in range(0, len(idx), chunk):
            sl = idx[c0:c0 + chunk]
            z = self.tz[sl]
            mu = z.mean(-1, keepdim=True)
            var = (z - mu).pow(2).mean(-1, keepdim=True)
            sig = torch.sqrt(var + 1e-6)
            zh = (z - mu) / sig
            L = (eye - 1.0 / D - zh.unsqueeze(-1) * zh.unsqueeze(-2) / D) / sig.unsqueeze(-1)   # [T, D, D]
            L = (1 + self.tgam[sl]).unsqueeze(-1) * L                                       # diag(1+gamma2) L
            K = (W2f.unsqueeze(0) * self.td[sl].unsqueeze(1)) @ W1f                         # W2 diag(d) W1
            if self.gates == "sample":
                g2 = self.tg2[sl]
            else:
                g2 = torch.stack([f["g2"][self.meta[s][0]].float() for s in sl])
            yield sl, eye + g2.unsqueeze(-1) * (K @ L)


def eig_summary(H):
    """min / median / max eigenvalue (fp64), for a matrix or a stack (then over all heads: min of mins, median of
    medians, max of maxes)."""
    H = H.double()
    try:
        e = torch.linalg.eigvalsh(H)
    except RuntimeError:
        e = torch.linalg.eigvalsh(H.cpu())
    e = e.reshape(-1, e.shape[-1])
    return {"min": float(e[:, 0].min()), "median": float(e.median(-1).values.median()), "max": float(e[:, -1].max())}


def spreads(H):
    e = torch.linalg.eigvalsh(H.double())
    lmax, lmin, lmed = float(e[-1]), float(e[0]), float(e.median())
    return {"lmax_over_lmin": lmax / lmin if lmin > 0 else float("inf"), "lmax_over_lmedian": lmax / lmed,
            "min": lmin, "median": lmed, "max": lmax}


def diag_normalized(H):
    d = H.diagonal().clamp_min(1e-30).rsqrt()
    return d[:, None] * H * d[None, :]
