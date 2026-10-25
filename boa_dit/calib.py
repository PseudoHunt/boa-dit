"""Stage 2 / 5.1: calibration capture and per-bucket Gram accumulation.

Cache layout: rows ordered (step, branch, image): row r = step * 2N + branch * N + m, N = #calibration images.
X: [S*2N, 256, 1152] fp16 block inputs (block-0 input = patch embed + pos embed); C: [S*2N, 1152] fp32
conditioning vector c = t_emb + y_emb (pre-SiLU; each block's adaLN_modulation applies its own SiLU).
"""
import torch

from boa_dit.model import LINEARS, block_linear, labels_for_seeds, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample


@torch.no_grad()
def capture_block0_inputs(model, diffusion, seeds, cfg, dev, chunk=None):
    """Run the FP sampler on `seeds` and cache block-0 inputs and c for every (step, branch, image)."""
    S, N = diffusion.num_timesteps, len(seeds)
    X = torch.empty(S * 2 * N, 256, 1152, dtype=torch.float16, device=dev)
    C = torch.empty(S * 2 * N, 1152, dtype=torch.float32, device=dev)

    @torch.autocast("cuda", enabled=False)
    def pre(mod, args):
        x, c = args
        r0 = CTX.step * 2 * N
        assert torch.isfinite(x).all()
        X[r0:r0 + 2 * N] = x.to(torch.float16)
        C[r0:r0 + 2 * N] = c.float()

    h = model.blocks[0].register_forward_pre_hook(pre)
    try:
        ddim_sample(model, diffusion, noise_for_seeds(seeds, dev), labels_for_seeds(seeds), cfg)
    finally:
        h.remove()
    assert not X.requires_grad and not C.requires_grad
    return X, C


def row_meta(S, N):
    """step and branch (0 cond, 1 uncond) for each cached row."""
    r = torch.arange(S * 2 * N)
    return r // (2 * N), (r % (2 * N)) // N


class GramAccumulator:
    """Forward pre-hooks on the 4 block linears; sums x x^T per (bucket, branch) in fp32 on GPU."""

    def __init__(self, block, B, dev):
        self.B, self.dev = B, dev
        self.G = {}       # (layer, bucket, branch) -> [d, d] fp32 sum
        self.n = {}       # (layer, bucket, branch) -> token count
        self.cur = None   # (bucket, branch) of the rows currently being run
        self.hooks = [block_linear(block, l).register_forward_pre_hook(self._hook(l)) for l in LINEARS]

    def _hook(self, lname):
        @torch.autocast("cuda", enabled=False)  # fp32 Grams even inside the fp16 autocast forward
        def h(mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            key = (lname, *self.cur)
            g = x.T @ x
            assert g.dtype == torch.float32
            if key in self.G:
                self.G[key] += g
                self.n[key] += x.shape[0]
            else:
                self.G[key], self.n[key] = g, x.shape[0]
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()

    def means(self):
        """(layer, bucket, branch) -> mean Gram E[x x^T] on CPU."""
        return {k: (v / self.n[k]).cpu() for k, v in self.G.items()}


@torch.no_grad()
def run_block(block, X, C, S, N, B, acc=None, write_back=False, rows_per_call=None):
    """Run `block` over all cached rows, one (step, branch) slab at a time so every call has a single
    (bucket, branch) for the Gram hooks. Optionally overwrite X with the block outputs."""
    steps_per_bucket = S // B
    for s in range(S):
        for br in range(2):
            r0 = s * 2 * N + br * N
            if acc is not None:
                acc.cur = (s // steps_per_bucket, br)
            with torch.autocast("cuda", dtype=torch.float16):
                y = block(X[r0:r0 + N].float(), C[r0:r0 + N])
            if write_back:
                assert torch.isfinite(y).all(), "non-finite block output"
                X[r0:r0 + N] = y.to(torch.float16)
