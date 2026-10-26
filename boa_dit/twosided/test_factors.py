"""Factor formula check: A_t = I + diag(g2) W2 diag(GELU'(h)) W1 L_t (factors.py) against the autograd Jacobian of
z -> z + g2 * MLP(LN(z)(1 + gamma2) + beta2), block 2's FP weights, fp64 on CPU; and attn_fp32 vs the timm module.
Run: python -m boa_dit.twosided.test_factors"""
import torch

from boa_dit.model import load_dit
from boa_dit.twosided.factors import D, attn_fp32, attn_weights, gelu_tanh_grad


def main():
    torch.manual_seed(0)
    m = load_dit("cpu")
    blk = m.blocks[2]
    W1, b1 = blk.mlp.fc1.weight.data.double(), blk.mlp.fc1.bias.data.double()
    W2, b2 = blk.mlp.fc2.weight.data.double(), blk.mlp.fc2.bias.data.double()
    z = torch.randn(D, dtype=torch.float64) * 3 + 0.5
    gam, beta, g2 = (torch.randn(D, dtype=torch.float64) * s for s in (0.3, 0.1, 0.2))

    def f(z):
        u = torch.nn.functional.layer_norm(z, (D,), eps=1e-6) * (1 + gam) + beta
        return z + g2 * (torch.nn.functional.gelu(u @ W1.T + b1, approximate="tanh") @ W2.T + b2)

    J = torch.autograd.functional.jacobian(f, z)
    mu = z.mean()
    sig = torch.sqrt((z - mu).pow(2).mean() + 1e-6)
    zh = (z - mu) / sig
    L = (1 + gam)[:, None] * ((torch.eye(D, dtype=torch.float64) - 1.0 / D - torch.outer(zh, zh) / D) / sig)
    u = torch.nn.functional.layer_norm(z, (D,), eps=1e-6) * (1 + gam) + beta
    d = gelu_tanh_grad(u @ W1.T + b1)
    A = torch.eye(D, dtype=torch.float64) + g2[:, None] * ((W2 * d[None, :]) @ W1 @ L)
    rel = float((A - J).norm() / J.norm())
    print(f"A_t vs autograd Jacobian: rel err {rel:.2e}")
    assert rel < 1e-6, rel
    x = torch.randn(2, 256, D)
    ref = blk.attn(x)
    e = float((attn_fp32(*attn_weights(blk.attn), x) - ref).norm() / ref.norm())
    print(f"attn_fp32 vs timm attention (fp32, CPU): rel err {e:.2e}; max |out| {float(ref.abs().max()):.1f}")
    assert e < 1e-5, e


if __name__ == "__main__":
    main()
