"""Required unit test for the carry (handoff §0). Run: python -m boa_dit.deflate.test_carry"""
import math

import torch

from boa_dit.deflate.carry import attach, make_carry
from boa_dit.model import load_dit
from boa_dit.qutils import rtn_weight


def toy():
    W = torch.tensor([[1., 2.], [3., 4.]])
    Wq = torch.tensor([[1., 2.], [3., 3.]])
    v = torch.tensor([[1.], [1.]]) / math.sqrt(2)
    c = make_carry(W, Wq, v)
    assert torch.allclose(c["delta"].float(), torch.tensor([[0.], [1 / math.sqrt(2)]]), atol=1e-3)
    lin = torch.nn.Linear(2, 2, bias=False)
    lin.weight.data = Wq.clone()
    attach(lin, v, (W - Wq) @ v)                    # fp32 carry for the exact toy check
    y = lin(torch.tensor([[10., 10.]]))
    assert torch.allclose(y, torch.tensor([[30., 70.]]), atol=1e-5), y           # along v: exact, = W x
    y = lin(torch.tensor([[10., -10.]]))
    assert torch.allclose(y, torch.tensor([[-10., 0.]]), atol=1e-5), y           # orthogonal: uncorrected
    assert torch.allclose(torch.tensor([[-10., -10.]]) - y, torch.tensor([[0., -10.]]))
    print("toy: ok")


def real():
    m = load_dit("cpu")
    lin = m.blocks[13].attn.qkv
    W, b = lin.weight.data.clone(), lin.bias.data.clone()
    torch.manual_seed(0)
    V, _ = torch.linalg.qr(torch.randn(W.shape[1], 8, dtype=torch.float64))
    V = V.float()
    x = torch.randn(64, W.shape[1])
    # W_q = W: output equals FP
    l1 = torch.nn.Linear(W.shape[1], W.shape[0]); l1.weight.data, l1.bias.data = W.clone(), b.clone()
    attach(l1, V, (W - W) @ V)
    e1 = float((l1(x) - (x @ W.T + b)).abs().max())
    assert e1 < 1e-5, e1
    # W_q = RTN(W): y = W_q P x + W (I - P) x + b
    Wq = rtn_weight(W, 4)
    l2 = torch.nn.Linear(W.shape[1], W.shape[0]); l2.weight.data, l2.bias.data = Wq.clone(), b.clone()
    attach(l2, V, (W - Wq) @ V)
    P = torch.eye(W.shape[1]) - V @ V.T
    ref = x @ P.T @ Wq.T + x @ (torch.eye(W.shape[1]) - P).T @ W.T + b
    e2 = float((l2(x) - ref).abs().max())
    assert e2 < 1e-4, e2
    print(f"real layer (b13 attn.qkv, r = 8): W_q = W max err {e1:.1e}; W_q = RTN4 max err {e2:.1e}: ok")


if __name__ == "__main__":
    toy()
    real()
