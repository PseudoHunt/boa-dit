"""PixArt-Σ §3: per-block attention core at N = 4096 (64 x 64 grid), fp16, batch 2, synthetic inputs.

The shared-map form is out = floor * mean-of-V term + residual kernel applied to V, with ONE 2-D kernel per head shared
by that head's 72 channels:
  sdpa            : fused F.scaled_dot_product_attention (baseline), q, k, v [2, 16, 4096, 72]
  conv_loop       : mean of V + per head F.conv2d(V_h as [B*72, 1, 64, 64], [1, 1, k, k], padding k//2), 16 calls
  conv_grouped    : mean of V + one F.conv2d on [B*72, 16, 64, 64] with groups=16, weight [16, 1, k, k]
                    (the 16 single-channel convolutions in one call)
  fft             : mean of V + rfft2 of V channels and kernels (size 64 + k - 1), multiply, irfft2
k in {9, 13, 21, 31}. The median of 50 timed iterations, CUDA events on the pinned device. -> results/attn_share/pixart/compute.json
"""
import argparse
import os

import torch
import torch.nn.functional as F

from boa_dit.attn_share.pixart.common import G, OUT
from boa_dit.common import Run, atomic_write_json

H, HD, B = 16, 72, 2
N = G * G
KS = (9, 13, 21, 31)


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(torch.tensor(ts).median())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, HD, device=dev, dtype=torch.half) for _ in range(3))
    floor = torch.rand(H, device=dev, dtype=torch.half) / N
    res = {"N": N, "batch": B, "heads": H, "head_dim": HD, "dtype": "fp16", "iters": 50, "per_k": {}}
    with Run("attn_share", "pixart_compute", dev, vars(a)) as run:
        res["sdpa_ms"] = timeit(lambda: F.scaled_dot_product_attention(q, k, v))
        vg = v.permute(0, 1, 3, 2).reshape(B, H, HD, G, G)                        # [B, H, 72, 64, 64]

        def vmean():
            return floor[None, :, None] * v.float().sum(2)                         # [B, H, 72]
        res["mean_of_v_ms"] = timeit(vmean)
        for kk in KS:
            ker = torch.rand(H, 1, kk, kk, device=dev, dtype=torch.half)
            x_loop = [vg[:, h].reshape(B * HD, 1, G, G).contiguous() for h in range(H)]
            x_grp = vg.permute(0, 2, 1, 3, 4).reshape(B * HD, H, G, G).contiguous()
            S = G + kk - 1
            kf = torch.fft.rfft2(ker[:, 0].float(), s=(S, S))                       # [H, S, S/2+1]

            def conv_loop():
                m = vmean()
                return m, [F.conv2d(x_loop[h], ker[h:h + 1], padding=kk // 2) for h in range(H)]

            def conv_grouped():
                return vmean(), F.conv2d(x_grp, ker, padding=kk // 2, groups=H)

            def fft():
                y = torch.fft.irfft2(torch.fft.rfft2(vg.float(), s=(S, S)) * kf[None, :, None], s=(S, S))
                return vmean(), y[..., kk // 2:kk // 2 + G, kk // 2:kk // 2 + G]

            # the three paths compute the same thing (conv2d is cross-correlation, the FFT product is a convolution:
            # with a symmetric check kernel they agree)
            sym = (ker + ker.flip(-1, -2)) / 2
            ker_save, ker = ker, sym
            kf_save, kf = kf, torch.fft.rfft2(sym[:, 0].float(), s=(S, S))
            a1 = torch.stack([o for o in conv_loop()[1]], 1).reshape(B, HD, H, G, G).float()
            a2 = conv_grouped()[1].reshape(B, HD, H, G, G).float()
            a3 = fft()[1].permute(0, 2, 1, 3, 4).float()
            agree = {"loop_vs_grouped": float((a1 - a2).abs().max() / a2.abs().max()),
                     "fft_vs_grouped": float((a3 - a2).abs().max() / a2.abs().max())}
            ker, kf = ker_save, kf_save
            r = {"conv_loop_ms": timeit(conv_loop), "conv_grouped_ms": timeit(conv_grouped), "fft_ms": timeit(fft),
                 "check_rel_diff": agree, "macs_per_token_per_channel": kk * kk}
            r.update({f"ratio_{p}_to_sdpa": r[f"{p}_ms"] / res["sdpa_ms"] for p in ("conv_loop", "conv_grouped", "fft")})
            res["per_k"][str(kk)] = r
            print(kk, {x: y for x, y in r.items() if x != "check_rel_diff"}, agree, flush=True)
        res["dense_attention_macs_per_token_per_channel"] = N
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "compute.json"), res)
    print("sdpa ms", res["sdpa_ms"], "mean_of_v ms", res["mean_of_v_ms"])


if __name__ == "__main__":
    main()
