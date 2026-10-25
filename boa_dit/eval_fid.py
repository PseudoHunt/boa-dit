"""pytorch-fid against the ADM reference batch (VIRTUAL_imagenet256_labeled.npz).

  --ref        : compute and cache reference Inception stats once (results/stage0/ref_stats.npz)
  --ckpt X     : generate N images with quantized model state X and score them
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json

REF_STATS = os.path.join(RESULTS, "stage0", "ref_stats.npz")


@torch.no_grad()
def inception_stats(images_uint8_nhwc, device, bs=200):
    """images: iterable/array of NHWC uint8. Returns mu, sigma of pool3 features (pytorch-fid, 299x299)."""
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    feats = []
    for i in range(0, len(images_uint8_nhwc), bs):
        x = torch.as_tensor(np.asarray(images_uint8_nhwc[i:i + bs])).to(device).permute(0, 3, 1, 2).float() / 255
        f = net(x)[0]  # InceptionV3 resizes to 299 internally (resize_input=True)
        feats.append(f.squeeze(-1).squeeze(-1).double().cpu())
    f = torch.cat(feats).numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def fid_from_stats(mu1, s1, mu2, s2):
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def compute_ref(device):
    arr = np.load(os.path.join(CKPT_DIR, "VIRTUAL_imagenet256_labeled.npz"))["arr_0"]
    assert arr.dtype == np.uint8 and arr.shape[-1] == 3, arr.shape
    mu, sigma = inception_stats(arr, device)
    tmp = REF_STATS + ".tmp.npz"
    np.savez(tmp, mu=mu, sigma=sigma, n=len(arr))
    os.replace(tmp, REF_STATS)
    return len(arr)


@torch.no_grad()
def generate_and_score(name, n, device, bs=32, stage="stage2"):
    """FID-n of model `name` ('fp' or ckpt/ckpt_<name>/model.pt). Seeds 0..n-1, cfg 1.5, DDIM-50."""
    from boa_dit.model import decode, load_dit, load_vae, make_diffusion
    from boa_dit.sample import sample_seeds
    model, diffusion = load_dit(device), make_diffusion(50)
    cdir = os.path.join(CKPT_DIR, f"ckpt_{name}")
    os.makedirs(cdir, exist_ok=True)
    if name != "fp":
        model.load_state_dict(torch.load(os.path.join(cdir, "model.pt")))
    part = os.path.join(cdir, f"fid{n}_latents.partial")
    x = sample_seeds(model, diffusion, list(range(n)), 1.5, bs, partial=part, resume=True)
    del model
    torch.cuda.empty_cache()
    vae = load_vae(device)
    imgs = decode(vae, x.to(device)).permute(0, 2, 3, 1).numpy()       # NHWC uint8
    del vae
    mu, sigma = inception_stats(imgs, device)
    ref = np.load(REF_STATS)
    os.remove(part)
    return fid_from_stats(mu, sigma, ref["mu"], ref["sigma"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--name", default=None, help="'fp' or a quantized run name")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--stage", default="stage2")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.ref:
        with Run("stage0", "fid_ref_stats", args.device, vars(args)) as run:
            n = compute_ref(args.device)
            rec = {"n_ref": n, **run.record()}
        atomic_write_json(os.path.join(RESULTS, "stage0", "ref_stats.json"), rec)
        print(rec)
    if args.name:
        with Run(args.stage, f"fid{args.n}_{args.name}", args.device, vars(args)) as run:
            fid = generate_and_score(args.name, args.n, args.device, stage=args.stage)
            rec = {"name": args.name, "n": args.n, "fid": fid, **run.record()}
        atomic_write_json(os.path.join(RESULTS, args.stage, "eval", f"fid{args.n}_{args.name}.json"), rec)
        print(rec)


if __name__ == "__main__":
    main()
