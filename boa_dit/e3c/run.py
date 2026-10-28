"""E3 confirmation (results/e3c/PROTOCOL.md Part 1): three training seeds, C1 time- and update-matched, C2, 3x-longer runs,
the original heig recipe at rank 16 (H16) as the matched-bits comparator, M refitted on the C2 carry, FID-2k.
  train  --arm C1t|C1u|C2|C2L|C1L --seed s   one carry (E3 seed-0 carries of C1t / C2 are reused, not retrained)
  mfit                                       M for C2 s0 (state_corr.fit_M on seeds 7000-7063, endpoints)
  eval                                       paired metrics on seeds 20000-20127 for every carry present (incremental)
  fid    --arm A                             FID-2k on seeds 30000-31999, per-image Inception features saved
Run: python -m boa_dit.e3c.run <cmd> ...
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.f5 import STAGE
from boa_dit.followup import carry_opt as E3
from boa_dit.followup import state_corr as SC
from boa_dit.model import decode, labels_for_seeds, load_dit, load_quant_state, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import ddim_sample, sample_seeds

OUT = os.path.join(RESULTS, "e3c")
CK = os.path.join(CKPT_DIR, "e3c")
EVAL, FIDS, MCAL = list(range(20000, 20128)), list(range(30000, 32000)), list(range(7000, 7064))
LR, LAM, HOURS, UPD = 1e-4, 0.5, 0.3, 841
H16 = "deflate_heig_r16_w4"
H16_STATE = os.path.join(CKPT_DIR, f"ckpt_{H16}", "model.pt")
H16_CARRY = os.path.join(RESULTS, "deflate", "quant", H16, "carry.pt")
SPEC = {"C1t": ("C1", HOURS, None), "C1u": ("C1", 100.0, UPD), "C2": ("C2", HOURS, None),
        "C2L": ("C2", 3 * HOURS, None), "C1L": ("C1", 3 * HOURS, None)}
REUSED = {("C1t", 0): os.path.join(CKPT_DIR, "followup", "e3", "carry_final_C1.pt"),
          ("C2", 0): os.path.join(CKPT_DIR, "followup", "e3", "carry_final_C2.pt")}


def carry_file(arm, seed):
    return os.path.join(CK, f"carry_{arm}_s{seed}.pt")


def train(dev, arm, seed):
    os.makedirs(CK, exist_ok=True)
    dst, f = carry_file(arm, seed), os.path.join(OUT, "train", f"{arm}_s{seed}.json")
    if (arm, seed) in REUSED:
        shutil.copyfile(REUSED[(arm, seed)], dst)
        src = {"C1t": "train_C1.json", "C2": "train_C2.json"}[arm]
        lg = json.load(open(os.path.join(RESULTS, "followup", "e3", src)))
        lg.update({"reused_from": REUSED[(arm, seed)], "note": "E3 final run (same code, seed 0), reused unchanged"})
        atomic_write_json(f, lg)
        return
    obj, hours, mu = SPEC[arm]
    with Run(STAGE, f"e3c_train_{arm}_s{seed}", dev, {"arm": arm, "obj": obj, "hours": hours, "max_updates": mu, "seed": seed, "lr": LR}) as run:
        p, lg = E3.train(dev, obj, LAM if obj == "C2" else 0.0, LR, hours, f"e3c_{arm}_s{seed}", seed=seed, max_updates=mu)
        shutil.move(p, dst)
        lg.update({"carry": dst, **run.record()})
    atomic_write_json(f, lg)
    print(f"  {arm} s{seed}: {lg['updates']} updates in {lg['train_seconds']:.0f} s", flush=True)


def heig16_model(dev):
    m = load_dit(dev)
    load_quant_state(m, H16_STATE)
    m.requires_grad_(False)
    return m


def arms_present():
    """name -> (model kind, carry path, M?) for every evaluable arm."""
    A = {"C0": ("h8", E3.CARRY, False), "H16": ("h16", H16_CARRY, False)}
    for arm in SPEC:
        for s in range(3):
            if os.path.exists(carry_file(arm, s)):
                A[f"{arm}_s{s}"] = ("h8", carry_file(arm, s), False)
    if os.path.exists(os.path.join(CK, "m_C2_s0.pt")) and "C2_s0" in A:
        A["C2M_s0"] = ("h8", carry_file("C2", 0), True)                      # must follow C2_s0 (eval reuses its latents)
    return A


@torch.no_grad()
def sample(dev, kind, carry, seeds, d, cache):
    if cache.get("kind") != kind:
        cache.clear()
        cache["m"], cache["kind"] = (E3.heig_base(dev) if kind == "h8" else heig16_model(dev)), kind
    m = cache["m"]
    hs = install(m, torch.load(carry, map_location="cpu"))
    x = torch.cat([ddim_sample(m, d, noise_for_seeds(seeds[b:b + 32], dev), labels_for_seeds(seeds[b:b + 32]), E3.CFG).cpu()
                   for b in range(0, len(seeds), 32)])
    for h in hs:
        h.remove()
    return x


@torch.no_grad()
def mfit(dev):
    R = torch.load(os.path.join(CKPT_DIR, "observability", "traces_cal.pt"), map_location="cpu")
    assert list(R["seeds"]) == MCAL
    with Run(STAGE, "e3c_mfit", dev, {"seeds": [MCAL[0], MCAL[-1]], "carry": "C2_s0"}) as run:
        q = sample(dev, "h8", carry_file("C2", 0), MCAL, make_diffusion(E3.T), {})
        p = SC.fit_M(q, R["x0fp"])
        res = {"coef": {f"band{b + 1}": {"g": v["g"], "mq": v["mq"].tolist(), "mf": v["mf"].tolist()} for b, v in p.items()},
               "heig8_e1_coef": json.load(open(os.path.join(RESULTS, "followup", "e1", "screen.json")))["placements"]["final"]["arms"]["M"].get("coef"),
               **run.record()}
    atomic_torch_save(p, os.path.join(CK, "m_C2_s0.pt"))
    atomic_write_json(os.path.join(OUT, "mfit.json"), res)
    print("  M for C2 s0:", res["coef"], flush=True)


def apply_m(x):
    return SC.apply("M", torch.load(os.path.join(CK, "m_C2_s0.pt")), x).float()


@torch.no_grad()
def eval_(dev):
    import lpips
    f = os.path.join(OUT, "eval.json")
    out = json.load(open(f)) if os.path.exists(f) else {"seeds": [EVAL[0], EVAL[-1]], "arms": {}}
    lat_f = os.path.join(CK, "eval_latents.pt")
    lats = torch.load(lat_f) if os.path.exists(lat_f) else {}
    todo = {k: v for k, v in arms_present().items() if k not in out["arms"]}
    print(f"  eval: {len(todo)} new arms: {list(todo)}", flush=True)
    with Run(STAGE, f"e3c_eval_{len(out['arms'])}", dev, {"seeds": [EVAL[0], EVAL[-1]], "arms": list(todo)}) as run:
        d = make_diffusion(E3.T)
        if "fp" not in lats:
            fp = E3.fp_model(dev)
            lats["fp"] = torch.cat([ddim_sample(fp, d, noise_for_seeds(EVAL[b:b + 32], dev), labels_for_seeds(EVAL[b:b + 32]), E3.CFG).cpu()
                                    for b in range(0, len(EVAL), 32)])
            del fp
        vae = load_vae(dev)
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        imf = decode(vae, lats["fp"].to(dev))
        cache = {}
        for name, (kind, carry, m) in todo.items():
            if m:
                x = apply_m(lats[name.replace("C2M", "C2")])
            else:
                x = sample(dev, kind, carry, EVAL, d, cache)
            lats[name] = x
            im = decode(vae, x.to(dev))
            ps, lp = [], []
            for i in range(len(im)):
                a, b = imf[i:i + 1].to(dev).float(), im[i:i + 1].to(dev).float()
                ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
                lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
            out["arms"][name] = {"rel_mse": E3.rel(x, lats["fp"]).tolist(), "psnr": ps, "lpips": lp}
            print(f"  {name}: latent {np.mean(out['arms'][name]['rel_mse']):.5f} PSNR {np.mean(ps):.2f} LPIPS {np.mean(lp):.4f}", flush=True)
            atomic_torch_save(lats, lat_f)
            atomic_write_json(f, out)
        out.setdefault("runs", []).append(run.record())
    atomic_write_json(f, out)


@torch.no_grad()
def inception_feats(imgs_nhwc, dev, bs=200):
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    fs = []
    for i in range(0, len(imgs_nhwc), bs):
        x = torch.as_tensor(np.asarray(imgs_nhwc[i:i + bs])).to(dev).permute(0, 3, 1, 2).float() / 255
        fs.append(net(x)[0].squeeze(-1).squeeze(-1).double().cpu())
    return torch.cat(fs).numpy()


@torch.no_grad()
def fid(dev, arm):
    """FID-2k against the fixed ADM reference stats; latents and per-image features saved for the paired bootstrap."""
    from boa_dit.eval_fid import REF_STATS, fid_from_stats
    os.makedirs(os.path.join(CK, "fid"), exist_ok=True)
    A = arms_present()
    kind, carry, m = A[arm]
    lat_f = os.path.join(CK, "fid", f"latents_{arm}.pt")
    with Run(STAGE, f"e3c_fid2000_{arm}", dev, {"arm": arm, "seeds": [FIDS[0], FIDS[-1]]}) as run:
        if m:
            x = apply_m(torch.load(os.path.join(CK, "fid", f"latents_{arm.replace('C2M', 'C2')}.pt")))
        else:
            model = E3.heig_base(dev) if kind == "h8" else heig16_model(dev)
            hs = install(model, torch.load(carry, map_location="cpu"))
            x = sample_seeds(model, make_diffusion(E3.T), FIDS, E3.CFG, 32, partial=lat_f + ".partial", resume=True, log=False)
            for h in hs:
                h.remove()
            del model
            torch.cuda.empty_cache()
        atomic_torch_save(x, lat_f)
        vae = load_vae(dev)
        imgs = torch.cat([decode(vae, x[b:b + 100].to(dev)) for b in range(0, len(x), 100)]).permute(0, 2, 3, 1).numpy()
        del vae
        F = inception_feats(imgs, dev)
        np.save(os.path.join(CK, "fid", f"feats_{arm}.npy"), F.astype(np.float32))
        ref = np.load(REF_STATS)
        val = fid_from_stats(F.mean(0), np.cov(F, rowvar=False), ref["mu"], ref["sigma"])
        rec = {"arm": arm, "n": len(F), "fid": val, **run.record()}
    if os.path.exists(lat_f + ".partial"):
        os.remove(lat_f + ".partial")
    os.makedirs(os.path.join(OUT, "fid"), exist_ok=True)
    atomic_write_json(os.path.join(OUT, "fid", f"fid2000_{arm}.json"), rec)
    print(f"  FID-2k {arm}: {val:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "mfit", "eval", "fid"])
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
    {"train": lambda: train(dev, a.arm, a.seed), "mfit": lambda: mfit(dev), "eval": lambda: eval_(dev), "fid": lambda: fid(dev, a.arm)}[a.cmd]()


if __name__ == "__main__":
    main()
