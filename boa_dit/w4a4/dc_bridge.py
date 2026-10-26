"""Run deepcompressor's SVDQuant-r32 PixArt model through OUR screen and image eval (rule.md, Methods).
Must run with /root/dcvenv/bin/python from /root/ext/deepcompressor/examples/diffusion (configs are relative).
  python /home/jl_fs/boa-dit/boa_dit/w4a4/dc_bridge.py --load /root/ext/dc_runs/svdq_r32_model --device cuda:0
Rebuilds the quantized pipeline (DiffusionModelStruct + ptq with load_dirpath: no re-quantization), screens the
unquantized fp16 pipeline (dtype floor, arm 'fp16') and the quantized one (arm 'svdq_dc_r32', per-block table vs the
pipeline's own unquantized fp16 blocks), then MJHQ-256 images + PSNR/LPIPS vs our bf16 images.
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, "/home/jl_fs/boa-dit")
import torch  # noqa: E402

from boa_dit.common import RESULTS, Run, atomic_write_json  # noqa: E402

ARM = "svdq_dc_r32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq
    argv = ["configs/model/pixart-sigma.yaml", "configs/svdquant/int4.yaml", "--pipeline-path", "/home/jl_fs/pixart_sigma_1024",
            "--output-root", "/root/ext/dc_runs/bridge_out", "--cache-root", "/root/ext/dc_runs/cache"]
    config, *_ = DiffusionPtqRunConfig.get_parser().parse_known_args(argv)
    from boa_dit.w4a4.screen import run_screen
    from boa_dit.w4a4 import gen
    with Run("w4a4", f"dc_bridge_{ARM}", a.device, vars(a)) as run:
        pipeline = config.pipeline.build()
        pipeline.set_progress_bar_config(disable=True)
        tr = pipeline.transformer
        fp_blocks = [copy.deepcopy(b) for b in tr.transformer_blocks]
        with torch.no_grad():
            r0 = run_screen(tr, None, "fp16", a.device)
        atomic_write_json(os.path.join(RESULTS, "w4a4", "screen", "fp16.json"), {**r0, "note": "unquantized fp16 pipeline (deepcompressor build)"})
        print("fp16 floor proxy", r0["rel_mse_mean"], flush=True)
        model = DiffusionModelStruct.construct(pipeline)
        ptq(model, config.quant, cache=config.cache, load_dirpath=a.load, save_dirpath="/root/ext/dc_runs/bridge_tmp",
            copy_on_save=False, save_model=False)
        with torch.no_grad():
            r = run_screen(pipeline.transformer, fp_blocks, ARM, a.device)
        r["note"] = "deepcompressor SVDQuant INT4 r32 (its own recipe, 128 calibration prompts, fp16), our harness"
        atomic_write_json(os.path.join(RESULTS, "w4a4", "screen", f"{ARM}.json"), r)
        print("svdq proxy", r["rel_mse_mean"], flush=True)
        gen.generate(ARM, a.n, a.device, pipe=pipeline)
        psnr, lp = gen.metrics(ARM, a.n, a.device)
        res = {"arm": ARM, "n": a.n, "psnr_mean": sum(psnr) / len(psnr), "lpips_mean": sum(lp) / len(lp),
               "lpips_net": "vgg (lpips 0.1.4)", "psnr_per_image": psnr, "lpips_per_image": lp, **run.record()}
    atomic_write_json(os.path.join(RESULTS, "w4a4", "eval", f"img{a.n}_{ARM}.json"), res)
    print({k: v for k, v in res.items() if not k.endswith("per_image") and k != "config"}, flush=True)


if __name__ == "__main__":
    main()
