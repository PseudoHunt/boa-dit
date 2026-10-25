"""VAR-d30 loading, exactly as the repo demo (external/VAR @ 78b9539): weights from /home/models/var, TF32 on as in the demo,
fp16 autocast, demo sampling settings (cfg 4, top_k 900, top_p 0.95, more_smooth False)."""
import os
import sys

import torch

VAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "VAR")
sys.path.insert(0, VAR_DIR)
W = "/home/models/var"
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
DEMO = {"cfg": 4.0, "top_k": 900, "top_p": 0.95, "more_smooth": False}


def _var_models():
    """VAR's `models` package under a private name (the DiT harness puts its own models.py first on sys.path)."""
    import importlib.util
    if "var_models" in sys.modules:
        return sys.modules["var_models"]
    spec = importlib.util.spec_from_file_location("var_models", os.path.join(VAR_DIR, "models", "__init__.py"),
                                                  submodule_search_locations=[os.path.join(VAR_DIR, "models")])
    mod = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "models" or k.startswith("models.")}
    sys.modules["var_models"] = sys.modules["models"] = mod                      # VAR uses absolute `models.` imports
    try:
        spec.loader.exec_module(mod)
    finally:
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            sys.modules["var_" + k] = sys.modules.pop(k)
        sys.modules.update(saved)                                                 # restore DiT's `models`
    return mod


def load(dev):
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    build_vae_var = _var_models().build_vae_var
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    vae, var = build_vae_var(V=4096, Cvae=32, ch=160, share_quant_resi=4, device=dev, patch_nums=PATCH_NUMS,
                             num_classes=1000, depth=30, shared_aln=False)
    vae.load_state_dict(torch.load(os.path.join(W, "vae_ch160v4096z32.pth"), map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(os.path.join(W, "var_d30.pth"), map_location="cpu"), strict=True)
    vae.eval(); var.eval()
    for p in list(vae.parameters()) + list(var.parameters()):
        p.requires_grad_(False)
    return vae, var


@torch.inference_mode()
def sample(var, labels, g_seed, dev):
    lab = torch.as_tensor(labels, device=dev)
    with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
        return var.autoregressive_infer_cfg(B=len(labels), label_B=lab, g_seed=g_seed, **DEMO)   # [B, 3, 256, 256] in [0, 1]
