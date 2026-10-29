"""Shared helpers: paths, atomic writes, run records, GPU-hour logging."""
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(ROOT, "ckpt")
for _p in (os.path.join(ROOT, "external", "DiT"), os.path.join(ROOT, "external", "boa")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HOME", os.environ.get("BOA_HF_HOME", os.path.join(ROOT, "data", "hf")))   # configurable; old VM used /home/jl_fs/hf
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_savefig(fig, path):
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=110, bbox_inches="tight")
    os.replace(tmp, path)


class Run:
    """Context manager: times a run, logs GPU-hours, returns a record dict to embed in JSON."""

    def __init__(self, stage, name, device, config):
        self.stage, self.name, self.device, self.config = stage, name, str(device), config

    def __enter__(self):
        self.t0 = time.time()
        self.start = utcnow()
        return self

    def record(self):
        wall = time.time() - self.t0
        return {"stage": self.stage, "run": self.name, "device": self.device, "config": self.config,
                "git_sha": git_sha(), "start_utc": self.start, "wall_s": wall,
                "gpu_hours": wall / 3600.0}

    def __exit__(self, *exc):
        r = self.record()
        path = os.path.join(RESULTS, "gpu_hours.csv")
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.writer(f).writerow([self.stage, self.name + ("" if exc[0] is None else "(FAILED)"),
                                    self.device, self.start, utcnow(), f"{r['gpu_hours']:.4f}"])
        return False


def assert_no_grad(*tensors):
    for t in tensors:
        if isinstance(t, torch.Tensor):
            assert not t.requires_grad, "leaked requires_grad"
