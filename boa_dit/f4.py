"""Follow-up protocol (results/followup/PROTOCOL.md, 2026-10-28): shared 4.0 A30-hour hard cap, stage `f4`, counted across
every checkout's gpu_hours.csv. Planning ceilings (transferable, total fixed): state correction 0.75 h, spatial shaping
1.0 h, carry optimization 2.0 h, reserve 0.25 h. Run names start with e1_/e2_/e3_.
Run: python -m boa_dit.f4 [est_hours]   (exit 1 if spent + est would cross the cap)"""
import csv
import os
import sys

STAGE = "f4"
CAP = 4.0
CSVS = [f"/home/{d}/results/gpu_hours.csv" for d in ("boa-dit", "boa-dit-wj", "boa-dit-fw", "boa-dit-fa")]


def spent(prefix=None):
    """f4 hours, optionally only runs whose name starts with prefix."""
    s = 0.0
    for f in CSVS:
        if os.path.exists(f):
            for r in csv.reader(open(f)):
                if r and r[0] == STAGE and (prefix is None or r[1].startswith(prefix)):
                    s += float(r[-1])
    return s


def guard(est, label=""):
    s = spent()
    ok = s + est <= CAP
    print(f"[f4 budget] {label}: spent {s:.3f} h + est {est:.3f} h vs cap {CAP} -> {'OK' if ok else 'STOP (unrun due to cost)'}", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if guard(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0, "check") else 1)
