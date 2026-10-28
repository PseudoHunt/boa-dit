"""Stage `f5` (results/e3c/PROTOCOL.md, 2026-10-28): no GPU cap (user instruction), hours are still logged and summed
across every checkout's gpu_hours.csv. The heig-16 build logs under stage `deflate` and is added explicitly.
Run: python -m boa_dit.f5"""
import csv
import glob

STAGE = "f5"
EXTRA_RUNS = {("deflate", "quant_deflate_heig_r16_w4")}


def spent(prefix=None):
    s = 0.0
    for f in glob.glob("/home/boa-dit*/results/gpu_hours.csv"):
        for r in csv.reader(open(f)):
            if r and (r[0] == STAGE or (r[0], r[1]) in EXTRA_RUNS) and (prefix is None or r[1].startswith(prefix)):
                s += float(r[-1])
    return s


if __name__ == "__main__":
    print(f"f5 spent {spent():.3f} h")
