"""Prints the dated attention-sharing close entry for REPORT.md from the verdict/result JSONs (append with >>)."""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(ROOT, "results", "attn_share")
j = lambda n: json.load(open(os.path.join(O, n)))
v1, v2, v3, c, cv = j("gate1_verdict.json"), j("gate2_verdict.json"), j("gate3_verdict.json"), j("compute.json"), j("convpath.json")
d = v2["delta_fid"]
gh = sum(float(r["gpu_hours"]) for r in csv.DictReader(open(os.path.join(ROOT, "results", "gpu_hours.csv"))) if r["stage"] == "attn_share")
t = c["measured"]["timings"]
f4 = cv["bench"]["per_N"]["4096"]["ms_block_attention_sublayer"]
print(f"""
## 2026-10-25 — CLOSE: sample-independent attention (Gate 3 STOP)
- Gate 1 {v1['verdict']} (s_max {v1['s_max']}). Gate 2 {v2['verdict']}: ΔFID-2k vs FP {v2['fp_fid2k']:.2f}: k5 {d['5']:+.2f}, k10 {d['10']:+.2f},
  k15 {d['15']:+.2f}, k20 {d['20']:+.2f}; k_pass {v2['k_pass']}; uniform-map control {j('eval/gate2_k10-uniform.json')['fid2k'] - v2['fp_fid2k']:+.2f}.
  Label accuracy is flat across the six arms (every CI for Δ vs FP includes 0); the control changes it by
  {100 * (j('eval/label_acc_k10-uniform.json')['label_acc_512'] - j('eval/gate2_fp.json')['label_acc_512']):.1f} points.
- Gate 3 **{v3['verdict']}** (k = {v3['k_pass']}): no variant met both conditions. V3 kernel {v3['V3']['delta_vs_V1']:+.2f} at {100 * v3['V3']['frac_of_V1']:.2f}% of V1
  (bar +0.5); V2 {v3['V2']['delta_vs_V1']:+.2f} at {100 * v3['V2']['frac_of_V1']:.0f}%; V4 {v3['V4']['delta_vs_V1']:+.1f}; V5 {v3['V5']['delta_vs_V1']:+.2f}. V2 + V5 (exactly 10%) not run (gate already failed).
- Compute: a fully shared forward at N = 256 is {100 * (t['shared map, V-only projection']['median_ms'] / t['baseline (fused SDPA)']['median_ms'] - 1):+.1f}% latency in plain PyTorch; per block at N = 4096,
  one dense map takes {f4['V1 dense stored map']:.1f} ms vs {f4['full attention (fused SDPA)']:.1f} ms for fused attention; the V3 cuDNN depthwise conv takes {f4['V3 conv (31x31 depthwise)']:.0f} ms.
  V3 kernel support is full (31×31) in {100 * j('kernel_support.json')['per_branch']['cond']['frac_full_31']:.0f}% of cases. No speed claim; next step PixArt-Σ 1024 px (see SUMMARY.md).
- Bug caught: the first latency runs recorded CUDA events on cuda:0 while the work ran on cuda:1, so those numbers
  (pushed by the heartbeat) were invalid. They were rerun with the device pinned; only the rerun is in compute.json and convpath.json.
- Unlogged: about 4 GPU-minutes of smoke tests. GPU-hours: {gh:.2f} of 5.0.
- `results/attn_share/SUMMARY.md` written. Heavy processes stopped; nothing is running. The instance can be shut down
  once this commit is on GitHub. Checkpoints and maps under `ckpt/` and `results/attn_share/maps/` are not in git.
""")
