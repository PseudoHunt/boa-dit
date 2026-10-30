## Answers to the six questions (written after the queue finished; pre-registered rules decided)
1. **Does feedback improve the bridge sampler at FP K4? No — it is harmful.** ACB / CB = 1.690 [1.335, 2.250] on 256 fresh images, 59/256 better,
   ΔLPIPS +0.012 [+0.008, +0.016] (pre-registered reading "harmful"). The bridge itself replicated: CB / C 0.257 [0.170, 0.384], 234/256.
2. **Does that survive the matched-range stateless control? The harm does; there is no gain to survive.** ACB / GB = 1.677 [1.381, 2.096]. The matched
   range alone changes nothing (G / C 1.007 [0.955, 1.067]; GB / CB 1.007), so the harm is attributable to the carried feedback state. Without the bridge,
   feedback is inconclusive (AC / G 1.038 [0.920, 1.174]); the matched interaction is negative (−0.0024 [−0.0049, −0.0000]): feedback hurts more with the
   bridge. Diagnostics (8 dev images, own trajectories) are consistent: feedback raises held-error energy (1171 vs 1090; 1217 vs 1144 with bridge) and lowers
   the bridge estimate's alignment with the true hold residual (cos 0.665 vs 0.686–0.689). This is descriptive, not a causal decomposition.
3. **Does the bridge work with W8 anchors? Yes (first 128 images).** CB.w8 / C.w8 = 0.279 [0.192, 0.412], 107/128; with the bridge, W8 anchors match FP
   anchors (CB.w8 / CB 1.010 [0.770, 1.376]). Feedback is harmful here too (ACB.w8 / CB.w8 1.848 [1.279, 2.752]; vs GB.w8 1.702). H8 run at every step is
   still more accurate than CB.w8 (CB.w8 / H8_50 2.675); W8 anchors measure cheap-minus-W8, not FP error.
4. **Can seven anchors match thirteen-anchor ordinary reuse within the margins? Unresolved.** CB.k8 / C(K4) 0.925 [0.698, 1.218] (upper bound misses 1.05;
   ΔLPIPS upper +0.016 misses 0.005); W8: CB.w8.k8 / C.w8 0.986 [0.749, 1.316], unresolved. Against the K4 bridge, K8 is clearly worse (CB.k8 / CB 3.64).
   The bridge still helps at K8 (CB.k8 / C.k8 0.443 [0.349, 0.561]). Seven anchors are a call reduction (50 + 7 vs 50 + 13), not a measured speedup.
5. **Does the existing local refit improve bridge accuracy without training? Not established.** LCB / CB 0.715 [0.529, 1.047] (inconclusive; LPIPS slightly
   better, −0.004 [−0.008, −0.000]); LACB / ACB 1.182 [0.920, 1.505] (inconclusive). LACB / LCB 2.39 changes checkpoint and feedback together.
6. **Generation quality: not measured** — the full three-arm FID-2k needed ≈ 3.15 A30-h and did not fit (2.81 spent + 3.15 + 0.35 reserve > 6.0); skipped
   whole, no substitute. **Efficiency: measured only in fake-quant simulation, where no anchored arm beats FP50** (batch 32: FP50 19.3 s, C / CB 57.5 s,
   AC / ACB 89.3 s per trajectory; batch 1: 1.4 / 5.0 / 7.5 s). The bridge itself costs ≈ 0 (≤ 0.15 s per trajectory); feedback adds ≈ 32 s at batch 32.
   Dequantized simulation is not packed INT4/INT8 evidence; the cost condition must be re-measured with real kernels. SANA timing not run (no validated
   real-kernel command installed).

**Overall:** the bridge is a robust accuracy improvement for output-anchored sampling with FP or W8 anchors at K4 and K8. Adding DCT-4 activation feedback
made results worse, so the three components do **not** form a better combined method under these conditions.
