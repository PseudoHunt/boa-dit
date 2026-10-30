## Interpretation (written 2026-10-30 after the development selection; observed facts, hypotheses and unrun parts kept separate)

**Observed facts**
- *Measurement.* The instrumented trace is bit-exact with deployed W4A8 + DCT-4 sampling, and the guarded-feedback identity closes at float64 roundoff (≈ 4e-33 relative).
- *Headroom (A1).* With the E_L u100 carry, removing A8 lowers endpoint MSE by 25.4% [12.9, 36.4] on dev and 37.9% [22.6, 50.9] on fit. The dev point is above the 20% spending threshold ("proceed").
  - The cross term 2⟨d_W, d_A⟩ is negative (dev −0.028 [−0.052, −0.013] of FP energy), so the weight and activation endpoint errors partly cancel. The gain is not an additive share of the total.
  - With the original carry (dev, carry-transfer check), the gain is 36.6% [28.3, 45.7]. Its cross term is smaller (−0.007), and its WA MSE is higher (0.095 vs 0.085).
  - So the refitted carry lowers WA error mainly by interacting with the activation error (more cancellation); its A16 error is similar (0.063 vs 0.061).
- *Site accounting* (consumed-activation error q − x; fit batch; not endpoint attribution):
  - discarded non-retained modes (I−P)r carry 68% of the energy;
  - **initial input clipping carries 27%**, despite a rate of only 3.8e-5. The old counter, which recorded only the second clip, reported 0 here;
  - the state injection p and the removed next state p_next are about 13% each, with a −0.215 cross term;
  - state guard, second clip, rounding clamp, fp32 rounding and fp16 storage are ≈ 0.
- *Aligned local errors (A2):*
  - e_W is slowly varying (lag-1 cos 0.96, lag-5 0.72, c = 0.22);
  - e_A|W is much less smooth (lag-1 0.60, c = 0.27–0.32);
  - both are far above the iid coherence 0.02. Period-2 energy is ≈ 0 (0.04–0.7%).
  - The 4×4 spatial low band, the observer target, holds only 2–4% of local error energy.
- *Observers (B/C, development):*
  - Ridge regression from the output/history baseline H explains essentially none of the held-out low-band error. The ratio vs zero prediction is 1.007 (e_W), 1.042 (e_A|W) and 1.025 (e_total), slightly worse than predicting zero; it is about 0.998 vs the per-step training mean.
  - Adding the carry bottlenecks (C) changes this by ≤ 0.03%. Adding activation signals changes it by ≤ 0.13% on e_A|W (H+R+E+S −0.13% [−0.19, −0.08]) and ≤ 0.05% on e_total.
  - These changes are statistically resolvable but practically nil. They are far below the 1% development-value threshold and the 10% gate.
  - CV chose λ = 100 (the grid maximum) for every deployable model, i.e. maximal shrinkage.
  - The non-deployable oracle (previous true error) removes 28% of e_total, so the error is temporally predictable from itself but not from these inexpensive signals.
  - The selection was H, so the gate cannot pass.

**Null and its scope.** No useful sample-specific information about the remaining low-band error was found in compact carry bottlenecks (2×2 spatial DCT of z = xV) or quantizer summaries (8-channel projections and energies/rates), under linear ridge with ≤ 32 PCA components per family on 64 training images.
This bounds these features, this regression capacity, this target band, this model and this budget. It does not show that weight error is unobservable, or that dedicated residual probes, richer decoders or other targets cannot work.

**Hypotheses (untested):**
- Input clipping is rare but energetic. A clip-aware treatment, such as range or spare-code handling, or clipping-error feedback, might matter more than the feedback state. This is a candidate for a separately registered experiment.
- The large oracle gain against the null deployable gain suggests the persistent error lives in directions these compressions (2×2 spatial, 8 random channels) do not capture.

**Not run (by rule):**
- observer confirmation (90200–90263, unopened);
- the live correction and its confirmation (90300–90363, unopened);
- the H+C+R+E+S combination (not triggered);
- the confirmation-set headroom;
- separate conditional/unconditional diagnostics (cost);
- any scalar-schedule derivation (no signed same-actuator responses exist; see `propagation_audit.md`).
