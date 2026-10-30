# Central seed / prompt registry (all branches, compiled 2026-10-29 at parking)

Built by sweeping every branch (`git grep` over .py/.md/.sh for `range(a, b)` and literal seeds) plus the protocols. A DiT "seed" s sets the
noise (`noise_for_seeds`, torch.Generator(s)) and the class label (`labels_for_seeds`: np.random.default_rng(s).integers(0, 1000)).
PixArt uses prompt indices; the seed is the prompt index.

## DiT sample seeds
| range | use (branch) | status |
|---|---|---|
| 0–1999 | FID-2k set 1: stage FIDs, temporal D3, sampler_corr base/Q-Drift (several branches) | evaluated |
| 2000–3999 | tsens g3 sampling; reserved as FID set 2 (D5, never run) | used |
| 1000–1031 | quantizer calibration (`quantize.CALIB_SEEDS`; heig builder) | calibration |
| 1000–1047, 1100–1163 | temporal / static-A8 calibration and checks (temporal-oracles) | calibration |
| 2000–2015 | temporal diagnostics | used |
| 3000–3199 | eval sets s3000/s3100 (`fp_eval_*_s3000/s3100`) | evaluated |
| 4000–4031, 4100–4227, 4300–4427 | weights-joint design / reserved / eval2 | used |
| 5000–5023 | weights-scale fit / validation | used |
| 6000–6095 | sampler_corr Q-Drift calibration (6000–6063) / held-out (6064–6095) | calibration / used |
| 7000–7231 | observability traces (7000–7063 cal, 7100–7163 eval, 7200–7231 design) | inspected |
| 8000–8527 | follow-up E1 (8000–8163) and E3 (8200–8527) | used |
| 20000–20127 | E3C paired evaluation; DNS paired diagnostics (shared, both evaluated) | **evaluated** |
| 30000–31999 | E3C FID-2k; DNS / C-DNS FID-2k (shared) | **evaluated** |
| 40000–42999 | DNS calibration FP samples (labels 3 per class) | calibration |
| 43000–43031 | DNS design / stability | used |
| 44000–44063 / 44100–44163 / 44200–44231 | C-DNS fit / held-out / (selection, unused) | used |
| generator seeds 45000+i | DNS forward-noise per step i (not sample seeds) | — |
| generator seeds 9000+100·t+s, 100003·site+t | temporal noise directions / dither (not sample seeds) | — |
| **60000–60031** | **DiT joint pilot: activation-range calibration** | **reserved (verified unused, 2026-10-29)** |
| **60100–60163** | **DiT joint pilot: training trajectories** | **reserved** |
| **60200–60231** | **DiT joint pilot: development / checkpoint selection** | **reserved** |
| **61000–61127** | **DiT joint pilot: sealed pilot evaluation (128); evaluated after first use** | **reserved, sealed** |
| 72000–72063 / 72100–72131 / 72200–72327 / 74000–75999 | DNS response pilot (r1): fit / dev / confirmation / FID-2k | **evaluated** |
| 82000–82327 / 83000–83327 | analytic-carry E1 / E2: fit, dev, confirmation (see analytic-carry-pilot) | **evaluated** |
| 90000–90063 / 90100–90131 | quantizer-observer pilot (qo1): fit / development (traced, opened 2026-10-30) | **used** |
| 90200–90263 / 90300–90363 | qo1 observer confirmation / live confirmation | **untouched (registered, unopened; stopped by gate)** |
| 92000–92031 / 92100–92163 | weight-anchor pilot (wa1): live development / live confirmation | **evaluated (2026-10-30)** |
| 93000–93063 | weight-anchor controls (wac1): fresh confirmation | **evaluated (2026-10-30)** |
| 94000–94031 / 94100–94163 | W8-anchor pilot (wa8): development / confirmation | **evaluated (2026-10-30)** |
| 95100–95163 | shared-anchor pilot (ws1): fresh confirmation (dev reuses 94000–94031) | **evaluated (2026-10-30)** |
| 96100–96163 | activation-anchor pilot (aa1): fresh confirmation (dev reuses 60200–60215; G1 bitwise test on 61000–61031) | **evaluated (2026-10-30)** |
| 97100–97163 | anchor-drift pilot (ad1): fresh confirmation (ler1 reserved it but stopped at E1 without opening it); dev 60208–60223 | **reserved (2026-10-30)** |
| 98100–98163 | anchor-bridge pilot (br1): fresh confirmation (dev 60208–60223; bitwise test 60200–60215) | **evaluated (2026-10-30)** |
| 110000–110255 | overnight abn1 (activation-bridge-overnight): confirmation cohort (Stages 2–4 use 110000–110127) | **reserved (2026-10-30)** |
| 112000–113999 | overnight abn1: FID-2k cohort (labels index // 2, two per class) | **reserved (2026-10-30)** |

## PixArt prompt sets (indices into the named list; seed = index)
| set | indices | use | status |
|---|---|---|---|
| `calib` (deepcompressor qdiff, first 32) | 0–31 | W4A4 calibration | calibration |
| `design` (qdiff 32–63) | 0–23 | follow-up E2 probes/screen | used |
| `design` | 24–31 | PixArt refit design (8 prompts; many selections made) | **heavily used for selection** |
| `design` | 0–31 | Part 3 pilot training states (pilot never ran) | registered |
| `screen` (MJHQ, 64) | 0–63 | early W4A4 screen | used |
| `mjhq1k` | 0–63 | MJHQ-64 (SOTA-table set); activation screen set | evaluated |
| `mjhq1k` | 0–255 | MJHQ-256 base validation | evaluated |
| `mjhq1k` | 64–95 | D2 second set | evaluated |
| `mjhq1k` | 256–287 | follow-up E2 confirmation; registered (unrun) as Part 3 pilot evaluation | evaluated |
| `mjhq1k` | 288–351 | PixArt Part 2 confirmation | **untouched (unopened)** |
| `mjhq1k` | 400–431 | PixArt fresh development set | **untouched (registered, unopened)** |
| `mjhq1k` | 432–463 | reserve dev set for the conditional trajectory-anchored target | **untouched (reserved)** |
| `mjhq1k` | 600–663 | PixArt refit training states | training |
| `mjhq1k` | 352–399, 464–599, 664–999 | — | free |

Rule (all protocols): a range is untouched only if this registry says so. A range does not count as untouched just because one branch never used it.
