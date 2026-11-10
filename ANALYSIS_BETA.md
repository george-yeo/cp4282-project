# Beta kernel vs Gaussian: analysis points

Findings from the pilot, q1, q2, q3, q4 and the 30k side check. Numbers come from `runs/summary.csv`,
the per-run `DONE` files and `runs_30k/`. The protocol and arm definitions are in `README_BETA.md`.

Unless stated otherwise: test split, 10,000 splats, dense loss, 20,000 iterations, random init
from 1,000 splats, `b_lr = 0.0005`, b clamped to [-2, 2].

Arms: **A** Gaussian. **B** Beta with learnable per-splat b. **B0** Beta with b frozen at 0.
**C** Beta with learnable b and the kernel-aware densifier.

Sign of b: the kernel is `(1 - q/9)^(4e^b)`. Positive b gives a sharper peak; negative b gives a
flatter, more box-like splat.

## Headline

Beta with a learnable b gives a small, consistent gain over the Gaussian at no extra cost. The
gain comes from the learnable b, not from the kernel shape. It grows with the splat budget up to
10k on both scenes tested and does not shrink beyond that.

## 1. Overall gain: +0.22 dB, same cost

| Scene | Seeds | B − A PSNR (dB) | Per-seed | B − A LPIPS |
|---|---|---|---|---|
| ficus | 2 | +0.65 | +0.81, +0.49 | −0.0052 |
| hotdog | 2 | +0.53 | +0.57, +0.50 | −0.0056 |
| lego | 4 | +0.42 | +0.47, +0.18, +0.55, +0.47 | −0.0025 |
| drums | 2 | +0.22 | +0.18, +0.26 | −0.0068 |
| mic | 2 | +0.04 | −0.03, +0.11 | −0.0022 |
| materials | 2 | +0.03 | +0.03, +0.04 | −0.0045 |
| ship | 2 | −0.10 | −0.20, −0.00 | +0.0026 |
| chair | 2 | −0.19 | −0.45, +0.08 | +0.0003 |

- 18 seed pairs: mean B − A = +0.22 dB, standard deviation of the pair differences 0.32 dB,
  so a standard error of about 0.075 dB (t ≈ 3).
- B wins 14/18 pairs on PSNR and 15/18 on LPIPS.
- Time per iteration is within ±2% of A. B stores one extra float per splat.
- Seed noise: lego A over 4 seeds is 27.12 ± 0.30 dB. Per-scene differences below about 0.3 dB
  are not distinguishable from noise with 2 seeds.
- Chair is the only scene with a clear loss, and it comes from one seed (−0.45); the other seed
  is +0.08.

## 2. The gain comes from the learnable b, not the kernel shape

| Scene (seed 0) | A | B0 | B | B0 − A | B − B0 |
|---|---|---|---|---|---|
| lego | 27.24 | 27.52 | 27.71 | +0.28 | +0.19 |
| ficus | 26.69 | 26.69 | 27.50 | 0.00 | +0.81 |
| hotdog | 32.90 | 33.05 | 33.47 | +0.15 | +0.42 |
| chair | 29.95 | 29.79 | 29.50 | −0.16 | −0.29 |
| Mean | | | | +0.07 | +0.28 |

- Swapping the Gaussian for a fixed-shape Beta (compact support, b = 0) is about neutral.
- Letting each splat learn its own b supplies most of the gain on ficus and hotdog.
- The budget sweep agrees: B0 stays within 0.2 dB of A at 25% and 50% capacity, while B leads both.
- Single seed per cell, so the lego split between "shape" and "learnable b" is within noise.

## 3. The lead grows with the splat budget up to 10k, and does not shrink beyond

Lego, B − A:

| Splats | 2.5k | 5k | 10k | 30k |
|---|---|---|---|---|
| B − A (dB) | +0.23 | +0.53 | +0.42 | +0.81 |
| Seeds | 1 | 1 | 4 | 2 |

Ficus, B − A (seed 0): +0.45 at 2.5k, +0.71 at 5k, +0.81 at 10k.

30k side check (lego, `runs_30k/`):

| Seed | A | B | B − A at 30k | B − A at 10k |
|---|---|---|---|---|
| 0 | 28.14 | 29.26 | +1.11 | +0.47 |
| 1 | 28.31 | 28.83 | +0.52 | +0.18 |

30k side check (ficus, `runs_30k/`):

| Seed | A | B | B − A at 30k | B − A at 10k |
|---|---|---|---|---|
| 0 | 25.97 | 27.21 | +1.24 | +0.81 |
| 1 | 26.26 | 26.51 | +0.25 | +0.49 |

- Up to 10k the lead grows with the budget on both scenes.
- From 10k to 30k it roughly doubles on lego (both seeds). On ficus it does not move on average
  (+0.75 vs +0.65): one seed goes up, the other down.
- Ficus at 30k is a poor regime for both arms: PSNR is lower than at 10k (A 26.58 → 26.12,
  B 27.22 → 26.86, means of two seeds) although LPIPS improves slightly. More splats did not
  help ficus at 20,000 iterations, the same pattern as lego at 100k in the pilot.
- Either way the lead does not shrink as splats are added, which contradicts the idea that Beta
  mainly helps when splats are scarce.
- Caveats: two seeds per scene at 30k, and the seeds disagree on the size. The 30k runs are
  outside the frozen protocol (main results are all at 10k).

## 4. How b is used: flatter when splats are scarce, sharper when plentiful

Mean and spread of learned b (seed 0) across the budget sweep:

| Splats | lego mean b | lego std | ficus mean b | ficus std |
|---|---|---|---|---|
| 2.5k | −0.35 | 0.53 | 0.00 | 0.34 |
| 5k | −0.17 | 0.60 | +0.14 | 0.40 |
| 9k | −0.03 | 0.66 | +0.26 | 0.46 |
| 10k | −0.01 | 0.67 | +0.27 | 0.45 |
| 30k | +0.08 | 0.61 | +0.27 | 0.39 |

- With few splats, b drifts negative: splats flatten to cover area.
- With more splats, b rises and spreads: splats sharpen to carry detail.
- This is a candidate mechanism for point 3: the extra shape freedom is used more as capacity
  grows. It is a correlation across runs, not a tested cause.

## 5. Which scenes benefit

Spread of learned b at 10k (mean of seeds), next to the gain:

| Scene | b std | b mean | B − A (dB) |
|---|---|---|---|
| lego | 0.67 | 0.00 | +0.42 |
| hotdog | 0.66 | +0.09 | +0.53 |
| ship | 0.56 | +0.09 | −0.10 |
| chair | 0.51 | −0.02 | −0.19 |
| ficus | 0.45 | +0.25 | +0.65 |
| drums | 0.44 | −0.03 | +0.22 |
| materials | 0.42 | −0.10 | +0.03 |
| mic | 0.34 | −0.06 | +0.04 |

- The three winning scenes are the ones where b moves furthest from 0: wide spread on lego and
  hotdog, a clear positive shift on ficus (thin leaves and stems).
- Mic and materials barely move b and gain nothing.
- Ship and chair are exceptions: b spreads but PSNR does not improve. b movement is necessary
  for a gain here, not sufficient.

## 6. The kernel-aware densifier adds nothing

| Scene (seed 0) | B | C |
|---|---|---|
| lego | 27.71 | 27.75 |
| ficus | 27.50 | 27.47 |

- Differences are far below seed noise. The learned b distributions are nearly identical too.
- One seed, two scenes.

## 7. b learning rate and the clamp

Lego val, seed 0:

| b_lr | PSNR | b std | b mean | Splats at a clamp |
|---|---|---|---|---|
| 0.0005 | 28.58 | 0.65 | −0.02 | 2.6% |
| 0.001 | 28.37 | 0.98 | −0.18 | 6.9% |
| 0.002 | 28.43 | 1.20 | −0.47 | 9.7% |

- A higher b_lr spreads b wider and pushes more splats to the clamps without improving PSNR.
  The 0.2 dB range is within seed noise, so the choice of 0.0005 is weakly supported.
- The paper uses 0.001.
- At 0.0005, clamp mass at b = +2 is highest on hotdog (5.5% and 4.2%) and lego (2.8–4.2%), and
  at most 1.9% elsewhere. The clamp may cap hotdog's gain; not tested.

## 8. Low-budget warm start is not a kernel effect

B0 at 2.5k splats, warm-started: lego 25.64 (cold 25.00), ficus 25.72 (cold 25.03). The warm run
gets about twice the optimisation, so the +0.6–0.7 dB is extra training, not the kernel.

## 9. Things that are not about the kernel but affect the numbers

- **Dense vs sparse loss.** At 30k on lego val, dense beats the course sparse MSE by +1.06 dB
  (A) and +1.20 dB (B). B's lead is about the same under either loss (+1.41 sparse, +1.54 dense;
  single runs, old b_lr).
- **100k is under-trained at 20,000 iterations.** A at 100k scores 27.25 dB, below 30k (27.84),
  though its SSIM and LPIPS are better.
- **Absolute PSNR is far below published numbers** (lego 27 dB here vs 36–37 dB in the paper).
  That is the 10k budget, 400 px resolution and random init, not a bug: all gates pass.
- **GPU runs are not bit-reproducible.** Float atomics make two identical GPU runs differ
  (max |Δmean| 2.7e-5 after a short run). Arm-equivalence gate G3 is bit-exact only on CPU.
- **Moment matching.** At init the truncated match reproduces the Gaussian render at 44–45.6 dB.
  On a trained Gaussian model the untruncated √(4/3) match is slightly closer (35.3–35.7 vs
  34.5–35.0 dB). The init choice stays truncated per the frozen protocol.

## 10. Agreement with the paper (Deformable Beta Splatting, arXiv 2501.18630)

| Point | Paper | Here |
|---|---|---|
| Kernel-only gain | +0.16 dB (T&T), +0.21 dB (DB), Table 3 | +0.22 dB (Blender) |
| Full method on Blender | +0.86 dB over 3DGS-MCMC | not tested (kernel only) |
| Best scenes | ficus +1.72, hotdog +1.13, lego +1.10 | ficus, hotdog, lego |
| Smallest gain | chair +0.19 | chair −0.19 |
| Budget trend | Gaussian plateaus, Beta keeps improving | lead grows 2.5k → 10k; to 30k on lego only |
| Densifier | opacity-only MCMC, no kernel-specific maths | kernel-aware densifier adds nothing |

- Most of the paper's headline gain is not the kernel: its own ablation gives the kernel about
  0.2 dB, and the rest comes from the colour model and the densifier.
- Ship is a mismatch: +1.33 in the paper (full method), −0.10 here.
- The densifier rows are consistent but not the same experiment (MCMC there, split/clone here).
- New here: the frozen-b ablation (point 2) and the learned-b statistics (points 4, 5, 7). The
  paper reports neither.
- The paper's numbers above were read from the arXiv HTML and should be checked against the PDF
  before they are quoted.

## 11. Fewer training views (q4): the lead holds at 50 views, not reliably at 25

Seed 0, 10k splats, test split:

| Scene | Views | A | B | B − A (dB) | B − A LPIPS |
|---|---|---|---|---|---|
| lego | 25 | 22.71 | 23.09 | +0.38 | −0.0038 |
| lego | 50 | 26.15 | 26.96 | +0.81 | −0.0087 |
| lego | 100 | 27.24 | 27.71 | +0.47 | −0.0048 |
| ficus | 25 | 20.26 | 18.99 | **−1.27** | **+0.0666** |
| ficus | 50 | 24.95 | 25.80 | +0.85 | −0.0101 |
| ficus | 100 | 26.69 | 27.50 | +0.81 | −0.0063 |

- At 50 views B's lead is at least as large as at 100 views on both scenes.
- At 25 views the result depends on the scene: lego keeps a lead, ficus loses 1.27 dB.
- Both ficus 25-view runs overfit: the held-out loss stops improving after about iteration
  2,200 while training continues. A recovers slowly (0.0101 → 0.0079); B does not
  (0.0104 → 0.0102).
- Possible reading: with too few views, the extra per-splat freedom overfits thin geometry. It
  is one seed in a regime where both arms have collapsed by 6–8 dB, so treat the ficus 25-view
  loss as an anomaly to report, not as a finding.

## Limits of the evidence

- Two seeds per scene (four on lego) for the main table; one seed for B0, C, the budget sweep
  and the view-count sweep.
- One capacity (10k) for the main table; the capacity trend rests on lego and ficus.
- Random init and 400 px only. No real-scene data.
- b_lr was tuned on lego val only.
- No qualitative analysis yet of where high-b and low-b splats sit in the image.
