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
10k on both scenes tested and does not shrink beyond that. It shows up mainly at object
boundaries. On lego it stays between +0.3 and +0.7 dB under every training variant tried
(full-capacity start, opacity reset, twice the iterations, 200 to 800 px).

The mean gain is the same from a full-capacity start (+0.22 dB over the same 18 pairs,
point 19), but which scenes gain is not. Ficus and lego lead in both settings. Hotdog's lead
almost disappears from a full start, and ship and chair turn from losses into small gains.

The protocol itself is noisy: 1,000 starting splats with no opacity reset leaves floaters that
cost about 0.8 dB on lego and account for most of the seed-to-seed spread (points 15 and 16).

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
- Counting each scene once instead of each pair: the mean of the eight scene leads is +0.20 dB
  with a standard error of 0.11 dB (t ≈ 1.9, six of eight scenes positive). Pairs from one
  scene are not independent, so this is the fairer test of "does it hold for a new scene", and
  eight scenes are too few to settle it.
- B takes about 3% longer per iteration (7.1 against 6.9 ms on average; the ratio within a
  pair runs from 0.98 to 1.15). B stores one extra float per splat.
- Seed noise: lego A over 4 seeds is 27.12 ± 0.30 dB. Per-scene differences below about 0.3 dB
  are not distinguishable from noise with 2 seeds.
- Chair is the only scene with a clear loss, and it comes from one seed (−0.45); the other seed
  is +0.08.

## 2. The gain comes from the learnable b, not the kernel shape

| Scene (seed 0) | A | B0 | B | B0 − A | B − B0 |
|---|---|---|---|---|---|
| lego | 27.24 | 27.52 | 27.71 | +0.28 | +0.19 |
| ficus | 26.69 | 26.69 | 27.50 | 0.00 | +0.81 |
| hotdog | 32.90 | 33.05 | 33.47 | +0.16 | +0.42 |
| chair | 29.95 | 29.79 | 29.50 | −0.17 | −0.29 |
| drums | 23.78 | 23.73 | 23.96 | −0.05 | +0.23 |
| materials | 24.51 | 24.48 | 24.54 | −0.03 | +0.06 |
| mic | 30.54 | 30.29 | 30.51 | −0.25 | +0.22 |
| ship | 26.46 | 26.28 | 26.26 | −0.18 | −0.02 |
| Mean (8 scenes) | | | | −0.03 | +0.20 |

- Swapping the Gaussian for a fixed-shape Beta (compact support, b = 0) is about neutral:
  −0.03 dB on average (standard error 0.06), positive on 2 of 8 scenes.
- Letting each splat learn its own b adds +0.20 dB on average (standard error 0.11), positive
  on 6 of 8 scenes. Most of it is on ficus and hotdog.
- The budget sweep agrees: B0 stays within 0.2 dB of A at 25% and 50% capacity, while B leads both.
- Single seed per cell, so the lego split between "shape" and "learnable b" is within noise.
  Only mic's frozen-b loss (−0.25 dB) is clearly outside the ±0.16 dB spread of one seed pair.
- The first version of this table had four scenes (lego, ficus, hotdog, chair) and gave
  +0.07 and +0.28 dB. Drums, materials, mic and ship were added on 12 Nov 2026 under the same
  protocol. The conclusion is the same and the frozen kernel now looks slightly worse, not
  slightly better, than the Gaussian.
- At seed 0 alone B − A is +0.17 dB over the eight scenes; the +0.22 dB headline is over all
  18 seed pairs.

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
- Ficus at 30k is a poor regime for both arms: PSNR is lower than at 10k (A 26.57 → 26.12,
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
- This ranking belongs to the protocol. From a full-capacity start (point 19) hotdog's lead
  falls to +0.05 dB and ship's rises to +0.24 dB; only ficus and lego stay on top.

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
- The paper's numbers above were read from the arXiv HTML and re-read on 12 Nov 2026 (v2): they
  match. The kernel-only figures are the first two rows of its Table 3, which compare the two
  kernels with MCMC densification and SH colour on both sides.

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

## 12. Where sharp and flat splats sit (b-maps)

Figures: `figures/b_map_B_lego_s0.png` and `figures/b_map_B_ficus_s0.png`, made by
`scripts/b_map.py`. Each pixel shows the visibility-weighted mean b of the splats it sees
(blue = flat, grey = 0, red = sharp).

| Run | Splats with b > +0.5 | Splats with b < −0.5 | Rank correlation of b with splat size |
|---|---|---|---|
| lego, seed 0 | 16% | 16% | −0.49 |
| ficus, seed 0 | 28% | 1% | −0.41 |

- **Ficus:** the leaves and thin stems are sharp (b > 0); the pot is flat (b < 0).
- **Lego:** broad surfaces (baseplate, bucket faces, body panels) are flat; the bucket rim, the
  thin rods and the stud outlines are sharp.
- Large splats flatten and small splats sharpen (size = largest scale; Spearman correlation).
- Reading: b does two jobs. Flat-topped splats tile smooth surfaces, and sharp splats draw edges
  and thin structures. That fits point 5: the scenes that gain are the ones with a lot of thin
  structure (ficus) or a mix of both (lego, hotdog). It is an interpretation of two scenes, not
  a tested cause.
- The ficus b-map also shows a faint haze of low-opacity floaters around the plant. They are in
  the colour render too but nearly invisible on white. Whether the Gaussian arm has the same
  floaters was not checked.

b-maps for the other six scenes are in `figures/b_map_B_<scene>_s0.png`. They show the same
pattern: ship's rigging and hull edges are sharp and its water is flat; chair's frame edges are
sharp and its cushions flat. The low-opacity floaters around every object are flat (b < 0). The
rank correlation of b with splat size is negative in all eight scenes (−0.42 to −0.69).

## 13. Where in the image Beta gains: object boundaries

`scripts/error_regions.py` splits every test pixel into four regions using the ground truth
only: background (more than 2 px outside the silhouette), silhouette (within 2 px of it), detail
(inside the object, top quarter of image gradient) and smooth (the rest of the inside). The table
gives B's change in total squared error, and how much of it comes through each region
(two seeds per scene, 200 test views each).

| Scene | Total | Background | Silhouette | Detail | Smooth |
|---|---|---|---|---|---|
| ficus | −13.3% | −0.2 | **−11.8** | −0.3 | −1.0 |
| hotdog | −8.9% | −1.1 | −0.8 | −3.0 | −4.0 |
| lego | −6.5% | −0.3 | **−3.3** | −1.0 | −1.9 |
| drums | −5.0% | −0.2 | **−2.9** | −0.7 | −1.2 |
| materials | −0.6% | −0.1 | −0.8 | +0.1 | +0.2 |
| mic | −0.5% | +0.1 | −0.4 | −0.6 | +0.4 |
| chair | +4.4% | +1.7 | −0.7 | +0.6 | +2.8 |
| ship | +16.3% | **+20.4** | +0.2 | −1.4 | −2.8 |

- **Silhouettes improve in 7 of 8 scenes.** Error inside the silhouette band falls by 15% on
  ficus, 17% on lego, 7% on drums, 3–4% on chair, hotdog and materials, 1% on mic. Ship is flat
  (+0.8%).
- **Ficus is almost all silhouette.** 78% of the Gaussian's error on ficus sits in the
  silhouette band (thin leaves against white), and that is where Beta's gain is.
- **Hotdog gains inside the object**, on both smooth and detailed areas.
- **The learnable b is what sharpens boundaries.** At seed 0, B0 changes silhouette error
  against A by −7% (lego), −1% (ficus), +25% (hotdog), +4% (chair), +4% (drums), −2%
  (materials), +8% (mic) and +2% (ship): no consistent effect. B against B0 changes it by
  −10%, −17%, −20%, −2%, −9%, −2%, −7% and +3% in the same order: lower in 7 of 8, ship being
  the exception.
- **Ship's loss is floaters, not the kernel's fit.** Inside the ship B is better (smooth −8%,
  detail −4%), but its background error is 5.4× the Gaussian's on seed 0 and 1.9× on seed 1.
  Without the background region B would be ahead by about 4%.
- These percentages are of pooled squared error, which weights bad views more than the mean of
  per-view PSNR used in the main table.

## 14. Per view, Beta wins more often than the means suggest

| Scene | B wins (share of test views, per seed) | Median per-view B − A (dB, per seed) |
|---|---|---|
| ficus | 96%, 90% | +0.84, +0.45 |
| hotdog | 95%, 91% | +0.57, +0.58 |
| lego | 96%, 81%, 90%, 90% | +0.46, +0.25, +0.61, +0.52 |
| drums | 90%, 96% | +0.21, +0.21 |
| ship | 64%, 85% | +0.11, +0.13 |
| mic | 56%, 62% | +0.08, +0.14 |
| materials | 57%, 56% | +0.04, +0.04 |
| chair | 36%, 54% | −0.18, +0.08 |

- By the median view, B is ahead in 17 of 18 seed pairs. The only exception is chair seed 0.
- On ship B wins most views by a little and loses a few badly: its worst 5% of views are 1.4 to
  3.0 dB behind, in runs of neighbouring cameras that look at the underside of the basin.
- The worst views for B share one cause (`figures/worst_views_A_vs_B.png`): a few stray splats in
  places the training views constrain poorly, such as under ship's basin, behind the chair,
  at the base of the mic stand and under hotdog's plate.
- So "where Beta does not shine" is mostly not a property of the scene's content. It is whether
  a run happened to leave a visible floater.

## 15. Why the seed matters: leftover floaters, shared by both arms

Lego, four seeds per arm:

- **The seed effect is spatially clustered.** After removing each seed's mean and each view's
  mean, the remaining per-view PSNR is strongly correlated between neighbouring test views
  (0.95 at one view apart, 0.5 at five apart, about 0 at twenty). Defects span about ten
  consecutive cameras, so 200 test views behave like roughly 20 independent ones.
- **The worst seed fails in the same place in both arms.** Seed 2 is the lowest seed for both
  A and B, and both lose more than 1 dB on the same views (77–91 and 111–119). The renders show
  the same white streak floaters beside the tracks in both arms
  (`figures/seed2_floaters_lego.png`).
- **But the seed is not the only cause of that defect.** One of three same-seed repeats of
  Gaussian seed 0 (point 16) lost more than 1 dB on views 77–88, the same place, with nothing
  changed. The spot beside the tracks is a weak point of the scene that a run falls into partly
  by chance. A seed can make it more likely; it does not decide it.
- **Pairing by seed already halves the noise.** The spread of B − A is 0.16 dB when both arms
  use the same seed, and 0.31 dB across mismatched seeds.
- Robust statistics do not remove it: the spread of the per-seed median is the same as the
  spread of the mean (0.31 vs 0.30 dB for A), because a seed also shifts typical views a little.

## 16. How much of the seed spread is chance, and how to shrink it

Lego, 10k splats, test split. Side runs outside the frozen protocol (`runs_rep*/`,
`runs_init10k/`).

| Setting | Runs per arm | A mean ± sd | B mean ± sd | B − A | sd of paired B − A |
|---|---|---|---|---|---|
| Protocol (1,000 starting splats), seeds 0–3 | 4 | 27.12 ± 0.30 | 27.54 ± 0.19 | +0.42 | 0.16 |
| Protocol, seed 0 run four times | 4 | 27.13 ± 0.12 | 27.78 ± 0.13 | +0.65 | not paired |
| 10,000 starting splats, seeds 0–3 | 4 | 28.02 ± 0.12 | 28.40 ± 0.16 | +0.38 | 0.07 |
| Opacity reset every 3,000 iterations, seeds 0–3 | 4 | 27.90 ± 0.16 | 28.32 ± 0.07 | +0.41 | 0.13 |

- **Opacity reset removes the floater clusters.** With resets (`runs_reset/`), no seed has a
  view more than 1 dB below the other seeds, in either arm. The seed-by-view variance falls about
  tenfold (0.28 → 0.04 for A, 0.42 → 0.03 for B). The seed spread halves for the Gaussian and
  falls to 0.07 dB for Beta. This is the direct evidence that floaters cause the spread.
- **Reset also raises quality by 0.8 dB for both arms.** A cost was expected, because the code
  notes that resets interfere with late pruning. There was none on lego.
- **Beta's lead is again unchanged:** +0.41 dB with resets.
- **About 0.12 dB of spread is pure chance.** The same seed run four times on the GPU varies by
  0.12 dB (A) and 0.13 dB (B). GPU float atomics make runs diverge, and training amplifies it.
  No handling of seeds can get a single run below this.
- **The rest comes from growing 10,000 splats out of 1,000.** Starting from 10,000 random splats
  brings the Gaussian's seed spread down from 0.30 to 0.12 dB, which is the chance floor.
- **A full start is also better:** +0.9 dB for both arms, and no run has a collapsed cluster of
  views. The worst single view is 23.3 dB or better in all eight runs, against 20.7 and 19.6 dB
  for seed 2 under the protocol.
- **Beta's lead does not depend on the start.** It is +0.38 dB from a full start and +0.42 dB
  under the protocol. From a full start all four seeds agree closely (+0.46, +0.41, +0.31,
  +0.33), so the spread of the paired difference falls from 0.16 to 0.07 dB.
- **One protocol pair understates or overstates by chance.** Seed 0's single pair gave +0.47 dB;
  the mean of four runs per arm at seed 0 gives +0.65 dB.
- With a full start b spreads less (std 0.38 against 0.67) and nothing reaches the clamp, yet
  the lead is the same size. The link between b spread and gain in point 5 is therefore loose.
- With resets b ends flatter (mean −0.23, std 0.38), again with the same lead.
- Ways to shrink the variance: start from full capacity, or reset opacity (each roughly halves
  the spread and adds 0.8–0.9 dB; the two together were not tested); pair by seed; average
  repeated runs (chance noise falls with the square root of the number of runs).
- Across the four training variants tried on lego, the lead is +0.42 (protocol), +0.38 (full
  start), +0.41 (opacity reset) and +0.45 dB (40,000 iterations, point 17).
- Caveats: lego only (point 19 repeats the full start on all eight scenes). Each spread is
  estimated from four runs, so it is itself uncertain by
  roughly 40%; the 0.07 dB values are below the 0.12 dB same-seed spread measured under the
  protocol, which is within that uncertainty or means the cleaner settings also reduce chance
  noise.

## 17. Longer training: both arms improve, the lead stays

Lego, 40,000 iterations, seeds 0 and 1 (`runs_it40k/`).

| | A | B | B − A |
|---|---|---|---|
| 20,000 iterations (protocol), seeds 0, 1 | 27.24, 27.42 | 27.71, 27.61 | +0.47, +0.18 |
| 40,000 iterations, seeds 0, 1 | 27.81, 27.92 | 28.16, 28.47 | +0.35, +0.55 |

- Both arms gain about 0.5 dB, so 20,000 iterations is under-trained at this budget.
- The lead does not close with more training (+0.45 dB against +0.33 dB for the same seeds), so
  it is not a matter of Beta converging faster.
- b keeps moving: its spread grows from 0.67 to 0.87 and 5% of splats reach the +2 clamp. b has
  not settled at 20,000 iterations, and with longer training the clamp starts to bind.
- Two seeds only.

## 18. Image resolution: the lead holds from 200 px to 800 px

Lego, 10k splats, seeds 0 and 1 (`runs_res200/`, `runs_res800/`).

| Resolution | A | B | B − A (dB) | B − A LPIPS | Mean b |
|---|---|---|---|---|---|
| 200 px | 28.14, 28.31 | 28.53, 28.90 | +0.39, +0.58 | −0.0009, −0.0025 | +0.09 |
| 400 px (protocol) | 27.24, 27.42 | 27.71, 27.61 | +0.47, +0.18 | −0.0048, −0.0001 | −0.01 |
| 800 px | 26.94, 27.22 | 27.82, 27.66 | +0.88, +0.44 | −0.0168, −0.0086 | −0.25 |

- Beta leads at every resolution: +0.49, +0.33 and +0.66 dB on average.
- No trend can be claimed. With two seeds each mean is uncertain by about 0.1 dB, and the
  order (400 lowest) is not monotone.
- The LPIPS gap is clearly largest at 800 px (9–17% lower than the Gaussian's).
- Mean b falls as resolution rises. At 800 px the same 10,000 splats must cover more detail
  each, and b goes flat; at 200 px splats are plentiful and b goes sharp. This matches point 4,
  from a different direction.
- Time per iteration stays equal between the arms (10.3 vs 10.6 ms at 800 px).

## 19. All eight scenes from a full-capacity start: same mean lead, different scenes

The main comparison (point 1) repeated with 10,000 starting splats instead of 1,000
(`runs_init10k/`, side runs outside the frozen protocol; same seeds, everything else unchanged).

| Scene | Seeds | A | B | B − A (dB) | Per-seed | B − A LPIPS | Protocol B − A |
|---|---|---|---|---|---|---|---|
| ficus | 2 | 25.81 | 26.59 | +0.77 | +0.78, +0.77 | −0.0069 | +0.65 |
| lego | 4 | 28.02 | 28.40 | +0.38 | +0.46, +0.41, +0.31, +0.33 | −0.0045 | +0.42 |
| ship | 2 | 27.04 | 27.28 | +0.24 | +0.28, +0.20 | −0.0061 | −0.10 |
| drums | 2 | 23.94 | 24.04 | +0.10 | +0.08, +0.12 | −0.0056 | +0.22 |
| chair | 2 | 30.21 | 30.30 | +0.09 | +0.06, +0.12 | −0.0016 | −0.19 |
| hotdog | 2 | 34.02 | 34.07 | +0.05 | +0.05, +0.04 | +0.0000 | +0.53 |
| mic | 2 | 30.15 | 30.18 | +0.03 | −0.01, +0.07 | +0.0002 | +0.04 |
| materials | 2 | 24.85 | 24.83 | −0.02 | −0.01, −0.04 | −0.0030 | +0.03 |

- **The mean lead is the same: +0.22 dB over 18 pairs** (protocol: +0.22). The standard
  deviation of the pair differences is 0.25 dB (protocol 0.32), so the standard error is
  0.059 dB (t ≈ 3.8). B wins 15/18 pairs on PSNR and 16/18 on LPIPS (protocol 14/18, 15/18).
  Counting each scene once: +0.20 dB, standard error 0.09 dB (t ≈ 2.2, seven of eight scenes
  positive).
- **The lead is far more repeatable.** The two seeds of a scene disagree on B − A by 0.04 dB on
  average, against 0.21 dB under the protocol. The largest disagreement is 0.08 dB (mic);
  under the protocol it was 0.54 dB (chair). Most of this comes from Beta: the gap between a
  scene's two seeds averages 0.25 dB for B under the protocol and 0.09 dB from a full start
  (A: 0.10 and 0.08 dB).
- **Ship and chair turn positive.** Ship goes from −0.10 to +0.24 dB, with B ahead on 94% and
  91% of views and no view more than 1 dB behind (20 and 18 such views under the protocol).
  Chair goes from −0.19 to +0.09 dB. This is what point 13 implied: their protocol losses were
  floaters, not a poor fit of the kernel.
- **Hotdog's lead almost disappears: +0.53 → +0.05 dB.** A full start helps the Gaussian by
  1.14 dB on hotdog but Beta by only 0.65 dB. By the median view B is still ahead (+0.17,
  +0.18 dB, 75–77% of views), but each seed has a run of 8–10 adjacent views where B is more
  than 1 dB behind (views 11–20 on seed 0, 107–113 on seed 1). Leaving those views out, the
  lead is +0.14 and +0.10 dB. So most of hotdog's protocol lead belongs to the sparse start,
  not to the scene.
- **Ficus and lego keep their lead** (+0.77 and +0.38 dB), and within each scene the seed
  pairs agree to within 0.15 dB. These are the two scenes where the lead is established in
  both settings.
- **The scene ranking is only loosely preserved.** Rank correlation of the per-scene lead
  between the two settings is 0.40 over eight scenes, which is not distinguishable from none.
- **A full start is not better everywhere.** Both arms improve on hotdog, lego, ship, chair
  and materials (+0.2 to +1.1 dB) but get worse on ficus (−0.76 A, −0.64 B) and mic (−0.39,
  −0.41); drums is unchanged. The 0.9 dB gain measured on lego in point 16 does not generalise.
- **A full start does not remove floaters either, except on ship.** Counting test views where
  a scene's two seeds differ by more than 1 dB: 149 (A) and 218 (B) of 1,600 under the
  protocol, 137 and 167 from a full start. Ship drops to none in both arms. Hotdog and chair
  still have runs of adjacent bad views, in different places for each seed; the renders show
  white streaks in front of the object in both arms, with more of them in B on the views
  where B loses (`figures/full_start_floaters_hotdog_chair.png`).
- b moves less in every scene (std 0.22–0.38 against 0.33–0.68) and almost nothing reaches the
  clamp. Ficus keeps its positive shift (mean b +0.22).

Where the change in error sits (`figures/error_regions_init10k.csv`, seeds 0 and 1, same
regions as point 13):

| Scene | Total | Background | Silhouette | Detail | Smooth |
|---|---|---|---|---|---|
| ficus | −15.7% | −0.0 | **−11.8** | −1.0 | −2.9 |
| lego | −8.7% | −0.2 | **−3.1** | −2.5 | −2.9 |
| ship | −5.7% | −1.0 | **−2.0** | −1.6 | −1.2 |
| drums | −1.3% | −0.0 | **−2.0** | −0.1 | +0.9 |
| mic | −0.4% | +0.0 | −0.0 | −0.4 | +0.0 |
| chair | +0.3% | +0.2 | −0.9 | −0.5 | +1.4 |
| materials | +0.6% | +0.0 | −0.5 | −0.2 | +1.3 |
| hotdog | +2.3% | −0.2 | −0.4 | +0.0 | +2.8 |

- **Silhouettes are the consistent gain.** Error inside the silhouette band falls in seven
  scenes (18% on lego and ficus, 11% on ship, 5% on drums and chair, 2% on materials and
  hotdog) and is flat on mic (−0.1%). In neither setting does a scene get clearly worse there;
  the worst case is ship under the protocol (+0.8%).
- **Smooth interiors are mixed.** Error on smooth areas inside the object falls on ficus, lego
  and ship (4–12%) and rises on hotdog, materials, chair and drums (2–7%). On hotdog and chair
  the floater streaks lie over the object, so part of that rise is floaters; on materials and
  drums no cause was identified.
- Ship's background, which cost B 20% of the total error under the protocol, now has 4% less
  error than the Gaussian's.
- Caveats: two seeds per scene (four on lego); pooled squared error weights bad views more than
  mean PSNR does, which is why hotdog and chair show a small rise in error next to a small
  PSNR lead.

## Limits of the evidence

- Two seeds per scene (four on lego) for the main table; one seed for B0, C, the budget sweep
  and the view-count sweep.
- One capacity (10k) for the main table; the capacity trend rests on lego and ficus.
- Which scenes gain depends on the starting splat count (point 19). Only ficus and lego lead
  in both settings; hotdog's lead is specific to the 1,000-splat start.
- Random init only; resolutions other than 400 px were tried on lego alone. No real-scene data.
- b_lr was tuned on lego val only.
- The b-maps cover two scenes, one seed each, two views each.
