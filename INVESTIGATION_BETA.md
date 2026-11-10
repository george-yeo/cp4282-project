# Beta kernel vs Gaussian: how the findings were reached

`ANALYSIS_BETA.md` states what was found. This file records how: the question asked at each
step, what was expected, the test, the result, and what changed. It includes the claims that
turned out wrong and were corrected. State as of 10 Nov 2026; the last section lists runs still
in progress.

## Rules the investigation followed

1. **Gates before experiments.** No training run counted until the correctness checks passed on
   the GPU: finite-difference gradients, sparse loss equal to dense loss, arm equivalences,
   moment-matched init.
2. **Protocol frozen before results.** Capacity, loss, init, densifier settings and the b clamp
   were fixed before the main runs. The pilot chose capacity and loss by a rule written down in
   advance (`runs/chosen.json` keeps the rule next to the choice).
3. **Choose on val, report on test.** The pilot and the b learning-rate sweep used the val split
   only. Test numbers were never used to pick a setting.
4. **Pair by seed.** Every A run has a B run with the same seed, and comparisons are made
   within the pair.
5. **One change at a time.** Arms differ in one thing: A → B0 changes the kernel shape, B0 → B
   makes b learnable, B → C changes the densifier.
6. **Off-protocol runs stay separate.** Anything outside the frozen protocol goes in its own
   `runs_*/` directory and is labelled as a side check wherever it is quoted.
7. **Anomalies are reported, not smoothed over.** Every batch was checked for NaNs, capacity not
   reached, b piling at a clamp, and results that contradicted the current story.
8. **Nothing in the trainer kernels, tile workspace or `checks/` was patched** to make a result
   come out. When a check failed, the cause was diagnosed and written down.
9. **A claim carries its sample size.** One seed is called one seed.

## The path, step by step

### 1. Is the implementation right?

- **Test:** the gates, on `cuda:0`.
- **Result:** all passed except the arm-equivalence gate (G3), which demands bit-identical
  results for two arms that should behave the same.
- **Diagnosis:** two identical GPU runs of the same arm also differ (max |Δmean| 2.7e-5 after a
  short run), with identical logs. The cause is the order of float atomic adds on the GPU. G3
  passes on CPU.
- **Decision:** `checks/` left unchanged; the limitation noted in `checks_gpu.log`.
- **Why it mattered later:** it was the first sign that a run's result is not fully determined
  by its seed (step 12).

### 2. Does the pipeline work end to end on this machine?

- **Test:** a smoke run through training and evaluation.
- **Result:** it failed. A relative `--runs` path nested the outputs, so the evaluator could not
  find the model. Fixed by resolving the path, in its own commit.

### 3. What capacity and loss should every run use?

- **Rule fixed in advance:** smallest capacity whose Gaussian val PSNR is within 0.5 dB of the
  100k run; dense loss if the projected GPU time fits 12 h.
- **Result:** 10k splats, dense loss.
- **Anomalies kept on record:** 100k scored below 30k (under-trained at 20,000 iterations), and
  dense beat sparse by more than 1 dB.
- **A number that later proved misleading:** the pilot showed Beta ahead by 1.5 dB at 30k. It was
  one val run with an untuned b learning rate. See step 7.

### 4. Which b learning rate?

- **Test:** three values on lego val.
- **Result:** 0.0005 chosen (28.58 vs 28.37 and 28.43 dB).
- **Caveat stated at the time:** the 0.2 dB range is inside seed noise, so the choice is weakly
  supported. The stronger reason is that higher rates push more splats to the clamp.

### 5. Does Beta beat the Gaussian?

- **First answer given:** per-scene gains, with the comment that most were inside the 0.3 dB
  seed noise and the result looked marginal.
- **Sharper test:** treat the 18 seed pairs as paired differences.
- **Result:** mean +0.22 dB, standard error 0.075 dB, B ahead in 14 of 18. Small but not noise.
- **Check against outside evidence:** the paper's own ablation gives the kernel +0.16 to
  +0.21 dB. The paper's headline gain (+0.86 dB on Blender) includes its colour model and
  densifier.

### 6. Is the gain from the kernel shape or from learning b?

- **Test:** arm B0, the Beta kernel with b frozen at 0.
- **Result:** B0 ≈ A (+0.07 dB on four scenes); B − B0 = +0.28 dB. The learnable b carries it.
- **Expected otherwise:** the compact kernel shape was expected to help on its own. It did not.

### 7. Does Beta help most when splats are scarce?

- **Expectation:** yes; that is the usual argument for a more flexible primitive.
- **Test:** budget sweep at 25%, 50%, 90% and 100% of capacity.
- **Result:** the opposite. The lead grew with the budget on both scenes.
- **Follow-up claim:** "the lead keeps growing with capacity", supported by the pilot's 1.5 dB.
- **Test of that claim:** lego at 30k on test, two seeds. Lead +0.81 dB, double the 10k lead.
  The pilot's 1.5 dB was an overestimate.
- **Replication on a second scene:** ficus at 30k. Lead +0.75 dB against +0.65 dB at 10k: no
  change, and the two seeds moved in opposite directions.
- **Correction made:** the claim was narrowed to "grows up to 10k on both scenes; continues to
  30k on lego only; never shrinks". The headline, the section and the paper-comparison table
  were edited to match.
- **Extra anomaly:** ficus is worse at 30k than at 10k for both arms.

### 8. Does a kernel-aware densifier help?

- **Test:** arm C on lego and ficus.
- **Result:** no difference (27.75 vs 27.71; 27.47 vs 27.50).

### 9. Does Beta help with fewer training views?

- **Test:** 25 and 50 views on lego and ficus.
- **Result:** the lead holds at 50 views. At 25 views lego keeps it and ficus loses 1.27 dB.
- **Checked before reporting:** the held-out loss curves. Both ficus 25-view runs overfit from
  about iteration 2,200.
- **Status:** one seed, recorded as an anomaly, not a finding.

### 10. What is b doing?

- **Statistics first:** mean b falls when splats are scarce and rises when they are plentiful;
  the winning scenes are the ones where b moves most.
- **Stated limit:** ship and chair spread b without gaining, so b movement did not explain
  everything. This was left open and resolved in step 11.
- **Picture second:** a b-map, rendered by recolouring a trained model and passing it through
  the unmodified evaluator. Leaves, rims and thin rods are sharp; flat surfaces are flat.
- **Limit:** an interpretation of images, one seed per scene.

### 11. Where in the image is the gain, and why do ship and chair lose?

- **Test:** split test pixels into background, silhouette, detail and smooth regions using the
  ground truth only, and attribute the error change to each.
- **Result:** silhouette error falls in 7 of 8 scenes. On ship, Beta is better inside the object
  and far worse in the background.
- **Looked at the worst views** instead of trusting the table: a few stray splats in places the
  training cameras barely see.
- **Correction made:** "ship and chair do not benefit" became "ship and chair are spoiled by
  floaters in some runs". By the median view Beta is ahead in 17 of 18 pairs.
- **Not claimed:** that Beta makes more floaters in general. Its background error is lower in
  five scenes and higher in three.

### 12. Why does the seed matter so much?

- **Starting belief:** seed noise is ordinary run-to-run scatter, about 0.3 dB.
- **Test 1:** if it were independent scatter across views, 200 views would average it down to
  about 0.04 dB. The observed spread is 0.30 dB, so it is not.
- **Test 2:** correlation of the leftover error between neighbouring test views: 0.95. The
  differences come in runs of about ten adjacent cameras.
- **Test 3:** which views make the worst seed bad. Lego seed 2 loses more than 1 dB on the same
  views in both arms. The renders show the same white streaks in both.
- **Conclusion:** a seed fixes the random starting splats, and a floater that grows from them
  is inherited by both arms. Pairing by seed removes this shared part (0.31 → 0.16 dB).
- **Tried and ruled out:** a robust statistic. The per-seed median is as spread as the mean.

### 13. How much of the seed spread is chance, and can it be reduced?

- **Test:** seed 0 run four times per arm; four seeds starting from 10,000 splats; two seeds at
  40,000 iterations. Predictions were written first (see the section below).
- **Result 1:** the same seed varies by 0.12 dB. That is the floor for a single GPU run.
- **Result 2, which overturned part of step 12:** one same-seed repeat of the Gaussian lost more
  than 1 dB on views 77–88, the exact place where seed 2 fails. Nothing had changed between the
  repeats. So the defect is a weak spot of the scene that runs fall into partly by chance, not
  something a seed's starting splats decide.
- **Result 3:** starting from 10,000 splats cuts the Gaussian's seed spread from 0.30 to
  0.12 dB, raises both arms by 0.9 dB and removes the collapsed view clusters.
- **Result 4:** Beta's lead is unchanged from a full start (+0.38 against +0.42 dB) and far more
  consistent across seeds (spread 0.07 against 0.16 dB). The main finding survived a change in
  a protocol setting it could have depended on.
- **Result 5:** at 40,000 iterations both arms gain 0.5 dB and the lead stays.
- **Consequence for the protocol:** the frozen choice of 1,000 starting splats costs about
  0.9 dB and more than doubles the Gaussian's seed spread. It does not change the conclusion.
  It was not changed after the fact; the full-start runs are reported as a side check.

### 14. Do floaters really cause the spread, and does resolution change the lead?

- **Test of the floater explanation:** if floaters cause the seed spread, removing them should
  remove it. Opacity reset every 3,000 iterations, four seeds per arm.
- **Result:** no run of bad views in any seed; seed-by-view variance down about tenfold; seed
  spread halved for the Gaussian and down to 0.07 dB for Beta. The explanation passed a test it
  could have failed.
- **Prediction that was wrong:** a quality cost was expected from resets. Quality rose 0.8 dB.
- **Resolution:** two readings predicted opposite directions. The lead is +0.49, +0.33 and
  +0.66 dB at 200, 400 and 800 px. With two seeds that is neither a rise nor a fall, so neither
  reading is supported and no trend is claimed.
- **A check that came for free:** step 10 said b goes flat when splats are scarce. Scarcity was
  varied there by cutting the splat count. Raising resolution makes the same 10,000 splats
  scarcer in a different way, and mean b fell from +0.09 (200 px) to −0.25 (800 px). This was
  noticed after the runs, not predicted before them.
- **Consequence:** two frozen protocol choices (1,000 starting splats, no opacity reset) each
  cost about 0.8 dB and most of the seed spread. Neither changes Beta's lead on lego, which is
  +0.42, +0.38, +0.41 and +0.45 dB under the protocol, a full start, resets and 40,000
  iterations.

## Claims that were corrected

| Earlier claim | What replaced it | What forced the change |
|---|---|---|
| Most per-scene gains are inside the noise; the kernel looks marginal | +0.22 dB over 18 pairs, standard error 0.075 dB | Treating the runs as paired differences |
| Beta leads by 1.5 dB at 30k | +0.81 dB on lego test | Two test runs with the tuned b learning rate |
| The lead keeps growing with capacity | Grows to 10k on both scenes; to 30k on lego only | Ficus at 30k |
| Beta should help most with few splats | The lead is smallest with few splats | Budget sweep |
| The Beta kernel shape itself helps | Only the learnable b helps | Frozen-b arm |
| Ship and chair do not suit Beta | Their losses are floaters in a few views | Region split and worst-view renders |
| Seed spread is ordinary noise | Clustered floaters shared by both arms | Per-view decomposition |
| b spread predicts the gain, with two unexplained exceptions | The exceptions are the floater scenes | Region split |
| The seed's starting splats decide where a floater forms | The same floater appears in a same-seed repeat; it is a weak spot of the scene, hit partly by chance | Same-seed repeats |
| b spread tracks the gain | From a full start b spreads half as much and the gain is the same | Full-start runs |
| Opacity reset will cost quality | It adds 0.8 dB on lego | Opacity-reset runs |

## Predictions for the runs in progress, written before their results

All are lego, both arms, 10k splats, off-protocol side runs.

- **Same seed repeated three times.** If the seed's starting splats decide everything, repeats
  should agree to well under 0.3 dB. Partial result at the time of writing: the Gaussian's four
  same-seed results span 27.00 to 27.24 dB and Beta's three span 27.71 to 27.96 dB, a spread of
  about 0.12 dB each. So GPU nondeterminism alone accounts for a real share of the seed spread,
  and more of Beta's than of the Gaussian's.
- **Start from 10,000 random splats.** If floaters grow from a sparse start, a full start
  should change the seed spread. The direction is not predicted: more random splats could cover
  space evenly or leave more debris.
- **40,000 iterations.** If the defects are under-training, the spread should shrink. If
  floaters are stable, it should not.
- **Opacity reset every 3,000 iterations.** If floaters cause the spread, resetting should cut
  it and remove the runs of bad views. Expected cost: lower PSNR, because the code notes that
  resets interfere with late pruning.
- **200 px and 800 px.** Two readings disagree. If Beta's gain is boundary sharpness, the lead
  should grow with resolution. If it follows splats per unit of detail (step 7), the lead should
  shrink at 800 px, where 10k splats are relatively scarcer.

### How the predictions came out

| Prediction | Outcome |
|---|---|
| Same-seed repeats agree to well under 0.3 dB if the seed decides everything | Spread 0.12 dB: mostly, but chance alone is a real share, and it can reproduce the "seed 2" floater |
| A full start changes the seed spread, direction unknown | Spread falls to the chance floor for the Gaussian; quality rises 0.9 dB |
| 40,000 iterations shrinks the spread if defects are under-training | Both arms gain 0.5 dB, so 20,000 is under-trained; two seeds cannot say anything about spread |
| Opacity reset cuts the spread and removes runs of bad views, at a cost in PSNR | Spread cut and bad views gone: right. Cost in PSNR: wrong, it rose 0.8 dB |
| Resolution: lead grows (boundary sharpness) or shrinks (splats per detail) | Neither; +0.49, +0.33, +0.66 dB with two seeds each is no trend |

## Still untested

- Whether hotdog's gain is capped by the b clamp.
- Whether any of this holds with a point-cloud init, on real scenes, or above 400 px.
- Whether flat (b < 0) floaters are more damaging than Gaussian ones. The b-maps show floaters
  are flat, but damage per floater was not measured.
- The ficus 25-view loss, beyond one seed.
