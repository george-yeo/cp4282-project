# CP4282 Beta-kernel study: code notes

Built on George's AS2 submission (`george-yeo/cp4282-as2`, branch `main`, on top of the course starter `weitsang/cp4282-as2` @ 9e74d1e). Branch `beta` adds one commit per change; `git log main..beta` lists them.

## What changed

| File | Change |
|---|---|
| `3dgs_trainer.py` | AS2 backward factored into one shared adjoint helper (inclusive colour clamp). Beta kernel `(1 - q/9)^(4e^b)` in `alpha_at_pixel`. Per-splat `b` with its own Adam (`learning_rates.shape`) and a projected clamp to `[b_min, b_max]`. Moment-matched init. Children inherit `b`. Arm C densifier. Per-densify-event log. `.beta.npz` sidecar. `final.runlog.json`. |
| `gaussian_first_tile_workspace_gpu.py` | Tile bounds use the same per-splat support as shading (`kernel_support`). |
| `shared/training_config.py` | `kernel:` section (`type`, `learn_b`, `b_init`, `b_min/max`, `moment_match_init`, `moment_match`, `densifier`, `moment_dim`) and `learning_rates.shape`. |
| `3dgs_evaluator.py` | Renders with the Beta kernel when `<ply>.beta.npz` exists. `--override-b` sets every b to one value; `--max-views N` evaluates only the first N views. |
| `beta_runs.py` | Run driver and queues (`pilot`, q1–q4), `pick-pilot`, `pick-lr`, `sigma`, `status`, `summary`. |
| `checks/` | Gradient and equivalence gates (below). |

Arm A uses exactly the course kernel. Its gradients are bit-identical before and after the Beta change (same output from `3dgs_gradient_check_gpu.py`). They also match the AS2 submission, except that a colour channel sitting exactly at 0 or 1 now still receives gradient.

## Design decisions that differ from the proposal

1. **Compact Box.** The course rule (`compact_box.beta = 0.5`) cuts the Gaussian where alpha = o·(α_min/o)^0.5 = √(o·α_min), not at α_min. The Beta kernel uses the same α threshold: `q_max = 9(1 - (α_min/o)^(0.5/β))`. Both kernels are therefore truncated at the same alpha. At o = 1 the cut radii are 2.35σ (Gaussian) and 2.12σ (Beta, b = 0).
2. **Moments are 2D.** The rendered footprint is a 2D Beta on the projected covariance, so the untruncated variance factor is 9/(2β+4). ρ(β) = √(6/(β+2)). `kernel.moment_dim: 3` gives the proposal's 3D version.
3. **Moment matching is truncated and opacity-aware** (`kernel.moment_match: truncated`). It matches the second moment of the footprint the Compact Box actually renders, at each splat's opacity. The factor is 1.09 at o = 1 and 0.99 at random-init opacity (σ(−4) ≈ 0.018), against √(4/3) = 1.155 untruncated. On the lego init cloud, a Gaussian render reproduced as Beta b = 0 reaches **44–46 dB** with truncated matching, against 34 dB with √(4/3) and 40 dB with no scaling (`checks/check_arm_equivalences.py`).
4. **Arm C uses the untruncated ρ(β) and c(β)** as proposed, with the 2D fix. Opacity-aware versions are possible but not implemented.
5. **Budget.** The course densifier always fills `model.capacity` (it refills freed slots with no cutoff), so there is no unconstrained count to take a fraction of. Capacity *is* the budget: one value for all scenes, chosen by the pilot (`pick-pilot` → `runs/chosen.json`, overridable with `--capacity`), with the sweep at 25/50/90/100% of it.
6. **Opacity reset is off** (course default `opacity_reset_interval: 0`). With resets on, every opacity is capped at 0.01, which is below late prune's `opacity_cutoff = 0.02`, and late prune fires every 400 iterations once armed. So a reset risks pruning every splat that hasn't recovered within 400 iterations. I haven't tested this; it's the reason I left resets off.
7. **Protocol overrides of the course defaults:** 400×400, 20k iterations, `position_max_steps = iterations`, convergence (early stop) off, random init with 1,000 splats in every run (so even the 25% budget has room to grow), `densification.fraction = 0.2`, `opacity_reset_interval = 0`, one snapshot at the end. Loss is dense or the course sparse MSE (48 px/tile), chosen by the pilot (`--loss` overrides). All other densification, ADC and learning-rate settings are course defaults.

## Gates (run these first on the GPU box)

```bash
python 3dgs_gradient_check_gpu.py --device cuda:0          # course FD check (Gaussian)
python checks/check_sparse_matches_dense.py                # sparse == dense, both kernels (CPU, ~1 min)
python checks/check_beta_gradients.py                      # Beta FD: end-to-end + centre-to-rim (CPU)
python checks/check_arm_equivalences.py                    # G3 C == B at b = 0, G2 moment match (CPU)
```

Results in the cloud workspace (CPU):
- FD rel. error ≤ 1.2e-2 end-to-end for b ∈ {−2..2}, with the Compact Box on and off.
- Per-pixel error ≤ 9e-3 out to the rim.
- With no Compact Box and β < 1, ∂α/∂q diverges at q → 9 and a float32 FD can't resolve it; error is ≤ 3e-4 inside 97% of the support. Training always runs with the Compact Box on.
- Sparse == dense to 1e-7 for both kernels. C == B bit-identical through 3 densify events.

After the first trained A model exists, re-run G2 on it:
`python checks/check_arm_equivalences.py --skip-g3 --device cuda:0 --ply runs/A_lego_s0/final.ply`

## Running

Data: the eight NeRF-synthetic scenes in one folder (`<root>/{chair,drums,...}/transforms_*.json`). The checks read `data/lego` (copy `data/` from your AS2 clone).

```bash
python beta_runs.py pilot --data-root ~/data/nerf_synthetic  # 6 lego runs (val): capacity x loss
python beta_runs.py pick-pilot                               # writes capacity + loss to runs/chosen.json
python beta_runs.py q1 --data-root ~/data/nerf_synthetic     # 3 b-lr runs (val) + 4 A lego seeds (test)
python beta_runs.py pick-lr                                  # merges b_lr into runs/chosen.json
python beta_runs.py sigma                                    # writes runs/sigma.json
python beta_runs.py q2 --data-root ~/data/nerf_synthetic     # 32 core runs
python beta_runs.py q3 --data-root ~/data/nerf_synthetic     # B0 x4, budget, B@90%, warm start, C x2
python beta_runs.py q4 --data-root ~/data/nerf_synthetic     # optional view-count sweep
python beta_runs.py status ; python beta_runs.py summary
```

Add `--no-lpips` if torch/lpips aren't installed. Run `python beta_runs.py q1 --dry-run` to list runs without starting them. Queues are resumable: a run with `DONE` is skipped, and an unfinished run directory is wiped and restarted.

Each run directory contains: `config.yaml`, `train.log`, `final.ply` (+ `final.beta.npz`), `final.runlog.json` (eval series, densify events with parent/active b histograms, b summary incl. % at the clamps, ms/iter, peak mempool), `{test,val}_metrics.csv`, `{test,val}_renders/*.png` and `DONE`.
