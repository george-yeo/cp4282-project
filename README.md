# Beta kernel vs Gaussian in a small-budget 3DGS trainer

CP4282 individual project, George Yeo, November 2026.

The project swaps the Gaussian kernel in the course's Warp 3DGS trainer for the Beta kernel from
Deformable Beta Splatting, changes nothing else, and measures what the kernel alone is worth.

**Start with the report: [REPORT_BETA.md](REPORT_BETA.md).**

| File | What it is |
|---|---|
| [REPORT_BETA.md](REPORT_BETA.md) | The report |
| [ANALYSIS_BETA.md](ANALYSIS_BETA.md) | Every result with its table |
| [INVESTIGATION_BETA.md](INVESTIGATION_BETA.md) | The order the findings were reached in, the predictions made before each batch of runs, and the claims that were corrected |
| [README_BETA.md](README_BETA.md) | Code changes, design decisions and the correctness checks |
| [README_AS2.md](README_AS2.md) | The course's assignment 2 README, unchanged. This repository is built on that starter code |
| `results/` | The result files of all 154 runs (per-view metrics, learned b, training logs) |
| `figures/` | The figures, with a CSV of every plotted value |

## Check the numbers without a GPU

`results/` holds, for every run, its settings (`spec.json`, `config.yaml`), its summary metrics
(`DONE`), its per-view test metrics (`test_metrics.csv`), its training log (`final.runlog.json`)
and, for Beta runs, the learned b of every splat (`final.beta.npz`). Run folders are named
`<arm>_<scene>_s<seed>`: arm `A` is the Gaussian, `B` is Beta with learnable b, `B0` is Beta with
b frozen at 0, and `C` is `B` with a densifier that accounts for b.

To redraw the report's figures from those files:

```bash
python -m pip install matplotlib
python scripts/report_figures.py --root results --out figures_check
```

Each figure comes with a CSV of its plotted values, which should match the ones in
`figures/report/` exactly.

## Reproduce the runs

Setup (the runs used Python 3.14, Warp 1.16 and one RTX 5070):

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Data: the eight NeRF-synthetic scenes (chair, drums, ficus, hotdog, lego, materials, mic, ship)
in one folder, from the NeRF project page (https://www.matthewtancik.com/nerf). The lego scene
under `data/` is the course's copy and is only used by the correctness checks.

Correctness checks, which passed before any run was counted:

```bash
python 3dgs_gradient_check_gpu.py --device cuda:0   # the course's check of the Gaussian gradients
python checks/check_sparse_matches_dense.py         # sampled loss equals dense loss, both kernels
python checks/check_beta_gradients.py               # finite-difference check of the Beta kernel
python checks/check_arm_equivalences.py             # arm C equals arm B at b = 0; scale matching
```

Every training run, in the order they were made:

```bash
bash scripts/reproduce.sh /path/to/nerf_synthetic
```

- It trains and evaluates 154 runs: about 6.5 hours of training plus evaluation on one RTX 5070.
- It is resumable. A finished run is skipped, so it can be stopped and restarted.
- `DRY_RUN=1 bash scripts/reproduce.sh /path/to/nerf_synthetic` lists the runs without starting
  them, and the script itself is the list of exact commands, one block per experiment.
- By default it uses the protocol's frozen choices (10,000 splats, dense loss, b learning rate
  0.0005, from `results/runs/chosen.json`). `--repick` reruns the selection rules instead.
- A single run is `python beta_runs.py one --arm B --scene lego --seed 0 --data-root <folder>`,
  about 2.5 minutes of training.

Then the figures and the two analyses that need the renders:

```bash
python scripts/report_figures.py                                   # figures/report/
python scripts/error_regions.py --data-root /path/to/nerf_synthetic  # error by image region
python scripts/b_map.py --help                                     # maps of the learned b
```

**What to expect from a rerun.** Training on the GPU is not bit-reproducible: the same seed run
four times varied by 0.12 dB on lego (report, section 4.5). A rerun should match the numbers in
`results/` to within a few tenths of a dB per run, but not exactly.

## Where the runs go

| Folder | Runs | What |
|---|---|---|
| `runs/` | 82 | The frozen protocol: pilot, main comparison, frozen-b ablation, budget sweep, arm C, fewer views |
| `runs_init10k/` | 36 | The main comparison again from 10,000 starting splats |
| `runs_30k/` | 8 | 30,000-splat budget, lego and ficus |
| `runs_reset/` | 8 | Opacity reset every 3,000 iterations, lego |
| `runs_it40k/` | 4 | 40,000 iterations, lego |
| `runs_res200/`, `runs_res800/` | 4 + 4 | 200 px and 800 px, lego |
| `runs_rep1/` to `runs_rep3/` | 2 + 2 + 2 | The same seed run again, lego |
| `runs_clamp4/` | 2 | b clamped to [−4, 4], lego and hotdog |

These folders are not in the repository (they hold the renders and models, about 3 GB).
`results/` mirrors them with the small files only; `python scripts/snapshot_results.py` rebuilds
it from local run folders.
