#!/usr/bin/env bash
# Train and evaluate every run behind the report, in the order they were made.
#
#   scripts/reproduce.sh <nerf_synthetic folder>             # the frozen protocol choices
#   scripts/reproduce.sh <nerf_synthetic folder> --repick    # also redo the pilot's choices
#
# <nerf_synthetic folder> holds the eight scene folders (chair, drums, ficus, hotdog, lego,
# materials, mic, ship), each with transforms_{train,val,test}.json.
#
# 154 runs, about 6.5 hours of training plus evaluation on one RTX 5070. Every command is
# resumable: a finished run (a folder with a DONE file) is skipped.
#
# The protocol's budget, loss and b learning rate were chosen by rules on lego's validation
# split and then frozen (results/runs/chosen.json). By default this script uses those frozen
# values, so it reproduces the report's protocol even if a rerun of the pilot would land on a
# different choice. --repick reruns the selection rules instead.
#
# Environment: RUNS_PREFIX=/some/dir/ writes the run folders there; DRY_RUN=1 only lists runs;
# PYTHON=python3 picks the interpreter.
set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,20p' "$0"
  exit 1
fi
cd "$(dirname "$0")/.."
DATA="$1"
MODE="${2:-}"
OUT="${RUNS_PREFIX:-}"
PY="${PYTHON:-python}"
DRY="${DRY_RUN:+--dry-run}"

run() { "$PY" beta_runs.py "$@" --data-root "$DATA" $DRY; }
decide() { [ -n "$DRY" ] || "$PY" beta_runs.py "$@"; }
frozen() { mkdir -p "$1" && cp results/runs/chosen.json "$1/chosen.json"; }

# ---- 1. Protocol runs (runs/): 82 runs
P="${OUT}runs"
if [ "$MODE" = "--repick" ]; then
  run pilot --runs "$P"                 # 6 lego runs on the validation split
  decide pick-pilot --runs "$P"         # budget and loss -> runs/chosen.json
  run q1 --runs "$P"                    # 3 b learning rates (validation) + 4 Gaussian lego seeds
  decide pick-lr --runs "$P"            # b learning rate -> runs/chosen.json
else
  frozen "$P"
  run pilot --runs "$P"
  run q1 --runs "$P"
fi
decide sigma --runs "$P"                # seed spread of the Gaussian on lego -> runs/sigma.json
run q2 --runs "$P"                      # the main comparison: 18 seed pairs (with q1)
run q3 --runs "$P" --b0-scenes chair drums ficus hotdog lego materials mic ship
                                        # frozen-b ablation, budget sweep, warm start, arm C
run q4 --runs "$P"                      # 25 and 50 training views
for scene in mic ship chair; do         # second frozen-b seed where seed 0 lost
  run one --arm B0 --scene "$scene" --seed 1 --runs "$P"
done

# ---- 2. Side checks: one setting changed, each in its own folder (72 runs)
S="${OUT}runs_init10k"; frozen "$S"     # full start: 10,000 starting splats, all 18 pairs
run q2 --runs "$S" --initial-splats 10000
for seed in 2 3; do
  run one --arm A --scene lego --seed "$seed" --runs "$S" --initial-splats 10000
done

S="${OUT}runs_30k"; frozen "$S"         # 30,000-splat budget
for scene in lego ficus; do for seed in 0 1; do for arm in A B; do
  run one --arm "$arm" --scene "$scene" --seed "$seed" --runs "$S" --capacity 30000
done; done; done

S="${OUT}runs_reset"; frozen "$S"       # opacity reset every 3,000 iterations
for seed in 0 1 2 3; do for arm in A B; do
  run one --arm "$arm" --scene lego --seed "$seed" --runs "$S" --opacity-reset-interval 3000
done; done

S="${OUT}runs_it40k"; frozen "$S"       # 40,000 iterations
for seed in 0 1; do for arm in A B; do
  run one --arm "$arm" --scene lego --seed "$seed" --runs "$S" --iterations 40000
done; done

for res in 200 800; do                  # 200 px and 800 px
  S="${OUT}runs_res${res}"; frozen "$S"
  for seed in 0 1; do for arm in A B; do
    run one --arm "$arm" --scene lego --seed "$seed" --runs "$S" --resolution "$res"
  done; done
done

for k in 1 2 3; do                      # the same seed run again, three times
  S="${OUT}runs_rep${k}"; frozen "$S"
  for arm in A B; do
    run one --arm "$arm" --scene lego --seed 0 --runs "$S"
  done
done

S="${OUT}runs_clamp4"; frozen "$S"      # b clamped to [-4, 4]
for scene in lego hotdog; do
  run one --arm B --scene "$scene" --seed 0 --runs "$S" --b-clamp 4
done

# ---- 3. Tables and figures
decide summary --runs "$P"              # runs/summary.csv
echo "Done. Figures: $PY scripts/report_figures.py (needs matplotlib)."
