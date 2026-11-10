"""Run driver for the CP4282 Beta-kernel study.

Every run is a directory under --runs (default runs/) holding config.yaml, train.log, final.ply
(+ final.beta.npz for Beta arms), final.runlog.json, test_metrics.csv, test_renders/ and a DONE
marker with summary metrics. Queues skip runs that already have DONE, so they are resumable.

Typical use on the GPU machine:

    python beta_runs.py q1 --data-root ~/data/nerf_synthetic        # lr sweep + 4 A lego seeds
    python beta_runs.py pick-lr && python beta_runs.py sigma         # freeze b lr and sigma
    python beta_runs.py q2 --data-root ~/data/nerf_synthetic        # core A/B x 8 scenes x 2 seeds
    python beta_runs.py q3 --data-root ~/data/nerf_synthetic        # B0, budget, warm start, C
    python beta_runs.py status
    python beta_runs.py summary                                      # runs/summary.csv

    python beta_runs.py one --arm B --scene lego --seed 0 --data-root ...   # a single run
    python beta_runs.py q1 --dry-run                                         # list, don't run

Arms:  A  Gaussian                       course densifier
       B  Beta, learnable b              course densifier
       B0 Beta, b frozen at 0            course densifier
       C  Beta, learnable b              kernel-aware densifier (rho(beta) split, c(beta) rank)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
LR_SWEEP = [0.0005, 0.001, 0.002]
PILOT_CAPACITIES = [10_000, 30_000, 100_000]
PILOT_PSNR_TOLERANCE_DB = 0.5   # budget: smallest capacity within this of the 100k run (A, sparse, val)
PILOT_TIMED_CAPACITY = 30_000   # dense (and B) timings are measured only at this capacity
GPU_HOURS_BUDGET = 12.0         # dense loss if q1+q2+q3 project to at most this, else sparse
DEFAULT_B_LR = 0.001
INITIAL_SPLATS = 1000  # random-init count, identical in every run (room to grow even at 25% budget)
DENSIFY_FRACTION = 0.2

ARMS = {
    "A": {"type": "gaussian"},
    "B": {"type": "beta", "learn_b": True, "densifier": "course"},
    "B0": {"type": "beta", "learn_b": False, "densifier": "course"},
    "C": {"type": "beta", "learn_b": True, "densifier": "kernel_aware"},
}


# --------------------------------------------------------------------------------------------
# Run specification


def run_name(spec: dict) -> str:
    if spec.get("pilot"):
        return f"pilot_{spec['arm']}_{spec['scene']}_c{spec['capacity']}_{spec['loss']}"
    name = f"{spec['arm']}_{spec['scene']}_s{spec['seed']}"
    if spec.get("budget", 1.0) != 1.0:
        name += f"_cap{int(round(spec['budget'] * 100))}"
    if spec.get("views", 100) != 100:
        name += f"_v{spec['views']}"
    if spec.get("lr_tag"):
        name += f"_lr{spec['b_lr']:g}"
    if spec.get("warm_from"):
        name += "_warm"
    return name


def read_chosen(runs: Path) -> dict:
    """runs/chosen.json: frozen protocol choices (capacity and loss from pick-pilot, b_lr from pick-lr)."""
    path = runs / "chosen.json"
    return json.loads(path.read_text()) if path.exists() else {}


def update_chosen(runs: Path, **values) -> Path:
    """Merge into chosen.json; each pick-* command owns its own keys and keeps the others."""
    chosen = read_chosen(runs)
    chosen.update(values)
    path = runs / "chosen.json"
    path.write_text(json.dumps(chosen, indent=1))
    return path


def chosen_b_lr(runs: Path) -> float:
    return float(read_chosen(runs).get("b_lr", DEFAULT_B_LR))


def resolve_protocol(spec: dict, args) -> tuple[int | None, str | None]:
    """(capacity, loss) for one run. A spec that names them (pilot runs) wins; otherwise
    --capacity / --loss, otherwise runs/chosen.json. None when nothing provides a value."""
    chosen = read_chosen(Path(args.runs))
    if "capacity" in spec:
        capacity = int(spec["capacity"])
    else:
        base = args.capacity if args.capacity is not None else chosen.get("capacity")
        capacity = None if base is None else int(round(int(base) * spec.get("budget", 1.0)))
    loss = spec.get("loss") or args.loss or chosen.get("loss")
    return capacity, loss


def make_config(spec: dict, run_dir: Path, args, capacity: int, loss: str) -> dict:
    scene_data = Path(args.data_root).expanduser().resolve() / spec["scene"]
    if not (scene_data / "transforms_train.json").exists():
        raise FileNotFoundError(f"No NeRF-synthetic scene at {scene_data}")
    if spec.get("views", 100) != 100:
        scene_data = subset_views(scene_data, run_dir / "data", spec["views"])
    if args.initial_splats > capacity:
        raise ValueError(f"{run_name(spec)}: initial_splats {args.initial_splats} exceeds capacity {capacity}")
    if loss not in ("sparse", "dense"):
        raise ValueError(f"loss must be 'sparse' or 'dense', got {loss!r}")
    kernel = {
        "type": "gaussian", "learn_b": True, "b_init": 0.0, "b_min": -2.0, "b_max": 2.0,
        "moment_match_init": True, "moment_match": args.moment_match,
        "densifier": "course", "moment_dim": 2,
    }
    kernel.update(ARMS[spec["arm"]])
    init_ply = None
    if spec.get("warm_from"):
        init_ply = str((Path(args.runs) / spec["warm_from"] / "final.ply").resolve())
    iterations = args.iterations
    return {
        "paths": {"data": str(scene_data), "output": str(run_dir / "final.png"),
                  "init_ply": init_ply},
        "runtime": {"arch": args.arch, "seed": int(spec["seed"])},
        "model": {"width": args.resolution, "height": args.resolution, "capacity": capacity,
                  "initial_splats": args.initial_splats, "background": "white"},
        "training": {"iterations": iterations},
        # Opacity reset off: with resets, every opacity is capped below late prune's cutoff.
        "densification": {"fraction": DENSIFY_FRACTION, "opacity_reset_interval": 0},
        # sparse: the course sparse MSE (sparse.samples_per_tile px per tile); dense: full-image MSE.
        "sparse": {"enabled": loss == "sparse"},
        "learning_rates": {"position_max_steps": iterations, "shape": float(spec["b_lr"])},
        "convergence": {"enabled": False},  # equal iterations across arms (protocol)
        "reporting": {"log_every": max(1, iterations // 100), "eval_every": max(1, iterations // 100),
                      "snapshot_every": iterations, "save_ply": True},
        "kernel": kernel,
    }


def subset_views(src: Path, dst: Path, views: int) -> Path:
    """A dataset dir whose transforms_train.json keeps `views` evenly spaced training frames.

    Every manifest is copied with absolute frame paths (resolve_frame_image accepts them), so the
    images stay in `src`. No symlinks: those need Developer Mode or admin rights on Windows."""
    src = src.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for manifest_path in sorted(src.glob("transforms_*.json")):
        manifest = json.loads(manifest_path.read_text())
        frames = manifest["frames"]
        if manifest_path.name == "transforms_train.json":
            keep = np.unique(np.round(np.linspace(0, len(frames) - 1, views)).astype(int))
            frames = [frames[i] for i in keep]
        manifest["frames"] = [{**f, "file_path": str((src / f["file_path"]).resolve())} for f in frames]
        (dst / manifest_path.name).write_text(json.dumps(manifest, indent=1))
    return dst


# --------------------------------------------------------------------------------------------
# Execution


def run(spec: dict, args) -> dict | None:
    runs = Path(args.runs)
    name = run_name(spec)
    run_dir = runs / name
    capacity, loss = resolve_protocol(spec, args)
    if (run_dir / "DONE").exists() and not args.force:
        done = json.loads((run_dir / "DONE").read_text())
        if (done.get("capacity"), done.get("loss")) != (capacity, loss):
            sys.exit(f"{name}: DONE has capacity={done.get('capacity')} loss={done.get('loss')} but this "
                     f"queue resolves to capacity={capacity} loss={loss}. Use a fresh --runs or --force.")
        return done
    if args.dry_run:
        print(f"[dry-run] {name}  capacity={capacity} loss={loss}  ({spec})")
        return None
    if capacity is None or loss is None:
        sys.exit(f"{name}: no capacity/loss. Run `pick-pilot` (writes {Path(args.runs) / 'chosen.json'}) "
                 "or pass --capacity and --loss.")
    if spec.get("warm_from") and not (runs / spec["warm_from"] / "DONE").exists():
        print(f"[skip] {name}: warm-start source {spec['warm_from']} not finished yet")
        return None
    if run_dir.exists():
        shutil.rmtree(run_dir)  # unfinished: start clean
    run_dir.mkdir(parents=True)
    config = make_config(spec, run_dir, args, capacity, loss)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "spec.json").write_text(json.dumps(spec))
    print(f"[train] {name}", flush=True)
    started = time.time()
    with (run_dir / "train.log").open("w") as log:
        code = subprocess.call([sys.executable, str(ROOT / "3dgs_trainer.py"), str(run_dir / "config.yaml")],
                               stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if code != 0:
        print(f"[FAIL] {name}: trainer exited {code}; see {run_dir / 'train.log'}")
        return None
    train_seconds = time.time() - started
    split = spec.get("eval_split", "test")
    started = time.time()
    metrics = evaluate(run_dir, Path(config["paths"]["data"]), split, args)
    if metrics is None:
        return None
    eval_seconds = time.time() - started
    runlog = json.loads((run_dir / "final.runlog.json").read_text())
    summary = {"name": name, **spec, "capacity": capacity, "loss": loss,
               "initial_splats": args.initial_splats, "split": split, **metrics,
               "train_seconds": train_seconds, "eval_seconds": eval_seconds, "ms_per_iteration": runlog["ms_per_iteration"],
               "peak_mempool_bytes": runlog["peak_mempool_bytes"], "final_active": runlog["final_active"],
               "b_summary": runlog.get("b_summary")}
    (run_dir / "DONE").write_text(json.dumps(summary, indent=1))
    print(f"[done] {name}: {split} psnr={metrics['psnr_db']:.3f} ssim={metrics['ssim']:.5f}"
          + (f" lpips={metrics['lpips']:.5f}" if "lpips" in metrics else "")
          + f" active={runlog['final_active']} {runlog['ms_per_iteration']:.1f} ms/it", flush=True)
    return summary


def evaluate(run_dir: Path, data: Path, split: str, args) -> dict | None:
    manifest = {"test": "transforms_test.json", "val": "transforms_val.json"}[split]
    metrics = ["psnr", "ssim"] + ([] if args.no_lpips else ["lpips"])
    cmd = [sys.executable, str(ROOT / "3dgs_evaluator.py"), str(run_dir / "final.ply"), str(data),
           "--manifest", manifest, "--resolution", str(args.resolution), "--metrics", *metrics,
           "--arch", "cpu" if args.arch == "cpu" else "gpu",
           "--output-dir", str(run_dir / f"{split}_renders"), "--csv", str(run_dir / f"{split}_metrics.csv")]
    if args.max_eval_views:
        cmd += ["--max-views", str(args.max_eval_views)]
    with (run_dir / f"eval_{split}.log").open("w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if code != 0:
        print(f"[FAIL] eval {run_dir.name}: exited {code}; see eval_{split}.log")
        return None
    with (run_dir / f"{split}_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for key in ("psnr_db", "ssim", "lpips", "mse"):
        if rows and key in rows[0]:
            out[key] = float(np.mean([float(r[key]) for r in rows]))
    out["eval_views"] = len(rows)
    return out


# --------------------------------------------------------------------------------------------
# Queues (see the execution plan)


def spec(arm, scene, seed, b_lr, **extra):
    return {"arm": arm, "scene": scene, "seed": seed, "b_lr": b_lr, **extra}


def queue(name: str, args) -> list[dict]:
    lr = chosen_b_lr(Path(args.runs))
    pair = args.sweep_scenes
    if name == "pilot":
        # Chooses capacity and loss (pick-pilot). Lego, seed 0, scored on VAL only so that no
        # protocol choice is made on test data.
        def p(arm, capacity, loss):
            return spec(arm, "lego", 0, lr, pilot=True, capacity=capacity, loss=loss, eval_split="val")
        return ([p("A", c, "sparse") for c in PILOT_CAPACITIES]
                + [p("A", 30_000, "dense"), p("B", 30_000, "sparse"), p("B", 30_000, "dense")])
    if name == "q1":
        return ([spec("B", "lego", 0, x, eval_split="val", lr_tag=True) for x in LR_SWEEP]
                + [spec("A", "lego", s, lr) for s in range(4)])
    if name == "q2":
        core = [spec(arm, scene, seed, lr) for scene in SCENES for seed in (0, 1) for arm in ("A", "B")]
        return core + [spec("B", "lego", s, lr) for s in (2, 3)]
    if name == "q3":
        out = [spec("B0", scene, 0, lr) for scene in args.b0_scenes]
        for scene in pair:
            for budget in (0.25, 0.5):
                out += [spec(arm, scene, 0, lr, budget=budget) for arm in ("A", "B", "B0")]
            out.append(spec("B", scene, 0, lr, budget=0.9))
        out += [spec("B0", scene, 0, lr, budget=0.25, warm_from=f"A_{scene}_s0_cap25") for scene in pair]
        out += [spec("C", scene, 0, lr) for scene in args.c_scenes]
        return out
    if name == "q4":
        return [spec(arm, scene, 0, lr, views=v) for scene in pair for v in (25, 50) for arm in ("A", "B")]
    raise ValueError(name)


# --------------------------------------------------------------------------------------------
# Decisions and summaries


def load_done(runs: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(runs.glob("*/DONE"))]


def pick_lr(args):
    runs = Path(args.runs)
    rows = [d for d in load_done(runs) if d.get("lr_tag")]
    if len(rows) < len(LR_SWEEP):
        sys.exit(f"Only {len(rows)}/{len(LR_SWEEP)} lr-sweep runs finished.")
    for d in sorted(rows, key=lambda d: d["b_lr"]):
        b = d.get("b_summary") or {}
        print(f"b_lr={d['b_lr']:<7g} val psnr={d['psnr_db']:.3f} ssim={d['ssim']:.5f} "
              f"b median={b.get('percentiles', [0] * 7)[3]:+.3f} at clamp={b.get('at_min', 0) + b.get('at_max', 0):.1%}")
    best = max(rows, key=lambda d: (round(d["ssim"], 4), d["psnr_db"]))
    path = update_chosen(runs, b_lr=best["b_lr"], b_lr_rule="max val SSIM (4 dp), then PSNR")
    print(f"chosen b_lr = {best['b_lr']} -> {path}")


def pick_pilot(args):
    runs = Path(args.runs)
    names = [run_name(s) for s in queue("pilot", args)]
    done = {n: json.loads((runs / n / "DONE").read_text()) for n in names if (runs / n / "DONE").exists()}
    missing = [n for n in names if n not in done]
    if missing:
        sys.exit(f"pilot runs not finished: {', '.join(missing)}")

    print(f"{'run':30s} {'cap':>7s} {'loss':6s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>7s} {'active':>7s} "
          f"{'ms/it':>6s} {'train':>6s} {'eval':>6s} {'mempool':>8s}")
    table = []
    for n in names:
        d = done[n]
        row = {k: d.get(k) for k in ("name", "arm", "capacity", "loss", "psnr_db", "ssim", "lpips", "final_active",
                                     "ms_per_iteration", "train_seconds", "eval_seconds", "eval_views",
                                     "peak_mempool_bytes")}
        table.append(row)
        lp = f"{row['lpips']:.4f}" if row.get("lpips") is not None else "-"
        mem = f"{row['peak_mempool_bytes'] / 2**20:.0f} MB" if row.get("peak_mempool_bytes") else "-"
        print(f"{n:30s} {row['capacity']:7d} {row['loss']:6s} {row['psnr_db']:7.3f} {row['ssim']:7.4f} {lp:>7s} "
              f"{row['final_active']:7d} {row['ms_per_iteration']:6.1f} {row['train_seconds'] / 60:5.1f}m "
              f"{row['eval_seconds'] / 60:5.1f}m {mem:>8s}")

    def get(arm, capacity, loss):
        return done[run_name({"pilot": True, "arm": arm, "scene": "lego", "capacity": capacity, "loss": loss})]

    # Budget: smallest capacity whose A sparse val PSNR is no more than the tolerance below 100k's.
    top = get("A", PILOT_CAPACITIES[-1], "sparse")["psnr_db"]
    capacity = next(c for c in PILOT_CAPACITIES
                    if get("A", c, "sparse")["psnr_db"] >= top - PILOT_PSNR_TOLERANCE_DB)
    print()
    print("budget: A sparse val PSNR " + ", ".join(
        f"{c // 1000}k {get('A', c, 'sparse')['psnr_db']:.3f}" for c in PILOT_CAPACITIES)
          + f" -> smallest within {PILOT_PSNR_TOLERANCE_DB} dB of 100k ({top - PILOT_PSNR_TOLERANCE_DB:.3f}): {capacity}")

    # Projected GPU time of q1+q2+q3 at that capacity, per loss mode. Every Beta arm (B, B0, C) is
    # timed by the B pilot. Timings measured at 30k are used unchanged for capacity <= 30k (an
    # overestimate at 10k) and scaled linearly by capacity / 30k at 100k. Budget-fraction runs in q3
    # are charged the full-capacity time. Eval time is charged per view of the run's split.
    split_views = {}
    for split in ("val", "test"):
        path = Path(args.data_root).expanduser() / "lego" / f"transforms_{split}.json"
        split_views[split] = len(json.loads(path.read_text())["frames"]) if path.exists() else {"val": 100, "test": 200}[split]

    def seconds(arm, loss, split):
        group = "A" if arm == "A" else "B"
        measured = capacity if (group == "A" and loss == "sparse") else PILOT_TIMED_CAPACITY
        d = get(group, measured, loss)
        scale = capacity / measured if capacity > measured else 1.0
        return scale * (d["train_seconds"] + d["eval_seconds"] / d["eval_views"] * split_views[split])

    queued = {q: queue(q, args) for q in ("q1", "q2", "q3")}
    projection = {}
    for loss in ("dense", "sparse"):
        per_queue = {q: sum(seconds(s["arm"], loss, s.get("eval_split", "test")) for s in specs) / 3600
                     for q, specs in queued.items()}
        projection[loss] = {"hours": sum(per_queue.values()), **{f"{q}_hours": h for q, h in per_queue.items()}}
    n_runs = sum(len(v) for v in queued.values())
    for loss in ("dense", "sparse"):
        pr = projection[loss]
        print(f"projected q1+q2+q3 ({n_runs} runs) @ {capacity}, {loss:6s}: {pr['hours']:5.1f} h  "
              + "  ".join(f"{q} {pr[f'{q}_hours']:.1f} h" for q in queued))
    loss = "dense" if projection["dense"]["hours"] <= GPU_HOURS_BUDGET else "sparse"
    print(f"loss: dense projects to {projection['dense']['hours']:.1f} h vs {GPU_HOURS_BUDGET:g} h limit -> {loss}")

    rule = (f"Budget = smallest of {PILOT_CAPACITIES} whose arm-A sparse-loss lego val PSNR is at least "
            f"(100k PSNR - {PILOT_PSNR_TOLERANCE_DB} dB). Loss = dense if the projected GPU time (train + eval) of "
            f"q1+q2+q3 at that budget is <= {GPU_HOURS_BUDGET:g} h, else sparse. Projection: per-run pilot "
            f"train time plus eval time per view x views in the run's split; A runs timed by pilot A, B/B0/C by "
            f"pilot B; dense and B timings come from {PILOT_TIMED_CAPACITY} and are used unscaled at or below "
            f"{PILOT_TIMED_CAPACITY} and scaled by capacity/{PILOT_TIMED_CAPACITY} above it; q3 budget-fraction "
            f"runs are charged full-capacity time.")
    path = update_chosen(runs, capacity=capacity, loss=loss, pilot_rule=rule, pilot_table=table,
                         pilot_projection={"capacity": capacity, "runs": n_runs, "split_views": split_views,
                                           **projection})
    print(f"chosen capacity = {capacity}, loss = {loss} -> {path}")


def sigma(args):
    runs = Path(args.runs)
    rows = [d for d in load_done(runs) if d["arm"] == "A" and d["scene"] == "lego"
            and d.get("budget", 1.0) == 1.0 and d.get("views", 100) == 100 and d["split"] == "test"]
    if len(rows) < 2:
        sys.exit("Need at least two finished A lego seeds.")
    out = {"seeds": [d["seed"] for d in rows], "n": len(rows)}
    for key in ("psnr_db", "ssim", "lpips"):
        if key in rows[0]:
            values = np.array([d[key] for d in rows])
            out[key] = {"mean": float(values.mean()), "sigma": float(values.std(ddof=1))}
    (runs / "sigma.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def status(args):
    runs = Path(args.runs)
    for q in ("pilot", "q1", "q2", "q3", "q4"):
        names = [run_name(s) for s in queue(q, args)]
        done = sum((runs / n / "DONE").exists() for n in names)
        print(f"{q}: {done}/{len(names)} done")
        for n in names:
            if not (runs / n / "DONE").exists() and (runs / n).exists():
                print(f"    started/failed: {n}")


def summary(args):
    runs = Path(args.runs)
    rows = load_done(runs)
    keys = ["name", "arm", "scene", "seed", "budget", "capacity", "loss", "views", "b_lr", "split",
            "psnr_db", "ssim", "lpips", "final_active", "ms_per_iteration", "peak_mempool_bytes",
            "train_seconds", "eval_seconds"]
    with (runs / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for d in rows:
            d.setdefault("budget", 1.0)
            d.setdefault("views", 100)
            writer.writerow(d)
    print(f"{len(rows)} runs -> {runs / 'summary.csv'}")


# --------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "q1", "q2", "q3", "q4", "one", "pick-pilot", "pick-lr",
                                        "sigma", "status", "summary"])
    ap.add_argument("--data-root", default=str(ROOT / "data"), help="Directory holding the 8 scene folders.")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--arch", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--resolution", type=int, default=400)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--capacity", type=int, default=None,
                    help="Splat budget at 100%% (all scenes). Default: runs/chosen.json (pick-pilot).")
    ap.add_argument("--initial-splats", type=int, default=INITIAL_SPLATS,
                    help="Random-init splat count (protocol: 1000 in every run).")
    ap.add_argument("--loss", default=None, choices=["sparse", "dense"],
                    help="Training loss. Default: runs/chosen.json (pick-pilot).")
    ap.add_argument("--moment-match", default="truncated", choices=["truncated", "untruncated"])
    ap.add_argument("--sweep-scenes", nargs="+", default=["lego", "ficus"])
    ap.add_argument("--b0-scenes", nargs="+", default=["lego", "ficus", "chair", "hotdog"])
    ap.add_argument("--c-scenes", nargs="+", default=["lego", "ficus"])
    ap.add_argument("--no-lpips", action="store_true", help="Skip LPIPS (no torch installed).")
    ap.add_argument("--max-eval-views", type=int, default=0, help="Smoke tests: evaluate only N views.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run even if DONE exists.")
    # `one`
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--views", type=int, default=100)
    ap.add_argument("--b-lr", type=float, default=None)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()
    # Absolute: config.yaml paths are resolved relative to the config file, so a relative --runs
    # would nest (runs/X/runs/X/final.ply).
    args.runs = str(Path(args.runs).expanduser().resolve())
    Path(args.runs).mkdir(parents=True, exist_ok=True)

    if args.command in ("pilot", "q1", "q2", "q3", "q4"):
        specs = queue(args.command, args)
        print(f"{args.command}: {len(specs)} runs")
        for s in specs:
            run(s, args)
    elif args.command == "one":
        b_lr = args.b_lr if args.b_lr is not None else chosen_b_lr(Path(args.runs))
        run(spec(args.arm, args.scene, args.seed, b_lr, budget=args.budget, views=args.views,
                 eval_split=args.split), args)
    elif args.command == "pick-pilot":
        pick_pilot(args)
    elif args.command == "pick-lr":
        pick_lr(args)
    elif args.command == "sigma":
        sigma(args)
    elif args.command == "status":
        status(args)
    elif args.command == "summary":
        summary(args)


if __name__ == "__main__":
    main()
