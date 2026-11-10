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


def chosen_b_lr(runs: Path) -> float:
    path = runs / "chosen.json"
    if path.exists():
        return float(json.loads(path.read_text())["b_lr"])
    return DEFAULT_B_LR


def make_config(spec: dict, run_dir: Path, args) -> dict:
    scene_data = Path(args.data_root).expanduser().resolve() / spec["scene"]
    if not (scene_data / "transforms_train.json").exists():
        raise FileNotFoundError(f"No NeRF-synthetic scene at {scene_data}")
    if spec.get("views", 100) != 100:
        scene_data = subset_views(scene_data, run_dir / "data", spec["views"])
    capacity = int(round(args.capacity * spec.get("budget", 1.0)))
    if args.initial_splats > capacity:
        raise ValueError(f"{run_name(spec)}: initial_splats {args.initial_splats} exceeds capacity {capacity}")
    if args.loss not in ("sparse", "dense"):
        raise ValueError(f"loss must be 'sparse' or 'dense', got {args.loss!r}")
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
        "sparse": {"enabled": args.loss == "sparse"},
        "learning_rates": {"position_max_steps": iterations, "shape": float(spec["b_lr"])},
        "convergence": {"enabled": False},  # equal iterations across arms (protocol)
        "reporting": {"log_every": max(1, iterations // 100), "eval_every": max(1, iterations // 100),
                      "snapshot_every": iterations, "save_ply": True},
        "kernel": kernel,
    }


def subset_views(src: Path, dst: Path, views: int) -> Path:
    """A dataset dir whose transforms_train.json keeps `views` evenly spaced training frames."""
    dst.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "transforms_train.json").read_text())
    frames = manifest["frames"]
    keep = np.unique(np.round(np.linspace(0, len(frames) - 1, views)).astype(int))
    manifest["frames"] = [frames[i] for i in keep]
    (dst / "transforms_train.json").write_text(json.dumps(manifest, indent=1))
    for item in src.iterdir():
        target = dst / item.name
        if item.name == "transforms_train.json" or target.exists():
            continue
        target.symlink_to(item.resolve())
    return dst


# --------------------------------------------------------------------------------------------
# Execution


def run(spec: dict, args) -> dict | None:
    runs = Path(args.runs)
    name = run_name(spec)
    run_dir = runs / name
    if (run_dir / "DONE").exists() and not args.force:
        return json.loads((run_dir / "DONE").read_text())
    if args.dry_run:
        print(f"[dry-run] {name}  ({spec})")
        return None
    if spec.get("warm_from") and not (runs / spec["warm_from"] / "DONE").exists():
        print(f"[skip] {name}: warm-start source {spec['warm_from']} not finished yet")
        return None
    if run_dir.exists():
        shutil.rmtree(run_dir)  # unfinished: start clean
    run_dir.mkdir(parents=True)
    config = make_config(spec, run_dir, args)
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
    metrics = evaluate(run_dir, Path(config["paths"]["data"]), split, args)
    if metrics is None:
        return None
    runlog = json.loads((run_dir / "final.runlog.json").read_text())
    summary = {"name": name, **spec, "split": split, **metrics,
               "train_seconds": train_seconds, "ms_per_iteration": runlog["ms_per_iteration"],
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
    (runs / "chosen.json").write_text(json.dumps({"b_lr": best["b_lr"], "rule": "max val SSIM (4 dp), then PSNR"}))
    print(f"chosen b_lr = {best['b_lr']} -> {runs / 'chosen.json'}")


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
    for q in ("q1", "q2", "q3", "q4"):
        names = [run_name(s) for s in queue(q, args)]
        done = sum((runs / n / "DONE").exists() for n in names)
        print(f"{q}: {done}/{len(names)} done")
        for n in names:
            if not (runs / n / "DONE").exists() and (runs / n).exists():
                print(f"    started/failed: {n}")


def summary(args):
    runs = Path(args.runs)
    rows = load_done(runs)
    keys = ["name", "arm", "scene", "seed", "budget", "views", "b_lr", "split", "psnr_db", "ssim",
            "lpips", "final_active", "ms_per_iteration", "peak_mempool_bytes", "train_seconds"]
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
    ap.add_argument("command", choices=["q1", "q2", "q3", "q4", "one", "pick-lr", "sigma", "status", "summary"])
    ap.add_argument("--data-root", default=str(ROOT / "data"), help="Directory holding the 8 scene folders.")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--arch", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--resolution", type=int, default=400)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--capacity", type=int, default=100_000, help="Splat budget at 100%% (all scenes).")
    ap.add_argument("--initial-splats", type=int, default=INITIAL_SPLATS,
                    help="Random-init splat count (protocol: 1000 in every run).")
    ap.add_argument("--loss", default="sparse", choices=["sparse", "dense"])
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
    Path(args.runs).mkdir(parents=True, exist_ok=True)

    if args.command in ("q1", "q2", "q3", "q4"):
        specs = queue(args.command, args)
        print(f"{args.command}: {len(specs)} runs")
        for s in specs:
            run(s, args)
    elif args.command == "one":
        b_lr = args.b_lr if args.b_lr is not None else chosen_b_lr(Path(args.runs))
        run(spec(args.arm, args.scene, args.seed, b_lr, budget=args.budget, views=args.views,
                 eval_split=args.split), args)
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
