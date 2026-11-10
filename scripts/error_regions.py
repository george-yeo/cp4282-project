"""Where in the image does Beta gain over the Gaussian? Splits test-view error by image region.

Uses the renders the evaluator already wrote (runs/<run>/test_renders); nothing is re-rendered.
Each test pixel falls in one of four regions, defined from the ground truth only:

  background   more than 2 px outside the object's silhouette
  silhouette   within 2 px of the silhouette (either side)
  detail       inside the object, top quarter of that view's image-gradient magnitude
  smooth       inside the object, the other three quarters

For each region the script reports its share of the pixels, its share of arm A's squared error,
and how much arm B changes the total squared error through that region. The last column sums to
B's overall change in MSE, so it says where the gain (or loss) comes from.

Usage:
  python scripts/error_regions.py --data-root /path/to/nerf_synthetic [--runs runs] [--csv out.csv]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ("background", "silhouette", "detail", "smooth")
BAND = 2  # px


def grow(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2r+1)^2 square."""
    out = mask.copy()
    padded = np.pad(mask, radius)
    h, w = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def regions_of(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ground truth on white, region index per pixel) for one resized RGBA frame in [0, 1]."""
    alpha = rgba[..., 3]
    truth = rgba[..., :3] * alpha[..., None] + (1.0 - alpha[..., None])
    inside = alpha > 0.5
    outer, inner = grow(inside, BAND), ~grow(~inside, BAND)
    luminance = truth @ np.array([0.299, 0.587, 0.114], np.float32)
    gy, gx = np.gradient(luminance)
    gradient = np.hypot(gx, gy)
    region = np.full(alpha.shape, REGIONS.index("silhouette"), np.int8)
    region[~outer] = REGIONS.index("background")
    region[inner] = REGIONS.index("smooth")
    if inner.any():
        region[inner & (gradient >= np.quantile(gradient[inner], 0.75))] = REGIONS.index("detail")
    return truth, region


def load_render(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--scenes", nargs="+",
                    default=["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--arms", nargs=2, default=["A", "B"], help="Baseline arm, then the arm compared to it.")
    ap.add_argument("--resolution", type=int, default=400)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    runs, data_root = Path(args.runs), Path(args.data_root).expanduser().resolve()
    base, other = args.arms

    rows = []
    for scene in args.scenes:
        manifest = json.loads((data_root / scene / "transforms_test.json").read_text())
        stems = [Path(f["file_path"]).stem for f in manifest["frames"]]
        pairs = [(runs / f"{base}_{scene}_s{s}", runs / f"{other}_{scene}_s{s}") for s in args.seeds]
        pairs = [p for p in pairs if all((d / "DONE").exists() for d in p)]
        if not pairs:
            print(f"{scene}: no finished {base}/{other} pair, skipped")
            continue
        pixels = np.zeros(len(REGIONS))
        sse = np.zeros((2, len(REGIONS)))
        for stem in stems:
            frame = Image.open(data_root / scene / "test" / f"{stem}.png").convert("RGBA")
            rgba = np.asarray(frame.resize((args.resolution, args.resolution), Image.LANCZOS), np.float32) / 255.0
            truth, region = regions_of(rgba)
            for k in range(len(REGIONS)):
                pixels[k] += (region == k).sum() * len(pairs)
            for pair in pairs:
                for arm, run in enumerate(pair):
                    error = ((load_render(run / "test_renders" / f"{stem}.png") - truth) ** 2).sum(-1)
                    sse[arm] += np.bincount(region.ravel(), weights=error.ravel(), minlength=len(REGIONS))
        total = sse[0].sum()
        print(f"\n{scene} ({len(pairs)} seed pair(s), {len(stems)} views): "
              f"{other} changes total squared error by {(sse[1].sum() - total) / total:+.1%}")
        print(f"  {'region':11s} {'pixels':>7s} {base + ' error':>8s} {'change in region':>17s} {'of total error':>15s}")
        for k, name in enumerate(REGIONS):
            row = {"scene": scene, "region": name, "pixel_share": pixels[k] / pixels.sum(),
                   "base_error_share": sse[0, k] / total,
                   "change_within_region": (sse[1, k] - sse[0, k]) / sse[0, k],
                   "change_of_total": (sse[1, k] - sse[0, k]) / total}
            rows.append(row)
            print(f"  {name:11s} {row['pixel_share']:7.1%} {row['base_error_share']:8.1%} "
                  f"{row['change_within_region']:+17.1%} {row['change_of_total']:+15.1%}")

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
