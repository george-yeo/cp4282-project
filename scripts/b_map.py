"""Where do sharp and flat Beta splats sit? Renders a per-pixel map of the learned b.

For a finished Beta run (a directory with final.ply and final.beta.npz) this writes one figure
per run: ground truth | the model's render | b-map, for a few held-out views.

How the b-map is made, without touching the trainer or its kernels:
  1. Copy final.ply with every splat's colour replaced by (t, 1, 0), where t maps b linearly from
     [-range, +range] onto [0, 1]. Geometry, opacity and the b sidecar are unchanged.
  2. Render that copy with 3dgs_evaluator.py on a black background. The red channel is then
     sum(w_i * t_i) and the green channel is sum(w_i), with w_i each splat's blending weight.
  3. red / green is the visibility-weighted mean b at that pixel. It is coloured with a diverging
     scale (blue = flat, grey = 0, red = sharp) and composited over white with green as alpha.

Usage:
  python scripts/b_map.py runs/B_lego_s0 runs/B_ficus_s0 --data-root /path/to/nerf_synthetic
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
C0 = 0.28209479177387814  # same constant as shared/gaussian_set.py

# Diverging scale: two poles and a neutral grey midpoint. The midpoint is darker than the page so
# that b = 0 regions still show the object's silhouette on white.
STOPS = [(0.00, (0x18, 0x4F, 0x95)), (0.25, (0x2A, 0x78, 0xD6)), (0.50, (0xC9, 0xC8, 0xC2)),
         (0.75, (0xD0, 0x3B, 0x3B)), (1.00, (0x8F, 0x1F, 0x1F))]
INK, INK_MUTED, PAGE = (0x0B, 0x0B, 0x0B), (0x52, 0x51, 0x4E), (255, 255, 255)


def colour_scale(t: np.ndarray) -> np.ndarray:
    """t in [0, 1] -> float RGB in [0, 1], piecewise linear through STOPS."""
    positions = np.array([p for p, _ in STOPS])
    colours = np.array([c for _, c in STOPS], np.float32) / 255.0
    return np.stack([np.interp(t, positions, colours[:, k]) for k in range(3)], axis=-1)


def font(size: int, bold: bool = False):
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def write_encoded_model(run: Path, work: Path, b_range: float) -> np.ndarray:
    """work/bmap.ply (+ .beta.npz): the run's splats with colour (t, 1, 0). Returns b."""
    with np.load(run / "final.beta.npz") as sidecar:
        b = np.asarray(sidecar["b"], np.float32)
    ply = PlyData.read(str(run / "final.ply"))
    vertex = ply["vertex"].data
    if len(vertex) != len(b):
        raise ValueError(f"{run}: {len(vertex)} splats in final.ply but {len(b)} b values.")
    t = np.clip((b + b_range) / (2.0 * b_range), 0.0, 1.0)
    vertex["f_dc_0"] = (t - 0.5) / C0
    vertex["f_dc_1"] = (1.0 - 0.5) / C0
    vertex["f_dc_2"] = (0.0 - 0.5) / C0
    ply.write(str(work / "bmap.ply"))
    shutil.copyfile(run / "final.beta.npz", work / "bmap.beta.npz")

    scales = np.exp(np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))
    opacity = 1.0 / (1.0 + np.exp(-vertex["opacity"].astype(np.float64)))
    print(f"  b: mean {b.mean():+.2f}, std {b.std():.2f}; "
          f"{(b > 0.5).mean():.0%} above +0.5, {(b < -0.5).mean():.0%} below -0.5")
    print(f"  Spearman of b with: largest scale {spearman(b, scales.max(1)):+.2f}, "
          f"anisotropy (largest/smallest scale) {spearman(b, scales.max(1) / scales.min(1)):+.2f}, "
          f"opacity {spearman(b, opacity):+.2f}")
    return b


def render_encoded(work: Path, scene: Path, views: list[int], resolution: int, arch: str) -> list[str]:
    """Render work/bmap.ply for the chosen test views. Returns the frame stems, in order."""
    manifest = json.loads((scene / "transforms_test.json").read_text())
    frames = [manifest["frames"][i] for i in views]
    manifest["frames"] = [{**f, "file_path": str((scene / f["file_path"]).resolve())} for f in frames]
    data = work / "data"
    data.mkdir(exist_ok=True)
    (data / "transforms_bmap.json").write_text(json.dumps(manifest, indent=1))
    cmd = [sys.executable, str(ROOT / "3dgs_evaluator.py"), str(work / "bmap.ply"), str(data),
           "--manifest", "transforms_bmap.json", "--resolution", str(resolution), "--metrics", "psnr",
           "--arch", arch, "--background", "black",
           "--output-dir", str(work / "renders"), "--csv", str(work / "metrics.csv")]
    with (work / "render.log").open("w") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if code != 0:
        sys.exit(f"evaluator exited {code}; see {work / 'render.log'}")
    return [Path(f["file_path"]).stem for f in frames]


def decode(encoded: Path, min_alpha: float = 0.02) -> Image.Image:
    """Encoded render -> coloured b-map on white."""
    pixels = np.asarray(Image.open(encoded).convert("RGB"), np.float32) / 255.0
    weighted_t, alpha = pixels[..., 0], pixels[..., 1]
    t = np.where(alpha > min_alpha, weighted_t / np.maximum(alpha, 1e-6), 0.5)
    colour = colour_scale(np.clip(t, 0.0, 1.0))
    alpha = np.where(alpha > min_alpha, alpha, 0.0)[..., None]
    return Image.fromarray(np.uint8((colour * alpha + (1.0 - alpha)) * 255.0 + 0.5))


def ground_truth(scene: Path, stem: str, resolution: int) -> Image.Image:
    image = Image.open(scene / "test" / f"{stem}.png").convert("RGBA")
    image = image.resize((resolution, resolution), Image.LANCZOS)
    page = Image.new("RGBA", image.size, PAGE + (255,))
    return Image.alpha_composite(page, image).convert("RGB")


def compose(rows: list[list[Image.Image]], title: str, b_range: float, out: Path) -> None:
    size, gap, margin = rows[0][0].width, 12, 24
    header, legend = 78, 74
    width = 2 * margin + 3 * size + 2 * gap
    height = header + len(rows) * size + (len(rows) - 1) * gap + legend + margin
    page = Image.new("RGB", (width, height), PAGE)
    draw = ImageDraw.Draw(page)
    draw.text((margin, 18), title, fill=INK, font=font(20, bold=True))
    for column, label in enumerate(("Ground truth", "Beta render", "Learned b (visibility-weighted mean)")):
        draw.text((margin + column * (size + gap), header - 24), label, fill=INK_MUTED, font=font(14))
    for r, row in enumerate(rows):
        for column, panel in enumerate(row):
            page.paste(panel, (margin + column * (size + gap), header + r * (size + gap)))

    # Colour bar under the b-map column.
    bar_x, bar_y = margin + 2 * (size + gap), height - margin - legend + 22
    bar = colour_scale(np.linspace(0.0, 1.0, size))[None].repeat(12, axis=0)
    page.paste(Image.fromarray(np.uint8(bar * 255.0 + 0.5)), (bar_x, bar_y))
    small = font(13)
    for value in (-b_range, 0.0, b_range):
        x = bar_x + (value + b_range) / (2 * b_range) * (size - 1)
        draw.line([(x, bar_y + 12), (x, bar_y + 17)], fill=INK_MUTED, width=1)
        text = f"{value:+.1f}" if value else "0"
        draw.text((x, bar_y + 20), text, fill=INK, font=small,
                  anchor="la" if value < 0 else "ra" if value > 0 else "ma")
    draw.text((bar_x, bar_y + 38), "flatter, box-like", fill=INK_MUTED, font=small, anchor="la")
    draw.text((bar_x + size - 1, bar_y + 38), "sharper peak", fill=INK_MUTED, font=small, anchor="ra")
    draw.text((bar_x + size / 2, bar_y - 18), "b", fill=INK, font=small, anchor="ma")
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="Finished Beta run directories.")
    ap.add_argument("--data-root", default=str(ROOT / "data"), help="Directory holding the scene folders.")
    ap.add_argument("--views", nargs="+", type=int, default=[0, 60], help="Indices into transforms_test.json.")
    ap.add_argument("--b-range", type=float, default=1.5,
                    help="b is mapped from [-range, +range]; values beyond saturate.")
    ap.add_argument("--resolution", type=int, default=400)
    ap.add_argument("--arch", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()

    for run in args.runs:
        run = run.expanduser().resolve()
        spec = json.loads((run / "spec.json").read_text())
        scene = Path(args.data_root).expanduser().resolve() / spec["scene"]
        work = args.out.resolve() / "_work" / run.name
        work.mkdir(parents=True, exist_ok=True)
        print(run.name)
        write_encoded_model(run, work, args.b_range)
        stems = render_encoded(work, scene, args.views, args.resolution, args.arch)
        rows = [[ground_truth(scene, stem, args.resolution),
                 Image.open(run / "test_renders" / f"{stem}.png").convert("RGB"),
                 decode(work / "renders" / f"{stem}.png")] for stem in stems]
        done = json.loads((run / "DONE").read_text())
        title = (f"{spec['scene']}: where the learned b sits "
                 f"({done['final_active']:,} splats, {done['psnr_db']:.2f} dB)")
        compose(rows, title, args.b_range, args.out.resolve() / f"b_map_{run.name}.png")


if __name__ == "__main__":
    main()
