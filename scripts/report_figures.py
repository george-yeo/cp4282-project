"""Report figures for the Beta vs Gaussian comparison.

Reads the per-run results the trainer and evaluator already wrote (test_metrics.csv and
final.beta.npz in each run directory); nothing is trained or re-rendered. Every number is the
mean of the per-view test PSNR of a run, the same quantity as in ANALYSIS_BETA.md.

Figures (each written as PNG, PDF and a CSV of the plotted values):

  fig0_kernel_profiles   the Gaussian and the Beta kernel at b = -2, 0 and +2
  fig1_lead_by_scene     B - A per scene, protocol and full-capacity start, every seed pair
  fig2_ablation          B0 - A and B - A on the four ablation scenes (seed 0)
  fig3_lead_vs_budget    B - A and mean learned b against the splat budget, lego and ficus
  fig4_seed_spread_lego  every lego run under five training settings, both arms

Arms: A Gaussian, B Beta with learnable b, B0 Beta with b frozen at 0.

With --for-slides the figures are written to figures/slides as PNG only, without the title line
(the slide carries it) and cropped to their content. That mode also writes a protocol-only
version of fig1, the panels of the lego b-map without its small labels, three images that
introduce the kernel (its equations, its values on a pixel grid, and the footprint of one splat
at b = -2, 0 and +2) and, when --data-root is given, a three-panel floater example.

Needs matplotlib, which the trainer environment does not have. To keep that environment
unchanged, use a separate one that reuses its packages:

  python -m venv .venv-figs --system-site-packages
  .venv-figs/Scripts/python -m pip install matplotlib
  .venv-figs/Scripts/python scripts/report_figures.py

Colours are three slots of a palette checked for colour-blind separation on white; the two
series in each chart also differ in marker shape or position, and are named in a legend.
"""

import argparse
import csv
import logging
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
ZERO = "#8f8e88"
BLUE = "#2a78d6"    # Beta with learnable b; the protocol in fig 1
ORANGE = "#eb6834"  # Beta with b frozen at 0
AQUA = "#1baf7a"    # full-capacity start in fig 1
GREY = "#77766f"    # Gaussian baseline
BLUE_LIGHT = "#86b6ef"  # same hue as BLUE, for an ordered set of b values
BLUE_DARK = "#104281"

SLIDES = False  # set by --for-slides

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 9.5,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "legend.frameon": False,
})


# ---------------------------------------------------------------- data

def run_dir(root, arm, scene, seed, suffix=""):
    return os.path.join(root, f"{arm}_{scene}_s{seed}{suffix}")


def psnr(root, arm, scene, seed, suffix=""):
    """Mean per-view test PSNR of one run, or None if the run is missing."""
    path = os.path.join(run_dir(root, arm, scene, seed, suffix), "test_metrics.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        values = [float(r["psnr_db"]) for r in csv.DictReader(f)]
    return float(np.mean(values))


def mean_b(root, scene, seed, suffix=""):
    path = os.path.join(run_dir(root, "B", scene, seed, suffix), "final.beta.npz")
    if not os.path.exists(path):
        return None
    return float(np.load(path)["b"].mean())


def leads(root, scene, seeds, arm="B", suffix=""):
    """Paired differences arm - A for the seeds that have both runs."""
    out = []
    for s in seeds:
        a, b = psnr(root, "A", scene, s, suffix), psnr(root, arm, scene, s, suffix)
        if a is not None and b is not None:
            out.append(b - a)
    return out


def seeds_for(scene):
    return [0, 1, 2, 3] if scene == "lego" else [0, 1]


# ---------------------------------------------------------------- drawing helpers

def clean_axes(ax, grid="x"):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ringed(ax, x, y, color, marker="o", size=52, zorder=4):
    """A marker with a surface-coloured ring so overlapping marks stay separate."""
    ax.scatter(x, y, s=size, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.2,
               zorder=zorder)


def rounded_hbar(ax, y, value, height, color, radius_px=3.0):
    """Horizontal bar from 0 to value: square at the baseline, rounded at the data end."""
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    p0 = inv.transform((0, 0))
    p1 = inv.transform((radius_px * fig.dpi / 72.0, radius_px * fig.dpi / 72.0))
    rx = min(abs(p1[0] - p0[0]), abs(value))
    ry = min(abs(p1[1] - p0[1]), height / 2)
    sign = 1.0 if value >= 0 else -1.0
    lo, hi = y - height / 2, y + height / 2
    tip = value
    verts = [(0, lo), (tip - sign * rx, lo), (tip, lo), (tip, lo + ry), (tip, hi - ry), (tip, hi),
             (tip - sign * rx, hi), (0, hi), (0, lo)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
             Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def title(fig, x, y, text):
    """The figure's one-line title; left out on slides, where the slide heading carries it."""
    if not SLIDES:
        fig.text(x, y, text, fontsize=10.5, fontweight="semibold")


def save(fig, out, name, rows, header):
    os.makedirs(out, exist_ok=True)
    if SLIDES:
        fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight",
                    pad_inches=0.06)
        plt.close(fig)
        print("wrote", os.path.join(out, name + ".png"))
        return
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)
    with open(os.path.join(out, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", os.path.join(out, name + ".{png,pdf,csv}"))


def signed(x, digits=2):
    """Signed number with a true minus sign; a value that rounds to zero has no sign."""
    if round(x, digits) == 0:
        return f"{0:.{digits}f}"
    return f"{x:+.{digits}f}".replace("-", "−")


def db_ticks(ax, lo=-0.6, hi=1.0, step=0.2):
    ticks = np.arange(lo, hi + 1e-9, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if round(v, 1) == 0 else signed(v, 1) for v in ticks])


# ---------------------------------------------------------------- figure 0

def fig_kernel_profiles(args):
    r = np.linspace(0, 4.2, 841)
    q = r * r
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.15)
    clean_axes(ax, "y")
    rows = []
    curves = [("Gaussian", GREY, np.exp(-q / 2))]
    for b, color in ((-2, BLUE_LIGHT), (0, BLUE), (2, BLUE_DARK)):
        beta = 4 * np.exp(b)
        label = "Beta, b = " + ("0" if b == 0 else signed(b, 0))
        curves.append((label, color, np.where(q < 9, np.clip(1 - q / 9, 0, None) ** beta, 0.0)))
    for label, color, y in curves:
        ax.plot(r, y, color=color, linewidth=2, solid_capstyle="round", label=label)
        rows += [[label, f"{ri:.2f}", f"{yi:.4f}"] for ri, yi in zip(r[::20], y[::20])]
    # Beyond 3 sigma the Beta kernel is exactly zero; the Gaussian only gets small.
    ax.axvspan(3, 4.2, facecolor="#efeeea", edgecolor="none", zorder=0)
    ax.axvline(3, color=ZERO, linewidth=0.8)
    ax.text(3.07, 0.99, "Beyond 3\u03c3", fontsize=8.5, fontweight="semibold", color=INK, va="top")
    ax.text(3.07, 0.91, "Beta: exactly 0\nGaussian: still above 0", fontsize=8, color=INK,
            va="top", linespacing=1.35)
    inset = ax.inset_axes([3.32, 0.13, 0.82, 0.50], transform=ax.transData)
    for label, color, y in curves:
        inset.plot(r, y, color=color, linewidth=2, solid_capstyle="round")
    inset.axvline(3, color=ZERO, linewidth=0.8)
    inset.set_xlim(2.7, 4.2)
    inset.set_ylim(-0.0025, 0.045)
    inset.set_xticks([3, 4])
    inset.set_xticklabels(["3\u03c3", "4\u03c3"])
    inset.set_yticks([0, 0.02, 0.04])
    inset.set_yticklabels(["0", "0.02", "0.04"])
    inset.tick_params(labelsize=7)
    inset.set_title("The tail, zoomed 20\u00d7", fontsize=7.5, loc="left", color=INK_2, pad=3)
    inset.text(3.2, 0.013, "Gaussian", fontsize=7, color=INK_2)
    inset.text(3.72, 0.0035, "Beta", fontsize=7, color=INK_2)
    inset.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        inset.spines[side].set_color(GRID)
    ax.text(2.1, 0.80, "flatter, box-like", fontsize=8, color=INK_2)
    ax.text(0.42, 0.20, "sharper\npeak", fontsize=8, color=INK_2, ha="center")
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["0", "1\u03c3", "2\u03c3", "3\u03c3", "4\u03c3"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Distance from the splat centre")
    ax.set_ylabel("Alpha / opacity")
    fig.legend(loc="upper left", bbox_to_anchor=(0.09, 0.93), ncols=4, fontsize=8,
               handlelength=1.6, columnspacing=1.4, borderpad=0)
    title(fig, 0.10, 0.955, "One extra parameter per splat changes the kernel's profile")
    save(fig, args.out, "fig0_kernel_profiles", rows,
         ["curve", "distance_sigma", "alpha_over_opacity"])


def floater_example(args, view=84, seed=2):
    """Ground truth, Gaussian and Beta renders of one lego test view where a floater shows."""
    from PIL import Image
    gt = Image.open(os.path.join(args.data_root, "lego", "test", f"r_{view}.png")).convert("RGBA")
    gt = gt.resize((400, 400), Image.LANCZOS)
    white = Image.new("RGBA", gt.size, (255, 255, 255, 255))
    panels = [Image.alpha_composite(white, gt).convert("RGB")]
    for arm in ("A", "B"):
        path = os.path.join(run_dir(args.runs, arm, "lego", seed), "test_renders", f"r_{view}.png")
        panels.append(Image.open(path).convert("RGB"))
    top, bottom = 30, 340  # the rows the object occupies in this view
    sheet = Image.new("RGB", (3 * 400 + 2 * 16, bottom - top), (255, 255, 255))
    for i, im in enumerate(panels):
        sheet.paste(im.crop((0, top, 400, bottom)), (i * 416, 0))
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"floaters_lego_view{view}_seed{seed}.png")
    sheet.save(out)
    print("wrote", out)


def view_psnr(root, arm, scene, seed, view):
    """Test PSNR of one view of one run."""
    path = os.path.join(run_dir(root, arm, scene, seed), "test_metrics.csv")
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["view"]) == view:
                return float(r["psnr_db"])
    raise ValueError(f"{path}: no view {view}")


def render_examples(args, scenes=("lego", "ficus", "materials", "ship"), view=0, seed=0):
    """Ground truth, Gaussian and Beta renders of one test view for four scenes.

    Each scene's three pictures share one square crop around the object, taken from the alpha of
    the ground truth, so the object fills the panel. The PSNR under a render is that view's.
    """
    from PIL import Image
    cell, gap, group_gap, head, foot, row_gap = 2.3, 0.08, 0.6, 0.34, 0.38, 0.14  # inches
    cols = 2
    rows = (len(scenes) + cols - 1) // cols
    triplet = 3 * cell + 2 * gap
    width = cols * triplet + (cols - 1) * group_gap
    row_h = head + cell + foot
    height = rows * row_h + (rows - 1) * row_gap
    fig = plt.figure(figsize=(width, height))
    for i, scene in enumerate(scenes):
        gt = Image.open(os.path.join(args.data_root, scene, "test", f"r_{view}.png"))
        gt = gt.convert("RGBA").resize((400, 400), Image.LANCZOS)
        ys, xs = np.nonzero(np.asarray(gt)[..., 3] > 8)
        side = min(400, int(1.08 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 1)
        x0 = int(np.clip((xs.min() + xs.max()) / 2 - side / 2, 0, 400 - side))
        y0 = int(np.clip((ys.min() + ys.max()) / 2 - side / 2, 0, 400 - side))
        box = (x0, y0, x0 + side, y0 + side)
        white = Image.new("RGBA", gt.size, (255, 255, 255, 255))
        panels = [(Image.alpha_composite(white, gt).convert("RGB"), "Ground truth")]
        for arm, name in (("A", "Gaussian"), ("B", "Beta")):
            run = run_dir(args.runs, arm, scene, seed)
            im = Image.open(os.path.join(run, "test_renders", f"r_{view}.png")).convert("RGB")
            panels.append((im, f"{name} {view_psnr(args.runs, arm, scene, seed, view):.2f} dB"))
        left = (i % cols) * (triplet + group_gap)
        top = height - (i // cols) * (row_h + row_gap)
        fig.text(left / width, (top - 0.02) / height, scene.capitalize(), fontsize=17,
                 fontweight="semibold", va="top")
        for j, (im, label) in enumerate(panels):
            x = left + j * (cell + gap)
            ax = fig.add_axes([x / width, (top - head - cell) / height, cell / width,
                               cell / height])
            ax.imshow(np.asarray(im.crop(box)), interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for side_name in ("top", "right", "left", "bottom"):
                ax.spines[side_name].set_color(GRID)
            fig.text(x / width, (top - head - cell - 0.07) / height, label, fontsize=16.5,
                     color=INK, va="top")
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"renders_truth_gaussian_beta_view{view}_seed{seed}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", out)


def beta_kernel(q, b):
    """Beta kernel value at squared distance q (in sigma^2): (1 - q/9)^(4 e^b), zero from q = 9."""
    return np.where(q < 9, np.clip(1 - q / 9, 0, None) ** (4 * np.exp(b)), 0.0)


def kernel_equations(args):
    """The two kernels and their shared distance q, typeset."""
    fig = plt.figure(figsize=(4.2, 2.7))
    lines = [
        (0.90, "Gaussian", INK_2, None),
        (0.74, r"$\alpha = o \, e^{-q/2}$", INK, 17),
        (0.54, "Beta, with one learnable b per splat", "#1c5cab", None),
        (0.38, r"$\alpha = o \, (1 - q/9)^{\,4e^{b}}$", INK, 17),
        (0.18, "The same distance q in both", INK_2, None),
        (0.03, r"$q = (\mathbf{x} - \mu)^{T} \, \Sigma^{-1} \, (\mathbf{x} - \mu)$", INK, 15),
    ]
    for y, text, color, size in lines:
        fig.text(0.02, y, text, color=color, fontsize=size or 10,
                 fontweight="normal" if size else "semibold", va="bottom")
    save(fig, args.out, "kernel_equations", [], [])


def kernel_matrix(args):
    """Each kernel as a grid of numbers: alpha / opacity at pixel offsets of -3..3 sigma."""
    offsets = np.arange(-3, 4)
    xx, yy = np.meshgrid(offsets, offsets)
    q = (xx ** 2 + yy ** 2).astype(float)
    grids = [("Gaussian", np.exp(-q / 2)), ("Beta, b = 0", beta_kernel(q, 0.0))]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.6))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.10, wspace=0.16)
    blue = np.array(matplotlib.colors.to_rgb(BLUE))
    for ax, (name, grid) in zip(axes, grids):
        for i, y in enumerate(offsets):
            for j, x in enumerate(offsets):
                v = float(grid[i, j])
                face = 1 - (1 - blue) * 0.9 * v
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face,
                                       edgecolor=SURFACE, linewidth=1.5))
                if v == 0:
                    label, color = "0", "#9a9992"
                elif v < 0.005:
                    label, color = "\u22480", INK_2  # small but not zero
                else:
                    label, color = (f"{v:.2f}" if v < 0.995 else "1"), (SURFACE if v > 0.55 else INK)
                ax.text(x, y, label, ha="center", va="center", fontsize=8, color=color)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_xticks(offsets)
        ax.set_xticklabels([("0" if o == 0 else signed(o, 0) + "\u03c3") for o in offsets], fontsize=7)
        ax.set_yticks(offsets)
        ax.set_yticklabels([("0" if o == 0 else signed(o, 0) + "\u03c3") for o in offsets], fontsize=7)
        ax.set_title(name, loc="left", fontsize=9.5)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    save(fig, args.out, "kernel_matrix", [], [])


def kernel_footprints(args, size=280, extent=3.4):
    """One splat seen from above at b = -2, 0 and +2: white where alpha is 0, dark where it is 1."""
    from PIL import Image
    axis = np.linspace(-extent, extent, size)
    xx, yy = np.meshgrid(axis, axis)
    q = xx ** 2 + yy ** 2
    ink = np.array([0x14, 0x21, 0x3D], dtype=float)
    os.makedirs(args.out, exist_ok=True)
    for b, tag in ((-2, "bm2"), (0, "b0"), (2, "bp2")):
        alpha = beta_kernel(q, float(b))[..., None]
        rgb = 255 * (1 - alpha) + ink * alpha
        out = os.path.join(args.out, f"kernel_footprint_{tag}.png")
        Image.fromarray(rgb.round().astype(np.uint8)).save(out)
        print("wrote", out)


def bmap_panels(args, source=os.path.join("figures", "b_map_B_lego_s0.png")):
    """The six panels of the lego b-map (scripts/b_map.py) without its title, labels and scale."""
    from PIL import Image
    if not os.path.exists(source):
        print("skipped the b-map panels:", source, "not found")
        return
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "b_map_lego_panels.png")
    Image.open(source).convert("RGB").crop((24, 88, 1248, 890)).save(out)
    print("wrote", out)


# ---------------------------------------------------------------- figure 1

def fig_lead_by_scene(args, protocol_only=False):
    settings = [("Protocol (1,000 starting splats)", args.runs, BLUE, "o", +0.17),
                ("Full start (10,000 starting splats)", args.runs_full, AQUA, "D", -0.17)]
    if protocol_only:
        settings = [settings[0][:4] + (0.0,)]
    data = {(label, sc): leads(root, sc, seeds_for(sc)) for label, root, *_ in settings
            for sc in SCENES}
    order = sorted(SCENES, key=lambda sc: np.mean(data[(settings[0][0], sc)]))

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.79, bottom=0.12)
    clean_axes(ax, "x")
    ax.axvline(0, color=ZERO, linewidth=1.0, zorder=1)
    for i in range(len(order)):
        if i % 2 == 0:
            ax.add_patch(Rectangle((-10, i - 0.5), 20, 1, facecolor="#f7f7f5", edgecolor="none",
                                   zorder=0))

    rows = []
    for label, root, color, marker, dy in settings:
        every = []
        for i, sc in enumerate(order):
            d = data[(label, sc)]
            every += d
            y = i + dy
            ax.plot([min(d), max(d)], [y, y], color=color, linewidth=2, alpha=0.35,
                    solid_capstyle="round", zorder=2)
            ax.scatter(d, [y] * len(d), s=13, color=color, alpha=0.75, linewidths=0, zorder=3)
            ringed(ax, [np.mean(d)], [y], color, marker=marker,
                   size=58 if marker == "o" else 44)
            rows.append([label, sc, len(d), f"{np.mean(d):.3f}"] + [f"{v:.3f}" for v in d])
        print(f"fig1 {label}: {len(every)} pairs, mean {np.mean(every):+.3f} dB")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color=INK)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.6, 1.0)
    db_ticks(ax)
    ax.set_xlabel("Beta − Gaussian, test PSNR (dB)")
    side = ax.get_xaxis_transform()
    ax.text(-0.02, 1.005, "Gaussian better", transform=side, ha="right", va="bottom", fontsize=8,
            color=INK_2)
    ax.text(0.02, 1.005, "Beta better", transform=side, ha="left", va="bottom", fontsize=8,
            color=INK_2)

    handles = [Line2D([], [], marker=m, color="none", markerfacecolor=c, markeredgecolor=SURFACE,
                      markersize=7.5 if m == "o" else 6.5, label=l) for l, _, c, m, _ in settings]
    if len(settings) > 1:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.115, 0.895), ncols=2,
                   fontsize=8, handletextpad=0.3, columnspacing=1.4, borderpad=0)
    title(fig, 0.125, 0.955, "Same mean lead under both starts, but different scenes gain")
    fig.text(0.125, 0.86 if protocol_only else 0.912,
             "Large marker: mean. Small dots: single seed pairs (four on lego, "
             "two elsewhere).", fontsize=8, color=INK_2)
    save(fig, args.out, "fig1_lead_by_scene" + ("_protocol" if protocol_only else ""), rows,
         ["setting", "scene", "seed_pairs", "mean_lead_db", "lead_seed0", "lead_seed1",
          "lead_seed2", "lead_seed3"])


# ---------------------------------------------------------------- figure 2

def fig_ablation(args):
    scenes = ["lego", "ficus", "hotdog", "chair"]
    series = [("Kernel shape only (b frozen at 0)", "B0", ORANGE, +0.19),
              ("Learnable b", "B", BLUE, -0.19)]
    val = {(arm, sc): leads(args.runs, sc, [0], arm=arm)[0] for _, arm, *_ in series
           for sc in scenes}
    order = sorted(scenes, key=lambda sc: val[("B", sc)])

    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.16)
    clean_axes(ax, "x")
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    band = args.pair_noise
    ax.add_patch(Rectangle((-band, -0.6), 2 * band, len(order) + 0.2, facecolor="#efeeea",
                           edgecolor="none", zorder=0))
    ax.axvline(0, color=ZERO, linewidth=1.0, zorder=2)
    rows = []
    for label, arm, color, dy in series:
        for i, sc in enumerate(order):
            v = val[(arm, sc)]
            rounded_hbar(ax, i + dy, v, 0.30, color)
            ax.text(v + (0.02 if v >= 0 else -0.02), i + dy, signed(v), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=INK)
            rows.append([label, arm, sc, f"{v:.3f}"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color=INK)
    db_ticks(ax)
    ax.set_xlabel("Difference from the Gaussian, test PSNR (dB), seed 0")
    ax.text(band + 0.015, -0.55, f"grey band: ±{band:.2f} dB, the spread of one seed pair on lego",
            fontsize=7.5, color=INK_2, va="bottom")

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none", label=l)
               for l, _, c, _ in series]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.115, 0.89), ncols=2,
               fontsize=8, handlelength=1.0, handleheight=0.8, columnspacing=1.6, borderpad=0)
    title(fig, 0.125, 0.93, "Most of the gain comes from learning b, not from the kernel shape")
    save(fig, args.out, "fig2_ablation", rows, ["series", "arm", "scene", "difference_from_A_db"])


# ---------------------------------------------------------------- figure 3

def fig_lead_vs_budget(args):
    budgets = [(2500, args.runs, "_cap25", [0]), (5000, args.runs, "_cap50", [0]),
               (10000, args.runs, "", None), (30000, args.runs_30k, "", [0, 1])]
    scenes = ["lego", "ficus"]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.4), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.83, bottom=0.11, hspace=0.22, wspace=0.26)
    rows = []
    for col, sc in enumerate(scenes):
        xs, lead_mean, b_mean, lead_all, b_all = [], [], [], [], []
        for cap, root, suffix, seeds in budgets:
            seeds = seeds if seeds is not None else seeds_for(sc)
            d = leads(root, sc, seeds, suffix=suffix)
            bs = [v for v in (mean_b(root, sc, s, suffix) for s in seeds) if v is not None]
            xs.append(cap)
            lead_mean.append(np.mean(d))
            b_mean.append(np.mean(bs))
            lead_all.append(d)
            b_all.append(bs)
            rows.append([sc, cap, len(d), f"{np.mean(d):.3f}", f"{np.mean(bs):.3f}",
                         " ".join(f"{v:.3f}" for v in d), " ".join(f"{v:.3f}" for v in bs)])
        for row, (means, every, ylabel) in enumerate([
                (lead_mean, lead_all, "Beta − Gaussian (dB)"),
                (b_mean, b_all, "Mean learned b")]):
            ax = axes[row][col]
            clean_axes(ax, "y")
            ax.axhline(0, color=ZERO, linewidth=1.0, zorder=1)
            ax.plot(xs, means, color=BLUE, linewidth=2, solid_capstyle="round",
                    solid_joinstyle="round", zorder=2)
            for x, vals in zip(xs, every):
                if len(vals) > 1:
                    ax.scatter([x] * len(vals), vals, s=13, color=BLUE, alpha=0.6, linewidths=0,
                               zorder=3)
            ringed(ax, xs, means, BLUE, size=46)
            ax.set_xscale("log")
            ax.set_xlim(1900, 40000)
            ax.set_xticks(xs)
            ax.set_xticklabels(["2.5k", "5k", "10k", "30k"])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if row == 0:
                ax.set_title(sc, loc="left")
                ax.set_ylim(-0.1, 1.4)
                ax.set_yticks([0, 0.4, 0.8, 1.2])
                ax.set_yticklabels(["0", "+0.4", "+0.8", "+1.2"])
            else:
                ax.set_ylim(-0.45, 0.45)
                ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
                ax.set_yticklabels(["−0.4", "−0.2", "0", "+0.2", "+0.4"])
                ax.set_xlabel("Splat budget")
            if col == 0:
                ax.set_ylabel(ylabel)
        axes[1][col].text(38000, 0.37, "sharper peak", fontsize=7.5, color=INK_2, va="top",
                          ha="right")
        axes[1][col].text(38000, -0.37, "flatter, box-like", fontsize=7.5, color=INK_2,
                          va="bottom", ha="right")
    title(fig, 0.125, 0.955, "Beta's lead and the learned b against the splat budget")
    fig.text(0.125, 0.905,
             "Line: mean over seeds. Small dots: single seeds where there is more than one "
             "(one seed at 2.5k and 5k).",
             fontsize=8, color=INK_2)
    save(fig, args.out, "fig3_lead_vs_budget", rows,
         ["scene", "splat_budget", "seed_pairs", "mean_lead_db", "mean_b", "lead_per_seed",
          "mean_b_per_seed"])


# ---------------------------------------------------------------- figure 4

def fig_seed_spread(args):
    reps = [args.runs] + [f"{args.rep_prefix}{i}" for i in (1, 2, 3)]
    settings = [
        ("Protocol\nseeds 0–3", [(args.runs, s) for s in range(4)]),
        ("Protocol\nseed 0, run four times", [(r, 0) for r in reps]),
        ("10,000 starting splats\nseeds 0–3", [(args.runs_full, s) for s in range(4)]),
        ("Opacity reset every 3,000 it.\nseeds 0–3", [(args.runs_reset, s) for s in range(4)]),
        ("40,000 iterations\nseeds 0–1", [(args.runs_40k, s) for s in range(2)]),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    fig.subplots_adjust(left=0.275, right=0.84, top=0.80, bottom=0.13)
    clean_axes(ax, "x")
    n = len(settings)
    rows = []
    for i, (label, runs) in enumerate(settings):
        y0 = n - 1 - i
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y0 - 0.5), 100, 1, facecolor="#f7f7f5", edgecolor="none",
                                   zorder=0))
        means = {}
        for arm, color, dy in (("A", GREY, +0.17), ("B", BLUE, -0.17)):
            vals = [psnr(root, arm, "lego", s) for root, s in runs]
            vals = [v for v in vals if v is not None]
            means[arm] = np.mean(vals)
            y = y0 + dy
            ax.plot([means[arm]] * 2, [y - 0.2, y + 0.2], color=color, linewidth=2,
                    solid_capstyle="round", zorder=3)
            ringed(ax, vals, [y] * len(vals), color, size=34)
            rows.append([label.replace("\n", ", ").replace("–", "-"), arm, len(vals),
                         f"{means[arm]:.3f}",
                         f"{np.std(vals, ddof=1):.3f}"] + [f"{v:.3f}" for v in vals])
        ax.text(1.03, y0, signed(means["B"] - means["A"]) + " dB", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=9, color=INK)
    ax.text(1.03, n - 0.42, "Beta − Gaussian\n(of means)", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=8, color=INK_2, linespacing=1.15)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s[0] for s in settings][::-1], color=INK, linespacing=1.2)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(26.6, 28.8)
    ax.set_xlabel("Test PSNR on lego (dB), one dot per run")
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=c, markeredgecolor=SURFACE,
                      markersize=7, label=l) for l, c in (("Gaussian", GREY), ("Beta", BLUE))]
    handles.append(Line2D([], [], marker="|", color=INK_2, linestyle="none", markersize=9,
                          markeredgewidth=2, label="mean of the runs"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.905), ncols=3,
               fontsize=8, handletextpad=0.3, columnspacing=1.4, borderpad=0)
    title(fig, 0.04, 0.945, "Training settings change the level and the spread, not Beta's lead")
    save(fig, args.out, "fig4_seed_spread_lego", rows,
         ["setting", "arm", "runs", "mean_psnr_db", "sd_db", "run1", "run2", "run3", "run4"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="Protocol runs.")
    ap.add_argument("--runs-full", default="runs_init10k", help="Runs from 10,000 starting splats.")
    ap.add_argument("--runs-30k", default="runs_30k", help="Runs with a 30,000-splat budget.")
    ap.add_argument("--runs-reset", default="runs_reset", help="Lego runs with opacity reset.")
    ap.add_argument("--runs-40k", default="runs_it40k", help="Lego runs with 40,000 iterations.")
    ap.add_argument("--rep-prefix", default="runs_rep", help="Prefix of the same-seed repeats.")
    ap.add_argument("--pair-noise", type=float, default=0.16,
                    help="Spread (sd) of one paired difference on lego, drawn as a band in fig 2.")
    ap.add_argument("--out", default=None, help="Default: figures/report, or figures/slides.")
    ap.add_argument("--for-slides", action="store_true",
                    help="PNG only, no title line, cropped; also writes the slide-only figures.")
    ap.add_argument("--data-root", default=None,
                    help="NeRF-synthetic folder; only needed for the slide images that show renders.")
    args = ap.parse_args()
    global SLIDES
    SLIDES = args.for_slides
    if args.out is None:
        args.out = os.path.join("figures", "slides" if SLIDES else "report")
    fig_kernel_profiles(args)
    if SLIDES:
        fig_lead_by_scene(args, protocol_only=True)
        bmap_panels(args)
        kernel_equations(args)
        kernel_matrix(args)
        kernel_footprints(args)
        if args.data_root:
            floater_example(args)
            render_examples(args)
    fig_lead_by_scene(args)
    fig_ablation(args)
    fig_lead_vs_budget(args)
    fig_seed_spread(args)


if __name__ == "__main__":
    main()
