"""Report figures for the Beta vs Gaussian comparison.

Reads the per-run results the trainer and evaluator already wrote (test_metrics.csv and
final.beta.npz in each run directory); nothing is trained or re-rendered. Every number is the
mean of the per-view test PSNR of a run, the same quantity as in ANALYSIS_BETA.md.

Figures (each written as PNG, PDF and a CSV of the plotted values):

  fig1_lead_by_scene     B - A per scene, protocol and full-capacity start, every seed pair
  fig2_ablation          B0 - A and B - A on the four ablation scenes (seed 0)
  fig3_lead_vs_budget    B - A and mean learned b against the splat budget, lego and ficus
  fig4_seed_spread_lego  every lego run under five training settings, both arms

Arms: A Gaussian, B Beta with learnable b, B0 Beta with b frozen at 0.

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


def save(fig, out, name, rows, header):
    os.makedirs(out, exist_ok=True)
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


# ---------------------------------------------------------------- figure 1

def fig_lead_by_scene(args):
    settings = [("Protocol (1,000 starting splats)", args.runs, BLUE, "o", +0.17),
                ("Full start (10,000 starting splats)", args.runs_full, AQUA, "D", -0.17)]
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
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.115, 0.895), ncols=2,
               fontsize=8, handletextpad=0.3, columnspacing=1.4, borderpad=0)
    fig.text(0.125, 0.955, "Same mean lead under both starts, but different scenes gain",
             fontsize=10.5, fontweight="semibold")
    fig.text(0.125, 0.912, "Large marker: mean. Small dots: single seed pairs (four on lego, "
             "two elsewhere).", fontsize=8, color=INK_2)
    save(fig, args.out, "fig1_lead_by_scene", rows,
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
    fig.text(0.125, 0.93, "Most of the gain comes from learning b, not from the kernel shape",
             fontsize=10.5, fontweight="semibold")
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
    fig.text(0.125, 0.955, "Beta's lead and the learned b against the splat budget",
             fontsize=10.5, fontweight="semibold")
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
    fig.text(0.04, 0.945, "Training settings change the level and the spread, not Beta's lead",
             fontsize=10.5, fontweight="semibold")
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
    ap.add_argument("--out", default=os.path.join("figures", "report"))
    args = ap.parse_args()
    fig_lead_by_scene(args)
    fig_ablation(args)
    fig_lead_vs_budget(args)
    fig_seed_spread(args)


if __name__ == "__main__":
    main()
