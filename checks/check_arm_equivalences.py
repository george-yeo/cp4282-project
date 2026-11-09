"""Gates G2 and G3.

G3: arm C (kernel-aware densifier) must equal arm B (course densifier) when every b = 0, since
    rho(4) = c(4) = 1. Trains both with b frozen at 0 through several densify events and requires
    identical parameters, active masks and densify logs.
G2: render an A (Gaussian) model as Beta b=0, (a) with the same scales, (b) with scales multiplied
    by the moment-matching factor sqrt(4/3). (b) should be close to the Gaussian render; (a) visibly
    thinner. Pass --ply to run this on a trained A model instead of the lego init cloud.

    python checks/check_arm_equivalences.py [--ply outputs/A_lego_s0/final.ply] [--device cuda:0]
"""
import argparse, importlib, sys
from pathlib import Path
import numpy as np
import warp as wp

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "shared")]
T = importlib.import_module("3dgs_trainer")
from gaussian_set import GaussianSet

BETA = {"type": "beta", "learn_b": False, "b_init": 0.0, "b_min": -2.0, "b_max": 2.0,
        "moment_match_init": True, "densifier": "course", "moment_dim": 2}


def load(res, n_views=6):
    bg = np.ones(3, np.float32)
    targets, cameras, focal, distance, _ = T.load_views(root / "data/lego", res, res, bg)
    return targets[:n_views], cameras[:n_views], focal, distance * 0.35, bg


def check_c_equals_b(device, res=32, iters=300):
    targets, cameras, focal, radius, bg = load(res)
    scene = GaussianSet.from_ply(str(root / "data/lego/init.ply"))
    sub = slice(0, 3000)
    scene = GaussianSet(scene.means[sub], scene.scales[sub], scene.rotations[sub], scene.opacities[sub], scene.colors[sub])
    states = []
    for densifier in ("course", "kernel_aware"):
        kernel = dict(BETA, densifier=densifier)
        tr = T.WarpImageTrainer(targets, cameras, focal, radius, 6000, 1, device, 7, init_scene=scene,
                                background=bg, training_options={"kernel": kernel,
                                                                 "multiview_adc": {"views": 3}})
        rng = np.random.default_rng(7)
        for it in range(1, iters + 1):
            view = np.asarray([rng.integers(len(targets))], np.int32)
            dens = it % 100 == 0
            _, g = tr.step(view, download_gradients=dens, iteration=it)
            if dens:
                tr.densify_and_prune(g, 0.2, view_ids=rng.choice(len(targets), 3, replace=False))
        states.append((tr.means.numpy(), tr.log_scales.numpy(), tr.opacity.numpy(), tr.active.copy(),
                       tr.densify_log))
    (mb, sb, ob, ab, lb), (mc, sc, oc, ac, lc) = states
    assert np.array_equal(ab, ac) and np.array_equal(mb, mc) and np.array_equal(sb, sc) and np.array_equal(ob, oc)
    assert lb == lc
    print(f"G3 C == B at b=0: identical after {iters} iters, {len(lb)} densify events, "
          f"active {int(ab.sum())} ({sum(e['split'] for e in lb)} splits, {sum(e['clone'] for e in lb)} clones)  PASS")


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12))


def check_moment_match(device, ply, res=128, views=(0, 1, 2)):
    targets, cameras, focal, radius, bg = load(res)
    scene = GaussianSet.from_ply(str(ply))
    cap = len(scene.means)
    def render(kernel, scales):
        s = GaussianSet(scene.means, scales, scene.rotations, scene.opacities, scene.colors)
        opts = {"kernel": kernel} if kernel else {}
        tr = T.WarpImageTrainer(targets, cameras, focal, radius, cap, 1, device, 7, init_scene=s,
                                background=bg, training_options=opts)
        # .copy(): on CPU, .numpy() aliases the render buffer, so the list would hold N copies
        # of the last view.
        return [tr.render_single_view_image(v)[0].copy() for v in views]
    compact = T.TrainingConfig.DEFAULTS["compact_box"]
    k_trunc = T.moment_match_scale(scene.opacities, 0.0, compact, "truncated")[:, None]
    gauss = render(None, scene.scales)
    same = render(dict(BETA, moment_match_init=False), scene.scales)
    untrunc = render(dict(BETA, moment_match_init=False), scene.scales * np.sqrt(4.0 / 3.0))
    trunc = render(dict(BETA, moment_match_init=False), scene.scales * k_trunc)
    print(f"G2 on {Path(ply).name}: {cap:,} splats, median opacity {np.median(scene.opacities):.3f}, "
          f"truncated match factor median {np.median(k_trunc):.3f}")
    for v, g, a, b, c, t in zip(views, gauss, same, untrunc, trunc, targets):
        print(f"  view {v}: PSNR vs Gaussian render -- Beta same scales {psnr(a, g):5.2f} dB, "
              f"x sqrt(4/3) {psnr(b, g):5.2f} dB, truncated-matched {psnr(c, g):5.2f} dB")
    return gauss, same, untrunc, trunc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ply", default=str(root / "data/lego/init.ply"))
    ap.add_argument("--skip-g3", action="store_true")
    args = ap.parse_args()
    wp.init()
    if not args.skip_g3:
        check_c_equals_b(args.device)
    check_moment_match(args.device, args.ply)
