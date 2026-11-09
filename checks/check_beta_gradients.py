"""Finite-difference checks for the Beta kernel (gate G1).

1. End-to-end: the course's one-splat scene, loss -> every parameter including b, at
   b in {-2, -1, 0, 1, 2}, with the Compact Box on and off.
2. Per-pixel, centre to rim: d alpha / d {b, mean, log-scale, opacity} from wp.grad against
   central differences of alpha itself, sampled along a line through the splat out to its
   truncation radius. A stencil whose two sides straddle the cutoff (alpha jumps there by
   design) is reported separately, not counted as an error.

Warp kernels are float32, so tolerances match the course check (rtol 5e-2) rather than fp64.
"""
import importlib, sys
from pathlib import Path
import numpy as np
import warp as wp

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "shared")]
T = importlib.import_module("3dgs_trainer")
GC = importlib.import_module("3dgs_gradient_check_gpu")

KERNEL = {"type": "beta", "learn_b": True, "b_init": 0.0, "b_min": -2.0, "b_max": 2.0,
          "moment_match_init": False, "densifier": "course", "moment_dim": 2}


def end_to_end(b, compact):
    targets = np.broadcast_to(np.array([0.2, 0.3, 0.4]), (1, 4, 4, 3)).copy().astype(np.float32)
    tr = T.WarpImageTrainer(
        targets, np.eye(4, dtype=np.float32)[None], 4.0, 1.0, 1, 1, "cpu", 7,
        init_scene=GC.one_splat_scene(), background=(1.0, 1.0, 1.0),
        training_options={"sparse": {"enabled": False}, "kernel": KERNEL,
                          "compact_box": {"enabled": compact}})
    tr.shape.assign(np.array([b], np.float32))
    tr.loss_and_gradients(np.array([0]))
    o = tr.optimizers
    analytic = {
        "mean x": float(o.mean_grad.numpy()[0, 0]),
        "log-scale y": float(o.scale_grad.numpy()[0, 1]),
        "quaternion y": float(o.quaternion_grad_flat.numpy()[2]),
        "opacity logit": float(o.opacity_grad.numpy()[0]),
        "base green": float(o.color_grad.numpy()[0, 1]),
        "shape b": float(o.shape_grad.numpy()[0]),
    }
    cases = [("mean x", tr.means, wp.vec3, (0, 0)), ("log-scale y", tr.log_scales, wp.vec3, (0, 1)),
             ("quaternion y", tr.quaternions, wp.vec4, (0, 2)), ("opacity logit", tr.opacity, wp.float32, 0),
             ("base green", tr.color, wp.vec3, (0, 1)), ("shape b", tr.shape, wp.float32, 0)]
    worst = 0.0
    for name, field, dtype, index in cases:
        numeric = GC.finite_difference(tr, field, dtype, index)
        err = abs(analytic[name] - numeric) / max(abs(numeric), 1e-3)
        worst = max(worst, err)
        np.testing.assert_allclose(analytic[name], numeric, rtol=5e-2, atol=2e-4,
                                    err_msg=f"b={b} compact={compact} {name}")
    return worst


@wp.kernel
def alpha_line(mean: wp.vec3, log_scale: wp.vec3, quat: wp.vec4, logit: float, b: float,
               camera: wp.mat44, xs: wp.array(dtype=float), py: float, size: float, focal: float,
               compact: int, eps: float, out: wp.array(dtype=float, ndim=2)):
    i = wp.tid()
    px = xs[i]
    a = T.alpha_at_pixel(mean, log_scale, quat, logit, b, camera, px, py, size, size, focal, 1,
                         compact, 0.5, T.ALPHA_CUTOFF)
    (g_m, g_s, g_q, g_o, g_b, g_c, g_px, g_py, g_w, g_h, g_f, g_k, g_ce, g_cb, g_ca) = wp.grad(
        T.alpha_at_pixel)(mean, log_scale, quat, logit, b, camera, px, py, size, size, focal, 1,
                          compact, 0.5, T.ALPHA_CUTOFF)
    out[i, 0] = a
    out[i, 1] = g_b
    out[i, 2] = g_m[0]
    out[i, 3] = g_s[0]
    out[i, 4] = g_o
    # central differences of alpha itself
    ap = T.alpha_at_pixel(mean, log_scale, quat, logit, b + eps, camera, px, py, size, size, focal, 1, compact, 0.5, T.ALPHA_CUTOFF)
    am = T.alpha_at_pixel(mean, log_scale, quat, logit, b - eps, camera, px, py, size, size, focal, 1, compact, 0.5, T.ALPHA_CUTOFF)
    out[i, 5] = (ap - am) / (2.0 * eps)
    out[i, 6] = wp.min(ap, am)
    dm = wp.vec3(eps, 0.0, 0.0)
    ap = T.alpha_at_pixel(mean + dm, log_scale, quat, logit, b, camera, px, py, size, size, focal, 1, compact, 0.5, T.ALPHA_CUTOFF)
    am = T.alpha_at_pixel(mean - dm, log_scale, quat, logit, b, camera, px, py, size, size, focal, 1, compact, 0.5, T.ALPHA_CUTOFF)
    out[i, 7] = (ap - am) / (2.0 * eps)
    out[i, 8] = wp.min(ap, am)
    ap = T.alpha_at_pixel(mean, log_scale + dm, quat, logit, b, camera, px, py, size, size, focal, 1, compact, 0.5, T.ALPHA_CUTOFF)
    am = T.alpha_at_pixel(mean, log_scale - dm, quat, logit, b, camera, px, py, size, size, focal, 1, compact, 0.5, T.ALPHA_CUTOFF)
    out[i, 9] = (ap - am) / (2.0 * eps)
    out[i, 10] = wp.min(ap, am)
    ap = T.alpha_at_pixel(mean, log_scale, quat, logit + eps, b, camera, px, py, size, size, focal, 1, compact, 0.5, T.ALPHA_CUTOFF)
    am = T.alpha_at_pixel(mean, log_scale, quat, logit - eps, b, camera, px, py, size, size, focal, 1, compact, 0.5, T.ALPHA_CUTOFF)
    out[i, 11] = (ap - am) / (2.0 * eps)
    out[i, 12] = wp.min(ap, am)


def rim_sweep(b, compact, n=4000):
    size, focal = 64.0, 64.0
    xs = np.linspace(32.0, 64.0, n).astype(np.float32)  # centre (x=32) to past the rim
    out = wp.zeros((n, 13), dtype=float)
    q = np.array([1, 0, 0, 0], np.float32)
    wp.launch(alpha_line, dim=n, inputs=[wp.vec3(0, 0, 2.0), wp.vec3(np.log(0.25), np.log(0.25), np.log(0.25)),
                                          wp.vec4(*q), 2.0, float(b), wp.mat44(np.eye(4, dtype=np.float32)),
                                          wp.array(xs), 32.0, size, focal, int(compact), 1e-3, out])
    r = out.numpy()
    inside = r[:, 0] > 0
    edge = xs[inside].max() - 32.0
    rows = []
    for name, an, fd, lo in (("b", 1, 5, 6), ("mean x", 2, 7, 8), ("log-scale x", 3, 9, 10), ("opacity", 4, 11, 12)):
        ok = inside & (r[:, lo] > 0)          # stencil fully inside support
        straddle = inside & ~(r[:, lo] > 0)
        scale = np.abs(r[ok, fd]).max() + 1e-12
        err = np.abs(r[ok, an] - r[ok, fd]) / scale
        frac = (xs[ok] - 32.0) / edge
        rim = frac > 0.9
        # Away from the outermost 3% of the support: with the Compact Box off and beta < 1,
        # d alpha / d q = -alpha beta / (9 - q) diverges as q -> 9, so a float32 central
        # difference there cannot resolve it. Training always runs with the Compact Box on.
        body = frac < 0.97
        rows.append((name, err.max(), err[rim].max() if rim.any() else np.nan, int(straddle.sum()),
                     err[body].max()))
    return edge, rows


if __name__ == "__main__":
    wp.init()
    print("== end-to-end loss gradients (rel err vs FD, worst over 6 params)")
    for compact in (True, False):
        for b in (-2.0, -1.0, 0.0, 1.0, 2.0):
            print(f"compact={compact!s:5s} b={b:+.0f}: worst rel err {end_to_end(b, compact):.2e}  PASS")
    print("== per-pixel alpha gradients, centre -> rim (max |err| / max |grad| along the line)")
    for compact in (True, False):
        for b in (-2.0, -1.0, 0.0, 1.0, 2.0):
            edge, rows = rim_sweep(b, compact)
            text = "  ".join(f"{n}: all {e:.1e} rim {r:.1e}" for n, e, r, _, _ in rows)
            print(f"compact={compact!s:5s} b={b:+.0f} support edge {edge:5.2f}px | {text} | straddling stencils {rows[0][3]}")
            singular = (not compact) and 4.0 * np.exp(b) < 1.0
            for n, e, r, _, body in rows:
                if singular:
                    print(f"    singular rim (beta<1, no Compact Box): {n} err inside 97% of support {body:.1e}")
                    assert body < 2e-2, (b, compact, n, body)
                else:
                    assert e < 2e-2, (b, compact, n, e)
    print("Beta gradients: PASS")
