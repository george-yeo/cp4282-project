"""At sparse.samples_per_tile == TILE*TILE the sparse path samples every pixel once, so its loss and
every gradient buffer must equal the dense path's (see 3dgs_trainer.py docstring)."""
import importlib, sys
from pathlib import Path
import numpy as np
import warp as wp

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "shared")]
T = importlib.import_module("3dgs_trainer")
from gaussian_set import GaussianSet


def build(sparse, kernel=None, res=48, device="cpu"):
    bg = np.ones(3, np.float32)
    targets, cameras, focal, distance, _ = T.load_views(root / "data/lego", res, res, bg)
    targets, cameras = targets[:2], cameras[:2]
    scene = GaussianSet.from_ply(str(root / "data/lego/init.ply"))
    opts = {"sparse": {"enabled": sparse, "samples_per_tile": T.TILE * T.TILE}}
    if kernel is not None:
        opts["kernel"] = kernel
    tr = T.WarpImageTrainer(targets, cameras, focal, distance * 0.35, len(scene.means), 1, device, 7,
                            init_scene=scene, background=bg, training_options=opts)
    return tr


def grads(tr):
    o = tr.optimizers
    out = {n: getattr(o, n).numpy().copy() for n in
           ("mean_grad_flat", "scale_grad_flat", "quaternion_grad_flat", "opacity_grad", "color_grad_flat")}
    if hasattr(o, "shape_grad"):
        out["shape_grad"] = o.shape_grad.numpy().copy()
    return out


def run(kernel=None, randomise_b=False):
    wp.init()
    d, s = build(False, kernel), build(True, kernel)
    if randomise_b:
        b = np.random.default_rng(0).uniform(-2, 2, d.capacity).astype(np.float32)
        d.shape.assign(b); s.shape.assign(b)
    ld = d.loss_and_gradients(np.array([0])); gd = grads(d)
    ls = s.loss_and_gradients(np.array([0])); gs = grads(s)
    print(f"loss dense={ld:.8f} sparse={ls:.8f}")
    assert abs(ld - ls) <= 1e-6 * max(1.0, abs(ld)), "loss mismatch"
    for k in gd:
        scale = np.abs(gd[k]).max() + 1e-20
        err = np.abs(gd[k] - gs[k]).max() / scale
        print(f"{k:22s} max|g|={scale:.3e} max rel diff={err:.2e}")
        assert err < 1e-3, k
    print("sparse == dense at full coverage: OK")


if __name__ == "__main__":
    run()
