"""Occupancy calibration of one npz written by tools/infer_uncertainty.py.

Predicted error of a voxel is min(p_occ, 1 - p_occ); actual error is prediction != GT occupancy.
Writes <name>_calibration.png (reliability diagram + 3D preview) and <name>_calibration_gap.ply:
GT- or prediction-occupied voxels as cubes, coloured by the local gap
(actual - predicted error rate within a window), red = overconfident, blue = underconfident.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

LOWER, VOXEL = np.array([0, -25.6, -2]), .2


def write_cubes(path, xyz, rgb, size):
    """Binary PLY with one cube (8 vertices, 12 triangles) per voxel centre."""
    corners = (np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)]) - .5) * size
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    n = len(xyz)
    vert = np.zeros(n * 8, [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v = (xyz[:, None] + corners).reshape(-1, 3)
    vert["x"], vert["y"], vert["z"] = v.T
    c = np.repeat((rgb * 255).astype(np.uint8), 8, 0)
    vert["red"], vert["green"], vert["blue"] = c.T
    face = np.zeros(n * 12, [("n", "u1"), ("i", "<i4", 3)])
    face["n"] = 3
    face["i"] = (faces[None] + 8 * np.arange(n)[:, None, None]).reshape(-1, 3)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n * 8}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"element face {n * 12}\nproperty list uchar int vertex_indices\nend_header\n".encode())
        f.write(vert.tobytes())
        f.write(face.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--window", type=int, default=5, help="voxels per side of the local window")
    args = parser.parse_args()
    d = np.load(args.npz)
    gt, pred, p = d["gt"], d["pred"], d["p_occ"].astype(np.float64)
    valid = gt != 255
    g_occ, p_occ = gt > 0, pred > 0
    predicted = np.minimum(p, 1 - p)
    wrong = (g_occ != p_occ) & valid

    # reliability diagram
    edges = np.linspace(0, .5, args.bins + 1)
    b = np.clip(np.digitize(predicted[valid], edges) - 1, 0, args.bins - 1)
    count = np.bincount(b, minlength=args.bins)
    mean_pred = np.bincount(b, predicted[valid], args.bins) / np.maximum(count, 1)
    actual = np.bincount(b, wrong[valid], args.bins) / np.maximum(count, 1)
    ece = (count * np.abs(actual - mean_pred)).sum() / count.sum()
    for lo, hi, n, mp, a in zip(edges[:-1], edges[1:], count, mean_pred, actual):
        print(f"predicted {lo:.2f}-{hi:.2f}: {n:>9,} voxels  mean predicted {mp:.3f}  actual {a:.3f}")
    print(f"ECE {ece:.4f}  overall predicted {predicted[valid].mean():.4f}  actual {wrong[valid].mean():.4f}")

    # local calibration gap: window averages over known voxels
    w = valid.astype(float)
    norm = uniform_filter(w, args.window, mode="constant")
    local_actual = uniform_filter(wrong * w, args.window, mode="constant") / np.maximum(norm, 1e-9)
    local_pred = uniform_filter(predicted * w, args.window, mode="constant") / np.maximum(norm, 1e-9)
    gap = local_actual - local_pred
    show = (g_occ | p_occ) & valid
    cmap, lim = plt.get_cmap("RdBu_r"), .5
    colors = cmap(np.clip(gap[show] / lim / 2 + .5, 0, 1))[:, :3]
    idx = np.argwhere(show)
    xyz = LOWER + (idx + .5) * VOXEL
    out = args.npz.replace(".npz", "_calibration")
    write_cubes(out + "_gap.ply", xyz, colors, VOXEL)
    print(f"local gap on {show.sum():,} shown voxels: mean {gap[show].mean():+.3f}, "
          f"overconfident (> +0.1) {(gap[show] > .1).mean():.1%}, underconfident (< -0.1) {(gap[show] < -.1).mean():.1%}")

    fig = plt.figure(figsize=(17, 7.5), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    width = edges[1] - edges[0]
    ax.bar(edges[:-1] + width / 2, actual, width * .9, color="#4c72b0", label="actual error rate (GT)")
    ax.plot([0, .5], [0, .5], "k--", label="perfect calibration")
    ax.plot(mean_pred, actual, "o-", color="#c44e52", label="mean predicted vs actual")
    for x, a, n in zip(edges[:-1] + width / 2, actual, count):
        ax.text(x, a + .01, f"{n:,}", ha="center", fontsize=8, rotation=90, va="bottom")
    ax.set_xlim(0, .5); ax.set_ylim(0, 1)
    ax.set_xlabel(r"predicted error $\min(p, 1-p)$")
    ax.set_ylabel("actual error rate (prediction != GT occupancy)")
    ax.set_title(f"Occupancy reliability diagram, ECE {ece:.3f} (numbers = voxels per bin)\n"
                 "above the diagonal = overconfident")
    ax.legend(loc="upper left")
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    ax3.scatter(*xyz.T, c=colors, s=.3, marker="s", depthshade=False)
    ax3.set_box_aspect((51.2, 51.2, 6.4 * 2))
    ax3.view_init(elev=35, azim=-160)
    ax3.set_xlabel("x forward [m]"); ax3.set_ylabel("y left [m]"); ax3.set_zlabel("z [m]")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
    fig.colorbar(sm, ax=ax3, fraction=.03, shrink=.6).set_label(
        f"local gap: actual - predicted error ({args.window * VOXEL:.0f} m window)")
    ax3.set_title("3D local calibration gap (GT- or prediction-occupied voxels)\nred = overconfident, blue = underconfident")
    fig.savefig(out + ".png", dpi=90)

    # legend image shown next to the PLY by tools/view_ply.sh
    leg = plt.figure(figsize=(3.2, 6))
    cax = leg.add_axes([.12, .3, .12, .55])
    leg.colorbar(sm, cax=cax).set_label("actual - predicted error rate")
    cax.text(1.2, 1.04, "overconfident\n(errs more than it says)", transform=cax.transAxes, va="bottom", fontsize=9, color="#b2182b")
    cax.text(1.2, -.04, "underconfident\n(errs less than it says)", transform=cax.transAxes, va="top", fontsize=9, color="#2166ac")
    leg.text(.05, .02, f"each cube: a GT- or prediction-\noccupied voxel; colour = gap in\nthe {args.window * VOXEL:.0f} m window around it\n"
             f"white = predicted error matches\nGT error rate (ECE {ece:.3f})", fontsize=9)
    leg.savefig(out + "_gap.legend.png", dpi=100)
    print("saved", out + ".png", out + "_gap.ply", out + "_gap.legend.png")


if __name__ == "__main__":
    main()
