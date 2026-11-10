"""Occupancy errors (TP / FP / FN against GT) and binary occupancy entropy for one npz
written by tools/infer_uncertainty.py. Writes a PNG and PLYs of the error voxels."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from infer_uncertainty import auroc
from plot_calibration import write_cubes

STATUS = ["TN", "TP", "FP", "FN"]
COLORS = np.array([[.93, .93, .93], [.20, .63, .17], [.89, .10, .11], [.12, .47, .71]])
LOWER, VOXEL = np.array([0, -25.6, -2]), .2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    args = parser.parse_args()
    d = np.load(args.npz)
    gt, pred, p = d["gt"], d["pred"], d["p_occ"].astype(np.float64)
    valid = gt != 255
    g_occ, p_occ = (gt > 0) & valid, (pred > 0)
    status = np.zeros(gt.shape, int)
    status[p_occ & g_occ] = 1
    status[p_occ & ~g_occ & valid] = 2
    status[~p_occ & g_occ] = 3
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    entropy = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))  # binary, 0..1 bit

    counts = {n: int((status[valid] == k).sum()) for k, n in enumerate(STATUS)}
    err = status[valid] >= 2
    print(counts, f"occupancy-error AUROC of binary entropy {auroc(entropy[valid], err):.3f}")

    # column view: show the worst status in each column (FN > FP > TP > TN), entropy max over height
    img = lambda a: a[::-1, ::-1]  # x forward = up, y left = left
    col = np.select([(status == 3).any(2), (status == 2).any(2), (status == 1).any(2)], [3, 2, 1], 0)
    is_err = (status >= 2)
    ent_err = np.where(is_err.any(2), (entropy * is_err).max(2), np.nan)
    ent_all = np.where(valid.any(2), np.where(valid, entropy, 0).max(2), np.nan)

    fig, ax = plt.subplots(2, 2, figsize=(13, 12.5), constrained_layout=True)
    ax[0, 0].imshow(img(COLORS[col]))
    ax[0, 0].legend(handles=[mpatches.Patch(color=COLORS[k], label=f"{n} ({counts[n]:,} voxels)")
                             for k, n in enumerate(STATUS)], loc="lower left")
    ax[0, 0].set_title("occupancy vs GT (column shows FN > FP > TP)")
    im = ax[0, 1].imshow(img(ent_err), cmap="magma_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax[0, 1], fraction=.046).set_label("binary occupancy entropy [bit]")
    ax[0, 1].set_title("entropy at FP / FN voxels (max over height)")
    im = ax[1, 0].imshow(img(ent_all), cmap="magma_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax[1, 0], fraction=.046).set_label("binary occupancy entropy [bit]")
    ax[1, 0].set_title("entropy at all known voxels (max over height)")
    for a in ax.flat[:3]:
        a.set_xticks([]); a.set_yticks([])
    bins = np.linspace(0, 1, 41)
    for k in range(4):
        e = entropy[valid & (status == k)]
        ax[1, 1].hist(e, bins, density=True, histtype="step", lw=2, color=COLORS[k] * .9,
                      label=f"{STATUS[k]} (median {np.median(e):.2f})")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("binary occupancy entropy [bit]")
    ax[1, 1].set_ylabel("density")
    ax[1, 1].legend()
    ax[1, 1].set_title(f"entropy per status (error AUROC {auroc(entropy[valid], err):.3f})")
    out = args.npz.replace(".npz", "_occ_error")
    fig.savefig(out + ".png", dpi=90)

    idx = np.argwhere(status >= 2)
    xyz = LOWER + (idx + .5) * VOXEL
    write_cubes(out + "_status.ply", xyz, COLORS[status[status >= 2]], VOXEL)
    write_cubes(out + "_entropy.ply", xyz, plt.get_cmap("magma_r")(entropy[status >= 2])[:, :3], VOXEL)

    # legend images shown next to the PLYs by tools/view_ply.sh
    leg = plt.figure(figsize=(3.2, 2))
    leg.legend(handles=[mpatches.Patch(color=COLORS[k], label=f"{STATUS[k]} ({counts[STATUS[k]]:,} voxels)")
                        for k in (2, 3)], loc="upper left", frameon=False,
               title="occupancy errors vs GT\nFP: predicted occupied, GT empty\nFN: predicted empty, GT occupied")
    leg.savefig(out + "_status.legend.png", dpi=100)
    leg = plt.figure(figsize=(3.2, 6))
    cax = leg.add_axes([.12, .3, .12, .55])
    leg.colorbar(plt.cm.ScalarMappable(cmap="magma_r", norm=plt.Normalize(0, 1)), cax=cax).set_label(
        "binary occupancy entropy [bit]")
    leg.text(.05, .05, f"FP and FN voxels only\n(error AUROC {auroc(entropy[valid], err):.3f})", fontsize=9)
    leg.savefig(out + "_entropy.legend.png", dpi=100)
    print("saved", out + ".png", out + "_status.ply", out + "_entropy.ply")


if __name__ == "__main__":
    main()
