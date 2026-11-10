"""Where is FoundationSSC over- or underconfident? Pools the npz files written by
tools/infer_uncertainty.py and splits the known voxels (GT != 255) into groups.

Predicted error of a voxel = min(p_occ, 1 - p_occ); actual error = prediction != GT occupancy.
Per group: voxels, mean predicted error, actual error rate, gap = actual - predicted
(> 0 overconfident), ECE, and the group's share of the total excess errors sum(actual - predicted).
Moving objects (raw voxel labels 252-259, the trails of the accumulated GT) form their own group
and are left out of every other split.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from io_data import _read_occluded_SemKITTI, _read_label_SemKITTI

BINS = 10
NAMES = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
         "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence",
         "vegetation", "trunk", "terrain", "pole", "traffic-sign"]
x, y = np.meshgrid(np.arange(256) * .2 + .1, np.arange(256) * .2 - 25.5, indexing="ij")
RANGE = np.broadcast_to(np.hypot(x, y)[..., None], (256, 256, 32))


def groups(gt, pred, moving, occluded):
    """{split: {group name: mask}} over known voxels."""
    known = gt != 255
    rest = known & ~moving
    g_occ, p_occ = gt > 0, pred > 0
    return {
        "GT moving objects": {"moving (trails)": known & moving, "everything else": rest},
        "visibility (moving excluded)": {"visible": rest & ~occluded, "occluded": rest & occluded},
        "predicted state (moving excluded)": {"predicted empty (FN side)": rest & ~p_occ,
                                              "predicted occupied (FP side)": rest & p_occ},
        "GT state (moving excluded)": {"GT empty": rest & ~g_occ, "GT occupied": rest & g_occ},
        "range (moving excluded)": {"0-12.8 m": rest & (RANGE < 12.8),
                                    "12.8-25.6 m": rest & (RANGE >= 12.8) & (RANGE < 25.6),
                                    "25.6 m+": rest & (RANGE >= 25.6)},
        "GT class (moving excluded)": {NAMES[c]: rest & (gt == c) for c in range(len(NAMES))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz_dir", default="uncertainty_out")
    parser.add_argument("--data_root", default="datasets/SemanticKITTI/dataset")
    args = parser.parse_args()
    files = sorted(glob.glob(f"{args.npz_dir}/*_[0-9][0-9][0-9][0-9][0-9][0-9].npz"))
    stats = {}  # (split, group) -> [count, sum predicted, sum wrong] per bin
    for f in files:
        seq, frame = os.path.basename(f)[:-4].split("_")
        d = np.load(f)
        gt, pred, p = d["gt"], d["pred"], d["p_occ"].astype(np.float64)
        npy = np.load(f"{args.data_root}/labels/{seq}/{frame}_1_1.npy")
        assert np.array_equal(gt, npy.astype(np.uint8)), "npz GT is not in the label-file layout"
        vox = f"{args.data_root}/sequences/{seq}/voxels/{frame}"
        raw = _read_label_SemKITTI(vox + ".label").reshape(256, 256, 32)
        occluded = _read_occluded_SemKITTI(vox + ".occluded").reshape(256, 256, 32).astype(bool)
        predicted = np.minimum(p, 1 - p)
        wrong = (gt > 0) != (pred > 0)
        b = np.clip((predicted / .5 * BINS).astype(int), 0, BINS - 1)
        for split, gs in groups(gt, pred, raw >= 252, occluded).items():
            for name, m in gs.items():
                s = stats.setdefault((split, name), np.zeros((3, BINS)))
                s[0] += np.bincount(b[m], minlength=BINS)
                s[1] += np.bincount(b[m], predicted[m], BINS)
                s[2] += np.bincount(b[m], wrong[m], BINS)
    print(f"{len(files)} frames from {args.npz_dir}\n")

    total_excess = sum(s[2].sum() - s[1].sum() for (sp, _), s in stats.items() if sp == "GT moving objects")
    rows = []
    for (split, name), (n, sp, sw) in stats.items():
        if n.sum() == 0:
            continue
        ece = np.abs(sw - sp).sum() / n.sum()
        rows.append((split, name, n.sum(), sp.sum() / n.sum(), sw.sum() / n.sum(), ece, (sw.sum() - sp.sum()) / total_excess))
    last = None
    for split, name, n, mp, a, ece, share in rows:
        if split != last:
            print(f"== {split}\n{'group':<30}{'voxels':>13}{'predicted':>11}{'actual':>9}{'gap':>9}{'ECE':>8}{'excess share':>14}")
            last = split
        print(f"{name:<30}{int(n):>13,}{mp:>11.4f}{a:>9.4f}{a - mp:>+9.4f}{ece:>8.4f}{share:>14.1%}")

    # reliability curves of the main splits
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    for ax, split in zip(axes, ["GT moving objects", "visibility (moving excluded)", "predicted state (moving excluded)"]):
        for (sp_, name), (n, sp, sw) in stats.items():
            if sp_ != split:
                continue
            ok = n > 100
            ax.plot(sp[ok] / n[ok], sw[ok] / n[ok], "o-", label=f"{name} ({int(n.sum()):,} voxels)")
        ax.plot([0, .5], [0, .5], "k--", label="perfect calibration")
        ax.set_xlim(0, .5); ax.set_ylim(0, 1)
        ax.set_xlabel(r"mean predicted error $\min(p, 1-p)$ per bin")
        ax.set_ylabel("actual error rate (GT)")
        ax.set_title(f"{split}\nabove the diagonal = overconfident")
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"FoundationSSC-small occupancy calibration, SemanticKITTI seq 08, {len(files)} frames")
    fig.savefig(f"{args.npz_dir}/calibration_by_group.png", dpi=90)
    print(f"\nsaved {args.npz_dir}/calibration_by_group.png")


if __name__ == "__main__":
    main()
