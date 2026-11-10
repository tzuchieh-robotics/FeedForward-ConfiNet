"""Which parts of the scene have a small or large calibration gap, across the npz files written by
tools/infer_uncertainty.py.

gap = actual occupancy error rate - mean predicted error min(p, 1 - p); > 0 is overconfident.
Groups are taken over GT-known voxels with moving objects (raw labels 252-259) left out, except
the "moving" group. "Seen" = seen by the LiDAR scan of that frame: free space its beams crossed
(not in the .occluded file, which flags every occupied voxel) or a voxel one of its points hit. Each group gets the pooled gap, a 95% interval from resampling frames,
the share of frames where its gap exceeds 0.1, and its share of all excess errors.
Writes gap_statistics.csv and gap_statistics.png next to the npz files.
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_cdt

from analyze_calibration import NAMES, RANGE
from plot_calibration import LOWER

Z = (np.arange(32) * .2 - 2 + .1)[None, None, :]  # voxel centre height in the LiDAR frame [m]


def seen_mask(vox, velodyne):
    occluded = np.unpackbits(np.fromfile(vox + ".occluded", np.uint8)).reshape(256, 256, 32).astype(bool)
    pts = np.fromfile(velodyne, np.float32).reshape(-1, 4)[:, :3]
    idx = np.floor((pts - LOWER) / .2).astype(int)
    idx = idx[((idx >= 0) & (idx < [256, 256, 32])).all(1)]
    hit = np.zeros((256, 256, 32), bool)
    hit[tuple(idx.T)] = True
    return ~occluded | hit


def groups(gt, pred, moving, seen):
    known = gt != 255
    rest = known & ~moving
    g_occ, p_occ = (gt > 0) & known, pred > 0
    # distance in voxels (chessboard) to the nearest voxel of the other GT state
    d = np.where(g_occ, distance_transform_cdt(g_occ, "chessboard"), distance_transform_cdt(~g_occ, "chessboard"))
    out = {("moving objects", "GT moving (trails)"): known & moving}
    for name, m in [("seen by this frame's LiDAR", seen), ("not seen", ~seen)]:
        out[("visibility", name)] = rest & m
    for name, m in [("on the GT surface (1 voxel)", d == 1), ("2 voxels from it", d == 2),
                    ("3-5 voxels", (d >= 3) & (d <= 5)), ("6+ voxels", d >= 6)]:
        out[("distance to GT surface", name)] = rest & m
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 52)]:
        out[("range", f"{lo}-{hi} m")] = rest & (RANGE >= lo) & (RANGE < hi)
    for lo, hi in [(-2, -1), (-1, 0), (0, 1), (1, 2), (2, 4.4)]:
        out[("height", f"{lo} to {hi} m")] = rest & (Z >= lo) & (Z < hi)
    for c in range(1, len(NAMES)):
        out[("GT class (FN side)", NAMES[c])] = rest & (gt == c)
    for c in range(1, len(NAMES)):
        out[("predicted class (FP side)", NAMES[c])] = rest & (pred == c)
    for vis, m in [("seen", seen), ("not seen", ~seen)]:
        out[("GT state x visibility", f"GT empty, {vis}")] = rest & ~g_occ & m
        out[("GT state x visibility", f"GT occupied, {vis}")] = rest & g_occ & m
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz_dir", default="uncertainty_out")
    parser.add_argument("--data_root", default="datasets/SemanticKITTI/dataset")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    files = sorted(glob.glob(f"{args.npz_dir}/*_[0-9][0-9][0-9][0-9][0-9][0-9].npz"))
    keys, per_frame = None, []
    for f in files:
        seq, frame = os.path.basename(f)[:-4].split("_")
        d = np.load(f)
        gt, pred, p = d["gt"], d["pred"], d["p_occ"].astype(np.float64)
        vox = f"{args.data_root}/sequences/{seq}/voxels/{frame}"
        raw = np.fromfile(vox + ".label", np.uint16).reshape(256, 256, 32)
        seen = seen_mask(vox, f"{args.data_root}/sequences/{seq}/velodyne/{frame}.bin")
        predicted = np.minimum(p, 1 - p)
        wrong = (gt > 0) != (pred > 0)
        gs = groups(gt, pred, raw >= 252, seen)
        keys = keys or list(gs)
        per_frame.append([(m.sum(), predicted[m].sum(), wrong[m].sum()) for m in gs.values()])
    s = np.array(per_frame, float)  # frames, groups, (count, sum predicted, sum wrong)
    n_frames = len(files)

    total = s.sum(0)
    excess_all = total[keys.index(("moving objects", "GT moving (trails)")), 2] - total[keys.index(("moving objects", "GT moving (trails)")), 1] \
        + sum(total[keys.index(("visibility", v)), 2] - total[keys.index(("visibility", v)), 1]
              for v in ("seen by this frame's LiDAR", "not seen"))
    rng = np.random.default_rng(0)
    boot = s[rng.integers(0, n_frames, (args.resamples, n_frames))].sum(1)  # resamples, groups, 3
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_gap = (boot[..., 2] - boot[..., 1]) / boot[..., 0]
        frame_gap = (s[..., 2] - s[..., 1]) / s[..., 0]
    rows = []
    for g, key in enumerate(keys):
        n, sp, sw = total[g]
        if n == 0:
            continue
        lo, hi = np.nanpercentile(boot_gap[:, g], [2.5, 97.5])
        present = s[:, g, 0] > 0
        rows.append(dict(category=key[0], group=key[1], voxels=int(n), frames=int(present.sum()),
                         predicted=sp / n, actual=sw / n, gap=(sw - sp) / n, gap_low=lo, gap_high=hi,
                         frames_gap_over_0_1=float((frame_gap[present, g] > .1).mean()),
                         excess_share=(sw - sp) / excess_all))
    out = f"{args.npz_dir}/gap_statistics"
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{n_frames} frames; gap = actual - predicted error rate, 95% interval from resampling frames\n")
    last = None
    for r in rows:
        if r["category"] != last:
            print(f"== {r['category']}\n{'group':<30}{'voxels':>13}{'pred':>8}{'actual':>8}{'gap':>8}"
                  f"{'95% interval':>17}{'frames >0.1':>13}{'excess':>9}")
            last = r["category"]
        print(f"{r['group']:<30}{r['voxels']:>13,}{r['predicted']:>8.3f}{r['actual']:>8.3f}{r['gap']:>+8.3f}"
              f"   [{r['gap_low']:+.3f}, {r['gap_high']:+.3f}]{r['frames_gap_over_0_1']:>12.0%}{r['excess_share']:>9.1%}")

    # one panel per category, groups sorted by gap, bars with frame-resampled intervals
    cats = list(dict.fromkeys(r["category"] for r in rows))
    heights = [len([r for r in rows if r["category"] == c]) for c in cats]
    fig, axes = plt.subplots(len(cats), 1, figsize=(10, .32 * sum(heights) + 1.2 * len(cats)),
                             gridspec_kw=dict(height_ratios=heights), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r")
    for ax, c in zip(axes, cats):
        rs = sorted([r for r in rows if r["category"] == c], key=lambda r: r["gap"])
        y = np.arange(len(rs))
        gap = np.array([r["gap"] for r in rs])
        ax.barh(y, gap, color=cmap(np.clip(gap + .5, 0, 1)), edgecolor="0.3", lw=.5)
        ax.errorbar(gap, y, xerr=[gap - [r["gap_low"] for r in rs], [r["gap_high"] for r in rs] - gap],
                    fmt="none", ecolor="0.2", lw=1, capsize=2)
        ax.set_yticks(y, [f"{r['group']} ({r['voxels']:,})" for r in rs], fontsize=8)
        ax.axvline(0, color="0.2", lw=.8)
        ax.set_xlim(-.1, .95)
        ax.set_title(c, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="x", lw=.4, alpha=.5)
    axes[-1].set_xlabel("gap = actual - predicted occupancy error rate  (> 0 overconfident; bar = 95% interval over frames)")
    fig.suptitle(f"FoundationSSC-small calibration gap by scene part, SemanticKITTI seq 08, {n_frames} frames "
                 "(moving objects excluded except their own row)", fontsize=10)
    fig.savefig(out + ".png", dpi=100)
    print(f"\nsaved {out}.csv {out}.png")


if __name__ == "__main__":
    main()
