"""Cube PLYs (+ legend images for tools/view_ply.sh) of the semantic prediction and GT
of one npz written by tools/infer_uncertainty.py. Empty and unknown (255) voxels are hidden."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config

from infer_uncertainty import COLORS
from plot_calibration import LOWER, VOXEL, write_cubes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    parser.add_argument("--config_path", default="configs/FoundationSSC-small-SemanticKITTI.py")
    args = parser.parse_args()
    names = Config.fromfile(args.config_path).class_names
    d = np.load(args.npz)
    for key, title in [("pred", "FoundationSSC prediction"), ("gt", "ground truth")]:
        labels = d[key]
        show = (labels > 0) & (labels < 255)
        out = args.npz.replace(".npz", f"_{key}")
        write_cubes(out + ".ply", LOWER + (np.argwhere(show) + .5) * VOXEL, COLORS[labels[show]], VOXEL)
        counts = np.bincount(labels[show], minlength=len(names))
        handles = [mpatches.Patch(color=COLORS[c], label=f"{names[c]} ({counts[c]:,})")
                   for c in range(1, len(names)) if counts[c]]
        leg = plt.figure(figsize=(3.2, .3 * len(handles) + 1))
        leg.legend(handles=handles, loc="upper left", title=f"{title}\n(voxels per class)", frameon=False)
        leg.savefig(out + ".legend.png", dpi=100)
        print("saved", out + ".ply", out + ".legend.png", f"{show.sum():,} voxels")


if __name__ == "__main__":
    main()
