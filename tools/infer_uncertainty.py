"""Run FoundationSSC on the test split and save per-voxel softmax uncertainty.

Per frame it writes <out>/<seq>_<frame>.npz (pred, gt, p_occ, entropy) and a PNG
with GT / prediction / uncertainty in bird's-eye view, plus a PLY of the predicted
occupied voxels coloured by entropy.
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from pytorch_lightning.utilities import move_data_to_device
from torch.utils.data import DataLoader

from mmdet3d_plugin import *  # noqa: registers modules
from LightningTools.pl_model import pl_model

# SemanticKITTI colours, index = train id
COLORS = np.array([
    [0, 0, 0], [100, 150, 245], [100, 230, 245], [30, 60, 150], [80, 30, 180],
    [100, 80, 250], [255, 30, 30], [255, 40, 200], [150, 30, 90], [255, 0, 255],
    [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0], [255, 120, 50],
    [0, 175, 0], [135, 60, 0], [150, 240, 80], [255, 240, 150], [255, 0, 0]]) / 255.


def logits(model, batch):
    """forward_test of FoundationSSC, returning the logits instead of argmax."""
    m = model.model
    feats, _, _ = m.extract_img_feat(batch["img_inputs"], batch["img_metas"])
    enc = m.occ_encoder(feats)
    enc = [enc[0]] if isinstance(enc, (list, tuple)) else [enc]
    return m.pts_bbox_head(voxel_feats=enc, img_metas=batch["img_metas"], img_feats=None,
                           gt_occ=batch["gt_occ"])["output_voxels"]


def auroc(score, positive):
    """Probability that a random positive outranks a random negative (ties count half)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    _, first, counts = np.unique(s, return_index=True, return_counts=True)
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)
    n1 = positive.sum()
    n0 = len(score) - n1
    return (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bev(labels):
    """Class of the highest non-empty voxel in each column (0 if none)."""
    occ = (labels > 0) & (labels < 255)
    top = occ.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=2)
    out = np.take_along_axis(labels, top[..., None], 2)[..., 0]
    return np.where(occ.any(2), out, 0)


def write_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(xyz))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, (rgb * 255).astype(np.uint8)):
            f.write("%.2f %.2f %.2f %d %d %d\n" % (*p, *c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--out", default="uncertainty_out")
    parser.add_argument("--no_figures", action="store_true", help="only write the npz files")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_path)
    cfg.update(save_path=None, test_mapping=False)
    model = pl_model(cfg)
    state = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    print(model.load_state_dict(state, strict=False))
    model = model.cuda().eval()

    dataset = build_dataset(cfg.data.test)
    os.makedirs(args.out, exist_ok=True)
    pc = np.array(cfg.point_cloud_range)
    voxel = (pc[3:] - pc[:3]) / np.array(cfg.occ_size)
    for batch in DataLoader(dataset, batch_size=1, num_workers=0):
        meta = batch["img_metas"]
        name = f'{meta["sequence"][0]}_{meta["frame_id"][0]}'
        batch = move_data_to_device(batch, "cuda")
        with torch.no_grad():
            prob = torch.softmax(logits(model, batch).float(), 1)[0].cpu().numpy()  # C,X,Y,Z
        gt = batch["gt_occ"][0].cpu().numpy().astype(np.int64)
        pred = prob.argmax(0)
        p_occ = 1 - prob[0]
        entropy = -(prob * np.log(np.clip(prob, 1e-12, 1))).sum(0) / np.log(len(prob))
        np.savez_compressed(f"{args.out}/{name}.npz", pred=pred.astype(np.uint8), gt=gt.astype(np.uint8),
                            p_occ=p_occ.astype(np.float16), entropy=entropy.astype(np.float16))

        # metrics on this frame (GT 255 = unknown, excluded)
        valid = gt != 255
        g, p = gt[valid], pred[valid]
        tp, fp, fn = ((g > 0) & (p > 0)).sum(), ((g == 0) & (p > 0)).sum(), ((g > 0) & (p == 0)).sum()
        ious = [((g == c) & (p == c)).sum() / max(((g == c) | (p == c)).sum(), 1)
                for c in range(1, len(prob)) if (g == c).any()]
        wrong = g != p
        print(f"{name}: IoU {tp / (tp + fp + fn):.3f}  mIoU(classes in frame) {np.mean(ious):.3f}  "
              f"error AUROC entropy {auroc(entropy[valid], wrong):.3f}  "
              f"occupancy-error AUROC min(p,1-p) {auroc(np.minimum(p_occ, 1 - p_occ)[valid], (g > 0) != (p > 0)):.3f}")

        if args.no_figures:
            continue
        # bird's-eye view figure
        img = lambda a: a[::-1, ::-1]  # x forward = up, y left = left
        occ = (pred > 0)
        unc_bev = np.where(occ.any(2), (entropy * occ).max(2), np.nan)
        fig, ax = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
        rgb = batch["img_inputs"][0][0, 0].permute(1, 2, 0).cpu().numpy()
        ax[0, 0].imshow((rgb - rgb.min()) / (rgb.max() - rgb.min()))
        ax[0, 0].set_title("left image (input)")
        ax[0, 1].imshow(img(COLORS[bev(gt)]))
        ax[0, 1].set_title("ground truth (top view)")
        ax[1, 0].imshow(img(COLORS[bev(pred)]))
        ax[1, 0].set_title(f"FoundationSSC prediction (IoU {tp / (tp + fp + fn):.3f})")
        im = ax[1, 1].imshow(img(unc_bev), cmap="magma_r", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax[1, 1], fraction=.046).set_label("normalized entropy (max over height)")
        ax[1, 1].set_title("uncertainty of predicted occupied voxels")
        for a in ax.flat[1:]:
            a.set_xticks([]); a.set_yticks([])
        fig.savefig(f"{args.out}/{name}.png", dpi=90)
        plt.close(fig)

        idx = np.argwhere(occ)
        write_ply(f"{args.out}/{name}_entropy.ply", pc[:3] + (idx + .5) * voxel,
                  plt.get_cmap("magma_r")(entropy[occ])[:, :3])
        write_ply(f"{args.out}/{name}_semantic.ply", pc[:3] + (idx + .5) * voxel, COLORS[pred[occ]])
        print("saved", f"{args.out}/{name}.*")


if __name__ == "__main__":
    main()
