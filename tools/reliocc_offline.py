"""Offline ReliOcc (Wang et al., IJCAI 2025) on a frozen FoundationSSC.

The network stays frozen. Trained on top of the 64-d voxel feature v that feeds the last 1x1 conv of
the occupancy head (logits z = W0 v):
  sigma = MLP(v) > 0                                  per-voxel uncertainty
  L_au  = CE(z + sigma * eps, y),  eps ~ N(0, I)      absolute uncertainty
  L_ru  = -sum((y_i + y_j) * log softmax(W0 v_hat)),  v_hat = lam v_i + (1 - lam) v_j,
          lam = sigma_i / (sigma_i + sigma_j), pairs shuffled within a frame   relative uncertainty
  L_cal = NLL(softmax((w * z + b) / (k1 sigma + k2)))  calibrator, w diagonal (init 1), b (init 0)
  loss  = 1.5 L_au + 1.0 L_ru + 4.0 L_cal, AdamW lr 1e-3, 20 epochs (paper's offline setting).

  extract: run the frozen model on a split, keep a uniform sample of GT-known voxels per frame
           (feature upsampled to 256x256x32 like the logits, label) -> one npz per frame
  train:   fit sigma-MLP and calibrator on the extracted samples -> reliocc.pt
  apply:   run the model on the test split, write npz like tools/infer_uncertainty.py
           (pred, gt, p_occ, entropy from the calibrated softmax, plus sigma)
Run from the repo root with PYTHONPATH=tools.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from infer_uncertainty import Config, DataLoader, build_dataset, move_data_to_device, pl_model

OCC_SIZE = (256, 256, 32)


def load_model(config, ckpt, split=None):
    cfg = Config.fromfile(config)
    cfg.update(save_path=None, test_mapping=False)
    if split:
        cfg.data.test.split = split
    model = pl_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    return cfg, model.cuda().eval()


def upsample(x):
    return F.interpolate(x[None], size=OCC_SIZE, mode="trilinear", align_corners=False)[0]


@torch.no_grad()
def voxel_features(model, batch, grid=False):
    """64-d feature before the last 1x1 conv, upsampled to the output grid; the conv is linear and
    trilinear weights sum to 1, so W0 @ upsampled v equals the model's upsampled logits.
    grid=True returns it at the head's 128x128x16 resolution instead."""
    m = model.model
    feats, _, _ = m.extract_img_feat(batch["img_inputs"], batch["img_metas"])
    enc = m.occ_encoder(feats)
    enc = enc[0] if isinstance(enc, (list, tuple)) else enc
    v = m.pts_bbox_head.occ_convs[0][:3](enc)[0]
    return v if grid else upsample(v)  # 64,X,Y,Z


class ReliOcc(nn.Module):
    def __init__(self, w0, hidden=64, context=False):
        """context=True: sigma from 3D convs over the 128x128x16 feature grid (two 3x3x3 layers, a 5x5x5
        neighbourhood) instead of a per-voxel MLP, so it can see surfaces and edges around each voxel."""
        super().__init__()
        self.register_buffer("w0", w0)  # C, D frozen last conv
        d, c = w0.shape[1], w0.shape[0]
        if context:
            self.sigma_net = nn.Sequential(nn.Conv3d(d, hidden, 3, padding=1), nn.ReLU(),
                                           nn.Conv3d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                           nn.Conv3d(hidden, 1, 1), nn.Softplus())
        else:
            self.sigma_mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                           nn.Linear(hidden, 1), nn.Softplus())
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.k1 = nn.Parameter(torch.zeros(()))
        self.k2 = nn.Parameter(torch.ones(()))

    def logits(self, v):
        return v @ self.w0.t()

    def sigma(self, v):
        return self.sigma_mlp(v).squeeze(-1) + 1e-6

    def sigma_grid(self, v_grid):
        """64,128,128,16 feature grid -> sigma on the 256x256x32 output grid, flattened."""
        return upsample(self.sigma_net(v_grid[None])[0]).flatten() + 1e-6

    def calibrated(self, z, sigma):
        return (self.w * z + self.b) / (self.k1 * sigma + self.k2).clamp_min(1e-3)[..., None]

    def losses(self, v, y, s=None):
        z, s = self.logits(v), self.sigma(v) if s is None else s
        l_au = F.cross_entropy(z + s[:, None] * torch.randn_like(z), y)
        j = torch.randperm(len(v), device=v.device)
        lam = (s / (s + s[j]))[:, None]
        z_mix = self.logits(lam * v + (1 - lam) * v[j])
        target = F.one_hot(y, z.shape[1]).float() + F.one_hot(y[j], z.shape[1]).float()
        l_ru = -(target * F.log_softmax(z_mix, 1)).sum(1).mean()
        l_cal = F.cross_entropy(self.calibrated(z, s), y)
        return l_au, l_ru, l_cal


def extract(args):
    cfg, model = load_model(args.config_path, args.ckpt_path, args.split)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    for k, batch in enumerate(DataLoader(build_dataset(cfg.data.test), batch_size=1, num_workers=2)):
        meta = batch["img_metas"]
        name = f'{meta["sequence"][0]}_{meta["frame_id"][0]}'
        if os.path.exists(f"{args.out}/{name}.npz"):
            continue
        batch = move_data_to_device(batch, "cuda")
        if args.grid:  # whole 128x128x16 feature grid and full label grid, for the context sigma net
            np.savez(f"{args.out}/{name}.npz", v=voxel_features(model, batch, grid=True).half().cpu().numpy(),
                     y=batch["gt_occ"][0].byte().cpu().numpy())
            print(k, name, flush=True)
            continue
        v = voxel_features(model, batch)
        gt = batch["gt_occ"][0].long()
        idx = torch.nonzero(gt.flatten() != 255)[:, 0]
        idx = idx[torch.from_numpy(rng.choice(len(idx), min(args.samples, len(idx)), replace=False)).cuda()]
        np.savez(f"{args.out}/{name}.npz", v=v.flatten(1)[:, idx].t().half().cpu().numpy(),
                 y=gt.flatten()[idx].byte().cpu().numpy())
        print(k, name, flush=True)
    torch.save(model.model.pts_bbox_head.occ_convs[0][3].weight.detach().cpu()[:, :, 0, 0, 0], f"{args.out}/w0.pt")


def train(args):
    files = sorted(glob.glob(f"{args.samples_dir}/*.npz"))
    data = [np.load(f) for f in files]
    v_all = [torch.from_numpy(d["v"]).cuda() for d in data]
    y_all = [torch.from_numpy(d["y"].astype(np.int64)).cuda() for d in data]
    net = ReliOcc(torch.load(f"{args.samples_dir}/w0.pt").cuda(), context=args.context).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    print(f"{len(files)} frames, {sum((y != 255).sum().item() for y in y_all):,} voxels")
    for epoch in range(args.epochs):
        sums = np.zeros(4)
        for k in np.random.permutation(len(files)):  # one frame per step so L_ru pairs voxels of one scene
            v, y = v_all[k].float(), y_all[k]
            if args.context:  # one step on every GT-known voxel of the frame
                known = (y.flatten() != 255).nonzero()[:, 0]
                s = net.sigma_grid(v)[known]
                with torch.no_grad():
                    v = upsample(v).flatten(1).t()[known]
                parts = [(v, y.flatten()[known], s)]
            else:
                parts = [(v[c], y[c], None) for c in torch.randperm(len(y), device=v.device).split(args.batch)]
            for vc, yc, sc in parts:
                l_au, l_ru, l_cal = net.losses(vc, yc, sc)
                loss = 1.5 * l_au + 1.0 * l_ru + 4.0 * l_cal
                opt.zero_grad()
                loss.backward()
                opt.step()
                sums += [loss.item(), l_au.item(), l_ru.item(), l_cal.item()]
        n = len(files) if args.context else sum(len(torch.arange(len(y)).split(args.batch)) for y in y_all)
        print(f"epoch {epoch + 1}: loss {sums[0] / n:.4f}  L_au {sums[1] / n:.4f}  L_ru {sums[2] / n:.4f}  "
              f"L_cal {sums[3] / n:.4f}  k1 {net.k1.item():.3f} k2 {net.k2.item():.3f}", flush=True)
    torch.save(net.state_dict(), args.weights)


@torch.no_grad()
def apply(args):
    cfg, model = load_model(args.config_path, args.ckpt_path)
    state = torch.load(args.weights)
    context = any(k.startswith("sigma_net") for k in state)
    net = ReliOcc(state["w0"], context=context).cuda()
    net.load_state_dict(state)
    os.makedirs(args.out, exist_ok=True)
    for batch in DataLoader(build_dataset(cfg.data.test), batch_size=1, num_workers=2):
        meta = batch["img_metas"]
        name = f'{meta["sequence"][0]}_{meta["frame_id"][0]}'
        batch = move_data_to_device(batch, "cuda")
        v_grid = voxel_features(model, batch, grid=True)
        v = upsample(v_grid).flatten(1).t()  # N, 64
        s_all = net.sigma_grid(v_grid) if context else None
        prob, sigma = [], []
        for i, part in zip(range(0, len(v), 1 << 20), v.split(1 << 20)):
            s = s_all[i:i + len(part)] if context else net.sigma(part)
            prob.append(torch.softmax(net.calibrated(net.logits(part), s), 1))
            sigma.append(s)
        prob = torch.cat(prob).t().reshape(-1, *OCC_SIZE).cpu().numpy()
        sigma = torch.cat(sigma).reshape(OCC_SIZE).cpu().numpy()
        entropy = -(prob * np.log(np.clip(prob, 1e-12, 1))).sum(0) / np.log(len(prob))
        np.savez_compressed(f"{args.out}/{name}.npz", pred=prob.argmax(0).astype(np.uint8),
                            gt=batch["gt_occ"][0].cpu().numpy().astype(np.uint8), p_occ=(1 - prob[0]).astype(np.float16),
                            entropy=entropy.astype(np.float16), sigma=sigma.astype(np.float16))
        print(name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--config_path", default="configs/FoundationSSC-small-SemanticKITTI.py")
        p.add_argument("--ckpt_path", default="ckpts/FoundationSSC-small-SemanticKITTI.ckpt")
        p.add_argument("--out", default="reliocc_samples" if name == "extract" else "uncertainty_out_reliocc")
    sub.choices["extract"].add_argument("--split", default="train")
    sub.choices["extract"].add_argument("--samples", type=int, default=30000, help="GT-known voxels kept per frame")
    sub.choices["extract"].add_argument("--grid", action="store_true", help="keep the whole feature grid (for --context)")
    sub.choices["apply"].add_argument("--weights", default="reliocc.pt")
    p = sub.choices["train"] = sub.add_parser("train")
    p.add_argument("--samples_dir", default="reliocc_samples")
    p.add_argument("--weights", default="reliocc.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--context", action="store_true", help="3D-conv sigma over --grid samples, one step per frame")
    args = parser.parse_args()
    {"extract": extract, "train": train, "apply": apply}[args.cmd](args)


if __name__ == "__main__":
    main()
