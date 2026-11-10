"""Offline ConfidNet (Corbiere et al., NeurIPS 2019) on a frozen FoundationSSC, for occupancy failure prediction.

The network stays frozen. A fully convolutional confidence head reads the 64-d feature grid v
(128x128x16, before the last 1x1 conv, see tools/reliocc_offline.py extract --grid) and regresses the
True Class Probability with an MSE loss on every GT-known voxel:
  c_hat = sigmoid(ConvNet3d(v)),  upsampled to 256x256x32
  TCP   = p_occ if the voxel is GT-occupied else 1 - p_occ,  p_occ = 1 - softmax(W0 v)[empty]
Binary TCP since failures here are occupancy errors. The paper's second stage (fine-tuning a copy of
the encoder) is not done.

  train: fit on grid npz files (use frames the main network never saw) -> confidnet.pt
  apply: write npz like tools/infer_uncertainty.py (pred, gt, p_occ from the frozen model) plus conf
Run from the repo root with PYTHONPATH=tools.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn

from reliocc_offline import upsample


class ConfidNet(nn.Module):
    def __init__(self, w0, hidden=64, layers=4):
        super().__init__()
        self.register_buffer("w0", w0)  # C, D frozen last conv
        # per-channel input standardization, set from the fit frames; raw v has outliers (~100) that kill the ReLUs
        self.register_buffer("mu", torch.zeros(w0.shape[1], 1, 1, 1))
        self.register_buffer("sd", torch.ones(w0.shape[1], 1, 1, 1))
        convs, d = [], w0.shape[1]
        for _ in range(layers):
            convs += [nn.Conv3d(d, hidden, 3, padding=1), nn.ReLU()]
            d = hidden
        self.net = nn.Sequential(*convs, nn.Conv3d(hidden, 1, 1))

    def forward(self, v_grid):
        """64,128,128,16 feature grid -> (confidence, model p_occ, model pred), each flat on 256x256x32."""
        conf = torch.sigmoid(upsample(self.net(((v_grid - self.mu) / self.sd)[None])[0])).flatten()
        with torch.no_grad():
            z = upsample(v_grid).flatten(1).t() @ self.w0.t()
            p_occ = 1 - torch.softmax(z, 1)[:, 0]
            pred = z.argmax(1)
        return conf, p_occ, pred


def load(f):
    d = np.load(f)
    return torch.from_numpy(d["v"]).cuda(), torch.from_numpy(d["y"]).cuda().flatten()


def train(args):
    files = sorted(glob.glob(f"{args.grid_dir}/*.npz"))
    data = [load(f) for f in files]
    net = ConfidNet(torch.load(f"{args.grid_dir}/w0.pt").cuda()).cuda()
    flat = torch.cat([v.float().flatten(1) for v, _ in data], 1)
    net.mu.copy_(flat.mean(1).view(-1, 1, 1, 1))
    net.sd.copy_(flat.std(1).clamp_min(1e-3).view(-1, 1, 1, 1))
    del flat
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    print(f"{len(files)} frames: {os.path.basename(files[0])} .. {os.path.basename(files[-1])}")
    for epoch in range(args.epochs):
        total = spread = 0.
        for k in np.random.permutation(len(files)):
            v, y = data[k]
            conf, p_occ, _ = net(v.float())
            known = y != 255
            tcp = torch.where(y > 0, p_occ, 1 - p_occ)
            loss = ((conf - tcp)[known] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            spread += conf[known].std().item()
        print(f"epoch {epoch + 1}: mse {total / len(files):.5f}  conf std {spread / len(files):.4f}", flush=True)
    torch.save(net.state_dict(), args.weights)


@torch.no_grad()
def apply(args):
    state = torch.load(args.weights)
    net = ConfidNet(state["w0"]).cuda()
    net.load_state_dict(state)
    os.makedirs(args.out, exist_ok=True)
    for f in sorted(glob.glob(f"{args.grid_dir}/*.npz")):
        v, y = load(f)
        conf, p_occ, pred = net(v.float())
        shape = (256, 256, 32)
        np.savez_compressed(f"{args.out}/{os.path.basename(f)}", pred=pred.reshape(shape).byte().cpu().numpy(),
                            gt=y.reshape(shape).cpu().numpy(), p_occ=p_occ.reshape(shape).half().cpu().numpy(),
                            conf=conf.reshape(shape).half().cpu().numpy())
        print(os.path.basename(f), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--grid_dir", default="reliocc_grid_08fit")
    p.add_argument("--weights", default="confidnet.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p = sub.add_parser("apply")
    p.add_argument("--grid_dir", default="reliocc_grid_08")
    p.add_argument("--weights", default="confidnet.pt")
    p.add_argument("--out", default="uncertainty_out_confidnet")
    args = parser.parse_args()
    {"train": train, "apply": apply}[args.cmd](args)


if __name__ == "__main__":
    main()
