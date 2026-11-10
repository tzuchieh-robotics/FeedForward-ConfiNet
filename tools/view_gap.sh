#!/usr/bin/env bash
# Interactive 3D calibration gap in a 1 m window for every npz written by tools/infer_uncertainty.py,
# using the occuq image (has open3d). Key G toggles the gap: predicted occupancy probability - real
# occupied fraction (default), or actual - predicted error rate (as in tools/plot_calibration.py).
# Buttons or keys N / P switch frames; buttons or keys 1-4 switch view:
# gap on all voxels, seen vs not seen by the LiDAR scan of that frame (free space its beams crossed,
# or a voxel one of its points hit), gap on visible voxels only, gap on occluded voxels only.
# Usage: tools/view_gap.sh [NPZ_DIR] [FRAME_ID]   Mouse: drag rotate, wheel zoom, ctrl+drag pan.
set -euo pipefail
dir=$(realpath "${1:-$(dirname "$0")/../uncertainty_out}")
docker run --rm -i --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="${DISPLAY:-:0}" -e XDG_RUNTIME_DIR=/tmp -e GAP_SOURCE="$(basename "$dir")" \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v "$dir":/data:ro \
    -v "$(realpath "$(dirname "$0")/../datasets/SemanticKITTI/dataset/sequences")":/seq:ro --entrypoint python occuq \
    - "${2:-}" <<'PYTHON'
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
from scipy.ndimage import uniform_filter

LOWER, VOXEL, WINDOW, LIM = np.array([0, -25.6, -2]), .2, 5, .5
files = sorted(glob.glob('/data/*_[0-9][0-9][0-9][0-9][0-9][0-9].npz'))
CORNERS = (np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)]) - .5) * VOXEL
FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


MODES = {'1': 'gap, all voxels', '2': 'visible / occluded from the vehicle', '3': 'gap, visible voxels only',
         '4': 'gap, occluded voxels only'}
VIS_COLORS = np.array([[.80, .80, .78], [.36, .23, .55]])  # visible, occluded
METRICS = {
    'occupancy': ('gap = mean predicted p_occ - real occupied\nfraction in the 1 m window around each cube',
                  '+0.5  predicts more occupancy\nthan real', '-0.5  predicts less occupancy\nthan real'),
    'error': ('gap = actual - predicted error rate\nin the 1 m window around each cube',
              '+0.5  overconfident\n(errs more than it says)', '-0.5  underconfident\n(errs less than it says)')}


def gap_mesh(path, mode, metric):
    d = np.load(path)
    seq, frame = os.path.basename(path)[:-4].split('_')
    # not seen by this frame's LiDAR: .occluded (it flags every occupied voxel) minus voxels a point hit
    occluded = np.unpackbits(np.fromfile(f'/seq/{seq}/voxels/{frame}.occluded', np.uint8)).reshape(256, 256, 32).astype(bool)
    pts = np.fromfile(f'/seq/{seq}/velodyne/{frame}.bin', np.float32).reshape(-1, 4)[:, :3]
    idx = np.floor((pts - LOWER) / VOXEL).astype(int)
    idx = idx[((idx >= 0) & (idx < [256, 256, 32])).all(1)]
    occluded[tuple(idx.T)] = False
    gt, pred, p = d['gt'], d['pred'], d['p_occ'].astype(np.float64)
    if 'conf' in d:  # ConfidNet: conf = chance the prediction is right, as an occupancy probability
        p = np.where(pred > 0, d['conf'], 1 - d['conf'].astype(np.float64))
    valid = gt != 255
    w = valid.astype(float)
    norm = np.maximum(uniform_filter(w, WINDOW, mode='constant'), 1e-9)
    wrong = ((gt > 0) != (pred > 0)) & valid
    # chance the prediction is wrong; after recalibration p_occ can sit on the other side of 0.5 from pred
    p_wrong = np.where(pred > 0, 1 - p, p)
    if metric == 'occupancy':
        gap = (uniform_filter(p * w, WINDOW, mode='constant')
               - uniform_filter(((gt > 0) & valid) * w, WINDOW, mode='constant')) / norm
    else:
        gap = (uniform_filter(wrong * w, WINDOW, mode='constant')
               - uniform_filter(p_wrong * w, WINDOW, mode='constant')) / norm
    show = ((gt > 0) | (pred > 0)) & valid
    show &= {'1': True, '2': True, '3': ~occluded, '4': occluded}[mode]
    xyz = LOWER + (np.argwhere(show) + .5) * VOXEL
    n = len(xyz)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector((xyz[:, None] + CORNERS).reshape(-1, 3)),
        o3d.utility.Vector3iVector((FACES[None] + 8 * np.arange(n)[:, None, None]).reshape(-1, 3)))
    if mode == '2':
        rgb = VIS_COLORS[occluded[show].astype(int)]
    else:
        rgb = plt.get_cmap('RdBu_r')(np.clip(gap[show] / LIM / 2 + .5, 0, 1))[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(rgb, 8, 0))
    mesh.compute_vertex_normals()
    known = valid & {'1': True, '2': True, '3': ~occluded, '4': occluded}[mode]
    g = gap[show]
    if metric == 'occupancy':
        pv, aw = p[known].mean(), (gt[known] > 0).mean()
        totals = f'over its known voxels: predicted\noccupancy {pv:.4f}, real {aw:.4f}\n'
        above, below = 'more occupancy than real', 'less occupancy than real'
    else:
        pv, aw = p_wrong[known].mean(), wrong[known].mean()
        totals = f'over its known voxels: predicted\nerror {pv:.4f}, actual {aw:.4f}\n'
        above, below = 'overconfident', 'underconfident'
    text = (f'{os.path.basename(path)[:-4]}  ({files.index(path) + 1}/{len(files)})\n'
            f'gap (G): {metric}\nview {mode}: {MODES[mode]}\n' + totals +
            f'shown voxels: {n:,}, occluded {occluded[show].mean():.0%}\n'
            f'{above} (gap > +0.1): {(g > .1).mean():.0%}\n'
            f'{below} (gap < -0.1): {(g < -.1).mean():.0%}')
    return mesh, text


app = gui.Application.instance
app.initialize()
window = app.create_window(f"FoundationSSC calibration gap  [{os.environ['GAP_SOURCE']}]", 1700, 1000)
em = window.theme.font_size
scene = gui.SceneWidget()
scene.scene = rendering.Open3DScene(window.renderer)
scene.scene.set_background([1., 1., 1., 1.])
scene.scene.view.set_post_processing(False)
mat = rendering.MaterialRecord()
mat.shader = 'defaultLit'
scene.scene.add_geometry('axes', o3d.geometry.TriangleMesh.create_coordinate_frame(2), mat)

panel = gui.Vert(.4 * em, gui.Margins(em, em, em, em))
info = gui.Label('')
row = gui.Horiz(.5 * em)
prev_b, next_b = gui.Button('< Prev (P)'), gui.Button('Next (N) >')
row.add_child(prev_b)
row.add_child(next_b)
panel.add_child(row)
for m, name in MODES.items():
    b = gui.Button(f'{m}  {name}')
    b.set_on_clicked(lambda m=m: show(state['k'], m))
    panel.add_child(b)
gap_b = gui.Button('G  switch gap: occupancy / error')
gap_b.set_on_clicked(lambda: show(state['k'], metric='error' if state['metric'] == 'occupancy' else 'occupancy'))
panel.add_child(gap_b)
panel.add_child(info)
grad = (plt.get_cmap('RdBu_r')(np.linspace(1, 0, 200))[:, :3] * 255).astype(np.uint8)
bar = gui.Horiz(.5 * em)
vis_legend = gui.Vert(.3 * em)
for rgb, text in zip(VIS_COLORS, ['visible from the vehicle', 'occluded (hidden from the LiDAR\nat this frame)']):
    r = gui.Horiz(.5 * em)
    r.add_child(gui.ImageWidget(o3d.geometry.Image(np.full((int(.9 * em), int(1.8 * em), 3), rgb * 255, np.uint8))))
    r.add_child(gui.Label(text))
    vis_legend.add_child(r)
bar.add_child(gui.ImageWidget(o3d.geometry.Image(np.ascontiguousarray(np.repeat(grad[:, None], int(1.8 * em), 1)))))
bar_label = gui.Label('')
bar.add_child(bar_label)
gap_legend = gui.Vert(.3 * em)
gap_title = gui.Label('')
gap_legend.add_child(gap_title)
gap_legend.add_child(bar)
panel.add_child(gap_legend)
panel.add_child(vis_legend)
panel.add_child(gui.Label('cubes: GT- or prediction-occupied,\nGT-known voxels\naxes: red x forward, green y left,\nblue z up (2 m)\nMouse: drag rotate, wheel zoom,\nctrl+drag pan'))
window.add_child(scene)
window.add_child(panel)
state = {'k': 0, 'mode': '1', 'metric': 'occupancy'}
if sys.argv[1:] and sys.argv[1]:
    state['k'] = next(i for i, f in enumerate(files) if sys.argv[1] in f)


def show(k, mode=None, metric=None):
    state['k'] = k % len(files)
    state['mode'] = mode or state['mode']
    state['metric'] = metric or state['metric']
    mesh, text = gap_mesh(files[state['k']], state['mode'], state['metric'])
    title, top, bottom = METRICS[state['metric']]
    gap_title.text = title
    bar_label.text = top + '\n' * 5 + '0  matches GT' + '\n' * 5 + bottom
    gap_legend.visible = state['mode'] != '2'
    vis_legend.visible = state['mode'] == '2'
    if scene.scene.has_geometry('mesh'):
        scene.scene.remove_geometry('mesh')
    scene.scene.add_geometry('mesh', mesh, mat)
    info.text = text
    window.set_needs_layout()


prev_b.set_on_clicked(lambda: show(state['k'] - 1))
next_b.set_on_clicked(lambda: show(state['k'] + 1))


def on_key(event):
    digits = [gui.KeyName.ONE, gui.KeyName.TWO, gui.KeyName.THREE, gui.KeyName.FOUR]
    if event.type == gui.KeyEvent.DOWN and event.key in digits:
        show(state['k'], str(digits.index(event.key) + 1))
        return gui.Widget.EventCallbackResult.HANDLED
    if event.type == gui.KeyEvent.DOWN and event.key == gui.KeyName.G:
        show(state['k'], metric='error' if state['metric'] == 'occupancy' else 'occupancy')
        return gui.Widget.EventCallbackResult.HANDLED
    if event.type == gui.KeyEvent.DOWN and event.key in (gui.KeyName.N, gui.KeyName.P):
        show(state['k'] + (1 if event.key == gui.KeyName.N else -1))
        return gui.Widget.EventCallbackResult.HANDLED
    return gui.Widget.EventCallbackResult.IGNORED


def on_layout(ctx):
    r = window.content_rect
    width = 19 * em
    scene.frame = gui.Rect(r.x, r.y, r.width - width, r.height)
    panel.frame = gui.Rect(r.get_right() - width, r.y, width, r.height)


window.set_on_layout(on_layout)
scene.set_on_key(on_key)
show(state['k'])
scene.setup_camera(60., scene.scene.bounding_box, [25, 0, 0])
scene.look_at([25, 0, 0], [-15, 0, 30], [0, 0, 1])
app.run()
PYTHON
