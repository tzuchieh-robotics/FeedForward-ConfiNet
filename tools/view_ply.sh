#!/usr/bin/env bash
# Open a PLY mesh in an interactive Open3D window, using the occuq image (has open3d).
# If FILE.legend.png exists next to FILE.ply, it is shown in a side panel.
# Usage: tools/view_ply.sh FILE.ply    Mouse: drag rotate, wheel zoom, ctrl+drag pan.
set -euo pipefail
ply=$(realpath "$1")
docker run --rm -i --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="${DISPLAY:-:0}" -e XDG_RUNTIME_DIR=/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v "$(dirname "$ply")":/data:ro --entrypoint python occuq \
    - "/data/$(basename "$ply")" <<'PYTHON'
import os
import sys
import open3d as o3d
from open3d.visualization import gui, rendering

path = sys.argv[1]
mesh = o3d.io.read_triangle_mesh(path)
mesh.compute_vertex_normals()
app = gui.Application.instance
app.initialize()
window = app.create_window(os.path.basename(path), 1700, 1000)
em = window.theme.font_size
scene = gui.SceneWidget()
scene.scene = rendering.Open3DScene(window.renderer)
scene.scene.set_background([1., 1., 1., 1.])
scene.scene.view.set_post_processing(False)
mat = rendering.MaterialRecord()
mat.shader = 'defaultLit'
scene.scene.add_geometry('mesh', mesh, mat)
scene.scene.add_geometry('axes', o3d.geometry.TriangleMesh.create_coordinate_frame(2), mat)
panel = gui.Vert(.3 * em, gui.Margins(em, em, em, em))
legend = path[:-4] + '.legend.png'
if os.path.exists(legend):
    panel.add_child(gui.ImageWidget(legend))
panel.add_child(gui.Label('axes: red x forward, green y left,\nblue z up (2 m)\nMouse: drag rotate, wheel zoom,\nctrl+drag pan'))
window.add_child(scene)
window.add_child(panel)


def on_layout(ctx):
    r = window.content_rect
    width = 18 * em
    scene.frame = gui.Rect(r.x, r.y, r.width - width, r.height)
    panel.frame = gui.Rect(r.get_right() - width, r.y, width, r.height)


window.set_on_layout(on_layout)
scene.setup_camera(60., scene.scene.bounding_box, [25, 0, 0])
scene.look_at([25, 0, 0], [-15, 0, 30], [0, 0, 1])
app.run()
PYTHON
