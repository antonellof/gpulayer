"""Render frames of a generated scene with Blender Cycles. Run by render.sh inside Blender:

    blender -b --factory-startup -P render.py -- DEVICE FRAMES SAMPLES [FIRST]

DEVICE is OPTIX, CUDA, HIP, METAL or CPU. A GPU device that Blender cannot use is an error,
not a silent fallback to the CPU. Frames are written to $GPULAYER_OUTPUT_DIR (or ./out).
"""

import math
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
device = (argv[0] if argv else "OPTIX").upper()
frames = int(argv[1]) if len(argv) > 1 else 4
samples = int(argv[2]) if len(argv) > 2 else 64
first = int(argv[3]) if len(argv) > 3 else 1
out_dir = os.environ.get("GPULAYER_OUTPUT_DIR", "out")

scene = bpy.context.scene
scene.render.engine = "CYCLES"
if device == "CPU":
    scene.cycles.device = "CPU"
    used = ["CPU"]
else:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = device
    prefs.get_devices()
    used = []
    for d in prefs.devices:
        d.use = d.type == device
        if d.use:
            used.append(d.name)
    if not used:
        found = sorted({f"{d.type}:{d.name}" for d in prefs.devices})
        sys.exit(f"no {device} device available to Blender (found: {found})")
    scene.cycles.device = "GPU"
print(f"blender {bpy.app.version_string}, device {device}: {', '.join(used)}", flush=True)

# A scene made here, so the job needs no input files: a subdivided monkey on a plane.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
bpy.ops.mesh.primitive_plane_add(size=20)
bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 1.2))
monkey = bpy.context.active_object
monkey.modifiers.new("subsurf", "SUBSURF").levels = 3
bpy.ops.object.shade_smooth()
mat = bpy.data.materials.new("glass")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.8, 0.3, 0.1, 1)
bsdf.inputs["Roughness"].default_value = 0.15
monkey.data.materials.append(mat)
bpy.ops.object.light_add(type="AREA", location=(3, -3, 6))
bpy.context.active_object.data.energy = 2000
bpy.ops.object.camera_add(location=(0, -7, 3.5), rotation=(math.radians(68), 0, 0))
scene.camera = bpy.context.active_object

scene.cycles.samples = samples
scene.cycles.use_denoising = False
scene.render.resolution_x, scene.render.resolution_y = 1280, 720
scene.render.image_settings.file_format = "PNG"
t_all = time.perf_counter()
for f in range(first, first + frames):
    monkey.rotation_euler[2] = math.radians(f * 15)
    scene.frame_set(f)
    scene.render.filepath = os.path.join(out_dir, f"frame_{f:04d}.png")
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    print(f"frame {f} rendered in {time.perf_counter() - t0:.2f}s", flush=True)
total = time.perf_counter() - t_all
print(f"RENDER device={device} frames={frames} samples={samples} seconds={total:.2f}", flush=True)
