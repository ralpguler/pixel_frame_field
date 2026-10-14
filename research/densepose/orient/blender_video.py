"""Blender: render the TRACKED framefield overlay for a whole clip in ONE session (fast).
Reads bg_XXXX.png + pts_XXXX.csv for every frame in DIR, renders render_XXXX.png each.
Run: blender --background --python blender_video.py -- DIR [samples]
"""
import bpy, os, sys, csv, math, glob
from mathutils import Quaternion, Euler

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
DIR = os.path.expanduser(argv[0]) if argv else os.path.expanduser("~/framefield_video")
SAMPLES = int(argv[1]) if len(argv) > 1 else 20
DARK = 0.20; ARROW = 0.032; EMISSION = 0.35; AXL = 0.85; TILT = (0, 0)


def make_arrow(color, length, shaft_r=0.035, head_r=0.11, head_frac=0.42, name="a"):
    sh = length * (1 - head_frac)
    bpy.ops.mesh.primitive_cylinder_add(radius=shaft_r, depth=sh, vertices=14, location=(0, 0, sh/2)); s = bpy.context.object
    bpy.ops.mesh.primitive_cone_add(radius1=head_r, radius2=0, depth=length*head_frac, vertices=16, location=(0, 0, sh + length*head_frac/2)); c = bpy.context.object
    s.select_set(True); c.select_set(True); bpy.context.view_layer.objects.active = s; bpy.ops.object.join(); s.name = name
    m = bpy.data.materials.new(name + "_m"); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    if "Roughness" in b.inputs: b.inputs["Roughness"].default_value = 0.35
    for k in ("Emission Color", "Emission"):
        if k in b.inputs: b.inputs[k].default_value = (*color, 1)
    if "Emission Strength" in b.inputs: b.inputs["Emission Strength"].default_value = EMISSION
    s.data.materials.append(m); bpy.ops.object.shade_smooth(); return s


def make_tripod():
    ax = make_arrow((0.70, 0.16, 0.16), AXL, name="ax"); ax.rotation_euler = (0, math.radians(90), 0)
    ay = make_arrow((0.20, 0.62, 0.26), AXL, name="ay"); ay.rotation_euler = (math.radians(-90), 0, 0)
    az = make_arrow((0.30, 0.55, 1.0), AXL, name="az")
    bpy.ops.object.select_all(action="DESELECT")
    for o in (ax, ay, az): o.select_set(True)
    bpy.context.view_layer.objects.active = ax
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True); bpy.ops.object.join(); ax.name = "tripod"; return ax


# --- one-time setup ---
bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = SAMPLES
try: sc.cycles.use_denoising = True
except Exception: pass
sc.view_settings.view_transform = "Standard"
sc.world.use_nodes = True
bgn = sc.world.node_tree.nodes.get("Background")
if bgn: bgn.inputs[0].default_value = (0, 0, 0, 1)
trip = make_tripod(); trip.hide_render = True; trip.hide_set(True)
tilt = Euler((math.radians(TILT[0]), math.radians(TILT[1]), 0)).to_quaternion()
# camera + lights (static; aspect from first frame)
frames = sorted(int(os.path.basename(f)[4:8]) for f in glob.glob(f"{DIR}/pts_*.csv"))
img0 = bpy.data.images.load(f"{DIR}/bg_{frames[0]:04d}.png"); W, H = img0.size; aspect = H / W
bpy.ops.object.camera_add(location=(0, 0, 3)); cam = bpy.context.object
cam.data.type = "ORTHO"; cam.data.ortho_scale = 1.0; cam.rotation_euler = (0, 0, 0); sc.camera = cam
sc.render.resolution_x = W; sc.render.resolution_y = H
for loc, en, sz in [((0.7, 0.5, 1.5), 22, 2.5), ((-0.9, 0.1, 1.0), 11, 3), ((0.1, -0.9, 1.0), 8, 3)]:
    bpy.ops.object.light_add(type="AREA", location=loc); Lt = bpy.context.object; Lt.data.energy = en; Lt.data.size = sz
sc.render.image_settings.file_format = "PNG"


def photo_mat(img):
    m = bpy.data.materials.new("photo"); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Strength"].default_value = DARK
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], e.inputs["Color"]); nt.links.new(e.outputs["Emission"], o.inputs["Surface"])
    return m


coll = bpy.context.collection
for fi in frames:
    ephemeral = [o for o in bpy.data.objects if o.get("frameobj")]
    bpy.ops.object.select_all(action="DESELECT")
    for o in ephemeral: o.select_set(True)
    if ephemeral: bpy.ops.object.delete()
    img = bpy.data.images.load(f"{DIR}/bg_{fi:04d}.png")
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.05)); plane = bpy.context.object
    plane.scale = (1.0, aspect, 1); plane["frameobj"] = 1; plane.data.materials.append(photo_mat(img))
    for r in list(csv.reader(open(f"{DIR}/pts_{fi:04d}.csv")))[1:]:
        u, v, qw, qx, qy, qz, tw = map(float, r)
        o = trip.copy(); coll.objects.link(o); o["frameobj"] = 1; o.hide_render = False; o.hide_set(False)
        o.location = ((u - 0.5), (v - 0.5) * aspect, 0.02); o.scale = (ARROW, ARROW, ARROW)
        o.rotation_mode = "QUATERNION"; o.rotation_quaternion = tilt @ Quaternion((qw, qx, qy, qz))
    sc.render.filepath = f"{DIR}/render_{fi:04d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED frame {fi}", flush=True)
print("VIDEO FRAMES DONE", flush=True)
