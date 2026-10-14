"""Blender reference render: the canonical TEMPLATE body with the SAME fixed FPS points and their surface
frames (tripods). Verifies the frames on known geometry — the blue NORMAL must radiate out of the surface.
Run: blender --background --python blender_template.py -- DIR   (needs tmpl_body.obj + tmpl_pts.csv in DIR)
"""
import bpy, os, sys, csv, math
from mathutils import Quaternion

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
DIR = os.path.expanduser(argv[0]) if argv else os.path.expanduser("~/framefield_template")
BODY = f"{DIR}/tmpl_body.obj"; PTS = f"{DIR}/tmpl_pts.csv"
ARROW = 0.055; EMISSION = 0.1; AXL = 0.85


def clean():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials): [c.remove(b) for b in list(c)]


def make_arrow(color, length, shaft_r=0.035, head_r=0.11, head_frac=0.42, name="a"):
    sh = length * (1 - head_frac)
    bpy.ops.mesh.primitive_cylinder_add(radius=shaft_r, depth=sh, vertices=16, location=(0, 0, sh/2)); s = bpy.context.object
    bpy.ops.mesh.primitive_cone_add(radius1=head_r, radius2=0, depth=length*head_frac, vertices=18, location=(0, 0, sh + length*head_frac/2)); c = bpy.context.object
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


clean(); sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 48
sc.view_settings.view_transform = "Standard"
sc.world.use_nodes = True
bg = sc.world.node_tree.nodes.get("Background")
if bg: bg.inputs[0].default_value = (0.03, 0.03, 0.035, 1)

# template body — build the mesh DIRECTLY from the OBJ verts (no importer axis conversion, so it shares
# the exact coordinate system as the points + camera; the importer's Y-up->Z-up rotation was flipping it)
verts, facesL = [], []
for line in open(BODY):
    if line.startswith("v "):
        verts.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("f "):
        facesL.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
me = bpy.data.meshes.new("bodymesh"); me.from_pydata(verts, [], facesL); me.update()
body = bpy.data.objects.new("body", me); bpy.context.collection.objects.link(body)
bm = bpy.data.materials.new("body"); bm.use_nodes = True
bb = bm.node_tree.nodes["Principled BSDF"]; bb.inputs["Base Color"].default_value = (0.12, 0.12, 0.14, 1)
if "Roughness" in bb.inputs: bb.inputs["Roughness"].default_value = 0.8
body.data.materials.append(bm); bpy.ops.object.shade_smooth()

# tripods at the fixed points, oriented by the template surface frame
trip = make_tripod(); trip.hide_render = True; trip.hide_set(True)
coll = bpy.context.collection
for r in list(csv.reader(open(PTS)))[1:]:
    x, y, z, qw, qx, qy, qz, tw = map(float, r)
    o = trip.copy(); coll.objects.link(o); o.hide_render = False; o.hide_set(False)
    o.location = (x, y, z); o.scale = (ARROW, ARROW, ARROW)
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = Quaternion((qw, qx, qy, qz))

# perspective camera aimed at the body via Track-To; use the front-facing position from tmpl_cam.txt if present
camloc = (1.7, -2.4, 0.3)
if os.path.exists(f"{DIR}/tmpl_cam.txt"):
    camloc = tuple(float(x) for x in open(f"{DIR}/tmpl_cam.txt").read().strip().split(","))
target = bpy.data.objects.new("tgt", None); coll.objects.link(target); target.location = (0, 0, 0)
bpy.ops.object.camera_add(location=camloc); cam = bpy.context.object; cam.data.lens = 60; sc.camera = cam
tc = cam.constraints.new("TRACK_TO"); tc.target = target; tc.track_axis = "TRACK_NEGATIVE_Z"; tc.up_axis = "UP_Y"
for loc, en in [((2, -2, 2), 120), ((-2, -1, 1), 60), ((0, 2, 2), 50)]:
    bpy.ops.object.light_add(type="AREA", location=loc); bpy.context.object.data.energy = en; bpy.context.object.data.size = 4
sc.render.resolution_x = 800; sc.render.resolution_y = 800
sc.render.filepath = f"{DIR}/render_template.png"; sc.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print("RENDERED", sc.render.filepath)
