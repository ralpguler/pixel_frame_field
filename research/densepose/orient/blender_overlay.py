"""Blender: render the TRACKED framefield as real shaded 3D tripods overlaid on the photo, aligned to the
body. Consumes the export from  framefield_overlay.py --export-blender DIR  (bg_XXXX.png + pts_XXXX.csv,
where each CSV row = u,v (normalized image coords, v up), qw,qx,qy,qz (frame in Blender-camera basis), twist).

Run:  blender --background --python blender_overlay.py -- DIR [FRAME]
  or open Blender, set DIR below, and run in the Scripting tab (drop --background to look around).

Why Blender: the arrows point along the surface normal, so on a frontal photo many point at the camera —
a flat 2D projector collapses them to slivers, but Blender's real camera renders them correctly (a cone
seen head-on is a shaded disc, at grazing angles a full arrow). Positions come from the tracked pixels;
orientation from the predicted per-point frame; darkening + glow make them shine on the dimmed photo.
"""
import bpy, os, sys, csv, math
from mathutils import Quaternion, Vector, Euler

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
DIR = os.path.expanduser(argv[0]) if argv else os.path.expanduser("~/framefield_overlay")
FRAME = argv[1] if len(argv) > 1 else sorted(f[3:7] for f in os.listdir(DIR) if f.startswith("pts_"))[0]
BG = f"{DIR}/bg_{FRAME}.png"; PTS = f"{DIR}/pts_{FRAME}.csv"
DARK = 0.22            # photo dimming so arrows shine
ARROW = 0.040         # tripod size (fraction of image width)
EMISSION = 0.10       # tiny self-glow; 3D form comes from the area lights (shading)
TILT = (0, 0)         # fixed view-tilt (deg, x/y); 0 = show TRUE directions (normal points straight out)
TRIPOD = True         # True = RGB coordinate tripod; False = single twist-colored arrow


def clean():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.node_groups):
        for b in list(c): c.remove(b)


def make_arrow(color, length=1.0, shaft_r=0.035, head_r=0.11, head_frac=0.42, name="arrow"):
    """chunky +Z arrow (shaft cylinder + cone head), single material `color` (emissive)."""
    sh = length * (1 - head_frac)
    bpy.ops.mesh.primitive_cylinder_add(radius=shaft_r, depth=sh, vertices=16, location=(0, 0, sh/2))
    shaft = bpy.context.object
    bpy.ops.mesh.primitive_cone_add(radius1=head_r, radius2=0, depth=length*head_frac, vertices=18,
                                    location=(0, 0, sh + length*head_frac/2))
    cone = bpy.context.object
    shaft.select_set(True); cone.select_set(True); bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join(); shaft.name = name
    m = bpy.data.materials.new(name + "_mat"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    if "Roughness" in b.inputs: b.inputs["Roughness"].default_value = 0.35
    for k in ("Emission Color", "Emission"):
        if k in b.inputs: b.inputs[k].default_value = (*color, 1)
    if "Emission Strength" in b.inputs: b.inputs["Emission Strength"].default_value = EMISSION
    shaft.data.materials.append(m)
    bpy.ops.object.shade_smooth()
    return shaft


def make_tripod():
    """ONE joined mesh with 3 arrows (X=red, Y=green, Z=blue/normal longest) + 3 material slots.
    Muted colors (not neon) so they don't blow out."""
    AXL = 0.85                                              # equal-length orthonormal tripod (all unit vectors)
    ax = make_arrow((0.70, 0.16, 0.16), AXL, name="ax"); ax.rotation_euler = (0, math.radians(90), 0)
    ay = make_arrow((0.20, 0.62, 0.26), AXL, name="ay"); ay.rotation_euler = (math.radians(-90), 0, 0)
    az = make_arrow((0.30, 0.55, 1.0), AXL, name="az")     # NORMAL = blue (Z)
    bpy.ops.object.select_all(action="DESELECT")
    for o in (ax, ay, az): o.select_set(True)
    bpy.context.view_layer.objects.active = ax
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)   # bake orientations
    bpy.ops.object.join()                                                       # -> single mesh, 3 mat slots
    ax.name = "tripod"; return ax


clean()
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = 48
try: sc.cycles.use_denoising = True
except Exception: pass
sc.view_settings.view_transform = "Standard"                     # keep colors punchy, no filmic desat

# --- background photo as a dimmed emissive plane (emission color=image, strength=DARK) ---
img = bpy.data.images.load(BG); W, H = img.size; aspect = H / W
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.05))
plane = bpy.context.object; plane.scale = (1.0, aspect, 1)          # span the ortho view (X: -0.5..0.5)
pm = bpy.data.materials.new("photo"); pm.use_nodes = True; nt = pm.node_tree; nt.nodes.clear()
tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
emit = nt.nodes.new("ShaderNodeEmission"); emit.inputs["Strength"].default_value = DARK
out = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(tex.outputs["Color"], emit.inputs["Color"]); nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
plane.data.materials.append(pm)

# --- ortho camera framing the unit-width plane, looking down -Z ---
bpy.ops.object.camera_add(location=(0, 0, 3)); cam = bpy.context.object
cam.data.type = "ORTHO"; cam.data.ortho_scale = 1.0; cam.rotation_euler = (0, 0, 0); sc.camera = cam
sc.render.resolution_x = W; sc.render.resolution_y = H
sc.world.use_nodes = True
bg = sc.world.node_tree.nodes.get("Background")
if bg: bg.inputs[0].default_value = (0, 0, 0, 1)

# --- place a tripod at each tracked point, oriented by its frame quaternion ---
trip = make_tripod(); trip.hide_render = True; trip.hide_set(True)
rows = list(csv.reader(open(PTS)))[1:]
coll = bpy.context.collection
tilt = Euler((math.radians(TILT[0]), math.radians(TILT[1]), 0)).to_quaternion()   # fixed off-axis view
for i, r in enumerate(rows):
    u, v, qw, qx, qy, qz, tw = map(float, r)
    x = (u - 0.5); y = (v - 0.5) * aspect                     # plane coords span -0.5..0.5 (v already up)
    o = trip.copy()                                            # linked-mesh duplicate (shares geometry+materials)
    coll.objects.link(o); o.hide_render = False; o.hide_set(False)
    o.location = (x, y, 0.02); o.scale = (ARROW, ARROW, ARROW)
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = tilt @ Quaternion((qw, qx, qy, qz))

# soft lights so the tripods are SHADED (show 3D form) without blowing out to white
for loc, en, sz in [((0.7, 0.5, 1.5), 22, 2.5), ((-0.9, 0.1, 1.0), 11, 3), ((0.1, -0.9, 1.0), 8, 3)]:
    bpy.ops.object.light_add(type="AREA", location=loc); Lt = bpy.context.object
    Lt.data.energy = en; Lt.data.size = sz
print(f"[blender_overlay] {len(rows)} tracked tripods placed from {os.path.basename(PTS)} on {W}x{H} photo")
sc.render.filepath = f"{DIR}/render_{FRAME}.png"; sc.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print("RENDERED", sc.render.filepath)
