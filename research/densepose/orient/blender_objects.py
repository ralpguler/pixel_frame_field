"""OBJECT PLACEMENT (stage 2) -- Cycles render of canonical-anchored objects posed by predicted R.

Consumes objects.json from viz_objects_bake.py. Each anchor gets a procedural object built in its
LOCAL frame (X = tangent, Y = bitangent, Z = outward normal) and posed per frame by the tracked
6-DoF anchor state: watch (twist readout on the wrist), badge (correspondence stability), crown,
butterflies (wings flap about the LOCAL tangent axis -- the frame is a coordinate system you can
animate in), map pins (normal direction, no aesthetics). Backdrop keeps the photo readable (AR).

  blender --background --python orient/blender_objects.py -- OBJ_DIR DUMP_DIR OUT_DIR
      [--samples 48] [--maxframes 0] [--still K]
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
OBJ, DUMP, OUT = argv[0], argv[1], argv[2]


def opt(name, default, cast):
    return cast(argv[argv.index(name) + 1]) if name in argv else default


SAMPLES = opt("--samples", 48, int)
MAXF = opt("--maxframes", 0, int)
STILL = opt("--still", -1, int)
os.makedirs(OUT, exist_ok=True)

data = json.load(open(f"{OBJ}/objects.json"))
ANCH = data["anchors"]; FR = data["frames"]; W = data["W"]
fids = sorted(int(k) for k in FR.keys())
if MAXF:
    fids = fids[:MAXF]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
sc.cycles.device = "CPU"
sc.render.resolution_x = W; sc.render.resolution_y = W
sc.render.image_settings.file_format = "PNG"

cam = bpy.data.cameras.new("cam"); cam.type = "ORTHO"; cam.ortho_scale = 1.0
camo = bpy.data.objects.new("cam", cam); camo.location = (0, 0, 2.0)
sc.collection.objects.link(camo); sc.camera = camo

I_alb = bpy.data.images.load(f"{DUMP}/f_{fids[0]:04d}.jpg")
mb = bpy.data.materials.new("ghost"); mb.use_nodes = True
ntb = mb.node_tree; ntb.nodes.clear()
o_ = ntb.nodes.new("ShaderNodeOutputMaterial")
tb = ntb.nodes.new("ShaderNodeTexImage"); tb.image = I_alb
em = ntb.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 0.32
ntb.links.new(tb.outputs["Color"], em.inputs["Color"])
ntb.links.new(em.outputs["Emission"], o_.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
bpy.context.active_object.data.materials.append(mb)


def mkmat(name, color, metal=0.9, rough=0.25, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if emit and "Emission Strength" in b.inputs:
        b.inputs["Emission Strength"].default_value = emit
        b.inputs["Emission Color"].default_value = (*color, 1)
    return m


GOLD = mkmat("gold", (1.0, 0.72, 0.25), 1.0, 0.18)
STEEL = mkmat("steel", (0.85, 0.87, 0.92), 1.0, 0.12)
RED = mkmat("red", (0.85, 0.08, 0.10), 0.1, 0.35)
DARK = mkmat("dark", (0.03, 0.03, 0.04), 0.3, 0.4)
WING1 = mkmat("wing1", (0.95, 0.35, 0.75), 0.0, 0.5, emit=0.10)
WING2 = mkmat("wing2", (0.30, 0.75, 0.95), 0.0, 0.5, emit=0.10)


def prim(fn, mat, parent, name, **kw):
    fn(**kw)
    o = bpy.context.active_object
    o.name = name; o.data.materials.append(mat); o.parent = parent
    return o


def empty(name):
    e = bpy.data.objects.new(name, None); sc.collection.objects.link(e)
    return e


def build(kind, name):
    """Objects live in mm units in their local frame (X=t, Y=b, Z=n outward), origin ON the skin."""
    root = empty(name)
    wings = []
    C = bpy.ops.mesh
    if kind == "badge":
        prim(C.primitive_cylinder_add, GOLD, root, name + "_d", radius=46, depth=8,
             location=(0, 0, 4))
        prim(C.primitive_torus_add, DARK, root, name + "_r", major_radius=46, minor_radius=4,
             location=(0, 0, 7))
        prim(C.primitive_cone_add, GOLD, root, name + "_s", radius1=14, depth=15, location=(0, 0, 13), rotation=(math.pi, 0, 0))
    elif kind == "watch":
        prim(C.primitive_cylinder_add, DARK, root, name + "_f", radius=28, depth=7, location=(0, 0, 5))
        prim(C.primitive_torus_add, GOLD, root, name + "_r", major_radius=29, minor_radius=4.5,
             location=(0, 0, 8))
        prim(C.primitive_cube_add, STEEL, root, name + "_h", size=1, location=(0, 9, 10),
             scale=(2.4, 18, 1.8))                              # minute hand along +Y (bitangent)
        prim(C.primitive_cube_add, STEEL, root, name + "_h2", size=1, location=(6, 0, 10),
             scale=(12, 2.4, 1.8))                               # hour hand along +X (tangent)
        for s in (-1, 1):
            prim(C.primitive_cube_add, DARK, root, name + f"_b{s}", size=1,
                 location=(0, s * 38, 0), scale=(24, 20, 6))
    elif kind == "crown":
        prim(C.primitive_torus_add, GOLD, root, name + "_b", major_radius=78, minor_radius=10,
             location=(0, 0, 9))
        for i in range(6):
            a2 = i / 6 * 2 * math.pi
            prim(C.primitive_cone_add, GOLD, root, name + f"_c{i}", radius1=13, depth=45,
                 location=(78 * math.cos(a2), 78 * math.sin(a2), 27))
    elif kind == "butterfly":
        prim(C.primitive_cylinder_add, DARK, root, name + "_body", radius=6, depth=52,
             rotation=(0, math.pi / 2, 0), location=(0, 0, 11))
        for s, mt in ((1, WING1), (-1, WING2)):
            h = empty(name + f"_hinge{s}")
            h.parent = root; h.location = (0, 0, 11)
            w = prim(C.primitive_plane_add, mt, h, name + f"_w{s}", size=1)
            w.scale = (46, 34, 1); w.location = (0, s * 35, 0)
            wings.append((h, s))
    elif kind == "pin":
        prim(C.primitive_cone_add, STEEL, root, name + "_n", radius1=3.5, depth=40, location=(0, 0, 20))
        prim(C.primitive_uv_sphere_add, RED, root, name + "_b", radius=13, location=(0, 0, 46))
    return root, wings


objs = []
for an in ANCH:
    root, wings = build(an["kind"], an["name"])
    objs.append({"an": an, "root": root, "wings": wings})

tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt)


def area(name, energy, size, color=(1, 1, 1)):
    la = bpy.data.lights.new(name, "AREA"); la.energy = energy; la.size = size; la.color = color
    o = bpy.data.objects.new(name, la); sc.collection.objects.link(o)
    c = o.constraints.new("TRACK_TO"); c.target = tgt
    c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return o


key = area("key", 200, 0.6)
fill = area("fill", 35, 1.6, (0.8, 0.85, 1.0)); fill.location = (-1.2, -0.4, 1.6)
rim = area("rim", 60, 0.4, (1.0, 0.85, 0.7)); rim.location = (1.4, 1.2, 0.7)
sc.world = bpy.data.worlds.new("w"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.01, 0.01, 0.015, 1)

todo = [fids[STILL]] if STILL >= 0 else fids
NF = max(len(fids) - 1, 1)
for j, fi in enumerate(todo):
    I_alb.filepath = f"{DUMP}/f_{fi:04d}.jpg"; I_alb.reload()
    st = FR[str(fi)]
    k = fids.index(fi)
    for o, s_ in zip(objs, st):
        root = o["root"]
        if not s_["vis"]:
            root.hide_render = True
            for ch in root.children_recursive:
                ch.hide_render = True
            continue
        root.hide_render = False
        for ch in root.children_recursive:
            ch.hide_render = False
        R = s_["R"]                                             # rows are the t/b/n BLENDER vectors
        mmscale = s_["ppmm"] / W                                # mm -> plane units
        M = Matrix(((R[0][0], R[1][0], R[2][0], s_["x"]),
                    (R[0][1], R[1][1], R[2][1], s_["y"]),
                    (R[0][2], R[1][2], R[2][2], 0.004),
                    (0, 0, 0, 1)))
        root.matrix_world = M @ Matrix.Scale(mmscale, 4)
        for h, sgn in o["wings"]:
            flap = 0.15 + 0.55 * abs(math.sin(2 * math.pi * (k * 0.16 + 0.13 * sgn)))
            h.rotation_euler = (sgn * flap, 0, 0)
    az = math.radians(-50 + 100 * (k / NF))
    key.location = (2.1 * math.sin(az), 0.9, 2.1 * math.cos(az))
    sc.render.filepath = f"{OUT}/objects_{fi:04d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"[obj] rendered {j + 1}/{len(todo)} (frame {fi})", flush=True)
print("OBJECTS RENDER DONE", flush=True)
