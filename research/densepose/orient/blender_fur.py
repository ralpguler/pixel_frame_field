"""FUR (stage 2) -- Cycles render of the baked 3D strands: real hair geometry, really lit.

Consumes strands_XXXX.bin from viz_fur_bake.py (raw float32; Blender's python has no numpy).
Per frame a Curves object is rebuilt (strand counts vary with visibility), per-curve colours ride
an attribute into the hair shader, and the light rig sweeps -- fur that is shaded, self-shadowing
and glossy along the strand: everything the flat OpenCV composites were missing.

  blender --background --python orient/blender_fur.py -- STRANDS_DIR DUMP_DIR OUT_DIR
      [--samples 48] [--maxframes 0] [--still K]
"""
import math
import os
import struct
import sys
from array import array

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
STR, DUMP, OUT = argv[0], argv[1], argv[2]


def opt(name, default, cast):
    return cast(argv[argv.index(name) + 1]) if name in argv else default


SAMPLES = opt("--samples", 48, int)
MAXF = opt("--maxframes", 0, int)
STILL = opt("--still", -1, int)
SHADER = opt("--shader", "hair", str)       # hair: Principled Hair (soft angora) | principled: saturated tinsel
os.makedirs(OUT, exist_ok=True)

fids = sorted(int(f[8:12]) for f in os.listdir(STR) if f.startswith("strands_") and f.endswith(".bin"))
if MAXF:
    fids = fids[:MAXF]

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
sc.cycles.device = "CPU"

probe = bpy.data.images.load(f"{DUMP}/f_{fids[0]:04d}.jpg")
W, H = probe.size
bpy.data.images.remove(probe)
sc.render.resolution_x = W; sc.render.resolution_y = H
sc.render.image_settings.file_format = "PNG"

cam = bpy.data.cameras.new("cam"); cam.type = "ORTHO"; cam.ortho_scale = 1.0
camo = bpy.data.objects.new("cam", cam); camo.location = (0, 0, 2.0)
sc.collection.objects.link(camo); sc.camera = camo

# ---------- backdrop: dim ghost of the photo ----------
I_alb = bpy.data.images.load(f"{DUMP}/f_{fids[0]:04d}.jpg")
mb = bpy.data.materials.new("ghost"); mb.use_nodes = True
ntb = mb.node_tree; ntb.nodes.clear()
ob_ = ntb.nodes.new("ShaderNodeOutputMaterial")
tb = ntb.nodes.new("ShaderNodeTexImage"); tb.image = I_alb
em = ntb.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 0.10
ntb.links.new(tb.outputs["Color"], em.inputs["Color"])
ntb.links.new(em.outputs["Emission"], ob_.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
back = bpy.context.active_object; back.data.materials.append(mb)

# ---------- hair material: colour from per-curve attribute ----------
mat = bpy.data.materials.new("fur"); mat.use_nodes = True
nt = mat.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "col"
shader = None
try:
    if SHADER != "hair":
        raise RuntimeError("principled requested")
    hb = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    try:
        hb.parametrization = "COLOR"
    except Exception:
        pass
    if "Color" in hb.inputs:
        nt.links.new(attr.outputs["Color"], hb.inputs["Color"])
    for nm, v in (("Roughness", 0.30), ("Radial Roughness", 0.25), ("Coat", 0.1), ("IOR", 1.55)):
        if nm in hb.inputs:
            hb.inputs[nm].default_value = v
    shader = hb
    print("[fur] Principled Hair BSDF")
except Exception as e:
    print("[fur] hair bsdf unavailable, falling back to Principled:", e)
    pb = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(attr.outputs["Color"], pb.inputs["Base Color"])
    pb.inputs["Roughness"].default_value = 0.45
    shader = pb
nt.links.new(shader.outputs[0], out.inputs["Surface"])

# ---------- lights (satin rig, slightly softened) ----------
tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt)


def area(name, energy, size, color=(1, 1, 1)):
    la = bpy.data.lights.new(name, "AREA"); la.energy = energy; la.size = size; la.color = color
    o = bpy.data.objects.new(name, la); sc.collection.objects.link(o)
    c = o.constraints.new("TRACK_TO"); c.target = tgt
    c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return o


key = area("key", 140, 0.6)
fill = area("fill", 25, 1.6, (0.75, 0.82, 1.0)); fill.location = (-1.2, -0.4, 1.6)
rim = area("rim", 60, 0.4, (1.0, 0.85, 0.7)); rim.location = (1.4, 1.2, 0.7)
sc.world = bpy.data.worlds.new("w"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.005, 0.009, 1)


def read_bin(path):
    with open(path, "rb") as f:
        N, S = struct.unpack("<ii", f.read(8))
        pos = array("f"); pos.fromfile(f, N * S * 3)
        col = array("f"); col.fromfile(f, N * 3)
        rad = array("f"); rad.fromfile(f, S)
    return N, S, pos, col, rad


fur_ob = None


def build_curves(N, S, pos, col, rad):
    global fur_ob
    cd = bpy.data.hair_curves.new("fur")
    cd.add_curves([S] * N)
    cd.attributes["position"].data.foreach_set("vector", pos)
    r = cd.attributes.new("radius", "FLOAT", "POINT")
    r.data.foreach_set("value", array("f", list(rad) * N))
    c = cd.attributes.new("col", "FLOAT_COLOR", "CURVE")
    rgba = array("f", [0.0]) * (N * 4)
    for i in range(N):
        rgba[4 * i] = col[3 * i]; rgba[4 * i + 1] = col[3 * i + 1]
        rgba[4 * i + 2] = col[3 * i + 2]; rgba[4 * i + 3] = 1.0
    c.data.foreach_set("color", rgba)
    cd.materials.append(mat)
    if fur_ob is None:
        fur_ob = bpy.data.objects.new("fur", cd)
        sc.collection.objects.link(fur_ob)
    else:
        old = fur_ob.data
        fur_ob.data = cd
        bpy.data.hair_curves.remove(old)


todo = [fids[STILL]] if STILL >= 0 else fids
NF = max(len(fids) - 1, 1)
for j, fi in enumerate(todo):
    N, S, pos, col, rad = read_bin(f"{STR}/strands_{fi:04d}.bin")
    build_curves(N, S, pos, col, rad)
    I_alb.filepath = f"{DUMP}/f_{fi:04d}.jpg"; I_alb.reload()
    k = fids.index(fi)
    az = math.radians(-65 + 130 * (k / NF))
    el = math.radians(28 + 14 * math.sin(2 * math.pi * k / NF))
    r = 2.1
    key.location = (r * math.sin(az) * math.cos(el), r * math.sin(el), r * math.cos(az) * math.cos(el))
    sc.render.filepath = f"{OUT}/fur{'' if SHADER == 'hair' else '_p'}_{fi:04d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"[fur] rendered {j + 1}/{len(todo)} (frame {fi}, {N} strands)", flush=True)
print("FUR RENDER DONE", flush=True)
