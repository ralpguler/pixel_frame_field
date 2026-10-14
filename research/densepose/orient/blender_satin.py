"""BEAUTY CONTEST #3 (stage 2) -- Cycles anisotropic satin relight of the predicted frame field.

A camera-facing plane whose Principled BSDF is driven entirely by the model's per-pixel prediction:
Normal socket <- baked predicted normal, Tangent socket <- baked predicted tangent. The anisotropic
highlight stretches PERPENDICULAR to the tangent and rotates with it, so the sheen combing around a
limb is the network's twist channel, physically lit. A key area light sweeps across the body over
the clip; a dim ghost of the photo sits behind.

Run (inside the container):
  blender --background --python orient/blender_satin.py -- BAKE_DIR DUMP_DIR OUT_DIR [--samples 48]
      [--maxframes 0] [--still K]   (--still: render only the K-th frame, for look iteration)
"""
import math
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
BAKE, DUMP, OUT = argv[0], argv[1], argv[2]


def opt(name, default, cast):
    return cast(argv[argv.index(name) + 1]) if name in argv else default


SAMPLES = opt("--samples", 48, int)
MAXF = opt("--maxframes", 0, int)
STILL = opt("--still", -1, int)
LOOK = opt("--look", "satin", str)          # satin: sweeping key light | chrome: rotating Nishita sky
os.makedirs(OUT, exist_ok=True)

fids = sorted(int(f[4:8]) for f in os.listdir(BAKE) if f.startswith("nrm_"))
if MAXF:
    fids = fids[:MAXF]

# ---------- clean scene ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
prefs = bpy.context.preferences.addons.get("cycles")
sc.cycles.device = "CPU"

probe = bpy.data.images.load(f"{BAKE}/msk_{fids[0]:04d}.png")
W, H = probe.size
bpy.data.images.remove(probe)
sc.render.resolution_x = W; sc.render.resolution_y = H
sc.render.image_settings.file_format = "PNG"

# ---------- camera ----------
cam = bpy.data.cameras.new("cam"); cam.type = "ORTHO"; cam.ortho_scale = 1.0
camo = bpy.data.objects.new("cam", cam); camo.location = (0, 0, 2.0)
sc.collection.objects.link(camo); sc.camera = camo

# ---------- images (one datablock per role, filepath swapped per frame) ----------
def im(path, noncolor):
    img = bpy.data.images.load(path)
    if noncolor:
        img.colorspace_settings.name = "Non-Color"
    return img


I_alb = im(f"{DUMP}/f_{fids[0]:04d}.jpg", False)
I_nrm = im(f"{BAKE}/nrm_{fids[0]:04d}.png", True)
I_tan = im(f"{BAKE}/tan_{fids[0]:04d}.png", True)
I_msk = im(f"{BAKE}/msk_{fids[0]:04d}.png", True)

# ---------- body plane material ----------
mat = bpy.data.materials.new("satin"); mat.use_nodes = True
nt = mat.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")


def tex(img):
    n = nt.nodes.new("ShaderNodeTexImage"); n.image = img; n.interpolation = "Linear"
    return n


def decode(img):
    """[0,1] png -> unit vector: *2-1 then normalize."""
    t = tex(img)
    mad = nt.nodes.new("ShaderNodeVectorMath"); mad.operation = "MULTIPLY_ADD"
    mad.inputs[1].default_value = (2, 2, 2); mad.inputs[2].default_value = (-1, -1, -1)
    nrm = nt.nodes.new("ShaderNodeVectorMath"); nrm.operation = "NORMALIZE"
    nt.links.new(t.outputs["Color"], mad.inputs[0])
    nt.links.new(mad.outputs["Vector"], nrm.inputs[0])
    return nrm.outputs["Vector"]


bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")


def set_in(node, name, val):
    if name in node.inputs:
        node.inputs[name].default_value = val
        return True
    return False


t_alb = tex(I_alb)
if LOOK == "chrome":
    I_met = im(f"{BAKE}/met_{fids[0]:04d}.png", True)
    t_met = tex(I_met)
    mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
    mixc.inputs["B"].default_value = (0.92, 0.94, 0.99, 1)      # chrome tint where metallic
    nt.links.new(t_met.outputs["Color"], mixc.inputs["Factor"])
    nt.links.new(t_alb.outputs["Color"], mixc.inputs["A"])
    nt.links.new(mixc.outputs["Result"], bsdf.inputs["Base Color"])
    nt.links.new(t_met.outputs["Color"], bsdf.inputs["Metallic"])
    mixr = nt.nodes.new("ShaderNodeMapRange")                   # roughness: 0.45 skin -> 0.12 chrome
    mixr.inputs["To Min"].default_value = 0.45
    mixr.inputs["To Max"].default_value = 0.12
    nt.links.new(t_met.outputs["Color"], mixr.inputs["Value"])
    nt.links.new(mixr.outputs["Result"], bsdf.inputs["Roughness"])
else:
    I_met = None
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    set_in(hsv, "Saturation", 0.75)
    nt.links.new(t_alb.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    set_in(bsdf, "Metallic", 0.35)
    set_in(bsdf, "Roughness", 0.30)
    if not set_in(bsdf, "Anisotropic", 0.95):
        set_in(bsdf, "Anisotropy", 0.95)
    set_in(bsdf, "Sheen Weight", 0.25)
nt.links.new(decode(I_nrm), bsdf.inputs["Normal"])
nt.links.new(decode(I_tan), bsdf.inputs["Tangent"])

t_msk = tex(I_msk)
trans = nt.nodes.new("ShaderNodeBsdfTransparent")
mix = nt.nodes.new("ShaderNodeMixShader")
nt.links.new(t_msk.outputs["Color"], mix.inputs["Fac"])
nt.links.new(trans.outputs["BSDF"], mix.inputs[1])
nt.links.new(bsdf.outputs["BSDF"], mix.inputs[2])
nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
body = bpy.context.active_object; body.data.materials.append(mat)

# ---------- backdrop: dim emissive ghost of the photo ----------
mb = bpy.data.materials.new("ghost"); mb.use_nodes = True
ntb = mb.node_tree; ntb.nodes.clear()
ob = ntb.nodes.new("ShaderNodeOutputMaterial")
tb = ntb.nodes.new("ShaderNodeTexImage"); tb.image = I_alb
em = ntb.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = 0.10
ntb.links.new(tb.outputs["Color"], em.inputs["Color"])
ntb.links.new(em.outputs["Emission"], ob.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, -0.02))
back = bpy.context.active_object; back.data.materials.append(mb)

# ---------- lights: sweeping key + static fill + rim ----------
tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt)


def area(name, energy, size, color=(1, 1, 1)):
    la = bpy.data.lights.new(name, "AREA"); la.energy = energy; la.size = size; la.color = color
    o = bpy.data.objects.new(name, la); sc.collection.objects.link(o)
    c = o.constraints.new("TRACK_TO"); c.target = tgt
    c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return o


# chrome is lit by the environment; big area lights would dominate a mirror as huge white blobs
key = area("key", 320 if LOOK == "satin" else 12, 0.55)
fill = area("fill", 45 if LOOK == "satin" else 6, 1.6, (0.75, 0.82, 1.0)); fill.location = (-1.2, -0.4, 1.6)
rim = area("rim", 90 if LOOK == "satin" else 15, 0.4, (1.0, 0.85, 0.7)); rim.location = (1.4, 1.2, 0.7)

sc.world = bpy.data.worlds.new("w")
sc.world.use_nodes = True
sky = None
if LOOK == "chrome":
    wt = sc.world.node_tree
    sky = wt.nodes.new("ShaderNodeTexSky")
    # Blender 5.0 sky outputs PHYSICAL radiance (thousands of W/m^2/sr) -> tiny strength
    for attr, val in (("sky_type", "MULTIPLE_SCATTERING"), ("sun_elevation", 0.30),
                      ("sun_intensity", 0.15), ("sun_size", 0.06)):
        try:
            setattr(sky, attr, val)
        except Exception as e:
            print(f"sky.{attr} skipped: {e}")
    wt.links.new(sky.outputs["Color"], wt.nodes["Background"].inputs["Color"])
    wt.nodes["Background"].inputs["Strength"].default_value = 0.0012
else:
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.005, 0.009, 1)

# ---------- render loop ----------  (bloom is applied outside, in the encode step)
todo = [fids[STILL]] if STILL >= 0 else fids
NF = max(len(fids) - 1, 1)
for j, fi in enumerate(todo):
    I_alb.filepath = f"{DUMP}/f_{fi:04d}.jpg"; I_alb.reload()
    I_nrm.filepath = f"{BAKE}/nrm_{fi:04d}.png"; I_nrm.reload()
    I_tan.filepath = f"{BAKE}/tan_{fi:04d}.png"; I_tan.reload()
    I_msk.filepath = f"{BAKE}/msk_{fi:04d}.png"; I_msk.reload()
    if LOOK == "chrome" and I_met is not None:
        I_met.filepath = f"{BAKE}/met_{fi:04d}.png"; I_met.reload()
    k = fids.index(fi)
    az = math.radians(-65 + 130 * (k / NF))                     # one full L->R sweep over the clip
    el = math.radians(28 + 14 * math.sin(2 * math.pi * k / NF))
    r = 2.1
    key.location = (r * math.sin(az) * math.cos(el), r * math.sin(el), r * math.cos(az) * math.cos(el))
    if sky is not None:                                         # chrome: the environment itself rotates
        try:
            sky.sun_rotation = math.radians(360.0 * k / NF)
            sky.sun_elevation = math.radians(14 + 10 * math.sin(2 * math.pi * k / NF))
        except Exception:
            pass
    sc.render.filepath = f"{OUT}/{LOOK}_{fi:04d}.png"
    bpy.ops.render.render(write_still=True)
    print(f"[satin] rendered {j + 1}/{len(todo)} (frame {fi})", flush=True)
print("SATIN RENDER DONE", flush=True)
