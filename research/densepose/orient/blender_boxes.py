"""Render the pixel frame field as oriented boxes (paper Fig. 1); consumes orient/framefield_boxes.py output.

  apptainer exec -B /lus/lfs1aip2,/projects ~/data/containers/blender.sif blender --background \
      --python ~/codes/wan/densepose/orient/blender_boxes.py -- image|template DIR [--scale 2] [--samples 64] [--dark 0.45]

image     dimmed photo plane + one box per sampled pixel, oriented by the PREDICTED camera-space frame
template  canonical template body + the SAME boxes at their template points, oriented by the template's
          own frame (transparent film; composite on white); --tview auto faces the side the boxes are on;
          --frame glyphs zooms the ortho camera onto the exported glyphs (template half of a zoom pair)
Both modes also write screen_<mode>[_<glyph>].json: every glyph centre in saved-image pixels (for fig1_compose links).
Boxes carry their colour on the object (Object Info > Color), so all instances share one mesh + material.
"""
import bpy, os, sys, csv, math, json
from mathutils import Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MODE, DIR = argv[0], os.path.expanduser(argv[1])
opt = dict(zip(argv[2::2], argv[3::2]))
SCALE = float(opt.get("--scale", 2)); SAMPLES = int(opt.get("--samples", 64))
STRIPE = tuple(float(v) for v in opt.get("--stripe", "0.97,0.97,0.95").split(",")); STRIPE_DEG = float(opt.get("--stripe-deg", 28))
SIDE_N = None if opt.get("--side-n", "0.06,0.20,0.85") == "location" else tuple(float(v) for v in opt.get("--side-n", "0.06,0.20,0.85").split(","))   # faces normal to n: ONE colour (Riza 09-15)
SIDE_T = tuple(float(v) for v in opt.get("--side-t", "0.85,0.07,0.05").split(","))   # faces normal to t
SIDE_B = tuple(float(v) for v in opt.get("--side-b", "0.05,0.62,0.10").split(","))   # faces normal to b
GLYPH = opt.get("--glyph", "box")      # box | cube (painted sides) | tripod | cone_p (normal cone PAINTED with a stripe towards +t: full frame) | cone_n (normal only) | cone_t
DARK = float(opt.get("--dark", 0.40)); EMIT = float(opt.get("--emit", 0.05 if MODE == "image" else 0.0))
meta = json.load(open(f"{DIR}/meta.json"))

bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
for c in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.lights, bpy.data.cameras):
    for b in list(c): c.remove(b)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = SAMPLES
try: sc.cycles.use_denoising = True
except Exception: pass
sc.render.threads_mode = "FIXED"; sc.render.threads = 32
sc.view_settings.view_transform = "Standard"
sc.world.use_nodes = True
bgn = sc.world.node_tree.nodes.get("Background")
if bgn: bgn.inputs[0].default_value = (0.02, 0.02, 0.02, 1); bgn.inputs[1].default_value = 0.3


def make_tripod(L):
    """RGB frame: three EQUAL-LENGTH unit arrows (shaft + shaded cone head) along local X = t (red),
    Y = b (green), Z = n (blue), muted colours -- Riza's 2026-07-13 feedback: a full tripod (not single arrows),
    Blender-shaded cones, not too bright, not chunky, unit axes."""
    arms = []                                   # Riza 09-16: "colours so pastel" -> saturated + emissive + thicker (--tri-thick/--tri-emit)
    TK = float(opt.get("--tri-thick", 1.8)); TE = float(opt.get("--tri-emit", 0.35))
    for rot, col in (((0, math.radians(90), 0), (0.95, 0.04, 0.02)), ((math.radians(-90), 0, 0), (0.03, 0.78, 0.06)),
                     ((0, 0, 0), (0.03, 0.22, 1.00))):
        sh, hd = 0.62 * L, 0.38 * L
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05 * TK * L, depth=sh, vertices=16, location=(0, 0, sh / 2)); s_ = bpy.context.object
        bpy.ops.mesh.primitive_cone_add(radius1=0.13 * TK * L, radius2=0, depth=hd, vertices=20, location=(0, 0, sh + hd / 2)); c_ = bpy.context.object
        bpy.ops.object.select_all(action="DESELECT"); s_.select_set(True); c_.select_set(True)
        bpy.context.view_layer.objects.active = s_; bpy.ops.object.join()
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)   # pivot = arrow BASE (was the shaft middle: a "+" sign)
        s_.rotation_euler = rot; bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        m = bpy.data.materials.new(f"axis{len(arms)}"); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*col, 1); b.inputs["Roughness"].default_value = 0.4
        if "Emission Color" in b.inputs:
            b.inputs["Emission Color"].default_value = (*col, 1); b.inputs["Emission Strength"].default_value = TE
        s_.data.materials.append(m); bpy.ops.object.shade_smooth(); arms.append(s_)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.055 * (1 + TK) * L, segments=20, ring_count=10, location=(0, 0, 0)); o_ = bpy.context.object
    m = bpy.data.materials.new("origin"); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]   # shared origin ball
    b.inputs["Base Color"].default_value = (0.85, 0.85, 0.82, 1); b.inputs["Roughness"].default_value = 0.4
    o_.data.materials.append(m); bpy.ops.object.shade_smooth(); arms.append(o_)
    bpy.ops.object.select_all(action="DESELECT")
    for o in arms: o.select_set(True)
    bpy.context.view_layer.objects.active = arms[0]; bpy.ops.object.join()
    ob = arms[0]; ob.hide_render = True; ob.hide_set(True)
    for k in range(3):                                          # every arm must start at the origin: bounds ~[-0.14L, L]
        v = [p.co[k] for p in ob.data.vertices]; print(f"[tripod] axis {'xyz'[k]} bounds [{min(v)/L:+.2f}L, {max(v)/L:+.2f}L]")
    return ob


def make_cube(L):
    """cube of side L: faces normal to n ONE fixed colour (blue; --side-n location = old per-box colour), faces normal to t
    red, faces normal to b green (tripod convention) -- Riza 2026-09-15: "cubes with sides painted with
    different colour" (the elongated boxes read worse)."""
    bpy.ops.mesh.primitive_cube_add(size=L); ob = bpy.context.object
    bev = ob.modifiers.new("bev", "BEVEL"); bev.width = 0.10 * L; bev.segments = 2
    bpy.ops.object.modifier_apply(modifier="bev")
    if SIDE_N is None:
        ob.data.materials.append(obj_colour_material())          # legacy: normal faces in template-location colour
    for name, col in ((("tface", SIDE_T), ("bface", SIDE_B)) if SIDE_N is None else (("nface", SIDE_N), ("tface", SIDE_T), ("bface", SIDE_B))):
        m = bpy.data.materials.new(name); m.use_nodes = True; bs = m.node_tree.nodes["Principled BSDF"]
        bs.inputs["Base Color"].default_value = (*col, 1); bs.inputs["Roughness"].default_value = 0.7
        for k_ in ("Specular IOR Level", "Specular"):
            if k_ in bs.inputs: bs.inputs[k_].default_value = 0.2
        if "Emission Color" in bs.inputs:
            bs.inputs["Emission Color"].default_value = (*col, 1); bs.inputs["Emission Strength"].default_value = EMIT
        ob.data.materials.append(m)
    for poly in ob.data.polygons:                              # bevel faces go to their dominant axis
        nx, ny, nz = (abs(c) for c in poly.normal)
        poly.material_index = 0 if nz >= max(nx, ny) else (1 if nx >= ny else 2)
    ob.hide_render = True; ob.hide_set(True)
    return ob


def make_box(dims):
    """glyph with frame dims (t, b, n) along local (X, Y, Z); one material reading the object colour.
    box: bevelled box.  cone_n: round cone standing on the surface along +n (base at the origin) -- shows the normal
    only.  cone_t: cone lying in the surface, tip along +t, flattened along n -- shows the full frame."""
    L, Wd, Tk = dims
    if GLYPH == "tripod":
        return make_tripod(L)
    if GLYPH == "cube":
        return make_cube(L)
    if GLYPH == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0); ob = bpy.context.object
        ob.scale = dims; bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bev = ob.modifiers.new("bev", "BEVEL"); bev.width = 0.22 * min(dims); bev.segments = 3
        bpy.ops.object.modifier_apply(modifier="bev")
    elif GLYPH == "cone_p":                                   # PAINTED normal cone: axis = n, a stripe faces +t
        h, r = 1.05 * L, 0.62 * Wd
        bpy.ops.mesh.primitive_cone_add(vertices=36, radius1=r, radius2=0, depth=h, location=(0, 0, h / 2)); ob = bpy.context.object
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
        bpy.ops.object.shade_smooth()
        m = obj_colour_material(); ob.data.materials.append(m)
        st = bpy.data.materials.new("stripe"); st.use_nodes = True; sb = st.node_tree.nodes["Principled BSDF"]
        sb.inputs["Base Color"].default_value = (*STRIPE, 1); sb.inputs["Roughness"].default_value = 0.35
        if "Emission Color" in sb.inputs:
            sb.inputs["Emission Color"].default_value = (*STRIPE, 1); sb.inputs["Emission Strength"].default_value = 0.15
        ob.data.materials.append(st)
        for poly in ob.data.polygons:                          # side faces within +-STRIPE_DEG of +X (= +t) get the stripe
            c = poly.center
            if c.z > 1e-6 * h and abs(math.degrees(math.atan2(c.y, c.x))) < STRIPE_DEG:
                poly.material_index = 1
        ob.hide_render = True; ob.hide_set(True)
        return ob
    elif GLYPH == "cone_n":
        h, r = 1.05 * L, 0.55 * Wd
        bpy.ops.mesh.primitive_cone_add(vertices=28, radius1=r, radius2=0, depth=h, location=(0, 0, h / 2)); ob = bpy.context.object
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    else:  # cone_t
        h, r = 1.15 * L, 0.62 * Wd
        bpy.ops.mesh.primitive_cone_add(vertices=28, radius1=r, radius2=0, depth=h, rotation=(0, math.radians(90), 0)); ob = bpy.context.object
        ob.scale = (1, 1, max(0.35, 1.15 * Tk / (2 * r)))            # flatten along the normal
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.shade_smooth()
    ob.data.materials.append(obj_colour_material()); ob.hide_render = True; ob.hide_set(True)
    return ob


def obj_colour_material():
    """Principled material whose base (and faint emission) colour is the OBJECT colour (Object Info)."""
    m = bpy.data.materials.new("objcol"); m.use_nodes = True; nt = m.node_tree
    bs = nt.nodes["Principled BSDF"]; info = nt.nodes.new("ShaderNodeObjectInfo")
    nt.links.new(info.outputs["Color"], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.38
    if "Emission Color" in bs.inputs:
        nt.links.new(info.outputs["Color"], bs.inputs["Emission Color"]); bs.inputs["Emission Strength"].default_value = EMIT
    return m


def place(proto, loc, q, rgb, lift, s=1.0):
    o = proto.copy(); bpy.context.collection.objects.link(o); o.hide_render = False; o.hide_set(False); o.scale = (s, s, s)
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = q
    nrm = q.to_matrix().col[2]; o.location = Vector(loc) + nrm * lift
    o.color = (*rgb, 1.0)
    return o.location.copy()


def area(loc, energy, size, target=(0, 0, 0)):
    bpy.ops.object.light_add(type="AREA", location=loc); L = bpy.context.object
    L.data.energy = energy; L.data.size = size
    d = Vector(target) - Vector(loc); L.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
sc.collection.objects.link(cam); sc.camera = cam; cam_data.type = "ORTHO"
suf = "" if GLYPH == "box" else f"_{GLYPH}"
centres = []

if MODE == "image":
    W, H = meta["W"], meta["H"]; aspect = H / W
    img = bpy.data.images.load(f"{DIR}/{opt.get('--bg', 'bg.png')}")
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, -0.05)); pl = bpy.context.object; pl.scale = (1, aspect, 1)
    pm = bpy.data.materials.new("photo"); pm.use_nodes = True; nt = pm.node_tree; nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Strength"].default_value = DARK
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"]); nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    pl.data.materials.append(pm)
    cam.location = (0, 0, 3); cam.rotation_euler = (0, 0, 0); cam_data.ortho_scale = 1.0
    sc.render.resolution_x, sc.render.resolution_y = int(W * SCALE), int(H * SCALE)
    if meta.get("roi"):                                               # zoom render: exactly the ROI
        x0, y0, x1, y1 = meta["roi"]; mg = 0.0
    else:                                                             # render only the person region
        x0, y0, x1, y1 = meta["person_bbox"]; mg = 0.04 * H
    sc.render.use_border = True; sc.render.use_crop_to_border = True
    v = meta.get("valid") or [0, 0, W, H]                              # never show the crop's edge-replicated padding
    sc.render.border_min_x, sc.render.border_max_x = max(0, x0 - mg, v[0]) / W, min(W, x1 + mg, v[2]) / W
    sc.render.border_min_y, sc.render.border_max_y = 1 - min(H, y1 + mg, v[3]) / H, 1 - max(0, y0 - mg, v[1]) / H
    proto = make_box(meta["box_img"]); rows = list(csv.DictReader(open(f"{DIR}/img_boxes.csv")))
    XS = float(opt.get("--extra-scale", 1.0)); n0 = len(rows) - int(meta.get("n_extra", 0))   # tracked anchors drawn bigger
    for i, r in enumerate(rows):
        f = {k: float(v) for k, v in r.items()}
        centres.append(place(proto, ((f["u"] - .5), (f["v"] - .5) * aspect, 0.02),
                             Quaternion((f["qw"], f["qx"], f["qy"], f["qz"])), (f["r"], f["g"], f["b"]), 0.0, XS if i >= n0 else 1.0))
    KEY = float(opt.get("--key", 45))
    for loc, en, sz in [((-0.8, 0.9, 1.3), KEY, 0.9), ((0.9, -0.2, 1.5), 0.25 * KEY, 3), ((0.0, -1.0, 0.9), 0.12 * KEY, 3)]:   # key upper-left
        area(loc, en, sz)
else:
    vs, fs = [], []
    for ln in open(f"{DIR}/tmpl_body.obj"):
        p = ln.split()
        if p and p[0] == "v": vs.append(tuple(map(float, p[1:4])))
        elif p and p[0] == "f": fs.append(tuple(int(i) - 1 for i in p[1:4]))
    me = bpy.data.meshes.new("body"); me.from_pydata(vs, [], fs); me.update()
    body = bpy.data.objects.new("body", me); sc.collection.objects.link(body)
    for poly in me.polygons: poly.use_smooth = True
    bm = bpy.data.materials.new("skin"); bm.use_nodes = True
    b = bm.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = (0.42, 0.42, 0.44, 1)
    b.inputs["Roughness"].default_value = 0.65; me.materials.append(bm)
    sc.render.film_transparent = True
    proto = make_box(meta["box_tmpl"]); rows = list(csv.DictReader(open(f"{DIR}/tmpl_boxes.csv")))
    lift = 0.0 if GLYPH in ("cone_n", "cone_p", "tripod") else 0.5 * meta["box_tmpl"][2]
    nsum = Vector((0, 0, 0))
    XS = float(opt.get("--extra-scale", 1.0)); n0 = len(rows) - int(meta.get("n_extra", 0))
    for i, r in enumerate(rows):
        f = {k: float(v) for k, v in r.items()}; q = Quaternion((f["qw"], f["qx"], f["qy"], f["qz"]))
        centres.append(place(proto, (f["x"], f["y"], f["z"]), q, (f["r"], f["g"], f["b"]), lift, XS if i >= n0 else 1.0)); nsum += q.to_matrix().col[2]
    # camera: 'auto' faces the side of the body the boxes are on (mean box normal, horizontal), 'front' = -Y
    d = Vector((nsum.x, nsum.y, 0)) if opt.get("--tview", "auto") == "auto" else Vector((0, -1, 0))
    d = d.normalized() if d.length > 1e-6 else Vector((0, -1, 0))
    el = math.radians(float(opt.get("--tview-el", 8)))
    fwd = Vector((d.x * math.cos(el), d.y * math.cos(el), math.sin(el)))          # body -> camera
    rot = (-fwd).to_track_quat("-Z", "Y"); Rm = rot.to_matrix(); right, up = Rm.col[0], Rm.col[1]
    FRAME = opt.get("--frame", "body")                    # body: whole template | glyphs: zoom onto the exported glyphs
    pts = [Vector(c) for c in centres] if FRAME == "glyphs" else [Vector(v) for v in vs]
    pr = [q.dot(right) for q in pts]; pu = [q.dot(up) for q in pts]; pf = [Vector(v).dot(fwd) for v in vs]
    w, h = max(pr) - min(pr), max(pu) - min(pu); mr, mu = (max(pr) + min(pr)) / 2, (max(pu) + min(pu)) / 2
    if FRAME == "glyphs":                                  # margin: glyph size + 15 % so edge glyphs are not cut
        pad = 2.0 * meta["box_tmpl"][0] + 0.15 * max(w, h); w, h = w + pad, h + pad
    cam.rotation_mode = "QUATERNION"; cam.rotation_quaternion = rot
    cam.location = right * mr + up * mu + fwd * (max(pf) + 3.0)
    cam_data.ortho_scale = 1.04 * max(w, h)
    sc.render.resolution_y = int(900 * SCALE); sc.render.resolution_x = int(900 * SCALE * w / h)
    ctr = right * mr + up * mu
    for off, en, sz in [((-1.6, 1.8, 3.0), 140, 1.5), ((2.0, 0.4, 2.6), 45, 3), ((0.3, 1.2, -2.5), 40, 2.5)]:
        area(tuple(ctr + right * off[0] + up * off[1] + fwd * off[2]), en, sz, target=tuple(ctr))
    print(f"[blender_boxes] template view dir {tuple(round(c, 2) for c in fwd)}", flush=True)
print(f"[blender_boxes] {MODE}: {len(rows)} glyphs ({GLYPH})", flush=True)

# 2D position of every glyph centre in the SAVED image (pixels, origin top-left) -> fig1_compose links boxes across panels
from bpy_extras.object_utils import world_to_camera_view
bpy.context.view_layer.update()
rx, ry = sc.render.resolution_x, sc.render.resolution_y
xy = []
for c in centres:
    pc = world_to_camera_view(sc, cam, c); x, y = pc.x * rx, (1 - pc.y) * ry
    if MODE == "image":
        x -= sc.render.border_min_x * rx; y -= (1 - sc.render.border_max_y) * ry
    xy.append([round(x, 2), round(y, 2)])
json.dump(dict(mode=MODE, glyph=GLYPH, xy=xy), open(f"{DIR}/screen_{MODE}{suf}.json", "w"))
sc.render.filepath = f"{DIR}/render_{MODE}{suf}.png"; sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA" if MODE == "template" else "RGB"
bpy.ops.render.render(write_still=True); print("RENDERED", sc.render.filepath, flush=True)
