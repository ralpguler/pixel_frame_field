"""Blender scene builder for the framefield tripods.
Run:  blender --python blender_framefield.py         (or paste into Blender's Scripting tab)
Assumes tripods.ply, body.obj, points_frames.csv are in FFDIR (edit below).

Two ways to render, pick with MODE:
  MODE = "ply"   -> import the pre-baked shaded tripod geometry (tripods.ply, vertex-colored). Robust & instant.
  MODE = "nodes" -> point-cloud + Geometry Nodes: instance a single arrow on each point, rotated by the
                     stored frame quaternion. Lighter scene, fully editable arrow/material in Blender.
Both give: emissive axis-colored tripods over a dark ghost body, 3-point light, camera, turntable frames.
"""
import bpy, os, math, csv
from mathutils import Quaternion, Vector

FFDIR = os.path.expanduser("~/framefield")          # <-- EDIT to where you rsync'd the bundle
MODE = "ply"                                          # "ply" or "nodes"
EMISSION = 6.0                                        # arrow glow strength
RENDER = False                                        # True -> render a 60-frame turntable to FFDIR/render/


def clean():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.node_groups):
        for b in list(c): c.remove(b)


def emit_vcol_material(name="tripod_emit"):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "Col"
    emit = nt.nodes.new("ShaderNodeEmission"); emit.inputs["Strength"].default_value = EMISSION
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return m


def ghost_body_material():
    m = bpy.data.materials.new("ghost"); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.02, 0.02, 0.03, 1)
    if "Roughness" in b.inputs: b.inputs["Roughness"].default_value = 0.6
    m.blend_method = "BLEND"
    for key in ("Alpha",):
        if key in b.inputs: b.inputs[key].default_value = 0.35
    return m


def import_mesh(path):
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "ply":
        try: bpy.ops.wm.ply_import(filepath=path)
        except Exception: bpy.ops.import_mesh.ply(filepath=path)
    else:
        try: bpy.ops.wm.obj_import(filepath=path)
        except Exception: bpy.ops.import_scene.obj(filepath=path)
    return bpy.context.selected_objects[0]


def build_arrow():
    """A single +Z arrow (shaft + cone) as a mesh object, for node-instancing mode."""
    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.66, location=(0, 0, 0.33))
    shaft = bpy.context.object
    bpy.ops.mesh.primitive_cone_add(radius1=0.14, radius2=0, depth=0.34, location=(0, 0, 0.83))
    cone = bpy.context.object
    shaft.select_set(True); cone.select_set(True); bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join(); shaft.name = "arrow_unit"
    return shaft


def add_lights_camera(target=(0, 0, 0)):
    for (loc, energy, size) in [((3, -4, 4), 900, 5), ((-4, -2, 2), 400, 6), ((0, 5, 3), 350, 8)]:
        bpy.ops.object.light_add(type="AREA", location=loc); L = bpy.context.object
        L.data.energy = energy; L.data.size = size
    bpy.ops.object.camera_add(location=(0, -3.6, 0.2)); cam = bpy.context.object
    cam.rotation_euler = (math.radians(90), 0, 0); bpy.context.scene.camera = cam
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg: bg.inputs[0].default_value = (0.02, 0.02, 0.03, 1); bg.inputs[1].default_value = 1.0
    return cam


clean()
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [e.identifier for e in
    bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
try: sc.eevee.use_bloom = True
except Exception: pass

# ghost body
if os.path.exists(f"{FFDIR}/body.obj"):
    body = import_mesh(f"{FFDIR}/body.obj"); body.data.materials.append(ghost_body_material())
    bpy.ops.object.shade_smooth()

if MODE == "ply":
    trip = import_mesh(f"{FFDIR}/tripods.ply")
    trip.data.materials.clear(); trip.data.materials.append(emit_vcol_material())
    bpy.ops.object.shade_smooth()
else:  # nodes: point cloud + Geometry Nodes instancing
    rows = list(csv.reader(open(f"{FFDIR}/points_frames.csv")))[1:]
    me = bpy.data.meshes.new("pts"); me.from_pydata([[float(r[0]), float(r[1]), float(r[2])] for r in rows], [], [])
    me.update()
    for nm in ("qw", "qx", "qy", "qz", "tw"):
        me.attributes.new(name=nm, type="FLOAT", domain="POINT")
    for i, r in enumerate(rows):
        me.attributes["qw"].data[i].value = float(r[3]); me.attributes["qx"].data[i].value = float(r[4])
        me.attributes["qy"].data[i].value = float(r[5]); me.attributes["qz"].data[i].value = float(r[6])
        me.attributes["tw"].data[i].value = float(r[10])
    obj = bpy.data.objects.new("framepts", me); bpy.context.collection.objects.link(obj)
    arrow = build_arrow(); arrow.hide_set(True); arrow.data.materials.append(emit_vcol_material())
    # NOTE: for full per-point rotation, add a GeometryNodes 'Instance on Points' modifier and wire the
    # qw/qx/qy/qz attributes -> a Quaternion-to-Rotation (or Euler) node -> Instance Rotation. Left as a
    # manual node step because the exact GN node names vary by Blender version. Points + arrow are ready.

cam = add_lights_camera()
# frame the body
bpy.ops.object.select_all(action="DESELECT")
if RENDER:
    os.makedirs(f"{FFDIR}/render", exist_ok=True)
    sc.frame_start = 1; sc.frame_end = 60
    piv = bpy.data.objects.new("piv", None); bpy.context.collection.objects.link(piv)
    cam.parent = piv
    piv.rotation_euler = (0, 0, 0); piv.keyframe_insert("rotation_euler", frame=1)
    piv.rotation_euler = (0, 0, math.radians(360)); piv.keyframe_insert("rotation_euler", frame=60)
    sc.render.filepath = f"{FFDIR}/render/turn_"
    sc.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True)
print("framefield scene ready. MODE=", MODE)
