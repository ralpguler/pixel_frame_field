"""Assemble the frame-field beauty-contest report HTML (self-contained: base64 stills + loops).

Reads $PFF_DATA/dp_viz_compare/beauty/report_assets (built by beauty_report_assets.py) and writes
framefield_beauty.html to --out (default viz/out/beauty_report/).
"""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
_env.take_out_arg()
import base64
import os

A = _env.data("dp_viz_compare/beauty/report_assets")
OUTD = _env.out("beauty_report")
os.makedirs(OUTD, exist_ok=True)
OUT = f"{OUTD}/framefield_beauty.html"


def b64(name, kind="image/jpeg"):
    with open(f"{A}/{name}", "rb") as f:
        return f"data:{kind};base64," + base64.b64encode(f.read()).decode()


def img(name, cap=""):
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f'<figure class="shot"><img src="{b64(name)}" alt="{cap}" loading="lazy">{c}</figure>'


def vid(name, cap=""):
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return (f'<figure class="shot"><video src="{b64(name, "video/mp4")}" muted loop autoplay '
            f'playsinline></video>{c}</figure>')


CSS = """
:root{--bg:#0c0e13;--panel:#141822;--line:#232838;--tx:#d5d2ca;--mut:#8a8f9c;
--acc:#b16cea;--gold:#d9a441;--good:#7fb069}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--tx);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:0 20px 90px}
.wrap{max-width:1060px;margin:0 auto}
header{padding:72px 0 18px;border-bottom:1px solid var(--line)}
.eyebrow{font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.22em;
color:var(--acc);text-transform:uppercase}
h1{font-family:"Iowan Old Style",Palatino,"Palatino Linotype",Georgia,serif;font-size:clamp(34px,6vw,58px);
font-weight:600;line-height:1.06;text-wrap:balance;margin:14px 0 10px}
.sub{color:var(--mut);max-width:62ch;font-size:17px}
.meta{display:flex;gap:22px;flex-wrap:wrap;margin-top:22px;
font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut)}
.meta b{color:var(--tx);font-weight:600}
.lead{max-width:68ch;margin:34px auto 0;font-size:17.5px}
.lead em{color:var(--acc);font-style:normal;font-weight:600}
nav{display:flex;gap:10px;flex-wrap:wrap;margin:30px 0 8px}
nav a{color:var(--mut);text-decoration:none;font:600 12.5px/1 ui-monospace,Menlo,monospace;
letter-spacing:.08em;border:1px solid var(--line);border-radius:999px;padding:8px 14px}
nav a:hover,nav a:focus-visible{color:var(--tx);border-color:var(--acc);outline:none}
section{margin-top:74px}
.no{font:600 12px/1 ui-monospace,Menlo,monospace;color:var(--mut);letter-spacing:.2em}
h2{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-size:clamp(26px,3.6vw,36px);
font-weight:600;margin:8px 0 4px}
.tag{color:var(--mut);font-size:15px;max-width:64ch}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 20px}
.chip{font:600 11px/1 ui-monospace,Menlo,monospace;letter-spacing:.1em;padding:6px 11px;
border-radius:999px;border:1px solid var(--line);color:var(--mut)}
.chip.on{color:var(--acc);border-color:color-mix(in srgb,var(--acc) 45%,transparent)}
.chip.gold{color:var(--gold);border-color:color-mix(in srgb,var(--gold) 45%,transparent)}
.hero{display:grid;grid-template-columns:1.35fr 1fr;gap:14px}
@media(max-width:760px){.hero{grid-template-columns:1fr}}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-top:14px}
.shot{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.shot img,.shot video{width:100%;display:block;aspect-ratio:1;object-fit:cover}
.shot figcaption{font:11.5px/1.5 ui-monospace,Menlo,monospace;color:var(--mut);padding:8px 12px}
.verdict{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}
@media(max-width:680px){.verdict{grid-template-columns:1fr}}
.v{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.v h3{font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.18em;text-transform:uppercase;
color:var(--mut);margin-bottom:8px}
.v.win h3{color:var(--gold)}
.v p{font-size:14.5px;color:var(--tx)}
table{width:100%;border-collapse:collapse;margin-top:20px;font-size:14.5px}
th{font:600 11px/1 ui-monospace,Menlo,monospace;letter-spacing:.15em;text-transform:uppercase;
color:var(--mut);text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}
td{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}
td:first-child{font-weight:600;white-space:nowrap}
.tablewrap{overflow-x:auto}
.rank{color:var(--gold);font:600 13px/1 ui-monospace,Menlo,monospace}
footer{margin-top:80px;padding-top:24px;border-top:1px solid var(--line);
font:12.5px/1.8 ui-monospace,Menlo,monospace;color:var(--mut)}
footer code{color:var(--tx)}
@media(prefers-reduced-motion:reduce){video{display:none}}
"""

M = {}

M["field"] = f"""
<section id="field">
<div class="no">THE GAUGE FIX</div><h2>A proper frame field for the body</h2>
<p class="tag">Riza's catch: particles and combing display t = R&nbsp;·&nbsp;t<sub>template</sub>, so
every defect of the template combing reads as a model artifact. The original choice — gradient of
geodesic distance from the crown — has a critical point at <i>every</i> extremity plus C⁰ creases
across the cut locus. Replaced by the globally-optimal smooth direction field (Knöppel et&nbsp;al.
2013: connection-Laplacian solve, soft head→toe alignment, cos&nbsp;0.82).</p>
<div class="chips"><span class="chip">EDGES &gt;30°: 16.2% → <b>2.45%</b></span>
<span class="chip">FLIPS &gt;90°: 4.2% → <b>0.48%</b></span>
<span class="chip">SINGULARITIES: every extremity → <b>~46, at soles·fingertips·ears</b></span>
<span class="chip gold">POINCARÉ–HOPF PAYS +2 MINIMUM</span></div>
<div class="hero">{img("28513779_lic_twist_f0075.jpg", "AFTER — smooth field: one continuous pelt, no side seams")}
<div class="strip" style="grid-template-columns:1fr;margin-top:0">
{img("28513779_lic_twist_f0075_v2geo.jpg", "BEFORE — geodesic gradient: cut-locus seam down the torso side, swirls at shoulders")}
</div></div>
<div class="verdict">
<div class="v win"><h3>What changed</h3><p>The cut-locus seams down the sides of the torso and limbs are
gone, shoulder combing is continuous, and particle streams stop braiding into rivers. The ~46 remaining
singular points are the topology's mandatory minimum neighbourhood — isolated dots at the foot soles,
fingertips and ears instead of creases through the torso.</p></div>
<div class="v"><h3>What it does NOT change</h3><p>The model's R is gauge-free; only the displayed section
of the tangent bundle changed. Twist colouring and satin anisotropy consume R directly and are
unaffected in content — but satin's sheen direction also inherits the cleaner tangent.</p></div>
</div></section>"""

M["wind"] = f"""
<section id="wind">
<div class="no">CONTESTANT 01</div><h2>Wind map — particle advection</h2>
<p class="tag">Particles ride the image-projected predicted tangent with fading trails; speed and
brightness drop where the tangent points at the camera, so foreshortening reads through the motion.
Density is kept uniform by respawning at fresh body pixels.</p>
<div class="chips"><span class="chip on">TANGENT ✓</span><span class="chip on">TWIST — as trail colour</span>
<span class="chip">2D COMPOSITE — superseded by FUR for shading</span></div>
<div class="hero">{vid("15731089_wind_twist.mp4", "hero loop · twist-coloured trails")}
<div class="strip" style="grid-template-columns:1fr;margin-top:0">
{img("15731089_wind_twist_f075.jpg", "still, frame 75")}
</div></div>
<div class="strip">{vid("15731089_wind_white.mp4", "monochrome variant")}
{img("15731089_wind_twist_f040.jpg", "frame 40")}{img("15731089_wind_twist_f110.jpg", "frame 110")}
{img("15731089_f_0075_photo.jpg", "source frame")}</div>
<div class="verdict">
<div class="v"><h3>Where it wins</h3><p>The most alive of the 2D methods — streams pour down the torso
and out along the arms, whole limbs share a trail colour. Cheap and real-time-able.</p></div>
<div class="v"><h3>Riza's verdict</h3><p>Flat — unshaded 2D composite, "looks like OpenCV". Kept as the
cheap/fast option; the shaded successor is the FUR section below.</p></div>
</div></section>"""

M["lic"] = f"""
<section id="lic">
<div class="no">CONTESTANT 02</div><h2>LIC — the combed body</h2>
<p class="tag">Line-integral convolution: white noise smeared along streamlines of the predicted
tangent, per pixel, lit by the predicted normal and coloured by twist.</p>
<div class="chips"><span class="chip on">TANGENT ✓</span><span class="chip on">NORMAL — as shading</span>
<span class="chip on">TWIST — as hue</span><span class="chip">2D COMPOSITE — superseded by FUR</span></div>
<div class="hero">{img("28513779_lic_twist_f0075.jpg", "portrait clip · the strongest 2D still")}
<div class="strip" style="grid-template-columns:1fr 1fr;margin-top:0">
{img("15731089_lic_twist_f0075.jpg", "runway clip")}{img("37828157_lic_twist_f0075.jpg", "street clip")}
{img("15153663_lic_twist_f0075.jpg", "low light clip")}{img("28513779_lic_normal_f0075.jpg", "normal-coloured variant")}
</div></div>
<div class="strip">{vid("15731089_lic_twist.mp4", "video — mild texture shimmer, fixed noise")}
{img("15731089_lic_normal_f0075.jpg", "normal-coloured, runway")}
{img("28513779_f_0075_photo.jpg", "source frame")}</div>
<div class="verdict">
<div class="v"><h3>Where it wins</h3><p>Densest statement of "this is a continuous field" — every body
pixel combed, fingers included. Doubles as an honest diagnostic (seam speckle shows directly).</p></div>
<div class="v"><h3>Riza's verdict</h3><p>Flat shading (2D flow smearing, not lit strands). The Cycles
fur below is the shaded replacement; LIC stays the quick-look tool.</p></div>
</div></section>"""

M["satin"] = f"""
<section id="satin">
<div class="no">CONTESTANT 03</div><h2>Satin — anisotropic relighting <span class="no">(Blender Cycles)</span></h2>
<p class="tag">A Cycles material whose Normal and Tangent sockets are driven per-pixel by the model's
prediction, under a sweeping area light. The anisotropic highlight stretches perpendicular to the
predicted tangent — the sheen combing around a limb IS the twist channel, lit.</p>
<div class="chips"><span class="chip on">NORMAL ✓</span><span class="chip on">TANGENT ✓</span>
<span class="chip on">TWIST ✓ — full SO(3)</span><span class="chip gold">RIZA-APPROVED LOOK</span></div>
<div class="hero">{vid("15731089_satin.mp4", "the light sweep · the highlight combs across limbs")}
<div class="strip" style="grid-template-columns:1fr;margin-top:0">
{img("15731089_satin_0075.jpg", "runway still")}
</div></div>
<div class="strip">{img("37828157_satin_0075.jpg", "street clip")}{img("15153663_satin_0075.jpg", "low light")}
{img("28513779_satin_0075.jpg", "portrait — the uncanny failure case")}{img("15731089_f_0075_photo.jpg", "source")}</div>
<div class="verdict">
<div class="v win"><h3>Where it wins</h3><p>On full-body shots the highlight band runs <i>across</i> each
limb exactly as anisotropy demands when the tangent runs along it, and it rotates with the light sweep.
The complete rotation, twist included, in one image a lay audience reads instantly.</p></div>
<div class="v"><h3>Honest flaws</h3><p>Close-ups go uncanny (porcelain face) — fixed for chrome via the
metallic map; hand/foot part-noise reads as harsh speckle under metal.</p></div>
</div></section>"""

M["hatch"] = f"""
<section id="hatch">
<div class="no">CONTESTANT 04</div><h2>Cross-hatching — pen &amp; ink</h2>
<p class="tag">Strokes are streamlines of the predicted tangent on a fixed jittered grid; density follows
Lambert shading of the predicted normal; cross-strokes along the bitangent build into shadow.</p>
<div class="chips"><span class="chip on">TANGENT ✓</span><span class="chip on">NORMAL — as tone</span>
<span class="chip gold">BEST PAPER FIGURE</span></div>
<div class="hero">{img("15731089_hatch_ink_f0075.jpg", "ink on paper · prints in grayscale")}
<div class="strip" style="grid-template-columns:1fr 1fr;margin-top:0">
{img("28513779_hatch_ink_f0075.jpg", "portrait")}{img("37828157_hatch_neon_f0075.jpg", "neon variant, street")}
{img("15153663_hatch_ink_f0075.jpg", "low light")}{img("28513779_hatch_neon_f0075.jpg", "neon, portrait")}
</div></div>
<div class="strip">{vid("15731089_hatch_ink.mp4", "video — steadiest of all methods")}
{img("15731089_hatch_neon_f0075.jpg", "neon, runway")}{img("37828157_hatch_ink_f0075.jpg", "ink, street")}
{img("37828157_f_0075_photo.jpg", "source frame")}</div>
<div class="verdict">
<div class="v win"><h3>Where it wins</h3><p>Reads at thumbnail size, prints in grayscale, and encodes two
channels in a visual language every reviewer trusts. The neon variant doubles as an AR-look demo.</p></div>
<div class="v"><h3>Honest flaws</h3><p>No twist channel; strokes sit still in image space while the body
moves under them; foreshortened tangents shrink strokes to stubble.</p></div>
</div></section>"""

M["round2"] = f"""
<section id="round2">
<div class="no">ROUND 2 · PHYSICALLY BASED</div><h2>Sequins, chrome, holograms</h2>
<p class="tag">Three more materials driven per-pixel by the prediction. Sequins: micro-mirrors anchored
in canonical UV (a forward pullback — no tracking), each canted randomly in the predicted tangent plane,
so its flash timing under the sweeping light is a physical readout of R, twist included. Diffraction:
Stam's grating model with grooves along the predicted tangent — the rainbow bands' orientation IS the
tangent field. Chrome: Cycles mirror under a rotating physical sky; the metallic map keeps the face
photographic.</p>
<div class="chips"><span class="chip on">SEQUINS — glint timing tests R over time</span>
<span class="chip on">DIFFRACTION — twist as colour physics</span>
<span class="chip on">CHROME — env relight, face kept photographic</span></div>
<div class="hero">{vid("15731089_diffract.mp4", "diffraction · the holographic suit — bands sweep as the light moves")}
<div class="strip" style="grid-template-columns:1fr;margin-top:0">
{img("15731089_diffract_f0075.jpg", "diffraction still")}
</div></div>
<div class="strip">{vid("15731089_sequin.mp4", "sequins · watch the glint clusters travel")}
{img("15731089_sequin_f0075.jpg", "sequin still")}
{vid("15731089_chrome.mp4", "chrome · rotating sky sweep")}
{img("15731089_chrome_0075.jpg", "chrome still")}</div>
<div class="strip">{img("28513779_diffract_f0075.jpg", "diffraction, portrait")}
{img("28513779_sequin_f0075.jpg", "sequins, portrait")}
{img("37828157_chrome_0075.jpg", "chrome, street")}
{img("28513779_chrome_0075.jpg", "chrome, portrait — photographic face")}</div>
<div class="verdict">
<div class="v win"><h3>Verdicts</h3><p><b>Diffraction</b> is the round-2 star — instantly legible,
unmistakably about directions, the only viz where twist becomes literal colour. <b>Sequins</b> are the
deepest test (temporal R-correctness as sparkle timing) but want a closer crop. <b>Chrome</b> is the
drama shot: liquid mercury, panel reflections streaking along limbs.</p></div>
<div class="v"><h3>What was fixed en route</h3><p>The v3 satin had silently reused stale geodesic-tangent
bakes — all G-buffers re-baked from the smooth field. Chrome needed two exposure fixes: the satin light
rig reflects as white blobs in a mirror (env-led lighting instead), and Blender 5.0's sky emits physical
radiance units (~10³ W/m²sr → strength 0.001).</p></div>
</div></section>"""

M["fur"] = f"""
<section id="fur">
<div class="no">ROUND 3 · REAL GEOMETRY</div><h2>Fur — combing with actual strands</h2>
<p class="tag">Riza's verdict on the 2D composites (wind, LIC): flat, unshaded, "looks like OpenCV".
Fixed with real geometry: ~17–48k hair curves per frame in Cycles, roots anchored in canonical UV
(they track the body like the sequins), each strand grown along the UNPROJECTED 3D predicted tangent —
its z-component tilts it out of the image plane, invisible to the ortho camera but real to the lights.
Per-cell deterministic shape → temporally coherent coat.</p>
<div class="chips"><span class="chip on">REAL SELF-SHADOWING</span><span class="chip on">3D TANGENT
(not the 2D projection)</span><span class="chip on">TWIST → COLOUR</span></div>
<div class="hero">{vid("15731089_fur_tinsel.mp4", "Principled shader — saturated, direction legible")}
<div class="strip" style="grid-template-columns:1fr;margin-top:0">
{vid("15731089_fur_hair.mp4", "Hair BSDF — soft angora, physically scattered")}
</div></div>
<div class="strip">{img("15731089_fur_0075.jpg", "hair BSDF still")}{img("15731089_fur_p_0075.jpg", "principled still")}
{img("28513779_fur_0075.jpg", "portrait")}{img("15153663_fur_0075.jpg", "low light")}</div>
<div class="verdict">
<div class="v win"><h3>Two shaders, one choice</h3><p><b>Principled ("tinsel")</b>: colours stay saturated
and the combing direction is clearly legible in the interior. <b>Hair BSDF ("angora")</b>: physically soft
with fur-trim pom-poms at the silhouettes, but multiple scattering launders the colours pastel — real
hair physics, not a bug.</p></div>
<div class="v"><h3>Honest flaws</h3><p>Interior strands seen tip-on read as texture at high density; the
silhouette overhang is where the 3D-ness sells. Fur inherits hand/foot part noise as unruly tufts.</p></div>
</div></section>"""

M["objects"] = f"""
<section id="objects">
<div class="no">ROUND 3 · THE AR DEMO</div><h2>Object placement — full 6-DoF anchors</h2>
<p class="tag">Anchors are picked on ONE reference frame (the predicted (part,uv) under a chosen image
point — click-to-attach AR) and tracked everywhere else by forward kNN in chart space; each object is
posed by the full predicted frame (X = tangent, Y = bitangent, Z = normal) and physically scaled by a
px/mm estimate read from the pixel-vs-UV distance ratio. A lay viewer instantly knows if a watch or a
crown sits wrong — this is the visualisation where rotation errors have nowhere to hide.</p>
<div class="chips"><span class="chip on">WATCH — twist readout</span>
<span class="chip on">BADGE — correspondence stability</span>
<span class="chip on">BUTTERFLIES — animate IN the local frame</span>
<span class="chip on">PINS — normal direction, no aesthetics</span></div>
<div class="hero">{vid("15731089_objects.mp4", "eight objects riding the frame field through the walk")}
<div class="strip" style="grid-template-columns:1fr;margin-top:0">
{img("15731089_objects_0075.jpg", "still, frame 75")}
</div></div>
<div class="verdict">
<div class="v win"><h3>What it shows</h3><p>The badge staying pinned through the whole walk is the
temporal-consistency claim in one shot; the tiara tilting with the head and the pins protruding along
the predicted normal read the frame directly. Butterfly wings flap about the LOCAL tangent axis —
proof the model hands you a coordinate system, not just a point.</p></div>
<div class="v"><h3>Honest limits</h3><p>No true depth: object–body occlusion is approximated by anchor
visibility (kNN median-distance test + hysteresis); the watch shows only ~16% of frames because that
wrist chart point genuinely faces away — correct behaviour that reads as popping. DensePose head charts
cover the face, not hair, so a "crown" is honestly a tiara. Objects are oversized ~1.5× — true physical
sizes are visually tiny at runway distance.</p></div>
</div></section>"""

html = f"""<title>Frame-Field Beauty Contest</title>
<style>{CSS}</style>
<div class="wrap">
<header>
<div class="eyebrow">Oriented DensePose · direction visualisations</div>
<h1>Nine ways to render a frame field</h1>
<p class="sub">A beauty contest. Every entrant composes with the forward prediction
f:&nbsp;pixel&nbsp;→&nbsp;(template&nbsp;point,&nbsp;frame) — none inverts it, so none inherits the
blinking, density and visibility pathologies of the tracked-point overlays.</p>
<div class="meta"><span>model <b>+MAMMA ckpt_120000</b></span><span>clips <b>4 × Pexels</b></span>
<span>hero <b>150 frames · 808²</b></span><span>renderers <b>wind · LIC · satin · hatch · sequins ·
chrome · diffraction · fur · objects</b></span></div>
</header>
<p class="lead">The runway clip is the shared benchmark; three clips of a different nature (street
action, close portrait, low light) check that each look survives outside the studio. Judging criteria:
<em>beauty</em>, <em>how much of the SO(3) it actually shows</em>, and <em>honesty</em> — whether model
errors stay visible or get cosmetically smoothed away.</p>
<nav><a href="#field">00 · the gauge fix</a><a href="#wind">01 · wind map</a><a href="#lic">02 · LIC</a>
<a href="#satin">03 · satin</a><a href="#hatch">04 · hatching</a><a href="#round2">05 · round 2 PBR</a>
<a href="#fur">06 · fur</a><a href="#objects">07 · objects</a><a href="#verdict">verdict</a></nav>
{M["field"]}{M["wind"]}{M["lic"]}{M["satin"]}{M["hatch"]}{M["round2"]}{M["fur"]}{M["objects"]}
<section id="verdict">
<div class="no">RESULT</div><h2>Verdict</h2>
<div class="tablewrap"><table>
<tr><th>method</th><th>beauty</th><th>SO(3) shown</th><th>honesty</th><th>best deployment</th></tr>
<tr><td><span class="rank">1·</span> Diffraction</td><td>the holographic suit; instantly legible</td>
<td>tangent + twist as colour physics</td><td>high</td><td>the demo video</td></tr>
<tr><td><span class="rank">1·</span> Satin/Chrome</td><td>spectacular full-body, physically lit</td>
<td><b>full SO(3)</b></td><td>medium — lighting flatters</td><td>the "we predict rotations" shot</td></tr>
<tr><td><span class="rank">3·</span> Fur</td><td>real shaded strands, silhouette pom-poms</td>
<td>3D tangent + twist-colour</td><td>high — tufts show noise</td><td>the shaded combing</td></tr>
<tr><td><span class="rank">3·</span> Objects</td><td>quiet, but the claim is readable</td>
<td>full 6-DoF at anchors</td><td>highest — errors have nowhere to hide</td><td>the AR/application story</td></tr>
<tr><td><span class="rank">5·</span> Sequins</td><td>glamour with a hidden test</td>
<td>R over TIME via glint timing</td><td>high</td><td>close-up crops</td></tr>
<tr><td><span class="rank">6·</span> Hatch</td><td>quietly excellent, prints well</td>
<td>tangent + normal-tone</td><td>high</td><td>paper figures</td></tr>
<tr><td><span class="rank">7·</span> LIC / Wind</td><td>dense/alive but flat (2D)</td>
<td>tangent + twist-hue</td><td>high</td><td>quick looks, debugging</td></tr>
</table></div>
<div class="verdict" style="margin-top:22px">
<div class="v win"><h3>Recommendation</h3><p><b>Diffraction</b> for the project-page loop, <b>satin/chrome</b>
for the headline "full rotations" demo, <b>fur (tinsel)</b> as the shaded combing, <b>objects</b> for the
application story, <b>hatch-ink</b> for the paper. The 2D methods stay as fast diagnostics.</p></div>
<div class="v"><h3>Iterate next</h3><p>Butterfly wing shapes + watch band wrap · closer-crop sequin clip ·
fur on the portrait clip · run the winners across the 300-clip dance test set.</p></div>
</div></section>
<footer>
<p>All renders: forward per-pixel composition of model outputs — no template→pixel matching anywhere.
Fields dumped once per clip (<code>orient/field_dump.py</code>), rendered by
<code>viz_wind · viz_lic · viz_hatch · viz_sequin · viz_diffract · blender_satin(chrome) ·
viz_fur_bake+blender_fur · viz_objects_bake+blender_objects</code>. Masters in
<code>~/data/dp_viz_compare/beauty/&lt;clip&gt;/render/</code>.</p>
</footer>
</div>"""

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
