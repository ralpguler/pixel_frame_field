"""Tables for E1/E6 (orient/eval_frames_e1e6.py outputs in ~/data/dp_frames_eval/).

  python -m orient.eval_frames_e1e6_report [--files ...] [--out ~/data/dp_frames_eval/report.txt]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

OUT = os.path.expanduser("~/data/dp_frames_eval")
NBIN = 9000
MET = ("geo", "normal", "twist")


def load(p):
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"])); d["extra"] = json.loads(str(d["extra"]))
    d["H"] = {tuple(k.split("|")): h for k, h in zip(d["hist_keys"], d["hist"])}
    d["I"] = {m: s for m, s in zip(d["img_methods"], d["img_sums"])}
    return d


def q_from_hist(h, q):
    c = np.cumsum(h); n = c[-1]
    if n == 0:
        return float("nan")
    i = np.searchsorted(c, q * n)
    return (i + 0.5) * 180.0 / NBIN


def pooled(d, meth, met, group="all"):
    h = d["H"][(meth, met, group)]
    mi = MET.index(met)
    if group == "all":
        mean = d["I"][meth][:, mi].sum() / d["img_cnt"].sum()
    else:
        cen = (np.arange(NBIN) + 0.5) * 180 / NBIN
        mean = (h * cen).sum() / max(h.sum(), 1)
    return mean, q_from_hist(h, 0.5), q_from_hist(h, 0.9), int(h.sum())


def per_image(d, meth, met):
    mi = MET.index(met)
    c = d["img_cnt"]; ok = c > 0
    m = d["I"][meth][ok, mi] / c[ok]
    return m


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # two-sided exact binomial, via log-space sum
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    mx = max(lp)
    p = 2 * math.exp(mx) * sum(math.exp(x - mx) for x in lp)
    return min(1.0, p)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(d, a, b):
    xa = per_image(d, a, "geo"); xb = per_image(d, b, "geo")
    diff = xb - xa                                 # >0: a better
    w = int((diff > 0).sum()); l = int((diff < 0).sum())
    lo, hi = boot_ci(diff)
    return w, l, sign_test(w, l), diff.mean(), lo, hi


def fmt_row(d, meth, label=None):
    s = f"{(label or meth):34s}"
    for met in MET:
        mean, med, p90, _ = pooled(d, meth, met)
        s += f" | {mean:6.2f} {med:6.2f} {p90:6.2f}"
    pi = per_image(d, meth, "geo")
    s += f" || {pi.mean():6.2f} {np.median(pi):6.2f} {np.percentile(pi, 90):6.2f}"
    return s


HDR = (f"{'method':34s} | {'geodesic mean/med/p90':20s} | {'normal mean/med/p90':20s} | {'twist mean/med/p90':20s}"
       f" || per-image geo mean/med/p90")


def best_r(d, corr, dis, cam="persp"):
    best = None
    for r in (1, 2, 3):
        m = f"D-{corr}-{dis}-{cam}-r{r}"
        if m in d["I"]:
            v = pooled(d, m, "geo")[0]
            if best is None or v < best[1]:
                best = (r, v)
    return best[0]


def report_one(p, lines, rsel=None):
    d = load(p)
    meta = d["meta"]; ex = d["extra"]
    L = lines.append
    L("=" * 150)
    L(f"{os.path.basename(p)}   split {meta['split']}   ckpt {meta['ckpt']}   images {meta['n_images']}")
    if "repro_eval512_geo_e2e" in meta:
        L(f"  SANITY eval_512 statistic (mean of per-batch medians, GT fg, pred-part gather): {meta['repro_eval512_geo_e2e']:.3f}"
          f"   | pixel-pooled MEAN over the same pixels: {ex['repro_pool_sum']/ex['repro_pool_n']:.3f}")
    if "repro_indomain_probe" in meta:
        r = meta["repro_indomain_probe"]
        L(f"  SANITY indomain_probe (native decode, pooled mean): geo_e2e {r['geo_e2e']:.3f}  geo_oracle {r['geo_oracle']:.3f}  n={r['n']}")
    if meta.get("missing_in_gt512"):
        L(f"  missing in gt512 shards: {len(meta['missing_in_gt512'])}")
    nfg = ex["n_fg"]
    L(f"  pixels: GT fg {nfg}  -> no hidden {ex['n_nohidden']} ({100*ex['n_nohidden']/nfg:.1f}%)  -> GT stencil "
      f"{ex['n_gt_stencil']} ({100*ex['n_gt_stencil']/nfg:.1f}%)  -> +pred stencil {ex['n_both_stencil']} "
      f"({100*ex['n_both_stencil']/nfg:.1f}%)  -> Omega (all solves finite) {ex['n_omega']} ({100*ex['n_omega']/nfg:.1f}%)")
    if rsel is None:
        rsel = {(c, s): best_r(d, c, s) for c in ("pred", "gt") for s in ("orc", "part", "smooth")}
    L("  main table (pixel-pooled over Omega; per-image = mean of per-image means); degrees")
    L("  " + HDR)
    rows = [("H", "H  rotation head (pred-part gather)"), ("H-orc", "H-orc  head, GT-part gather")]
    for c in ("pred", "gt"):
        for s in ("orc", "part", "smooth"):
            r = rsel[(c, s)]
            rows.append((f"D-{c}-{s}-persp-r{r}", f"D-{c}-{s} persp r{r}"))
        rows.append((f"D-{c}-orc-weak-r{rsel[(c, 'orc')]}", f"D-{c}-orc WEAK r{rsel[(c, 'orc')]}"))
        rows.append((f"D-{c}-smooth-weak-r{rsel[(c, 'smooth')]}", f"D-{c}-smooth WEAK r{rsel[(c, 'smooth')]}"))
    for m, lab in rows:
        L("  " + fmt_row(d, m, lab))
    L("  stencil sweep (pixel-pooled mean geodesic, persp):")
    for c in ("pred", "gt"):
        for s in ("orc", "part", "smooth"):
            L(f"    D-{c}-{s:6s}: " + "  ".join(f"r{r} {pooled(d, f'D-{c}-{s}-persp-r{r}', 'geo')[0]:6.2f}" for r in (1, 2, 3)))
    L("  two-fold ambiguity: share of Omega where the two solutions differ by >30 deg | derived normal back-facing")
    for c in ("pred", "gt"):
        for r in (1, 2, 3):
            L(f"    {c:4s} r{r}: persp {100*ex[f'amb30_{c}-persp-r{r}']/ex['n_omega']:5.1f}%   weak "
              f"{100*ex[f'amb30_{c}-weak-r{r}']/ex['n_omega']:5.1f}%   | back-facing persp "
              f"{100*ex[f'backfacing_{c}-persp-r{r}']/ex['n_omega']:5.2f}%")
    bp = f"D-pred-smooth-persp-r{rsel[('pred', 'smooth')]}"
    bpp = f"D-pred-part-persp-r{rsel[('pred', 'part')]}"
    bo = f"D-pred-orc-persp-r{rsel[('pred', 'orc')]}"
    L("  paired per image (per-image mean geodesic): wins = head better")
    for other in (bp, bpp, bo, f"D-gt-orc-persp-r{rsel[('gt', 'orc')]}", f"D-gt-smooth-persp-r{rsel[('gt', 'smooth')]}"):
        w, l, p, md, lo, hi = paired(d, "H", other)
        L(f"    H vs {other:24s}: head wins {w}, loses {l}, sign test p={p:.2e}, mean(other - H) {md:+.2f} deg "
          f"[95% CI {lo:+.2f}, {hi:+.2f}]")
    L("  breakdown (pixel-pooled mean geodesic / normal / twist; n px):")
    for gname in ("torso", "arms", "legs", "hands_feet", "head", "d<4", "d4-12", "d>=12"):
        s = f"    {gname:10s} n={pooled(d, 'H', 'geo', gname)[3]:9d} "
        for m, lab in (("H", "H"), (bpp, "Dpart"), (bp, "Dsmooth"), (bo, "Dorc"),
                       (f"D-gt-orc-persp-r{rsel[('gt', 'orc')]}", "GTorc"), (f"D-gt-smooth-persp-r{rsel[('gt', 'smooth')]}", "GTsmooth")):
            s += f"| {lab} " + "/".join(f"{pooled(d, m, mt, gname)[0]:.1f}" for mt in MET) + " "
        L(s)
    return rsel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--out", default=f"{OUT}/report.txt")
    a = ap.parse_args()
    files = a.files or sorted(f for f in glob.glob(f"{OUT}/S*_*.npz") if "_lim" not in f)
    lines = []
    rsel = {}
    for f in files:                                   # r chosen on C1 (dp_w_M) per split, reused for the others
        sp = os.path.basename(f)[:2]
        if "dp_w_M_ckpt" in f:
            d = load(f)
            rsel[sp] = {(c, s): best_r(d, c, s) for c in ("pred", "gt") for s in ("orc", "part", "smooth")}
    for f in files:
        report_one(f, lines, rsel.get(os.path.basename(f)[:2]))
    # compact cross-checkpoint table
    lines.append("=" * 150)
    lines.append("CROSS-CHECKPOINT (pixel-pooled over each run's Omega; mean geodesic / normal / twist; "
                 "RESULTS-statistic reproduction in brackets)")
    for f in files:
        d = load(f); sp = os.path.basename(f)[:2]; rs = rsel.get(sp) or {(c, s): best_r(d, c, s) for c in ("pred", "gt") for s in ("orc", "part", "smooth")}
        m = d["meta"]
        rep = (f"eval_512 {m['repro_eval512_geo_e2e']:.2f}" if "repro_eval512_geo_e2e" in m else
               f"probe {m['repro_indomain_probe']['geo_e2e']:.2f}" if "repro_indomain_probe" in m else "")
        s = f"  {sp} {m['ckpt'].split('dp_runs/')[1]:28s} [{rep:14s}]"
        for meth, lab in (("H", "H"), (f"D-pred-smooth-persp-r{rs[('pred','smooth')]}", "Dpred-smooth"),
                          (f"D-pred-orc-persp-r{rs[('pred','orc')]}", "Dpred-orc"),
                          (f"D-gt-orc-persp-r{rs[('gt','orc')]}", "Dgt-orc")):
            s += f" | {lab} " + "/".join(f"{pooled(d, meth, mt)[0]:.2f}" for mt in MET)
        w, l, p, md, lo, hi = paired(d, "H", f"D-pred-smooth-persp-r{rs[('pred','smooth')]}")
        s += f" | H wins {w}/{w+l}"
        lines.append(s)
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
