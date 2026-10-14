# Building the paper

TeX Live (TinyTeX) is installed and the CVPR template compiles. **Always set the PATH first:**

```bash
export PATH=$HOME/.TinyTeX/bin/aarch64-linux:$HOME/.local/bin:$PATH
cd ~/codes/wan/paper
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

Already installed and verified working: `cvpr.sty` deps (silence, currfile, cuted, soul, xspace,
cleveref, caption/subcaption, microtype, times/psnfss, lineno, flushend), plus booktabs, multirow,
amsmath, amsfonts, float, enumitem, siunitx, colortbl, array, threeparttable, makecell, wrapfig,
graphics, latexmk, natbib, `collection-latexrecommended`, `collection-fontsrecommended`.

Anything else: `tlmgr install <pkg>` (note: some packages live in bundles — `xspace`/`array` are in
`tools`, `cuted`/`flushend` in `sttools`, `subcaption` in `caption`).

Template source: `author-kit-main/` (cvpr.sty, preamble.tex, ieeenat_fullname.bst). Use
`\usepackage[review]{cvpr}` for the review version.

Figures live in `figs/` — see `FIGURES.md` for the full inventory and where each belongs.
Numbers come from `RESULTS.md` — that file is the single source of truth; never invent a number.

A smoke-test build of the stock template lives at /tmp/texcheck if you need a known-good reference.
