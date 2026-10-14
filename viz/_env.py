"""Where the page generators find things when run from this repository.

Imported first by every generator in viz/. It
  * puts research/densepose (the research model, data loader and orient/ helpers) and research/bedlam on sys.path;
  * makes the research modules read the lookup tables from the repository's assets/ (pff.assets) and the SMPL-X
    body model from $SMPLX_DIR, by patching their path constants as they are imported;
  * maps the research data locations "~/data/<rel>" to $PFF_DATA/<rel> (default ~/data, as on the cluster) and the
    CameraHMR files "~/codes/CameraHMR/<rel>" to $CAMERAHMR/<rel> (default ~/codes/CameraHMR);
  * gives generators without their own --out option one (take_out_arg / out), defaulting to viz/out/<name>.
"""
import importlib.abc
import importlib.machinery
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, os.path.join(REPO, "research", "bedlam"), os.path.join(REPO, "research", "densepose")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pff.assets import ASSETS as _ASSETS  # noqa: E402

ASSETS = str(_ASSETS)
SMPLX_DIR = os.environ.get("SMPLX_DIR", os.path.expanduser("~/data/smplx/models_lockedhead"))
DATA = os.environ.get("PFF_DATA", os.path.expanduser("~/data"))
CAMERAHMR = os.environ.get("CAMERAHMR", os.path.expanduser("~/codes/CameraHMR"))
OUT_ROOT = os.path.join(HERE, "out")

# research module -> {constant: value}, applied right after the module is imported
_BEDLAM2 = os.path.join(DATA, "bedlam2")
_PATCH = {"data": {"ASSETS": ASSETS, "ROOT": _BEDLAM2, "SHARDS": f"{_BEDLAM2}/dp_shards",
                   "SHARDS_ROT": f"{_BEDLAM2}/dp_shards_rot", "SHARDS_512": f"{_BEDLAM2}/dp_shards_512",
                   "SHARDS_ROT_MAMMA": os.path.join(DATA, "datasets", "dp_shards_rot_mamma"),
                   "SHARDS_512_MAMMA": os.path.join(DATA, "datasets", "dp_shards_512_mamma")},
          "viz_dataloader": {"ASSETS": ASSETS},
          "render_correspondence": {"SMPLX_DIR": SMPLX_DIR}}


class _PatchPaths(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PATCH:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module, run=run, values=_PATCH[name]):
            run(module)
            for k, v in values.items():
                setattr(module, k, v)
        spec.loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _PatchPaths) for f in sys.meta_path):
    sys.meta_path.insert(0, _PatchPaths())


def data(rel=""):
    """A research data location "~/data/<rel>", under $PFF_DATA."""
    return os.path.join(DATA, rel)


def camerahmr(rel=""):
    """A CameraHMR file "~/codes/CameraHMR/<rel>" (the 3DPW refinement study), under $CAMERAHMR."""
    return os.path.join(CAMERAHMR, rel)


_OUT = None


def take_out_arg():
    """For generators without their own --out: take `--out PATH` off the command line (before the script's own
    argument parser, if any, sees it)."""
    global _OUT
    argv = sys.argv
    for i, a in enumerate(argv[1:], 1):
        if a == "--out" and i + 1 < len(argv):
            _OUT = argv[i + 1]
            del argv[i:i + 2]
            return
        if a.startswith("--out="):
            _OUT = a.split("=", 1)[1]
            del argv[i]
            return


def out(name):
    """The output path: --out if given, else viz/out/<name>."""
    return os.path.abspath(os.path.expanduser(_OUT)) if _OUT else os.path.join(OUT_ROOT, name)
