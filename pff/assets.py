"""Location of the SMPL-X-derived lookup tables (assets/README.md). Override with PFF_ASSETS."""
import os
from pathlib import Path

ASSETS = Path(os.environ.get("PFF_ASSETS", Path(__file__).resolve().parents[1] / "assets"))


def asset(name):
    p = ASSETS / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing (see assets/README.md; set PFF_ASSETS to another folder)")
    return str(p)
