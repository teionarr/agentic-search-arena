"""Ensure the repo root (for the real ``arena`` package) and this dir (for ``_fakes``) are
importable, regardless of pytest's rootdir. Kept minimal: fakes live in ``_fakes.py``."""

import os
import sys

_here = os.path.dirname(__file__)
_root = os.path.abspath(os.path.join(_here, "..", ".."))
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
