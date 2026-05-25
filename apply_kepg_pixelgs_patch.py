#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restore files changed by apply_srap_pixelgs_patch.py.
Run:
  cd /root/Pixel-GS
  python restore_srap_pixelgs_patch.py
"""
from pathlib import Path
import shutil

root = Path.cwd().resolve()
files = [
    "arguments/__init__.py",
    "train.py",
    "scene/gaussian_model.py",
]

for rel in files:
    path = root / rel
    bak = path.with_suffix(path.suffix + ".bak_srap")
    if bak.exists():
        shutil.copy2(bak, path)
        print(f"[RESTORE] {rel}")
    else:
        print(f"[MISS] {bak}")

print("Done.")
