#!/usr/bin/env python3
"""Apply GodsEye's DEVA patches. Idempotent — safe to re-run."""
import sys
from pathlib import Path

DEVA_ROOT = Path.home() / "Desktop/Tracking-Anything-with-DEVA"

PATCHES = [
    ("deva/ext/MobileSAM/tiny_vit_sam.py",
     "return x, None",
     "return x",
     "TinyViT.forward: return Tensor (not tuple) for vanilla SAM decoder"),
]

def main():
    if not DEVA_ROOT.exists():
        sys.exit(f"DEVA repo not found at {DEVA_ROOT}")
    ok = True
    for relpath, old, new, desc in PATCHES:
        p = DEVA_ROOT / relpath
        if not p.exists():
            print(f"MISSING   {relpath}"); ok = False; continue
        src = p.read_text()
        if old not in src and new in src:
            print(f"OK        {relpath}: already patched"); continue
        if old not in src:
            print(f"UNKNOWN   {relpath}: '{old}' not found"); ok = False; continue
        p.write_text(src.replace(old, new))
        print(f"PATCHED   {relpath}: {desc}")
    print("\n" + ("All patches applied." if ok else "Some patches failed."))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
