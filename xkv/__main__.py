"""CLI for the xkv package: python -m xkv <cmd>.

Commands: patch | unpatch | verify | drift | selftest
Run from flash-optimizations (or with it on PYTHONPATH). `drift` is read-only
and exits non-zero when any anchor has moved in the tree at SG_LOWRANK_SRC.
"""

import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "patch":
        from . import patching

        patching.patch()
    elif cmd == "unpatch":
        from . import patching

        patching.unpatch()
    elif cmd == "verify":
        from . import patching

        patching.verify()
    elif cmd == "drift":
        from . import patching

        raise SystemExit(patching.drift())
    elif cmd == "selftest":
        from .tests import validity

        validity.run_reference()
    elif cmd in {"calib_finalize", "calibfinalize"}:
        from . import calib

        calib.run_finalize()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
