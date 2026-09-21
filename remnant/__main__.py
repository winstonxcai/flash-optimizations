"""CLI for the remnant package: python -m remnant <cmd>.

Commands: patch | unpatch | verify | drift
Run from flash-optimizations (or with it on PYTHONPATH).
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
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
