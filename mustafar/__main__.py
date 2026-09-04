"""CLI for the mustafar package: python -m mustafar <cmd>.

Commands: patch | unpatch | verify | selftest | packed_selftest
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
    elif cmd == "selftest":
        from .tests import unit

        unit.run_topmag()
    elif cmd in {"packed_selftest", "packedselftest"}:
        from .tests import unit

        unit.run_packed_reference()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
