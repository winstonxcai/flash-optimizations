"""CLI for the mustafar package: python -m mustafar <cmd>.

Commands: patch | unpatch | verify | selftest | sparseselftest
Run from flash-optimizations (or with it on PYTHONPATH).
"""
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "patch":
        from . import ops
        ops.patch()
    elif cmd == "unpatch":
        from . import ops
        ops.unpatch()
    elif cmd == "verify":
        from . import ops
        ops.verify()
    elif cmd == "selftest":
        from .tests import unit
        unit.run_topmag()
    elif cmd == "sparseselftest":
        from .tests import unit
        unit.run_sparse()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
