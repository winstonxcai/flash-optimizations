"""CLI for the mustafar package: python -m mustafar <cmd>.

Commands: patch | unpatch | verify | selftest | sparseselftest
Run from flash-optimizations (or with it on PYTHONPATH).
"""
import sys

from . import ops


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "patch":
        ops.patch()
    elif cmd == "unpatch":
        ops.unpatch()
    elif cmd == "verify":
        ops.verify()
    elif cmd == "selftest":
        from . import selftest
        selftest.run()
    elif cmd == "sparseselftest":
        from . import selftest_sparse
        selftest_sparse.run()
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
