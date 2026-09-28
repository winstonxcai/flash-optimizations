"""CLI for the starkv package: python -m starkv <cmd>.

Commands: patch | unpatch | verify | drift | selftest | layout
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
    elif cmd == "selftest":
        from .tests import test_analysis, test_patching, test_reference

        rc = 0
        for suite in (test_reference, test_analysis, test_patching):
            rc |= suite.run()
            print()
        raise SystemExit(1 if rc else 0)
    elif cmd == "layout":
        from . import config

        print(
            f"rank={config.RANK} bytes/token={config.BYTES_PER_TOKEN} "
            f"(native {config.NATIVE_RECORD_BYTES}, "
            f"{config.NATIVE_RECORD_BYTES / config.BYTES_PER_TOKEN:.3f}x)"
        )
        print(
            f"  z      [{config.Z_OFFSET}:{config.Z_OFFSET + config.RANK}] "
            f"{config.RANK} fp8"
        )
        print(
            f"  scales [{config.SCALE_OFFSET}:{config.SCALE_OFFSET + config.SCALE_TILES}] "
            f"{config.SCALE_TILES} u8"
        )
        print(
            f"  tail   [{config.ROPE_OFFSET}:{config.ROPE_OFFSET + config.ROPE_BYTES}] "
            f"{config.ROPE_BYTES} bf16"
        )
        print(f"  pad    {config.PAD_BYTES} B")
    else:
        raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
