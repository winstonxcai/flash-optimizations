"""Backward-compatible entrypoint for the xkv package."""
from xkv import *  # noqa: F401,F403
from xkv.ops import (
    decode_lowrank,
    dequantize_lowrank_k_cache_paged,
    lowrank_enabled,
    patch,
    set_basis_dir,
    set_cur_layer,
    store_compressed_lowrank,
    unpatch,
    verify,
)


def _main():
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if command == "patch":
        patch()
    elif command == "unpatch":
        unpatch()
    elif command == "verify":
        print(verify())
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    _main()
