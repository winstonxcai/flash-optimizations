#!/usr/bin/env python3
"""External harness copy of the test fixture's import shim (audit note).

The reference fixture `third_party/flashmla/tests/lib.py` does `import flash_mla`
(the standalone reference package). In the disposable pilot image only the
installed AOT shim `sgl_kernel.flash_mla` exists, so the pilot needs a fallback.
Rather than edit the tracked submodule, this external copy inserts that one-line
fallback into `lib` AFTER it is imported -- see `install_flash_mla_fallback`.
The tests/lib.py hunk used for the earlier runs has been reverted; nothing in the
submodule is required for the pilot anymore.
"""
import sys


def install_flash_mla_fallback():
    """Make `import flash_mla` resolve to the installed AOT shim if absent.

    Returns the module object that `lib.flash_mla`/`ref.flash_mla` will see.
    """
    try:
        import flash_mla as _fm  # standalone reference package, if present
    except ImportError:
        import sgl_kernel.flash_mla as _fm  # installed AOT shim
    # Pre-register so subsequent `import flash_mla` anywhere returns the shim.
    sys.modules.setdefault("flash_mla", _fm)
    return _fm
