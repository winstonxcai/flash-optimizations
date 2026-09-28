"""Shared import fragment for the SGLang source patches.

Spliced into every patched module so the hook can reach the live package. The
package root is resolved at patch time, so a tree patched from a mounted repo
imports the checkout rather than a baked copy. The ``try`` keeps a broken
package from turning the server into an ImportError instead of a no-op.
"""

from .. import config


def _import_block() -> str:
    return (
        "\n" + config.MARKER + " (import)\n"
        "import sys as _sg_lr_sys\n"
        f"if {config.PACKAGE_ROOT!r} not in _sg_lr_sys.path:\n"
        f"    _sg_lr_sys.path.insert(0, {config.PACKAGE_ROOT!r})\n"
        "try:\n    import starkv as _sg_lr\nexcept Exception:\n    _sg_lr = None\n"
    )
