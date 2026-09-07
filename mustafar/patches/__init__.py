"""Shared import fragment for the SGLang source patches."""

from .. import config


def _import_block() -> str:
    return (
        "\n" + config.MARKER + " (import)\n"
        "import sys as _sg_lr_sys\n"
        f"if {config.PACKAGE_ROOT!r} not in _sg_lr_sys.path:\n"
        f"    _sg_lr_sys.path.insert(0, {config.PACKAGE_ROOT!r})\n"
        "import mustafar as _sg_lr\n"
    )
