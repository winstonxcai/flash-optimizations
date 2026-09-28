"""Compressor edits: take over the c4 store and tag the current layer.

Both edits land in ``forward_unified``/``_forward_compress_all_in_one``, the one
live compressor path on CUDA (``_forward_unified_hip`` is unreferenced in
v0.5.18, and the unified-KV triton path is gated on ``is_hip()``).
"""

from .. import config
from . import _import_block


def _compressor_edits():
    return [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        # Layer tag. `forward_unified` is the only place a layer id is known
        # before the store, and it runs for every compressor (c4, c128 and the
        # indexer), so the hook itself is what filters by ratio/indexer.
        ("        if forward_batch.forward_mode.is_idle():\n            return\n",
         "        if forward_batch.forward_mode.is_idle():\n            return\n"
         "        if _sg_lr is not None:\n            _sg_lr.set_cur_layer(layer_id)\n"),
        # Store takeover. Everything the hook needs is a parameter or local of
        # _forward_compress_all_in_one: the caller has already resolved the
        # store target, so `kv_cache`/`page_size`/`bf16_store` are what decides
        # whether this write is the c4 paged pool we own.
        ("        # Step 2: norm + rope + store\n        compress_norm_rope_store(\n",
         "        # Step 2: norm + rope + store\n"
         "        " + config.MARKER + " (low-rank store)\n"
         "        if _sg_lr is not None and _sg_lr.store_lowrank(\n"
         "            kv_compressed, plan, norm, compress_ratio, is_indexer,\n"
         "            kv_cache, page_size, out_loc, freqs_cis_cache, bf16_store,\n"
         "        ):\n"
         "            return\n"
         "        compress_norm_rope_store(\n"),
    ]
