"""Compressor edits: select one TopMag mask and route packed writes."""

from . import _import_block


def _compressor_edits():
    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + _import_block(),
        ),
        (
            "        kv_cache: torch.Tensor,\n        is_indexer: bool,\n",
            "        kv_cache: Optional[torch.Tensor],\n        is_indexer: bool,\n",
        ),
        (
            "        bf16_store: bool = False,\n    ) -> None:\n",
            "        bf16_store: bool = False,\n"
            "        packed_pool=None,\n"
            "        packed_layer_id: Optional[int] = None,\n"
            "    ) -> None:\n",
        ),
        (
            "        # Step 2: norm + rope + store\n"
            "        compress_norm_rope_store(\n",
            "        ## MUSTAFAR (single-mask packed store)\n"
            "        if (\n"
            "            compress_ratio == 4\n"
            "            and not is_indexer\n"
            "            and _sg_lr.topmag_enabled()\n"
            "        ):\n"
            "            keep_mask = _sg_lr.topmag_keep_mask(\n"
            "                kv_compressed, _sg_lr.topmag_keep()\n"
            "            )\n"
            "            if _sg_lr.packed_enabled():\n"
            "                _sg_lr.validate_packed_static_config()\n"
            "                assert packed_pool is not None\n"
            "                assert packed_layer_id is not None\n"
            "                packed_pool.set_rope_freqs(\n"
            "                    packed_layer_id, freqs_cis_cache\n"
            "                )\n"
            "                _sg_lr.pack_rows(\n"
            "                    kv_compressed, keep_mask, norm.weight,\n"
            "                    norm.variance_epsilon, plan, out_loc,\n"
            "                    packed_pool, layer_id=packed_layer_id,\n"
            "                )\n"
            "                return\n"
            "            _sg_lr.topmag_zero_from_mask(kv_compressed, keep_mask)\n"
            "\n"
            "        # Step 2: norm + rope + store\n"
            "        assert kv_cache is not None\n"
            "        compress_norm_rope_store(\n",
        ),
        (
            "            bf16_store = False\n"
            "            if compressor.is_in_indexer:\n",
            "            bf16_store = False\n"
            "            packed_pool = None\n"
            "            packed_layer_id = None\n"
            "            if compressor.is_in_indexer:\n",
        ),
        (
            "                _, _, compress_kv_pool = token_to_kv_pool.layer_mapping[layer_id]\n",
            "                _, compress_layer_id, compress_kv_pool = (\n"
            "                    token_to_kv_pool.layer_mapping[layer_id]\n"
            "                )\n",
        ),
        (
            "                kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)\n"
            "                page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)\n",
            "                if (\n"
            "                    _sg_lr.packed_enabled()\n"
            "                    and compressor.ratio == 4\n"
            "                ):\n"
            "                    kv_cache = None\n"
            "                    packed_pool = compress_kv_pool\n"
            "                    packed_layer_id = compress_layer_id\n"
            "                else:\n"
            "                    kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)\n"
            "                page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)\n",
        ),
        (
            "                kv_cache=kv_cache.view(dtype=torch.uint8),\n"
            "                is_indexer=compressor.is_in_indexer,\n",
            "                kv_cache=(\n"
            "                    kv_cache.view(dtype=torch.uint8)\n"
            "                    if kv_cache is not None else None\n"
            "                ),\n"
            "                is_indexer=compressor.is_in_indexer,\n",
        ),
        (
            "                bf16_store=bf16_store,\n            )\n",
            "                bf16_store=bf16_store,\n"
            "                packed_pool=packed_pool,\n"
            "                packed_layer_id=packed_layer_id,\n"
            "            )\n",
        ),
    ]
