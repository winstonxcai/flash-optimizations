"""Injected packed pool, accessors, and allocator byte accounting.

The class remains source text inserted into SGLang, where its base class and
memory-management names already exist. Do not import SGLang from this module.
"""

from . import _import_block


def _packed_pool_class() -> str:
    return r'''

## MUSTAFAR (packed-pool)
class MustafarPackedKVPool(DeepSeekV4SingleKVPool):
    """Persistent 328-byte/page-row packed storage (no native shadow pool)."""

    def _create_buffers(self):
        num_pages = (self.size + self.page_size + 1) // self.page_size
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.packed_values = [
                    torch.zeros(
                        num_pages, self.page_size, 256,
                        dtype=torch.uint8, device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.packed_bitmaps = [
                    torch.zeros(
                        num_pages, self.page_size, 8,
                        dtype=torch.uint64, device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.packed_scales = [
                    torch.zeros(
                        num_pages, self.page_size, 8,
                        dtype=torch.uint8, device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
        # Compatibility only: generic lifecycle code expects this attribute.
        # It must never be treated as a native hybrid allocation.
        self.kv_buffer = self.packed_values
        self.kv_cache_total_dim = 328
        self.bytes_per_page_padded = self.page_size * 328
        self._mustafar_rope_freqs = [None] * self.layer_num
        logger.info(
            "Mustafar packed pool: layers=%d pages/layer=%d "
            "page_size=%d logical_row_bytes=328 allocated_bytes=%d",
            self.layer_num,
            num_pages,
            self.page_size,
            self.get_kv_size_bytes(),
        )

    def get_bytes_per_token(self) -> int:
        return 328

    def get_packed_buffers(self, layer_id: int):
        local = layer_id - self.start_layer
        return (
            self.packed_values[local],
            self.packed_bitmaps[local],
            self.packed_scales[local],
        )

    def set_rope_freqs(self, layer_id: int, freqs_cis) -> None:
        local = layer_id - self.start_layer
        if self._mustafar_rope_freqs[local] is None:
            if not freqs_cis.is_complex() or not freqs_cis.is_contiguous():
                raise RuntimeError(
                    "packed requires a contiguous complex RoPE table"
                )
            # Retain the compressor's existing table. Creating contiguous real
            # and imaginary copies costs 256 MiB per rank at 135168 context and
            # can OOM after KV-pool sizing has consumed the remaining HBM.
            self._mustafar_rope_freqs[local] = freqs_cis

    def get_rope_freqs(self, layer_id: int):
        local = layer_id - self.start_layer
        value = self._mustafar_rope_freqs[local]
        if value is None:
            raise RuntimeError("packed RoPE frequencies not initialized")
        return value

    def get_key_buffer(self, layer_id: int):
        raise RuntimeError("packed has no native key buffer; unpack first")

    def set_key_buffer(self, *args, **kwargs):
        raise RuntimeError("packed must be written through pack_rows")

    set_key_buffer_fused = set_key_buffer

    def get_kv_size_bytes(self):
        return sum(
            t.nbytes
            for group in (
                self.packed_values, self.packed_bitmaps, self.packed_scales
            )
            for t in group
        )

    def get_buf_infos(self):
        ptrs, lens, items = [], [], []
        for layer in range(self.layer_num):
            for buf in self.get_packed_buffers(layer):
                ptrs.append(buf.data_ptr())
                lens.append(buf.nbytes)
                items.append(buf[0].nbytes)
        return ptrs, lens, items
'''


def _memory_pool_edits():
    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + _import_block(),
        ),
        (
            "\n\nclass HiSparseC4DevicePool(DeepSeekV4SingleKVPool):\n",
            _packed_pool_class()
            + "\n\nclass HiSparseC4DevicePool(DeepSeekV4SingleKVPool):\n",
        ),
        (
            "            c4_kv_pool_type = DeepSeekV4SingleKVPool\n"
            "            if enable_hisparse:\n"
            "                c4_kv_pool_type = HiSparseC4DevicePool\n",
            "            c4_kv_pool_type = DeepSeekV4SingleKVPool\n"
            "            if _sg_lr.packed_enabled():\n"
            "                _sg_lr.validate_packed_static_config()\n"
            "                if enable_hisparse:\n"
            "                    raise RuntimeError(\n"
            "                        'packed is incompatible with HiSparse'\n"
            "                    )\n"
            "                c4_kv_pool_type = MustafarPackedKVPool\n"
            "            elif enable_hisparse:\n"
            "                c4_kv_pool_type = HiSparseC4DevicePool\n",
        ),
        (
            "        buf_groups = [\n"
            "            self.c4_kv_pool.kv_buffer,\n"
            "            self.c4_indexer_kv_pool.index_k_with_scale_buffer,\n"
            "            self.c128_kv_pool.kv_buffer,\n"
            "        ]\n",
            "        if _sg_lr.packed_enabled():\n"
            "            p, n, i = self.c4_kv_pool.get_buf_infos()\n"
            "            data_ptrs.extend(p)\n"
            "            data_lens.extend(n)\n"
            "            item_lens.extend(i)\n"
            "            buf_groups = [\n"
            "                self.c4_indexer_kv_pool.index_k_with_scale_buffer,\n"
            "                self.c128_kv_pool.kv_buffer,\n"
            "            ]\n"
            "        else:\n"
            "            buf_groups = [\n"
            "                self.c4_kv_pool.kv_buffer,\n"
            "                self.c4_indexer_kv_pool.index_k_with_scale_buffer,\n"
            "                self.c128_kv_pool.kv_buffer,\n"
            "            ]\n",
        ),
        (
            "    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:\n",
            "    ## MUSTAFAR (packed accessors)\n"
            "    def get_packed_pool(self, layer_id: int):\n"
            "        ratio, _, pool = self.layer_mapping[layer_id]\n"
            "        if ratio != 4 or not isinstance(pool, MustafarPackedKVPool):\n"
            "            raise RuntimeError('layer does not use packed')\n"
            "        self.wait_layer_transfer(layer_id)\n"
            "        return pool\n"
            "\n"
            "    def get_packed_buffers(self, layer_id: int):\n"
            "        ratio, local, pool = self.layer_mapping[layer_id]\n"
            "        if ratio != 4 or not isinstance(pool, MustafarPackedKVPool):\n"
            "            raise RuntimeError('layer does not use packed')\n"
            "        self.wait_layer_transfer(layer_id)\n"
            "        return pool.get_packed_buffers(local)\n"
            "\n"
            "    def get_packed_freqs(self, layer_id: int):\n"
            "        ratio, local, pool = self.layer_mapping[layer_id]\n"
            "        if ratio != 4 or not isinstance(pool, MustafarPackedKVPool):\n"
            "            raise RuntimeError('layer does not use packed')\n"
            "        return pool.get_rope_freqs(local)\n"
            "\n"
            "    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:\n",
        ),
    ]


def _pool_config_edits():
    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + _import_block(),
        ),
        (
            "            + c4_frac * kv_bytes * self.num_layers_ca4\n",
            "            + c4_frac\n"
            "            * (_sg_lr.PACKED_RECORD_BYTES if _sg_lr.packed_enabled() else kv_bytes)\n"
            "            * self.num_layers_ca4\n",
        ),
    ]
