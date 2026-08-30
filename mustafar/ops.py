"""Patch SGLang with Mustafar's dense TopMag and packed C4 paths."""

import os
import shutil
import subprocess

from . import config


# --- patch / unpatch / verify --------------------------------------------------
def _apply(path: str, edits) -> None:
    with open(path) as f:
        s = f.read()
    if config.MARKER in s:
        print(f"[mustafar] {path}: already patched")
        return
    backup = path + ".mustafar.orig"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    for anchor, new in edits:
        if s.count(anchor) != 1:
            raise AssertionError(
                f"[mustafar] anchor count != 1 in {path}: {anchor[:70]!r}")
        s = s.replace(anchor, new, 1)
    with open(path, "w") as f:
        f.write(s)
    print(f"[mustafar] {path}: ok")


def _import_block() -> str:
    return ("\n" + config.MARKER + " (import)\n"
            "import sys as _sg_lr_sys\n"
            f"if {config.PACKAGE_ROOT!r} not in _sg_lr_sys.path:\n"
            f"    _sg_lr_sys.path.insert(0, {config.PACKAGE_ROOT!r})\n"
            "import mustafar as _sg_lr\n")


def _packed_pool_class() -> str:
    return r'''

## MUSTAFAR (packed-c4-pool)
class MustafarPackedC4KVPool(DeepSeekV4SingleKVPool):
    """Persistent 328-byte/page-row C4 storage (no native shadow pool)."""

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
        # It must never be treated as a native hybrid C4 allocation.
        self.kv_buffer = self.packed_values
        self.kv_cache_total_dim = 328
        self.bytes_per_page_padded = self.page_size * 328
        self._mustafar_rope_freqs = [None] * self.layer_num
        logger.info(
            "Mustafar packed C4 pool: layers=%d pages/layer=%d "
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
                    "packed C4 requires a contiguous complex RoPE table"
                )
            # Retain the compressor's existing table. Creating contiguous real
            # and imaginary copies costs 256 MiB per rank at 135168 context and
            # can OOM after KV-pool sizing has consumed the remaining HBM.
            self._mustafar_rope_freqs[local] = freqs_cis

    def get_rope_freqs(self, layer_id: int):
        local = layer_id - self.start_layer
        value = self._mustafar_rope_freqs[local]
        if value is None:
            raise RuntimeError("packed C4 RoPE frequencies not initialized")
        return value

    def get_key_buffer(self, layer_id: int):
        raise RuntimeError("packed C4 has no native key buffer; unpack first")

    def set_key_buffer(self, *args, **kwargs):
        raise RuntimeError("packed C4 must be written through pack_c4_rows")

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


def _compressor_edits():
    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + _import_block(),
        ),
        (
            "        kv_cache: torch.Tensor,\n"
            "        is_indexer: bool,\n",
            "        kv_cache: Optional[torch.Tensor],\n"
            "        is_indexer: bool,\n",
        ),
        (
            "        bf16_store: bool = False,\n"
            "    ) -> None:\n",
            "        bf16_store: bool = False,\n"
            "        packed_c4_pool=None,\n"
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
            "            if _sg_lr.packed_c4_enabled():\n"
            "                _sg_lr.validate_packed_static_config()\n"
            "                assert packed_c4_pool is not None\n"
            "                assert packed_layer_id is not None\n"
            "                packed_c4_pool.set_rope_freqs(\n"
            "                    packed_layer_id, freqs_cis_cache\n"
            "                )\n"
            "                _sg_lr.pack_c4_rows(\n"
            "                    kv_compressed, keep_mask, norm.weight,\n"
            "                    norm.variance_epsilon, plan, out_loc,\n"
            "                    packed_c4_pool, layer_id=packed_layer_id,\n"
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
            "            packed_c4_pool = None\n"
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
            "                    _sg_lr.packed_c4_enabled()\n"
            "                    and compressor.ratio == 4\n"
            "                ):\n"
            "                    kv_cache = None\n"
            "                    packed_c4_pool = compress_kv_pool\n"
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
            "                bf16_store=bf16_store,\n"
            "            )\n",
            "                bf16_store=bf16_store,\n"
            "                packed_c4_pool=packed_c4_pool,\n"
            "                packed_layer_id=packed_layer_id,\n"
            "            )\n",
        ),
    ]


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
            "            if _sg_lr.packed_c4_enabled():\n"
            "                _sg_lr.validate_packed_static_config()\n"
            "                if enable_hisparse:\n"
            "                    raise RuntimeError(\n"
            "                        'packed C4 is incompatible with HiSparse'\n"
            "                    )\n"
            "                c4_kv_pool_type = MustafarPackedC4KVPool\n"
            "            elif enable_hisparse:\n"
            "                c4_kv_pool_type = HiSparseC4DevicePool\n",
        ),
        (
            "        buf_groups = [\n"
            "            self.c4_kv_pool.kv_buffer,\n"
            "            self.c4_indexer_kv_pool.index_k_with_scale_buffer,\n"
            "            self.c128_kv_pool.kv_buffer,\n"
            "        ]\n",
            "        if _sg_lr.packed_c4_enabled():\n"
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
            "    ## MUSTAFAR (packed-c4 accessors)\n"
            "    def get_packed_c4_pool(self, layer_id: int):\n"
            "        ratio, _, pool = self.layer_mapping[layer_id]\n"
            "        if ratio != 4 or not isinstance(pool, MustafarPackedC4KVPool):\n"
            "            raise RuntimeError('layer does not use packed C4')\n"
            "        self.wait_layer_transfer(layer_id)\n"
            "        return pool\n"
            "\n"
            "    def get_packed_c4_buffers(self, layer_id: int):\n"
            "        ratio, local, pool = self.layer_mapping[layer_id]\n"
            "        if ratio != 4 or not isinstance(pool, MustafarPackedC4KVPool):\n"
            "            raise RuntimeError('layer does not use packed C4')\n"
            "        self.wait_layer_transfer(layer_id)\n"
            "        return pool.get_packed_buffers(local)\n"
            "\n"
            "    def get_packed_c4_freqs(self, layer_id: int):\n"
            "        ratio, local, pool = self.layer_mapping[layer_id]\n"
            "        if ratio != 4 or not isinstance(pool, MustafarPackedC4KVPool):\n"
            "            raise RuntimeError('layer does not use packed C4')\n"
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
            "            * (_sg_lr.PACKED_C4_BYTES if _sg_lr.packed_c4_enabled() else kv_bytes)\n"
            "            * self.num_layers_ca4\n",
        ),
    ]


def _indexer_edits():
    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + _import_block(),
        ),
        (
            "        if capture_enabled:\n"
            "            raw_indices = torch.empty_like(c4_sparse_page_indices)\n",
            "        if capture_enabled and not _sg_lr.packed_c4_enabled():\n"
            "            raw_indices = torch.empty_like(c4_sparse_page_indices)\n",
        ),
        (
            "        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:\n"
            "            topk_transform_512_v2(\n"
            "                logits,\n"
            "                c4_seq_lens,\n"
            "                page_table,\n"
            "                c4_sparse_page_indices,\n"
            "                indexer_metadata.c4_page_size,\n"
            "                indexer_metadata.topk_metadata,\n"
            "            )\n",
            "        elif envs.SGLANG_OPT_USE_TOPK_V2.get():\n"
            "            ## MUSTAFAR (v2 raw-index auxiliary output)\n"
            "            topk_transform_512_v2(\n"
            "                logits,\n"
            "                c4_seq_lens,\n"
            "                page_table,\n"
            "                c4_sparse_page_indices,\n"
            "                indexer_metadata.c4_page_size,\n"
            "                indexer_metadata.topk_metadata,\n"
            "                raw_indices,\n"
            "            )\n",
        ),
    ]


def _backend_edits():
    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + _import_block(),
        ),
        (
            "        if is_prefill:\n"
            "            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)\n",
            "        if is_prefill or _sg_lr.packed_c4_enabled():\n"
            "            ## MUSTAFAR (retain existing v2 top-k raw output)\n"
            "            self.c4_sparse_raw_indices = torch.empty_like(\n"
            "                self.c4_sparse_page_indices\n"
            "            )\n",
        ),
        (
            "        self.sparse_prefill_workspace = SparsePrefillWorkspace(self.device)\n",
            "        self.sparse_prefill_workspace = SparsePrefillWorkspace(self.device)\n"
            "        ## MUSTAFAR (static native gather workspace)\n"
            "        self.mustafar_c4_workspace = None\n"
            "        if _sg_lr.packed_c4_enabled():\n"
            "            _sg_lr.validate_packed_static_config()\n"
            "            args = model_runner.server_args\n"
            "            if self.device.type != 'cuda':\n"
            "                raise RuntimeError('packed C4 requires CUDA')\n"
            "            if self.c4_topk != 512:\n"
            "                raise RuntimeError('packed C4 requires index_topk=512')\n"
            "            if self.token_to_kv_pool._unified_kv:\n"
            "                raise RuntimeError('packed C4 is incompatible with unified KV')\n"
            "            if self.hisparse_coordinator is not None or args.enable_hisparse:\n"
            "                raise RuntimeError('packed C4 is incompatible with HiSparse')\n"
            "            if args.speculative_algorithm is not None or self.mtp_enabled:\n"
            "                raise RuntimeError('packed C4 is incompatible with speculative decode')\n"
            "            if args.disaggregation_mode != 'null':\n"
            "                raise RuntimeError('packed C4 is incompatible with disaggregation')\n"
            "            if (\n"
            "                args.cpu_offload_gb > 0\n"
            "                or args.enable_hierarchical_cache\n"
            "                or args.disaggregation_decode_enable_offload_kvcache\n"
            "            ):\n"
            "                raise RuntimeError('packed C4 is incompatible with offload')\n"
            "            if (\n"
            "                args.enable_prefill_context_parallel\n"
            "                or args.enable_dsa_prefill_context_parallel\n"
            "                or get_parallel().attn_cp_size != 1\n"
            "            ):\n"
            "                raise RuntimeError('packed C4 is incompatible with context parallelism')\n"
            "            self.mustafar_c4_workspace = _sg_lr.NativeC4Workspace.allocate(\n"
            "                self.token_to_kv_pool.max_num_reqs, 512,\n"
            "                self.page_size // 4, self.device,\n"
            "            )\n",
        ),
        (
            "            if compress_ratio == 4:\n"
            "                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)\n"
            "                extra_indices = core_attn_metadata.c4_sparse_page_indices\n"
            "                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths\n",
            "            if compress_ratio == 4:\n"
            "                extra_indices = core_attn_metadata.c4_sparse_page_indices\n"
            "                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths\n"
            "                if _sg_lr.packed_c4_enabled():\n"
            "                    raw_indices = core_attn_metadata.c4_sparse_raw_indices\n"
            "                    assert raw_indices is not None\n"
            "                else:\n"
            "                    extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)\n",
        ),
        (
            "            extra_indices = match_num_queries(extra_indices, value=-1)\n"
            "            extra_topk_lengths = match_num_queries(extra_topk_lengths, value=1)\n",
            "            extra_indices = match_num_queries(extra_indices, value=-1)\n"
            "            extra_topk_lengths = match_num_queries(extra_topk_lengths, value=1)\n"
            "            if compress_ratio == 4 and _sg_lr.packed_c4_enabled():\n"
            "                raw_indices = match_num_queries(raw_indices, value=-1)\n",
        ),
        (
            "            if forward_batch.forward_mode.is_extend_without_speculative() and (\n"
            "                q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD\n"
            "                or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()\n"
            "            ):\n",
            "            if forward_batch.forward_mode.is_extend_without_speculative() and (\n"
            "                q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD\n"
            "                or (\n"
            "                    _sg_lr.packed_c4_enabled()\n"
            "                    and self.mustafar_c4_workspace is not None\n"
            "                    and q.shape[0] > self.mustafar_c4_workspace.max_queries\n"
            "                )\n"
            "                or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()\n"
            "            ):\n",
        ),
        (
            "            if _is_sm120:\n",
            "            if compress_ratio == 4 and _sg_lr.packed_c4_enabled():\n"
            "                ## MUSTAFAR (decode/small-extend native reconstruction)\n"
            "                assert self.mustafar_c4_workspace is not None\n"
            "                packed_indices = (\n"
            "                    extra_indices.squeeze(1)\n"
            "                    if extra_indices.ndim == 3 else extra_indices\n"
            "                )\n"
            "                packed_raw_indices = (\n"
            "                    raw_indices.squeeze(1)\n"
            "                    if raw_indices.ndim == 3 else raw_indices\n"
            "                )\n"
            "                extra_k_cache, extra_indices = (\n"
            "                    _sg_lr.unpack_gather_c4_native(\n"
            "                        token_to_kv_pool.get_packed_c4_buffers(layer_id),\n"
            "                        packed_indices, packed_raw_indices,\n"
            "                        extra_topk_lengths,\n"
            "                        token_to_kv_pool.get_packed_c4_freqs(layer_id),\n"
            "                        self.mustafar_c4_workspace,\n"
            "                    )\n"
            "                )\n"
            "                extra_page_size = token_to_kv_pool.page_size // 4\n"
            "                extra_k_cache = extra_k_cache[\n"
            "                    :, : extra_page_size * k_cache_total_dim\n"
            "                ].view(\n"
            "                    extra_k_cache.shape[0], extra_page_size, 1,\n"
            "                    k_cache_total_dim,\n"
            "                )\n"
            "                extra_indices = extra_indices.unsqueeze(1)\n"
            "\n"
            "            if _is_sm120:\n",
        ),
        (
            "            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)\n"
            "            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)\n"
            "            if compress_ratio == 128:\n",
            "            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)\n"
            "            if not (\n"
            "                compress_ratio == 4 and _sg_lr.packed_c4_enabled()\n"
            "            ):\n"
            "                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)\n"
            "            if compress_ratio == 128:\n",
        ),
        (
            "        if compressed_slice is not None:\n"
            "            dequantize_k_cache_paged(\n"
            "                extra_k_cache,\n"
            "                flat_token_ids,\n"
            "                page_size=extra_page_size,\n"
            "                out=compressed_slice,\n"
            "            )\n",
            "        if compressed_slice is not None:\n"
            "            if compress_ratio == 4 and _sg_lr.packed_c4_enabled():\n"
            "                c4_max = flat_token_ids.numel() // cache.num_reqs\n"
            "                if not hasattr(cache, '_mustafar_c4_raw_indices'):\n"
            "                    cache._mustafar_c4_raw_indices = (\n"
            "                        torch.arange(\n"
            "                            c4_max, dtype=torch.int32, device=q.device\n"
            "                        )[None, :].expand(cache.num_reqs, -1).contiguous()\n"
            "                    )\n"
            "                    cache._mustafar_c4_lengths = torch.clamp(\n"
            "                        cache.seq_lens // 4, min=0, max=c4_max\n"
            "                    ).to(torch.int32)\n"
            "                _sg_lr.unpack_gather_c4_bf16(\n"
            "                    token_to_kv_pool.get_packed_c4_buffers(layer_id),\n"
            "                    flat_token_ids.view(cache.num_reqs, c4_max),\n"
            "                    cache._mustafar_c4_raw_indices,\n"
            "                    cache._mustafar_c4_lengths,\n"
            "                    token_to_kv_pool.get_packed_c4_freqs(layer_id),\n"
            "                    compressed_slice,\n"
            "                )\n"
            "            else:\n"
            "                dequantize_k_cache_paged(\n"
            "                    extra_k_cache,\n"
            "                    flat_token_ids,\n"
            "                    page_size=extra_page_size,\n"
            "                    out=compressed_slice,\n"
            "                )\n",
        ),
    ]


def patch() -> None:
    _apply(config.COMPRESSOR_V2, _compressor_edits())
    _apply(config.MEM_POOL, _memory_pool_edits())
    _apply(config.POOL_CFG, _pool_config_edits())
    _apply(config.INDEXER, _indexer_edits())
    _apply(config.DSV4_BACKEND, _backend_edits())


def unpatch() -> None:
    """Restore all five patched SGLang files to their pristine versions."""
    for p in config.PATCH_FILES:
        bak = p + ".mustafar.orig"
        if os.path.exists(bak):
            shutil.copy2(bak, p)
            print(f"[mustafar] restored {p} from .mustafar.orig")
        else:
            subprocess.run(["git", "checkout", "--", p], cwd=config.SRC_ROOT,
                           check=False)
            print(f"[mustafar] git checkout {p}")


def verify() -> None:
    for p in config.PATCH_FILES:
        with open(p) as f:
            s = f.read()
        print(f"{p}: mustafar_markers={s.count(config.MARKER)}")
