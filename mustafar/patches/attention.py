"""Raw-index plumbing and attention reconstruction call-site edits.

Indexer scoring, selection policy, and attention kernels are unchanged.
"""

from pathlib import Path

from .. import config
from . import _import_block


def _indexer_edits():
    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + _import_block(),
        ),
        (
            "        if capture_enabled:\n"
            "            raw_indices = torch.empty_like(c4_sparse_page_indices)\n",
            "        if capture_enabled and not _sg_lr.packed_enabled():\n"
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


def _backend_edits(source: str | None = None):
    prefill_anchor = (
        "            if forward_batch.forward_mode.is_extend_without_speculative() and (\n"
        "                q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD\n"
        "                or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()\n"
        "            ):\n"
    )
    prefill_replacement = (
        "            if forward_batch.forward_mode.is_extend_without_speculative() and (\n"
        "                q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD\n"
        "                or (\n"
        "                    _sg_lr.packed_enabled()\n"
        "                    and self.mustafar_workspace is not None\n"
        "                    and q.shape[0] > self.mustafar_workspace.max_queries\n"
        "                )\n"
        "                or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()\n"
        "            ):\n"
    )
    # v0.5.17 added an SM120 guard to this condition. Keep the patch usable
    # against both the previously validated preview tree and the 0731 tree.
    current_backend = (
        Path(config.DSV4_BACKEND).read_text() if source is None else source
    )
    v17_prefill_anchor = (
        "            if (\n"
        "                forward_batch.forward_mode.is_extend_without_speculative()\n"
        "                and not _is_sm120\n"
        "                and (\n"
        "                    q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD\n"
        "                    or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()\n"
        "                )\n"
        "            ):\n"
    )
    if prefill_anchor not in current_backend:
        if v17_prefill_anchor not in current_backend:
            raise AssertionError(
                "[mustafar] unsupported DSV4 sparse-prefill anchor; "
                "inspect the pinned SGLang source before patching"
            )
        prefill_anchor = v17_prefill_anchor
        prefill_replacement = (
            "            if (\n"
            "                forward_batch.forward_mode.is_extend_without_speculative()\n"
            "                and not _is_sm120\n"
            "                and (\n"
            "                    q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD\n"
            "                    or (\n"
            "                        _sg_lr.packed_enabled()\n"
            "                        and self.mustafar_workspace is not None\n"
            "                        and q.shape[0] > self.mustafar_workspace.max_queries\n"
            "                    )\n"
            "                    or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()\n"
            "                )\n"
            "            ):\n"
        )

    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\n" + _import_block(),
        ),
        (
            "        if is_prefill:\n"
            "            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)\n",
            "        if is_prefill or _sg_lr.packed_enabled():\n"
            "            ## MUSTAFAR (retain existing v2 top-k raw output)\n"
            "            self.c4_sparse_raw_indices = torch.empty_like(\n"
            "                self.c4_sparse_page_indices\n"
            "            )\n",
        ),
        (
            "        self.sparse_prefill_workspace = SparsePrefillWorkspace(self.device)\n",
            "        self.sparse_prefill_workspace = SparsePrefillWorkspace(self.device)\n"
            "        ## MUSTAFAR (static native gather workspace)\n"
            "        self.mustafar_workspace = None\n"
            "        if _sg_lr.packed_enabled():\n"
            "            _sg_lr.validate_packed_static_config()\n"
            "            args = model_runner.server_args\n"
            "            if self.device.type != 'cuda':\n"
            "                raise RuntimeError('packed requires CUDA')\n"
            "            if self.c4_topk != 512:\n"
            "                raise RuntimeError('packed requires index_topk=512')\n"
            "            if self.token_to_kv_pool._unified_kv:\n"
            "                raise RuntimeError('packed is incompatible with unified KV')\n"
            "            if self.hisparse_coordinator is not None or args.enable_hisparse:\n"
            "                raise RuntimeError('packed is incompatible with HiSparse')\n"
            "            if args.speculative_algorithm is not None or self.mtp_enabled:\n"
            "                raise RuntimeError('packed is incompatible with speculative decode')\n"
            "            if args.disaggregation_mode != 'null':\n"
            "                raise RuntimeError('packed is incompatible with disaggregation')\n"
            "            if (\n"
            "                args.cpu_offload_gb > 0\n"
            "                or args.enable_hierarchical_cache\n"
            "                or args.disaggregation_decode_enable_offload_kvcache\n"
            "            ):\n"
            "                raise RuntimeError('packed is incompatible with offload')\n"
            "            if (\n"
            "                args.enable_prefill_context_parallel\n"
            "                or args.enable_dsa_prefill_context_parallel\n"
            "                or get_parallel().attn_cp_size != 1\n"
            "            ):\n"
            "                raise RuntimeError('packed is incompatible with context parallelism')\n"
            "            self.mustafar_workspace = _sg_lr.NativeWorkspace.allocate(\n"
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
            "                if _sg_lr.packed_enabled():\n"
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
            "            if compress_ratio == 4 and _sg_lr.packed_enabled():\n"
            "                raw_indices = match_num_queries(raw_indices, value=-1)\n",
        ),
        (
            prefill_anchor,
            prefill_replacement,
        ),
        (
            "            if _is_sm120:\n",
            "            if compress_ratio == 4 and _sg_lr.packed_enabled():\n"
            "                ## MUSTAFAR (decode/small-extend native reconstruction)\n"
            "                assert self.mustafar_workspace is not None\n"
            "                packed_indices = (\n"
            "                    extra_indices.squeeze(1)\n"
            "                    if extra_indices.ndim == 3 else extra_indices\n"
            "                )\n"
            "                packed_raw_indices = (\n"
            "                    raw_indices.squeeze(1)\n"
            "                    if raw_indices.ndim == 3 else raw_indices\n"
            "                )\n"
            "                extra_k_cache, extra_indices = (\n"
            "                    _sg_lr.unpack_gather_native(\n"
            "                        token_to_kv_pool.get_packed_buffers(layer_id),\n"
            "                        packed_indices, packed_raw_indices,\n"
            "                        extra_topk_lengths,\n"
            "                        token_to_kv_pool.get_packed_freqs(layer_id),\n"
            "                        self.mustafar_workspace,\n"
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
            "                compress_ratio == 4 and _sg_lr.packed_enabled()\n"
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
            "            if compress_ratio == 4 and _sg_lr.packed_enabled():\n"
            "                _max = flat_token_ids.numel() // cache.num_reqs\n"
            "                if not hasattr(cache, '_mustafar_raw_indices'):\n"
            "                    cache._mustafar_raw_indices = (\n"
            "                        torch.arange(\n"
            "                            _max, dtype=torch.int32, device=q.device\n"
            "                        )[None, :].expand(cache.num_reqs, -1).contiguous()\n"
            "                    )\n"
            "                    cache._mustafar_lengths = torch.clamp(\n"
            "                        cache.seq_lens // 4, min=0, max=_max\n"
            "                    ).to(torch.int32)\n"
            "                _sg_lr.unpack_gather_bf16(\n"
            "                    token_to_kv_pool.get_packed_buffers(layer_id),\n"
            "                    flat_token_ids.view(cache.num_reqs, _max),\n"
            "                    cache._mustafar_raw_indices,\n"
            "                    cache._mustafar_lengths,\n"
            "                    token_to_kv_pool.get_packed_freqs(layer_id),\n"
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
