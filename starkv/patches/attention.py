"""Backend edits: read-side hooks on the sparse-prefill and decode paths.

The c4 record is no longer the native 584-byte layout, so both paths that read
it have to change: the prefill gather swaps its dequantizer, and decode -- whose
``flash_mla_with_kvcache`` reads the byte record in-kernel by index, leaving no
python hook to intercept -- replaces the attention call itself.
"""

from .. import config
from . import _import_block


def _backend_edits():
    return [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        # Prefill: gather + dequantize into the workspace the sparse kernel reads.
        ("        if compressed_slice is not None:\n"
         "            dequantize_k_cache_paged(\n"
         "                extra_k_cache,\n"
         "                flat_token_ids,\n"
         "                page_size=extra_page_size,\n"
         "                out=compressed_slice,\n"
         "            )\n",
         "        if compressed_slice is not None:\n"
         "            # " + config.MARKER + " (prefill gather)\n"
         "            if (\n"
         "                _sg_lr is not None\n"
         "                and _sg_lr.starkv_enabled()\n"
         "                and compress_ratio == 4\n"
         "            ):\n"
         "                _sg_lr.dequantize_lowrank_k_cache_paged(\n"
         "                    extra_k_cache, flat_token_ids,\n"
         "                    page_size=extra_page_size, layer_id=layer_id,\n"
         "                    out=compressed_slice,\n"
         "                )\n"
         "            else:\n"
         "                dequantize_k_cache_paged(\n"
         "                    extra_k_cache,\n"
         "                    flat_token_ids,\n"
         "                    page_size=extra_page_size,\n"
         "                    out=compressed_slice,\n"
         "                )\n"),
        # Decode. Anchored after the q/index normalization above, so `q` is
        # (b, 1, h, d_qk) and the index/length tensors are the *matched* locals:
        # match_num_queries rebinds those to q's batch dimension without writing
        # back to the metadata, so the metadata fields can disagree with
        # q.shape[0] and must not be passed instead.
        #
        # The capture call repeats every guard the decode branch below needs,
        # for one reason each. compress_ratio == 4: only the CSA layers have the
        # compact c4 record, and a row from any other layer has no c4 slot to
        # join a Top-k entry through. The two `is not None` tests: this site is
        # reached by layers and modes where match_num_queries above leaves those
        # locals as None, and the hook dereferences both. is_decode(): the hook's
        # contract is a decode step, and its per-layer window is one-shot, so
        # recording a prefill batch would spend that layer's only window on rows
        # the measurement cannot use -- silently, since nothing would error.
        ("            assert attn_sink is not None\n\n"
         "            flashmla_metadata = "
         "core_attn_metadata.get_flashmla_metadata(compress_ratio)\n",
         "            assert attn_sink is not None\n\n"
         "            # " + config.MARKER + " (decode dispatch + capture)\n"
         "            if (\n"
         "                _sg_lr is not None\n"
         "                and _sg_lr.capture_enabled()\n"
         "                and compress_ratio == 4\n"
         "                and extra_indices is not None\n"
         "                and extra_topk_lengths is not None\n"
         "                and forward_batch.forward_mode.is_decode()\n"
         "            ):\n"
         "                _sg_lr.capture_decode_attn(\n"
         "                    layer_id, q=q, indices=extra_indices,\n"
         "                    lengths=extra_topk_lengths)\n"
         "            if (\n"
         "                _sg_lr is not None\n"
         "                and _sg_lr.starkv_enabled()\n"
         "                and compress_ratio == 4\n"
         "                and forward_batch.forward_mode.is_decode()\n"
         "            ):\n"
         "                return _sg_lr.decode_lowrank(\n"
         "                    self, q=q, layer_id=layer_id,\n"
         "                    token_to_kv_pool=token_to_kv_pool, attn_sink=attn_sink,\n"
         "                    swa_page_indices=swa_page_indices,\n"
         "                    swa_topk_lengths=swa_topk_lengths,\n"
         "                    c4_page_indices=extra_indices,\n"
         "                    c4_topk_lengths=extra_topk_lengths)\n\n"
         "            flashmla_metadata = "
         "core_attn_metadata.get_flashmla_metadata(compress_ratio)\n"),
        # Decode rebuilds its own workspace from raw c4 indices, so it needs them
        # kept, not just the padded page indices prefill uses.
        ("        self.c4_sparse_page_indices = "
         "_pad_last_dim(self.c4_sparse_page_indices)\n"
         "        if is_prefill:\n"
         "            self.c4_sparse_raw_indices = "
         "torch.empty_like(self.c4_sparse_page_indices)\n",
         "        self.c4_sparse_page_indices = "
         "_pad_last_dim(self.c4_sparse_page_indices)\n"
         "        if is_prefill or (\n"
         "            _sg_lr is not None and _sg_lr.starkv_enabled()\n"
         "        ):\n"
         "            self.c4_sparse_raw_indices = "
         "torch.empty_like(self.c4_sparse_page_indices)\n"),
        # A compact record cannot be reinterpreted as the native fp8 layout the
        # in-kernel reader assumes, so skip the reinterpretation entirely.
        ("            if extra_k_cache is not None:\n"
         "                page_sizes = {\n"
         "                    4: token_to_kv_pool.page_size // 4,\n"
         "                    128: token_to_kv_pool.page_size // 128,\n"
         "                }\n"
         "                extra_k_cache = extra_k_cache[\n"
         "                    :, : page_sizes[compress_ratio] * k_cache_total_dim\n"
         "                ].view(\n"
         "                    extra_k_cache.shape[0],\n"
         "                    page_sizes[compress_ratio],\n"
         "                    1,\n"
         "                    k_cache_total_dim,\n"
         "                )\n",
         "            if extra_k_cache is not None and not (\n"
         "                _sg_lr is not None\n"
         "                and _sg_lr.starkv_enabled()\n"
         "                and compress_ratio == 4\n"
         "            ):\n"
         "                page_sizes = {\n"
         "                    4: token_to_kv_pool.page_size // 4,\n"
         "                    128: token_to_kv_pool.page_size // 128,\n"
         "                }\n"
         "                extra_k_cache = extra_k_cache[\n"
         "                    :, : page_sizes[compress_ratio] * k_cache_total_dim\n"
         "                ].view(\n"
         "                    extra_k_cache.shape[0],\n"
         "                    page_sizes[compress_ratio],\n"
         "                    1,\n"
         "                    k_cache_total_dim,\n"
         "                )\n"),
        # Fixed-capacity decode workspace, allocated once at graph-capture setup
        # so capture never observes a reallocation.
        ("    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:\n",
         "    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:\n"
         "        if _sg_lr is not None:\n"
         "            _sg_lr.init_decode_workspace(\n"
         "                self, max_bs=max_bs, c4_topk=self.c4_topk,\n"
         "                swa_window=self.swa_page_size)\n"),
    ]
