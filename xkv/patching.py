"""Apply, restore, verify, and census xKV patches in the SGLang source tree.

Mirrors ``mustafar.patching``: edits are declared as ``(anchor, replacement)``
lists so the same declarations can be applied, reverted, or censused read-only
against a different SGLang revision (``drift``). Anchors are validated against
the file's pristine base -- its ``.xkv.orig`` backup when the tree is already
patched, else the file as it is on disk.
"""

import os
import shutil
import tempfile
from pathlib import Path

from . import config


def _import_block():
    """Splice `import xkv as _sg_lr` into a patched module.

    PACKAGE_ROOT is resolved at patch time, so a tree patched from a mounted
    repo imports the live package rather than a baked copy.
    """
    return (
        "\n" + config.MARKER + " (import)\n"
        "import sys as _sg_lr_sys\n"
        f"if {config.PACKAGE_ROOT!r} not in _sg_lr_sys.path:\n"
        f"    _sg_lr_sys.path.insert(0, {config.PACKAGE_ROOT!r})\n"
        "try:\n    import xkv as _sg_lr\nexcept Exception:\n    _sg_lr = None\n"
    )


def _pool_class():
    """The 200 B/token coefficient pool that replaces DeepSeekV4SingleKVPool."""
    return '''class DeepSeekV4LowRankPool(KVCache):
    coeff_buffer_dtype = torch.uint8
    def __init__(self, size, page_size, dtype, qk_nope_head_dim, qk_rope_head_dim, layer_num, device, enable_memory_saver, start_layer=None, end_layer=None):
        super().__init__(size, page_size, dtype, layer_num, device, enable_memory_saver, start_layer, end_layer)
        self._create_buffer()
    def get_bytes_per_token(self):
        return _sg_lr.BYTES_PER_TOKEN
    def _create_buffer(self):
        page_bytes = self.page_size * self.get_bytes_per_token()
        self.coeff_buffer = [torch.zeros((self.size + self.page_size + 1) // self.page_size, page_bytes, dtype=torch.uint8, device=self.device) for _ in range(self.layer_num)]
    def get_key_buffer(self, layer_id): return self.coeff_buffer[layer_id]
    def get_kv_buffer(self, *args, **kwargs): raise NotImplementedError()
    def get_value_buffer(self, *args, **kwargs): raise NotImplementedError()
    def set_kv_buffer(self, *args, **kwargs): raise NotImplementedError()
'''


def _compressor_edits():
    """Store-side hook: replace the write with a low-rank record."""
    return [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        ("        if forward_batch.forward_mode.is_idle():\n            return\n",
         "        if forward_batch.forward_mode.is_idle():\n            return\n"
         "        if _sg_lr is not None:\n            _sg_lr.set_cur_layer(layer_id)\n"),
        ("        # Step 2: norm + rope + store\n        compress_norm_rope_store(\n",
         "        if _sg_lr is not None and _sg_lr.store_compressed_lowrank(\n"
         "            kv_compressed, plan, norm, compress_ratio, is_indexer, kv_cache, page_size, out_loc, freqs_cis_cache):\n"
         "            return\n"
         "        # Step 2: norm + rope + store\n        compress_norm_rope_store(\n"),
    ]


def _memory_pool_edits():
    """Swap the C4 pool for the coefficient pool when the store is enabled."""
    return [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        ("class DeepSeekV4IndexerPool(KVCache):\n",
         _pool_class() + "\n\nclass DeepSeekV4IndexerPool(KVCache):\n"),
        ("            c4_kv_pool_type = DeepSeekV4SingleKVPool\n",
         "            c4_kv_pool_type = DeepSeekV4SingleKVPool\n"
         "            if _sg_lr is not None and _sg_lr.lowrank_enabled():\n"
         "                c4_kv_pool_type = DeepSeekV4LowRankPool\n"),
        ("        buf_groups = [\n            self.c4_kv_pool.kv_buffer,\n",
         "        _c4_buffers = (self.c4_kv_pool.coeff_buffer if _sg_lr is not None and _sg_lr.lowrank_enabled() else self.c4_kv_pool.kv_buffer)\n"
         "        buf_groups = [\n            _c4_buffers,\n"),
    ]


def _pool_config_edits():
    """Bill the pool at 200 B/token instead of the 584-byte C4 record."""
    return [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        ("            + c4_frac * kv_bytes * self.num_layers_ca4\n",
         "            + c4_frac * (kv_bytes if _sg_lr is None or not _sg_lr.lowrank_enabled() else _sg_lr.BYTES_PER_TOKEN) * self.num_layers_ca4\n"),
    ]


def _backend_edits():
    """Read-side hooks: reconstruct on both the sparse-prefill and decode paths."""
    return [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        # Prefill: the gather+dequantize swap.
        ("        if compressed_slice is not None:\n            dequantize_k_cache_paged(\n                extra_k_cache,\n                flat_token_ids,\n                page_size=extra_page_size,\n                out=compressed_slice,\n            )\n",
         "        if compressed_slice is not None:\n"
         "            if _sg_lr is not None and _sg_lr.lowrank_enabled() and compress_ratio == 4:\n"
         "                _sg_lr.dequantize_lowrank_k_cache_paged(extra_k_cache, flat_token_ids, page_size=extra_page_size, layer_id=layer_id, out=compressed_slice)\n"
         "            else:\n"
         "                dequantize_k_cache_paged(extra_k_cache, flat_token_ids, page_size=extra_page_size, out=compressed_slice)\n"),
        # Decode: replace the whole sparse-attention call. flash_mla_with_kvcache
        # reads the packed 584-byte C4 record in-kernel by index, so a 200-byte
        # record has no python-side hook to intercept -- the call itself has to go.
        # Anchored after the q/index normalization above, so `q` is (b, 1, h, d_qk)
        # and the indices are the *matched* locals -- `match_num_queries` rebinds
        # those without writing back to the metadata, so passing the metadata
        # fields instead would risk a first dimension that disagrees with q.
        ("            assert attn_sink is not None\n\n            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)\n",
         "            assert attn_sink is not None\n\n"
         "            if (_sg_lr is not None and _sg_lr.lowrank_enabled() and compress_ratio == 4\n"
         "                    and forward_batch.forward_mode.is_decode()):\n"
         "                return _sg_lr.decode_lowrank(\n"
         "                    self, q=q, layer_id=layer_id,\n"
         "                    token_to_kv_pool=token_to_kv_pool, attn_sink=attn_sink,\n"
         "                    swa_page_indices=swa_page_indices,\n"
         "                    swa_topk_lengths=swa_topk_lengths,\n"
         "                    c4_page_indices=extra_indices,\n"
         "                    c4_topk_lengths=extra_topk_lengths)\n\n"
         "            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)\n"),
        # Decode keeps raw SWA indices so the low-rank path can rebuild its own
        # workspace; prefill needs them for the sparse-prefill gather.
        ("        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)\n        if is_prefill:\n            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)\n",
         "        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)\n"
         "        if is_prefill or (_sg_lr is not None and _sg_lr.lowrank_enabled()):\n"
         "            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)\n"),
        # A 200-byte record cannot be reinterpreted as the fp8 record layout the
        # native view assumes.
        ("            if extra_k_cache is not None:\n                page_sizes = {\n                    4: token_to_kv_pool.page_size // 4,\n                    128: token_to_kv_pool.page_size // 128,\n                }\n                extra_k_cache = extra_k_cache[\n                    :, : page_sizes[compress_ratio] * k_cache_total_dim\n                ].view(\n                    extra_k_cache.shape[0],\n                    page_sizes[compress_ratio],\n                    1,\n                    k_cache_total_dim,\n                )\n",
         "            if extra_k_cache is not None and not (_sg_lr is not None and _sg_lr.lowrank_enabled() and compress_ratio == 4):\n"
         "                page_sizes = {\n                    4: token_to_kv_pool.page_size // 4,\n                    128: token_to_kv_pool.page_size // 128,\n                }\n"
         "                extra_k_cache = extra_k_cache[:, : page_sizes[compress_ratio] * k_cache_total_dim].view(extra_k_cache.shape[0], page_sizes[compress_ratio], 1, k_cache_total_dim)\n"),
        # Fixed-capacity decode workspace, allocated once at graph-capture setup
        # so capture never observes a reallocation.
        ("    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:\n",
         "    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:\n"
         "        if _sg_lr is not None:\n"
         "            _sg_lr.init_decode_workspace(self, max_bs=max_bs, c4_topk=self.c4_topk, swa_window=self.swa_page_size)\n"),
    ]


# Ordered targets for patch/unpatch/verify/drift.
_TARGETS = (
    (config.COMPRESSOR_V2, _compressor_edits),
    (config.MEM_POOL, _memory_pool_edits),
    (config.POOL_CFG, _pool_config_edits),
    (config.DSV4_BACKEND, _backend_edits),
)


def _render(path: Path, source: str, edits) -> str:
    """Validate every anchor without touching the source or its backup."""
    for anchor, new in edits:
        if source.count(anchor) != 1:
            raise AssertionError(f"[xkv] anchor count != 1 in {path}: {anchor[:70]!r}")
        source = source.replace(anchor, new, 1)
    compile(source, str(path), "exec")
    return source


def _plan(*, restoring: bool = False):
    """Rebuild expected patches from originals, never from already-patched text.

    Restoring is driven by the backup alone, not by re-rendering the current
    declarations: after an anchor is re-based, the text on disk is a *previous*
    patch that today's factory can no longer produce, and refusing to restore it
    would strand the tree. The backup is the authority for what the original was;
    the only thing checked is that the tree is one we patched (marker + backup),
    so a hand-edit is still never silently discarded.
    """
    plan = []
    for filename, factory in _TARGETS:
        path = Path(filename)
        current = path.read_text()
        backup = Path(filename + ".xkv.orig")
        if not backup.exists():
            if restoring:
                if config.MARKER in current:
                    raise RuntimeError(f"[xkv] missing original backup: {path}")
                continue  # Nothing owned by xKV to restore; never use git checkout.
        original = backup.read_text() if backup.exists() else current
        if config.MARKER in original:
            raise RuntimeError(f"[xkv] backup is not an unpatched original: {backup}")
        if restoring:
            plan.append((path, current, original, original))
            continue
        expected = _render(path, original, factory())
        if current not in (original, expected):
            raise RuntimeError(
                f"[xkv] unexpected edits or incompatible patch in {path}; "
                "preserve/reconcile them before patching or restoring"
            )
        plan.append((path, current, original, expected))
    return plan


def _atomic_write(path: Path, content: str) -> None:
    """Replace a source file without exposing a partially written Python module."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _commit(plan, *, restoring: bool = False) -> None:
    changed = []
    created_backups = []
    try:
        for path, current, original, expected in plan:
            if path.read_text() != current:
                raise RuntimeError(f"[xkv] source changed during patch operation: {path}")
            target = original if restoring else expected
            if current == target:
                continue
            backup = Path(str(path) + ".xkv.orig")
            if not restoring and not backup.exists():
                created_backups.append(backup)
                shutil.copy2(path, backup)
            _atomic_write(path, target)
            changed.append((path, current))
    except BaseException:
        # If rollback itself fails, keep backups for manual recovery.
        for path, current in reversed(changed):
            _atomic_write(path, current)
        for backup in created_backups:
            backup.unlink(missing_ok=True)
        raise
    for path, *_ in plan:
        print(f"[xkv] {path}: {'restored' if restoring else 'patched/verified'}")


def patch() -> None:
    """Prevalidate all targets, apply idempotently, and roll back on write errors."""
    os.makedirs(config.ctrl_dir(), exist_ok=True)
    _commit(_plan())


def unpatch() -> None:
    """Restore verified backups only; refuse to overwrite unexpected user edits."""
    _commit(_plan(restoring=True), restoring=True)


def verify() -> None:
    """Require the complete expected patch and its original backup in every file."""
    plan = _plan()
    for path, current, _, expected in plan:
        if current != expected:
            raise RuntimeError(f"[xkv] missing or incomplete patch: {path}")
    for path, current, *_ in plan:
        print(f"{path}: xkv_markers={current.count(config.MARKER)}")


def drift() -> int:
    """Read-only anchor census against the tree at SRC_ROOT (e.g. a v0.5.18 clone).

    Checks EVERY target file instead of raising on the first mismatch. For each
    file it reports every anchor whose count in the pristine source is != 1
    (missing=0, duplicated>1, or context changed) and the anchor count, printing
    a file -> anchor -> count table. Never writes. Returns the number of files
    that drifted (missing target or any broken anchor); 0 means every anchor of
    every edit still matches this tree one-to-one.
    """
    drifted = 0
    for filename, factory in _TARGETS:
        path = Path(filename)
        status = "patched" if Path(str(path) + ".xkv.orig").exists() else "pristine"
        if not path.exists():
            print(f"[drift] MISSING  {path}  (target dropped by this tree?)")
            drifted += 1
            continue
        source = path.read_text()
        backup = Path(str(path) + ".xkv.orig")
        base = backup.read_text() if backup.exists() else source
        edits = factory()
        broken = [(a, base.count(a)) for a, _ in edits if base.count(a) != 1]
        if broken:
            drifted += 1
            print(f"[drift] DRIFT {len(broken)}/{len(edits)}  {path}  ({status})")
            for anchor, count in broken:
                line = next((l.strip() for l in anchor.splitlines() if l.strip()), "")
                print(f"        count={count:<2}  {line[:78]!r}")
        else:
            print(f"[drift] OK    {len(edits)}/{len(edits)}  {path}  ({status})")
    if drifted:
        print(f"[drift] {drifted} file(s) drifted -- anchors need re-basing "
              f"(see results/xkv/drift-v0.5.18.md)")
    else:
        print("[drift] all anchors intact against this tree")
    return drifted
