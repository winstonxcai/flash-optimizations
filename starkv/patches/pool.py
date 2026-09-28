"""Pool edits: replace the c4 pool with the compact low-rank pool.

Only the c4 paged pool changes. The indexer, SWA and c128 pools keep their
native classes and their native records.
"""

from .. import config
from . import _import_block


def _pool_class():
    """The c4 pool that holds ``[z][scales][tail]`` records instead of 584 B
    native ones. Same page/num-page geometry as ``DeepSeekV4SingleKVPool``,
    different ``bytes_per_token``, so every index the model already computes
    (``token // compress_ratio``) lands on the same slot."""
    return '''class DeepSeekV4StarkvPool(KVCache):
    """STAR-CSA c4 pool: page_size * BYTES_PER_TOKEN uint8 bytes per page."""

    def __init__(self, size, page_size, dtype, qk_nope_head_dim, qk_rope_head_dim,
                 layer_num, device, enable_memory_saver, start_layer=None,
                 end_layer=None):
        super().__init__(size, page_size, dtype, layer_num, device,
                         enable_memory_saver, start_layer, end_layer)
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self._create_buffers()

    def get_bytes_per_token(self):
        """Read live, so flipping STARKV_RANK in a fresh server is enough --
        a rank change always implies a fresh allocation anyway."""
        return _sg_lr.BYTES_PER_TOKEN

    def _create_buffers(self):
        num_pages = (self.size + self.page_size + 1) // self.page_size
        page_bytes = self.page_size * self.get_bytes_per_token()
        # One flat [num_pages, page_bytes] uint8 buffer per layer, exactly as the
        # native pool: no per-token padding, so the record stride is uniform.
        self.kv_buffer = [
            torch.zeros(num_pages, page_bytes, dtype=torch.uint8,
                        device=self.device)
            for _ in range(self.layer_num)
        ]

    def get_key_buffer(self, layer_id):
        return self.kv_buffer[layer_id - (self.start_layer or 0)]

    def get_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError()

    def get_value_buffer(self, *args, **kwargs):
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError()
'''


def _memory_pool_edits():
    return [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        ("class DeepSeekV4IndexerPool(KVCache):\n",
         _pool_class() + "\n\nclass DeepSeekV4IndexerPool(KVCache):\n"),
        ("            c4_kv_pool_type = DeepSeekV4SingleKVPool\n",
         "            c4_kv_pool_type = DeepSeekV4SingleKVPool\n"
         "            " + config.MARKER + " (c4 pool swap)\n"
         "            if _sg_lr is not None and _sg_lr.starkv_enabled():\n"
         "                c4_kv_pool_type = DeepSeekV4StarkvPool\n"),
        # `buf_groups` needs no edit: the compact pool exposes `kv_buffer` under
        # the same name as the native one, so the allocator picks up the smaller
        # pages by itself.
    ]


def _pool_config_edits():
    """Bill the c4 term at the compact record size.

    `kv_bytes` is the native 584-byte record (448 + 64*2 + 8); substituting
    BYTES_PER_TOKEN keeps the same token-fraction model with the smaller
    record, so the reported capacity matches what the pool allocates.
    """
    return [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        ("            + c4_frac * kv_bytes * self.num_layers_ca4\n",
         "            # " + config.MARKER + " (compact c4 record)\n"
         "            + c4_frac * (\n"
         "                kv_bytes\n"
         "                if _sg_lr is None or not _sg_lr.starkv_enabled()\n"
         "                else _sg_lr.BYTES_PER_TOKEN\n"
         "            ) * self.num_layers_ca4\n"),
    ]
