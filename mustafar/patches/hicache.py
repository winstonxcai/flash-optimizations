"""Packed-aware DSV4 c4 host mirror for the hierarchical cache.

Packed c4 KV lives in three non-contiguous per-layer tensors
(packed_values / packed_bitmaps / packed_scales). The stock
DeepSeekV4PagedHostPool whole-page copy treats each c4 page row as one
contiguous item_bytes span and therefore silently corrupts packed data.

This module injects a packed-aware host pool into hybrid_pool_assembler.py and
branches the c4_host_pool construction to use it whenever the c4 pool exposes
get_packed_buffers (the packed ABI). Index semantics are unchanged from native:
host/device indices are FULL-KV token ids (the C4 sidecar reuses the KV
transfer indices), and page row = id // slot_page_size. A c4 page row holds the
compressed latents for one full page (256 tokens -> 64 latents), spread across
the three fragments.

The class stays source text inserted into SGLang, where its base class and
torch/psutil already exist. Do not import SGLang from this module.
"""


def _assembler_edits():
    return [
        (
            "def _deepseek_v4_num_host_pages(\n",
            _host_pool_class() + "\n\ndef _deepseek_v4_num_host_pages(\n",
        ),
        (
            "    if c4_layer_mapping:\n"
            "        c4_device_buffers, c4_item_bytes = _dsv4_compressed_region_buffers(kvcache, 4)\n"
            "        c4_host_pool = DeepSeekV4PagedHostPool(\n"
            "            pool_name=str(PoolName.DEEPSEEK_V4_C4),\n"
            "            device_buffers=c4_device_buffers,\n"
            "            item_bytes=c4_item_bytes,\n"
            "            num_host_pages=num_host_pages,\n"
            "            slot_page_size=page_size,\n"
            "            layout=server_args.hicache_mem_layout,\n"
            "            allocator_type=_get_allocator_type(server_args),\n"
            "        )\n",
            "    if c4_layer_mapping:\n"
            "        c4_device_buffers, c4_item_bytes = _dsv4_compressed_region_buffers(kvcache, 4)\n"
            "        if getattr(kvcache.c4_kv_pool, \"get_packed_buffers\", None) is not None:\n"
            "            ## MUSTAFAR (packed-aware c4 host mirror)\n"
            "            c4_host_pool = MustafarPackedHostPool(\n"
            "                pool_name=str(PoolName.DEEPSEEK_V4_C4),\n"
            "                device_pool=kvcache.c4_kv_pool,\n"
            "                num_host_pages=num_host_pages,\n"
            "                slot_page_size=page_size,\n"
            "                layout=server_args.hicache_mem_layout,\n"
            "                allocator_type=_get_allocator_type(server_args),\n"
            "            )\n"
            "        else:\n"
            "            c4_host_pool = DeepSeekV4PagedHostPool(\n"
            "                pool_name=str(PoolName.DEEPSEEK_V4_C4),\n"
            "                device_buffers=c4_device_buffers,\n"
            "                item_bytes=c4_item_bytes,\n"
            "                num_host_pages=num_host_pages,\n"
            "                slot_page_size=page_size,\n"
            "                layout=server_args.hicache_mem_layout,\n"
            "                allocator_type=_get_allocator_type(server_args),\n"
            "            )\n",
        ),
    ]


def _host_pool_class() -> str:
    return r'''

## MUSTAFAR (packed-aware c4 host mirror)
import threading as _mustafar_threading  # noqa: E402
import psutil as _mustafar_psutil  # noqa: E402
import torch as _mustafar_torch  # noqa: E402


class MustafarPackedHostPool(DeepSeekV4PagedHostPool):
    """Host mirror for a packed DSV4 c4 pool.

    Backs up / restores the three packed fragments (values, bitmaps, scales)
    page-row by page-row instead of copying one contiguous item_bytes span.
    Host/device indices are FULL-KV token ids (the C4 sidecar transfer reuses
    the KV anchor indices); page row = id // slot_page_size, exactly as in the
    native pool. Host storage mirrors the device ABI so backup and load are
    dtype-faithful .copy_()s with no repacking.
    """

    def __init__(
        self,
        *,
        pool_name: str,
        device_pool: Any,
        num_host_pages: int,
        slot_page_size: int,
        layout: str = "layer_first",
        allocator_type: str = "default",
    ):
        self.pool_name = pool_name
        self.device_pool = device_pool
        self.layer_num = device_pool.layer_num
        self.num_host_pages = num_host_pages
        self.slot_page_size = slot_page_size
        self.dtype = _mustafar_torch.uint8
        self.device = "cpu"
        self.pin_memory = True
        self.page_size = slot_page_size
        self.size = num_host_pages * slot_page_size
        self.layout = layout
        self.start_layer = 0
        self.end_layer = self.layer_num
        self.lock = _mustafar_threading.RLock()
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self.gpu_device = device_pool.packed_values[0].device

        # Host fragments mirror each packed device fragment, one host page row
        # per slot_page_size token group (one full page -> one c4 page row).
        self.host_values = []
        self.host_bitmaps = []
        self.host_scales = []
        requested_bytes = 0
        for layer in range(self.layer_num):
            vals, bms, scs = device_pool.get_packed_buffers(layer)
            # vals/bms/scs are [device_pages, page_size, cols] on GPU.
            hv = _mustafar_torch.zeros(
                (num_host_pages,) + tuple(vals.shape[1:]),
                dtype=_mustafar_torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            hb = _mustafar_torch.zeros(
                (num_host_pages,) + tuple(bms.shape[1:]),
                dtype=_mustafar_torch.uint64,
                device="cpu",
                pin_memory=True,
            )
            hs = _mustafar_torch.zeros(
                (num_host_pages,) + tuple(scs.shape[1:]),
                dtype=_mustafar_torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            self.host_values.append(hv)
            self.host_bitmaps.append(hb)
            self.host_scales.append(hs)
            requested_bytes += hv.nbytes + hb.nbytes + hs.nbytes
        self.item_bytes = sum(
            t[0].nbytes
            for t in (self.host_values[0], self.host_bitmaps[0], self.host_scales[0])
        )
        self.size_per_token = self.item_bytes
        host_mem = _mustafar_psutil.virtual_memory()
        if requested_bytes > host_mem.available:
            raise ValueError(
                f"Not enough host memory for packed V4 paged pool {pool_name}. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{host_mem.available / 1e9:.2f} GB free."
            )
        logger.info(
            "Allocating %.2f GB host memory for packed V4 paged pool '%s' "
            "(layers=%d, host_pages=%d, slot_page_size=%d).",
            requested_bytes / 1e9,
            self.pool_name,
            self.layer_num,
            num_host_pages,
            slot_page_size,
        )
        self.clear()

    # ---- whole-page fragment transfers -----------------------------------
    # Both directions receive full-KV token ids. Whole-page transfers (index
    # counts divisible by slot_page_size) are guaranteed by the unified radix
    # path (page_aligned keys). Packed rows are not one contiguous byte span,
    # so the token-granular fallback has no packed equivalent and is refused.

    def _aligned_rows(self, host_indices, device_indices):
        if not self._has_transfer_indices(host_indices, device_indices):
            return None
        if (
            host_indices.numel() % self.slot_page_size != 0
            or device_indices.numel() % self.slot_page_size != 0
        ):
            raise RuntimeError(
                f"{self.pool_name}: packed host mirror supports whole-page "
                f"transfers only (unified radix path); got "
                f"{host_indices.numel()} host / {device_indices.numel()} device "
                f"indices, slot_page_size={self.slot_page_size}"
            )
        return (
            self._to_page_indices(host_indices),
            self._to_page_indices(device_indices),
        )

    def _check_rows(self, rows, pages: int) -> None:
        if rows.numel() and rows.max() >= pages:
            raise RuntimeError(
                f"{self.pool_name}: packed device page {int(rows.max())} >= "
                f"pool capacity {pages}; c4 host mirror cannot serve this row"
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        rows = self._aligned_rows(host_indices, device_indices)
        if rows is None:
            return
        host_rows, device_rows = rows
        for layer in range(self.layer_num):
            vals, bms, scs = device_pool.get_packed_buffers(layer)
            self._check_rows(device_rows, vals.shape[0])
            self.host_values[layer][host_rows].copy_(
                vals[device_rows], non_blocking=True
            )
            # uint64 advanced indexing has no CUDA kernel; gather through a
            # byte view (host side is CPU, both sides are dtype-clean u8).
            self.host_bitmaps[layer].view(_mustafar_torch.uint8)[host_rows].copy_(
                bms.view(_mustafar_torch.uint8)[device_rows], non_blocking=True
            )
            self.host_scales[layer][host_rows].copy_(
                scs[device_rows], non_blocking=True
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        rows = self._aligned_rows(host_indices, device_indices)
        if rows is None:
            return
        host_rows, device_rows = rows
        vals, bms, scs = device_pool.get_packed_buffers(layer_id)
        self._check_rows(device_rows, vals.shape[0])
        vals[device_rows].copy_(
            self.host_values[layer_id][host_rows], non_blocking=True
        )
        bms.view(_mustafar_torch.uint8)[device_rows].copy_(
            self.host_bitmaps[layer_id].view(_mustafar_torch.uint8)[host_rows],
            non_blocking=True,
        )
        scs[device_rows].copy_(
            self.host_scales[layer_id][host_rows], non_blocking=True
        )

    # ---- the packed host layout is fragment mirrors, not the flat layout ----
    # Nothing on the sidecar path reads these (HostPoolGroup and the controller
    # only touch the anchor pool's data methods), but refuse loudly rather than
    # let generic code misinterpret the non-native layout.

    def get_data_page(self, index, flat: bool = True):
        raise NotImplementedError(
            f"{self.pool_name}: packed host mirror stores fragments, not a flat "
            "page; get_data_page is unsupported"
        )

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError(
            f"{self.pool_name}: packed host mirror has no flat data page"
        )

    def get_page_buffer_meta(self, indices):
        raise NotImplementedError(
            f"{self.pool_name}: packed host mirror has no flat page-buffer meta"
        )

    def get_dummy_flat_data_page(self):
        raise NotImplementedError(
            f"{self.pool_name}: packed host mirror has no dummy flat page"
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            f"{self.pool_name}: packed host mirror has no contiguous row buffers"
        )
'''
