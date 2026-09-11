"""Speed suite: native vs packed/Triton vs packed/fused on identical workloads.

Three legs, because a two-legged native-vs-packed table cannot distinguish
"packing costs us" from "the Triton reconstruct costs us" -- and the whole point
of the fused extension is to close the gap against native.

====================  ===================  =================  =================
op (call site)        native               packed / triton    packed / fused
====================  ===================  =================  =================
store                 ``compress_norm_``   ``pack_rows``      n/a -- Triton-
                      ``rope_store``                          only store
decode->native        no op (native hands  ``unpack_gather_`` same kernel,
layout                attention the        ``native``         fused dispatch
                      buffer)
decode->dense bf16    ``dequantize_k_``    ``unpack_gather_`` n/a -- no fused
                      ``cache_paged``      ``bf16``           variant
attention c4 leg      ``unpack_gather_``   n/a -- the direct  ``c4_leg``,
                      ``bf16`` +           read replaces the  reading the
                      ``flash_mla_``       reassembly itself  328-byte rows
                      ``sparse_fwd``
====================  ===================  =================  =================

Row 2's native leg has **no operator to time**: native passes the raw 584-byte
buffer straight to the attention kernel. It is reported as an explicit ``0.0``
baseline and the packed legs as *added reconstruct overhead*, not filled with a
fabricated number.

Row 4 is the one that justifies the sparse MLA kernel: identical inputs, both
legs reading the same 328 cache bytes per row, differing only in whether the
records are reassembled into dense BF16 before attention. Its ``native`` and
``fused`` columns therefore mean reassembled and direct -- the "fused" column
name is the table's slot for the packed-side optimization, not a reference to
``_FUSED``. ``fused_over_native`` is the headline ratio.

Every row is timed in eager and under CUDA-graph replay, on the same
:data:`mustafar.tests.harness.WORKLOADS` grid the validity suite uses. A leg
whose capture fails is re-timed eagerly and the reason is recorded in that
record's ``graph_fallbacks``.

Direct run prints per-workload JSON::

    python3 -m mustafar.tests.speed
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from .. import config
from . import harness

RESULTS_DIR = os.environ.get("MUSTAFAR_RESULTS_DIR")

# (name, native cache bytes moved per row, packed cache bytes moved per row,
# direction). These are the *cache* bytes -- 584 vs 328 is the whole capacity
# story -- not the scratch both legs write. A zero native figure is a genuine
# "no operator" and is rendered as such, never estimated.
OPS = (
    ("store", config.NATIVE_RECORD_BYTES, config.PACKED_RECORD_BYTES, "write"),
    ("decode_native_layout", 0, config.PACKED_RECORD_BYTES, "read"),
    ("decode_dense_bf16", config.NATIVE_RECORD_BYTES, config.PACKED_RECORD_BYTES, "read"),
    ("attention_c4_leg", config.PACKED_RECORD_BYTES, config.PACKED_RECORD_BYTES, "read"),
)


def _fused_available() -> bool:
    from ..fused import fused_available

    return bool(fused_available())


def _sparse_available() -> bool:
    from ..sparse import sparse_available

    return bool(sparse_available())


def _regime(fn, regime: str, *, warmup: int, repeats: int) -> dict[str, float]:
    if regime == "eager":
        return harness.timed(fn, warmup=warmup, repeats=repeats)
    return harness.timed(harness.captured(fn), warmup=warmup, repeats=repeats)


def _regime_or_eager(fn, regime: str, *, warmup: int, repeats: int):
    """``_regime`` plus a reason string when a graph capture had to be abandoned.

    ``flash_mla_sparse_fwd`` is a third-party op whose graph safety is not ours,
    and a capture failure would otherwise sink the whole suite. The fallback
    returns eager numbers, which the caller must record -- an eager figure
    labelled "graph" would be worse than no row at all.
    """
    try:
        return _regime(fn, regime, warmup=warmup, repeats=repeats), None
    except RuntimeError as error:
        if regime != "graph":
            raise
        return harness.timed(fn, warmup=warmup, repeats=repeats), str(error).splitlines()[0]


def _gbps(bytes_per_row: int, rows: int, p50_us: float | None) -> float | None:
    """Effective cache bandwidth, or None where the leg has no operator."""
    if not bytes_per_row or p50_us is None or p50_us <= 0:
        return None
    return bytes_per_row * rows / (p50_us * 1.0e-6) / 1.0e9


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Ratio, or None when either leg is absent (e.g. no fused variant)."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


@patch.dict(
    os.environ,
    SGLANG_OPT_TOPMAG="1",
    KEEP="0.5",
    SGLANG_OPT_TOPMAG_PACKED="1",
    SGLANG_OPT_TOPMAG_FUSED="0",
)
def run_speed() -> dict[str, object]:
    """Time native, packed/Triton, and packed/fused over every workload."""
    if not torch.cuda.is_available():
        raise RuntimeError("speed requires CUDA")
    from sglang.kernels.ops.attention.dsv4.compress import compress_norm_rope_store

    from ..packed import pack_rows, unpack_gather_bf16
    from .. import sparse

    device = torch.device("cuda")
    fused = _fused_available()
    sparse = _sparse_available()
    if not sparse:
        print("[speed] sparse MLA extension absent; attention_c4_leg row skipped")
    warmup, repeats = 10, 50
    results: list[dict[str, object]] = []

    for workload in harness.WORKLOADS:
        case = harness.build_case(workload, device)
        gathered = workload.gather_rows
        rows_per_leg = {
            "store": case.rows,
            "decode_native_layout": gathered,
            "decode_dense_bf16": gathered,
            "attention_c4_leg": gathered,
        }

        buffers = harness.packed_buffers(case)
        native_cache, locations = harness.native_store(case, case.masked_latent)
        native_out = torch.empty(
            gathered, 1, config.HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        native_locations = locations[:gathered]
        triton_workspace = harness.native_workspace(case, with_dense=True)
        fused_workspace = harness.native_workspace(case, with_dense=False)

        def native_store_fn() -> None:
            compress_norm_rope_store(
                case.latent,
                case.plan,
                norm_weight=case.weight,
                norm_eps=1.0e-6,
                freq_cis=case.freqs,
                out_loc=case.locations,
                kvcache=native_cache,
                page_size=harness.PAGE_SIZE,
            )

        def native_decode_fn() -> None:
            _dequant(native_cache, native_locations, harness.PAGE_SIZE, native_out)

        def packed_store_fn() -> None:
            # Pack into the existing buffers. The TopMag mask is precomputed by
            # the caller in production, exactly as it is for the native store,
            # so neither leg is charged for selection.
            pack_rows(
                case.latent,
                case.mask,
                case.weight,
                1.0e-6,
                case.plan,
                case.locations,
                buffers,
            )

        legs: dict[str, dict[str, object]] = {}

        # --- row 1: store -------------------------------------------------
        legs["store"] = {
            "native": native_store_fn,
            "triton": packed_store_fn,
            "fused": None,
            "notes": "both legs write one record per compressed row; no fused store variant",
        }

        # --- row 2: decode into the native layout --------------------------
        def triton_native_fn() -> None:
            with patch.dict(os.environ, {"SGLANG_OPT_TOPMAG_FUSED": "0"}):
                harness.packed_native(case, buffers, triton_workspace)

        def fused_native_fn() -> None:
            with patch.dict(os.environ, {"SGLANG_OPT_TOPMAG_FUSED": "1"}):
                harness.packed_native(case, buffers, fused_workspace)

        legs["decode_native_layout"] = {
            "native": None,  # native has no operator here; see module docstring
            "triton": triton_native_fn,
            "fused": fused_native_fn if fused else None,
            "notes": (
                "native passes the 584-byte buffer to attention unchanged; packed "
                "figures are reconstruct overhead added on top of native"
            ),
        }

        # --- row 3: decode into plain dense bf16 ---------------------------
        legs["decode_dense_bf16"] = {
            "native": native_decode_fn,
            "triton": lambda: unpack_gather_bf16(
                buffers, case.physical, case.raw, case.lengths, case.freqs, native_out
            ),
            "fused": None,
            "notes": "the one genuine apples-to-apples decode pair",
        }

        # --- row 4: the c4 attention leg, reassembled versus read directly ---
        # Both legs read the same 328 bytes/row from the packed cache; the
        # difference is that the bar leg first materialises dense BF16 KV through
        # the Triton gather and then runs FlashMLA sparse on it. The bar's dense
        # buffer is preallocated outside the timed region, so this is the
        # *charitable* version of the reassembling path: production allocates
        # that workspace too. The gate env is untouched here -- both legs drive
        # their kernels directly, exactly as the other rows drive the unpackers.
        if sparse:
            c4_q = harness.c4_query(case)
            bar_rows = torch.empty(
                gathered, config.HEAD_DIM, dtype=torch.bfloat16, device=device
            )
            bar_kv = bar_rows.view(gathered, 1, config.HEAD_DIM)
            bar_indices = harness.flat_indices(case)
            c4_scale = harness.SM_SCALE

            def c4_reassemble_fn() -> None:
                unpack_gather_bf16(
                    buffers, case.physical, case.raw, case.lengths, case.freqs,
                    bar_rows,
                )
                harness.c4_bar(c4_q, bar_kv, bar_indices, c4_scale)

            def c4_direct_fn() -> None:
                sparse.c4_leg(
                    c4_q, buffers.values, buffers.bitmaps, buffers.scales,
                    case.physical, case.raw, case.freqs, c4_scale,
                    topk_lengths=case.lengths,
                )

            legs["attention_c4_leg"] = {
                "native": c4_reassemble_fn,
                "triton": None,
                "fused": c4_direct_fn,
                "notes": (
                    "native = unpack_gather_bf16 + flash_mla_sparse_fwd (the "
                    "production c4 leg); fused = the direct 328-byte read. "
                    "fused_over_native is the direct-over-reassembled ratio; "
                    "both legs allocate their own output and read the same "
                    "cache bytes, so the row is comparable."
                ),
            }

        for op, native_bytes, packed_bytes, direction in OPS:
            leg = legs.get(op)
            if leg is None:
                continue
            row_count = rows_per_leg[op]
            for regime in ("eager", "graph"):
                timings: dict[str, dict[str, float]] = {}
                fallbacks: dict[str, str] = {}
                for name in ("native", "triton", "fused"):
                    fn = leg[name]
                    if fn is None:
                        continue
                    timings[name], reason = _regime_or_eager(
                        fn, regime, warmup=warmup, repeats=repeats
                    )
                    if reason:
                        fallbacks[name] = reason
                native_us = timings.get("native", {}).get("p50_us")
                packed_us = timings.get("triton", {}).get("p50_us")
                fused_us = timings.get("fused", {}).get("p50_us")
                per_leg_bytes = {
                    "native": native_bytes,
                    "triton": packed_bytes,
                    "fused": packed_bytes,
                }
                record = {
                    "workload": workload.name,
                    "op": op,
                    "regime": regime,
                    "direction": direction,
                    "gather_rows": gathered,
                    "context_rows": case.rows,
                    "rows_timed": row_count,
                    "timings": timings,
                    "cache_bytes_per_row": {
                        "native": native_bytes,
                        "triton": packed_bytes,
                        "fused": packed_bytes,
                    },
                    "effective_gbps": {
                        name: _gbps(per_leg_bytes[name], row_count, ms.get("p50_us"))
                        for name, ms in timings.items()
                    },
                    "projected_p50_us_over_csa_layers": {
                        name: ms.get("p50_us") * harness.CSA_LAYERS
                        for name, ms in timings.items()
                    },
                    "triton_over_native": _ratio(packed_us, native_us),
                    "fused_over_native": _ratio(fused_us, native_us),
                    "fused_over_triton": _ratio(fused_us, packed_us),
                    "graph_fallbacks": fallbacks,
                    "notes": leg["notes"],
                }
                results.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

        # The fused kernel exists to beat the Triton reconstruct. Gate it on the
        # smallest workload, in graph mode (what serving captures), so a fast
        # but wrong or merely equal kernel cannot pass quietly.
        if fused and workload.name == harness.WORKLOADS[0].name:
            row = next(
                r
                for r in results
                if r["workload"] == workload.name
                and r["op"] == "decode_native_layout"
                and r["regime"] == "graph"
            )
            fused_us = row["timings"]["fused"]["p50_us"]
            triton_us = row["timings"]["triton"]["p50_us"]
            assert fused_us < triton_us, (
                f"[{workload.name}] fused reconstruct is not faster than Triton: "
                f"fused={fused_us:.2f}us triton={triton_us:.2f}us"
            )

    summary = {
        "gpu": torch.cuda.get_device_name(),
        "csa_layers": harness.CSA_LAYERS,
        "fused_available": fused,
        "sparse_available": sparse,
        "notes": (
            "NativeWorkspace.allocate skips the dense-bf16 workspace under "
            "SGLANG_OPT_TOPMAG_FUSED=1, so the fused leg allocates less memory "
            "than the Triton leg. Relevant if a row is read as a memory "
            "comparison; the timings themselves are unaffected. "
            "attention_c4_leg's 'native' leg is the reassembling production path "
            "and its 'fused' leg the direct 328-byte read."
        ),
        "records": results,
    }
    _write(results, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _dequant(
    kvcache: torch.Tensor, locations: torch.Tensor, page_size: int, out: torch.Tensor
) -> None:
    """Native 584-byte decode into a preallocated buffer (graph-safe)."""
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    dequantize_k_cache_paged(kvcache, locations, page_size, out=out)


def _write(records: list[dict[str, object]], summary: dict[str, object]) -> None:
    if not RESULTS_DIR:
        return
    directory = Path(RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "speed.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (directory / "speed.csv").open("w", newline="") as stream:
        fields = [
            "workload",
            "op",
            "regime",
            "rows_timed",
            "native_p50_us",
            "triton_p50_us",
            "fused_p50_us",
            "triton_over_native",
            "fused_over_native",
            "fused_over_triton",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {k: v for k, v in record.items() if not isinstance(v, dict)}
            for leg in ("native", "triton", "fused"):
                row[f"{leg}_p50_us"] = record["timings"].get(leg, {}).get("p50_us")
            writer.writerow(row)
    print(f"[speed] wrote {directory / 'speed.json'} and speed.csv")


if __name__ == "__main__":
    run_speed()
