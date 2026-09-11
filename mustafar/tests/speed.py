"""Speed suite: every leg timed at every stage it has an operator, on one grid.

The same four **legs** and the same stage names as
:mod:`mustafar.tests.validity`, so a leg name means one thing across both suites:
validity asks "does this leg match native", this asks "what does it cost".
Columns are :data:`mustafar.tests.harness.COLUMNS` -- ``native`` first, then the
four candidates -- and rows are :data:`STAGES`.

The table itself is :data:`MATRIX`, written as data so the prose and the code
implementing it cannot drift: each row names the legs it times together with a
note saying what the timed region contains, and each leg it cannot time together
with the reason. Two rules hold everywhere in it:

  * **No number is fabricated.** Where a leg has no operator at a stage, the cell
    is reported as absent with its reason rather than as an estimated or zero
    figure. ``rows.native_layout`` is the case in point: native hands its
    584-byte buffer to attention unchanged, so there is nothing to time there and
    the bar is the Triton reconstruct the fused kernel has to beat.
  * **A timed region is the operator and nothing else.** Environment pins and
    imports are resolved around the timed run, never inside it -- a ``patch.dict``
    or an ``import`` measured alongside a kernel is Python time reported as kernel
    time -- and :func:`_prepare` allocates every buffer the callables write into
    before timing starts.

Every leg of the ``attention`` stage feeds the same consumer,
``harness.c4_bar`` -> ``flash_mla_sparse_fwd``, so the five differ only in how
the rows are produced. That includes ``sparse``, which reads the 328-byte records
directly; note that its softmax still runs in Python between the two kernel
launches (the v1 wiring), so its figure is a v1 cost and not a kernel-only one.

The grid is :data:`mustafar.tests.harness.WORKLOADS` at the ``identity`` pattern.
This suite answers "what does this cost", not "does it handle this shape", and a
pattern that changed the row counts would make stages incomparable across legs.

Direct run prints per-stage JSON::

    python3 -m mustafar.tests.speed
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch

from .. import config
from ..packed import NativeWorkspace, PackedBuffers
from . import harness

REGIMES = ("eager", "graph")


@dataclass(frozen=True)
class Row:
    """One stage of the table.

    ``bar`` names the leg every other leg at this stage is reported as a ratio
    against. ``present`` pairs each leg the stage can time with a note saying what
    the timed region contains; ``absent`` pairs each leg it cannot with the
    reason. Absences are recorded and reported rather than omitted, so a missing
    column is visible in the artifact instead of invisible in it.
    """

    stage: str
    bar: str
    present: tuple[tuple[str, str], ...]
    absent: tuple[tuple[str, str], ...]
    note: str = ""


@dataclass(frozen=True)
class Contrast:
    """A comparison the suite exists to answer, reported on every record.

    ``gate`` says whether the ratio is *asserted* rather than only reported, and
    ``reason`` says why when it is not: a comparison worth naming is worth saying
    why it is not a gate.
    """

    stage: str
    numerator: str
    denominator: str
    gate: bool
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.numerator}_over_{self.denominator}"


# The stage x leg table, and the bars it holds each leg to. Both candidate entry
# points of the packed leg appear under their own names rather than as one column:
# they render different products for different serving regimes, and they cost
# different amounts. See harness.LEGS.
MATRIX: tuple[Row, ...] = (
    Row(
        stage="store",
        bar=harness.NATIVE,
        present=(
            (harness.NATIVE, "compress_norm_rope_store, writing the 584-byte record"),
            ("packed.bf16", "pack_rows, writing the 328-byte record"),
        ),
        absent=(
            (
                "packed.native",
                "the packed store is one operator shared by both entry points; "
                "this leg reads what it wrote",
            ),
            ("fused", "no fused store variant"),
            ("sparse", "the sparse kernel consumes packed rows and has no store"),
        ),
        note=(
            "both legs are fed the same latent and neither is charged for TopMag "
            "selection, which production precomputes"
        ),
    ),
    Row(
        stage="rows.dense_bf16",
        bar=harness.NATIVE,
        present=(
            (harness.NATIVE, "dequantize_k_cache_paged over the 584-byte pages"),
            ("packed.bf16", "unpack_gather_bf16 over the 328-byte records"),
        ),
        absent=(
            (
                "packed.native",
                "renders the 584-byte layout, not dense BF16 -- see the "
                "rows.native_layout row",
            ),
            ("fused", "no fused variant of this product"),
            ("sparse", "no dense row output by construction"),
        ),
        note="the one genuine apples-to-apples decode pair",
    ),
    Row(
        stage="rows.native_layout",
        bar="packed.native",
        present=(
            (
                "packed.native",
                "unpack_gather_native with FUSED=0: the Triton gather, then a repack",
            ),
            ("fused", "unpack_gather_native with FUSED=1: the CUDA adapter"),
        ),
        absent=(
            (
                harness.NATIVE,
                "native hands its 584-byte buffer to attention unchanged, so "
                "there is no operator to time",
            ),
            (
                "packed.bf16",
                "renders dense BF16, not the 584-byte layout -- see the "
                "rows.dense_bf16 row",
            ),
            ("sparse", "no reassembly by construction"),
        ),
        note=(
            "the packed figures are reconstruct overhead measured against a "
            "native stage that does no work; the bar is therefore the Triton "
            "reconstruct the fused kernel has to beat, not a native figure that "
            "does not exist"
        ),
    ),
    Row(
        stage="attention",
        bar=harness.NATIVE,
        present=(
            (
                harness.NATIVE,
                "dequantize_k_cache_paged + flash_mla_sparse_fwd: the "
                "all-flags-off cost of this step",
            ),
            (
                "packed.bf16",
                "unpack_gather_bf16 + flash_mla_sparse_fwd: the multi-token-extend "
                "production path",
            ),
            (
                "packed.native",
                "unpack_gather_native (FUSED=0) + dequant + flash_mla_sparse_fwd: "
                "the decode/small-extend path",
            ),
            (
                "fused",
                "unpack_gather_native (FUSED=1) + dequant + flash_mla_sparse_fwd",
            ),
            (
                "sparse",
                "sparse.c4_leg: the 328-byte records read directly, no reassembly",
            ),
        ),
        absent=(),
        note=(
            "every leg is read back through dequantize_k_cache_paged into a "
            "preallocated dense buffer and fed to the same consumer, so the five "
            "differ only in how the rows are produced"
        ),
    ),
)

STAGES = tuple(row.stage for row in MATRIX)

# The comparisons this suite exists to answer. The gated one is the fused
# kernel's whole justification; the other is the sparse kernel's, reported with
# the measured ratio so a reader can see where v1 stands.
CONTRASTS: tuple[Contrast, ...] = (
    Contrast(
        stage="rows.native_layout",
        numerator="fused",
        denominator="packed.native",
        gate=True,
        reason="the fused kernel exists to beat the Triton reconstruct it replaces",
    ),
    Contrast(
        stage="attention",
        numerator="sparse",
        denominator="packed.bf16",
        gate=False,
        reason=(
            "reported, not asserted: v1 runs its softmax in Python between two "
            "kernel launches, so it is expected to lose to the reassembling path "
            "until the online-softmax pass lands. The number is the finding."
        ),
    ),
)


@dataclass(frozen=True)
class _Ops:
    """Every operator a timed callable drives, resolved once before timing.

    Resolved outside any timed region because a lookup inside one would be
    measured as though it were kernel time. It also keeps this module importable
    -- and the CPU matrix test runnable -- without SGLang, Triton or the CUDA
    extensions, none of which are touched until :func:`run_speed`.
    """

    store: Callable[..., object]
    dequant: Callable[..., object]
    pack_rows: Callable[..., object]
    gather_bf16: Callable[..., object]
    unpack_native: Callable[..., object]
    c4_leg: Callable[..., object] | None


@dataclass(frozen=True)
class _Ctx:
    """Everything one workload's timed callables write into, allocated once.

    All of it is preallocated because the callables write in place: production
    owns these buffers already, so charging the timer for allocating them would
    overstate every leg.
    """

    case: harness.Case
    buffers: PackedBuffers  # the 328-byte records
    native_cache: torch.Tensor  # the 584-byte pages
    native_locations: torch.Tensor  # the native records the gather reads
    # The three row buffers are all (gathered, 1, HEAD_DIM) bf16: that is the shape
    # dequantize_k_cache_paged asserts on its ``out``, and flash_mla_sparse_fwd
    # wants the same one for its ``kv``. unpack_gather_bf16 only reshapes whatever
    # it is handed, so it takes the same buffer shape rather than a second one.
    native_rows: torch.Tensor  # the native readback
    bf16_rows: torch.Tensor  # the packed.bf16 readback
    dense_rows: torch.Tensor  # the 584-byte layout readback
    workspaces: dict[str, NativeWorkspace]  # the legs that reconstruct a layout
    workspace_locations: dict[str, torch.Tensor]  # and the ids they were built for
    q: torch.Tensor  # (batch, HEAD_COUNT, HEAD_DIM) bf16 queries
    indices: torch.Tensor  # (batch, 1, TOPK) int32 ids for flash_mla


def _resolve_ops() -> _Ops:
    """Import and bind every operator the timed callables drive."""
    from sglang.kernels.ops.attention.dsv4.compress import compress_norm_rope_store
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    from ..packed import pack_rows, unpack_gather_bf16, unpack_gather_native

    c4_leg = None
    if harness.leg_available("sparse"):
        from ..sparse import c4_leg as sparse_c4_leg

        c4_leg = sparse_c4_leg
    return _Ops(
        store=compress_norm_rope_store,
        dequant=dequantize_k_cache_paged,
        pack_rows=pack_rows,
        gather_bf16=unpack_gather_bf16,
        unpack_native=unpack_gather_native,
        c4_leg=c4_leg,
    )


def _prepare(case: harness.Case) -> _Ctx:
    """Allocate every buffer ``case``'s timed callables write into.

    Runs once per workload, outside any timed region. The packed buffers can only
    be built under a packed pin because ``pack_rows`` asserts the ABI's static
    config at its head; both packed entry points share them.
    """
    gathered = case.workload.gather_rows
    with harness.leg_env("packed.bf16"):
        buffers = harness.packed_buffers(case)
    native_cache, locations = harness.native_store(case, case.latent)
    workspaces = {
        # with_dense is explicit on purpose: it is what production allocates for
        # each path, and giving the fused leg the dense workspace it never uses
        # would hide an allocation difference between the two.
        "packed.native": harness.native_workspace(case, with_dense=True),
        "fused": harness.native_workspace(case, with_dense=False),
    }

    def rows() -> torch.Tensor:
        return torch.empty(
            gathered, 1, config.HEAD_DIM, dtype=torch.bfloat16, device=case.device
        )

    return _Ctx(
        case=case,
        buffers=buffers,
        native_cache=native_cache,
        # The grid is the identity pattern, so the gathered native records are the
        # first ``gathered`` of the cache and no gather happens on the native side.
        native_locations=locations[:gathered],
        native_rows=rows(),
        bf16_rows=rows(),
        dense_rows=rows(),
        workspaces=workspaces,
        workspace_locations={
            leg: workspace.temporary_indices.reshape(-1).to(torch.int64)
            for leg, workspace in workspaces.items()
        },
        q=harness.c4_query(case),
        indices=harness.flat_indices(case),
    )


def _then(*fns: Callable[[], None]) -> Callable[[], None]:
    """Compose in-place callables.

    A leg that is another leg plus a readback is spelled that way instead of
    being rebuilt, so the two cannot drift apart.
    """

    def run() -> None:
        for fn in fns:
            fn()

    return run


def _cells(
    ctx: _Ctx, ops: _Ops, selected: tuple[str, ...]
) -> dict[str, dict[str, Callable[[], None]]]:
    """The timed callable for every selected leg :data:`MATRIX` declares.

    Verified against the declaration by :func:`_check_cells`, in both directions:
    a declared leg that is not built would silently narrow the table, and a built
    leg that is not declared would be timed and never reported.
    """
    case = ctx.case

    def reconstruct(leg: str) -> Callable[[], None]:
        # The same call harness.packed_native makes, with the import hoisted out
        # of the timed region.
        workspace = ctx.workspaces[leg]
        return lambda: ops.unpack_native(
            ctx.buffers, case.physical, case.raw, case.lengths, case.freqs, workspace
        )

    def read_back(leg: str) -> Callable[[], None]:
        """Read a reconstructed 584-byte layout back into dense rows.

        Every layout-rendering leg needs this before ``flash_mla`` can consume it,
        so doing it for all of them is what keeps the attention stage's legs
        comparable with each other.
        """
        workspace = ctx.workspaces[leg]
        locations = ctx.workspace_locations[leg]
        out = ctx.dense_rows

        def run() -> None:
            ops.dequant(workspace.native_bytes, locations, workspace.page_size, out)

        return run

    def c4(rows: torch.Tensor) -> Callable[[], None]:
        """flash_mla_sparse_fwd over one preallocated dense buffer.

        ``rows`` is already the ``(gathered, 1, HEAD_DIM)`` its ``kv`` wants, so
        nothing is reshaped inside the timed region.
        """
        return lambda: harness.c4_bar(ctx.q, rows, ctx.indices, harness.SM_SCALE)

    def native_store() -> None:
        ops.store(
            case.latent,
            case.plan,
            norm_weight=case.weight,
            norm_eps=1.0e-6,
            freq_cis=case.freqs,
            out_loc=case.locations,
            kvcache=ctx.native_cache,
            page_size=harness.PAGE_SIZE,
        )

    def packed_store() -> None:
        ops.pack_rows(
            case.latent,
            case.mask,
            case.weight,
            1.0e-6,
            case.plan,
            case.locations,
            ctx.buffers,
        )

    def gather_bf16() -> None:
        ops.gather_bf16(
            ctx.buffers,
            case.physical,
            case.raw,
            case.lengths,
            case.freqs,
            ctx.bf16_rows,
        )

    def native_rows() -> None:
        ops.dequant(
            ctx.native_cache, ctx.native_locations, harness.PAGE_SIZE, ctx.native_rows
        )

    def sparse_c4() -> None:
        ops.c4_leg(
            ctx.q,
            ctx.buffers.values,
            ctx.buffers.bitmaps,
            ctx.buffers.scales,
            case.physical,
            case.raw,
            case.freqs,
            harness.SM_SCALE,
            topk_lengths=case.lengths,
        )

    declared: dict[str, dict[str, Callable[[], None]]] = {
        "store": {harness.NATIVE: native_store, "packed.bf16": packed_store},
        "rows.dense_bf16": {harness.NATIVE: native_rows, "packed.bf16": gather_bf16},
        "rows.native_layout": {
            "packed.native": reconstruct("packed.native"),
            "fused": reconstruct("fused"),
        },
        "attention": {
            harness.NATIVE: _then(native_rows, c4(ctx.native_rows)),
            "packed.bf16": _then(gather_bf16, c4(ctx.bf16_rows)),
            "packed.native": _then(
                reconstruct("packed.native"),
                read_back("packed.native"),
                c4(ctx.dense_rows),
            ),
            "fused": _then(
                reconstruct("fused"), read_back("fused"), c4(ctx.dense_rows)
            ),
            "sparse": sparse_c4,
        },
    }
    return {
        stage: {leg: fn for leg, fn in stage_cells.items() if leg in selected}
        for stage, stage_cells in declared.items()
    }


def _check_cells(
    cells: dict[str, dict[str, Callable[[], None]]], selected: tuple[str, ...]
) -> None:
    """Assert the built cells and :data:`MATRIX` agree, in both directions."""
    declared = {(row.stage, leg) for row in MATRIX for leg, _ in row.present}
    wanted = {pair for pair in declared if pair[1] in selected}
    built = {
        (stage, leg) for stage, stage_cells in cells.items() for leg in stage_cells
    }
    missing = sorted(wanted - built)
    extra = sorted(built - declared)
    assert not missing, f"MATRIX declares legs that _cells does not build: {missing}"
    assert not extra, f"_cells builds legs MATRIX does not declare: {extra}"


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
        return (
            harness.timed(fn, warmup=warmup, repeats=repeats),
            str(error).splitlines()[0],
        )


def _gbps(bytes_per_row: int, rows: int, p50_us: float | None) -> float | None:
    """Effective cache bandwidth, or None where the leg has no operator."""
    if not bytes_per_row or p50_us is None or p50_us <= 0:
        return None
    return bytes_per_row * rows / (p50_us * 1.0e-6) / 1.0e9


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Ratio, or None when either leg is absent or the denominator is zero."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _bytes_per_row(leg: str) -> int:
    """Cache bytes one row moves for a leg.

    The 584-vs-328 figure is the whole capacity story, so it is reported per leg
    rather than per stage: the native column moves 584-byte records and every
    packed leg moves 328-byte ones.
    """
    if leg == harness.NATIVE:
        return config.NATIVE_RECORD_BYTES
    return config.PACKED_RECORD_BYTES


def _rows_timed(case: harness.Case, stage: str) -> int:
    """Rows one call processes: the whole cache when storing, else the gather."""
    return case.rows if stage == "store" else case.workload.gather_rows


def _contrast_value(contrast: Contrast, timings: dict) -> float | None:
    """A contrast's ratio, or None when one of its legs was not timed."""
    return _ratio(
        timings.get(contrast.numerator, {}).get("p50_us"),
        timings.get(contrast.denominator, {}).get("p50_us"),
    )


def _check_gates(case: harness.Case, row: Row, regime: str, timings: dict) -> None:
    """Assert the gated contrasts, at the point serving actually runs them.

    The smallest workload in graph mode: that is the regime serving captures, and
    it is where the kernels these gates compare get deployed. A gate that held
    only at some larger batch would not be saying anything useful.
    """
    if regime != "graph" or case.workload.name != harness.WORKLOADS[0].name:
        return
    for contrast in CONTRASTS:
        if contrast.stage != row.stage or not contrast.gate:
            continue
        numerator = timings.get(contrast.numerator, {}).get("p50_us")
        denominator = timings.get(contrast.denominator, {}).get("p50_us")
        if numerator is None or denominator is None:
            continue
        assert numerator < denominator, (
            f"[{case.workload.name}/{row.stage}] {contrast.label} is not below 1: "
            f"{contrast.numerator}={numerator:.2f}us "
            f"{contrast.denominator}={denominator:.2f}us "
            f"({contrast.reason})"
        )


def _record(
    case: harness.Case,
    row: Row,
    regime: str,
    timings: dict[str, dict[str, float]],
    fallbacks: dict[str, str],
    absent: dict[str, str],
) -> dict[str, object]:
    """One stage, one regime: every timed leg plus what it means."""
    rows_timed = _rows_timed(case, row.stage)
    bar_us = timings.get(row.bar, {}).get("p50_us")
    legs: dict[str, dict[str, object]] = {}
    for leg, timing in timings.items():
        bytes_per_row = _bytes_per_row(leg)
        legs[leg] = {
            "p50_us": timing["p50_us"],
            "p95_us": timing["p95_us"],
            "cache_bytes_per_row": bytes_per_row,
            "effective_gbps": _gbps(bytes_per_row, rows_timed, timing["p50_us"]),
            "projected_p50_us_over_csa_layers": timing["p50_us"] * harness.CSA_LAYERS,
        }
    return {
        "workload": case.workload.name,
        "stage": row.stage,
        "regime": regime,
        "bar": row.bar,
        "gather_rows": case.workload.gather_rows,
        "context_rows": case.rows,
        "rows_timed": rows_timed,
        "legs": legs,
        "absent": absent,
        "ratios_vs_bar": {
            leg: _ratio(timing["p50_us"], bar_us)
            for leg, timing in timings.items()
            if leg != row.bar
        },
        "contrasts": {
            contrast.label: _contrast_value(contrast, timings)
            for contrast in CONTRASTS
            if contrast.stage == row.stage
        },
        "graph_fallbacks": fallbacks,
        "notes": row.note,
    }


def _write(records: list[dict[str, object]], summary: dict[str, object]) -> None:
    # Read at write time rather than at import: the artifact's destination is the
    # caller's business, and a module-level snapshot ignores anyone who sets it
    # after this module is imported.
    results_dir = os.environ.get("MUSTAFAR_RESULTS_DIR")
    if not results_dir:
        return
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "speed.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    # Derived from the vocabulary so a new leg or contrast is one entry in
    # harness.LEGS / CONTRASTS and nothing else.
    fields = ["workload", "stage", "regime", "bar", "rows_timed"]
    fields += [f"{leg}_p50_us" for leg in harness.COLUMNS]
    fields += [contrast.label for contrast in CONTRASTS]
    with (directory / "speed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {k: v for k, v in record.items() if not isinstance(v, dict)}
            for leg in harness.COLUMNS:
                row[f"{leg}_p50_us"] = record["legs"].get(leg, {}).get("p50_us")
            for contrast in CONTRASTS:
                row[contrast.label] = record["contrasts"].get(contrast.label)
            writer.writerow(row)
    print(f"[speed] wrote {directory / 'speed.json'} and speed.csv")


@patch.dict(os.environ, **harness.OFF_ENV)
def run_speed(*, legs: tuple[str, ...] | None = None) -> dict[str, object]:
    """Time every selected leg at every stage :data:`MATRIX` gives it an operator.

    The all-flags-off base is pinned for the whole call and each leg's own flags
    are turned back on around its timed run, so a leg is timed under exactly the
    configuration it is defined by -- and the pinned set is proven legal by
    ``harness.leg_env`` rather than assumed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("speed requires CUDA")

    available = harness.available_legs()
    # The bar is always timed: it is what every ratio in the table is taken
    # against, so a table without it could not be read.
    selected = (harness.NATIVE, *harness.select_legs(legs))
    ops = _resolve_ops()
    print(
        f"[speed] legs available={list(available)} selected={list(selected)} "
        f"stages={list(STAGES)}",
        flush=True,
    )
    warmup, repeats = 10, 50
    results: list[dict[str, object]] = []

    for workload in harness.WORKLOADS:
        case = harness.build_case(workload, torch.device("cuda"))
        ctx = _prepare(case)
        cells = _cells(ctx, ops, selected)
        _check_cells(cells, selected)

        for row in MATRIX:
            stage_cells = cells[row.stage]
            absent = {leg: reason for leg, reason in row.absent if leg in selected}
            for regime in REGIMES:
                timings: dict[str, dict[str, float]] = {}
                fallbacks: dict[str, str] = {}
                for leg in harness.COLUMNS:
                    fn = stage_cells.get(leg)
                    if fn is None:
                        continue
                    # The pin wraps the timed run, not the callable: inside the
                    # callable it would be measured alongside the kernel.
                    with harness.leg_env(leg):
                        timings[leg], reason = _regime_or_eager(
                            fn, regime, warmup=warmup, repeats=repeats
                        )
                    if reason:
                        fallbacks[leg] = reason
                _check_gates(case, row, regime, timings)
                record = _record(case, row, regime, timings, fallbacks, absent)
                results.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    # The serving point: what a captured decode step costs at the smallest
    # workload, which is where both contrasts are decided.
    serving = [
        record
        for record in results
        if record["workload"] == harness.WORKLOADS[0].name
        and record["regime"] == "graph"
    ]
    summary = {
        "gpu": torch.cuda.get_device_name(),
        "csa_layers": harness.CSA_LAYERS,
        "legs": {leg: leg in available for leg in harness.LEGS},
        "selected": list(selected),
        "headline": {
            f"{record['stage']}/{label}": value
            for record in serving
            for label, value in record["contrasts"].items()
        },
        "notes": (
            "NativeWorkspace.allocate skips the dense-bf16 workspace under "
            "SGLANG_OPT_TOPMAG_FUSED=1, so the fused leg allocates less memory "
            "than the Triton leg. Relevant if a row is read as a memory "
            "comparison; the timings themselves are unaffected. "
            + " ".join(
                f"{contrast.label}: {contrast.reason}" for contrast in CONTRASTS
            )
        ),
        "records": results,
    }
    _write(results, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legs",
        default=None,
        help="comma-separated subset of "
        f"{','.join(harness.LEGS)} (default: all available)",
    )
    args = parser.parse_args()
    run_speed(legs=tuple(args.legs.split(",")) if args.legs else None)
