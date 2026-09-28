"""Collect per-layer (latent, query, selected-index) samples from a live server.

Runs on the **fork** tree with the store *off* -- the patch supplies the hooks,
`SGLANG_OPT_STARKV=0` keeps every value native, and `STARKV_CAPTURE=<tag>` turns
the hooks into collectors. That combination is what makes the samples usable as
ground truth: nothing here is recorded through the compression being measured.

Two hooks, because the quantities live in different places:

* A, at the store (``ops.store_lowrank``): the normed NoPE block the native store
  would have quantized, the post-rope tail, the c4 pool slot the row went to, and
  the rope position.
* B, at the decode dispatch (``deepseek_v4_backend``): the query and the c4
  indices the native attention selects, plus the per-row top-k length.

The join is on the pool slot: B's indices and A's locations are both c4 token
ids. ``spectrum.py`` therefore reconstructs the exact key behind every selected
index, and scores compression against the real attention, not a proxy.

**That join is currently broken, and the eps metric is off until it is fixed.**
The first two captures recorded only 17 distinct ``loc`` values for 2,708 rows
per layer -- a degenerate key, not the slots the attention selected -- so
``spectrum.score_error`` joined 109-295 of ~5.5M entries and now refuses to
report below its coverage floor. The write-path ``loc`` the vendor hands the
store hook is the thing to diagnose (``_get_out_loc``; the decode path zeroes
non-boundary rows onto slot 0). Note also that even a correct ``loc`` only
partially helps: this hook samples the rows being *written* in the window, while
each decode step selects 512 slots spanning the whole sequence history, so a
usable eps likely needs read-path capture rather than a repair here.

The two hooks sample a **common decode window**, and that is the point of the
`calls` counter below. Both fire exactly once per c4 layer per decode step, so
their per-layer call indices advance in lockstep; a capture that let A buffer
prefill rows while B buffered decode steps would fill its caps at unrelated
points and the join would come back empty. Rows are tagged with the call index
that produced them and the window closes at ``STARKV_CAPTURE_STEPS``, so the
overlap is a property of the file, not of how long the session ran.

Cost: capture runs eagerly (no decode graphs) and flushes on a cap, so a capture
session is diagnostic, not a throughput measurement.
"""

import atexit
import os
import threading

import torch

from .. import config

_lock = threading.Lock()
# layer -> {"calls", "loc", "pos", "boundary", "step", "nope", "tail"}
_A = {}
# layer -> {"calls", "step", "q", "idx", "lens"}
_B = {}
_A_done = set()  # layers whose store buffer is full and written
_B_done = set()


def enabled() -> bool:
    return bool(config.capture_tag())


def _rank() -> int:
    """Distributed rank, so a TP run writes one file per rank instead of racing."""
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def out_dir(tag: str = "") -> str:
    return os.path.join(config.captures_dir(), tag or config.capture_tag())


def _store_path(layer: int) -> str:
    return os.path.join(
        out_dir(), f"L{layer:03d}_store_r{config.RANK}_rank{_rank()}.pt"
    )


def _attn_path(layer: int) -> str:
    return os.path.join(out_dir(), f"L{layer:03d}_attn_rank{_rank()}.pt")


def _cpu_fp16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True).to(torch.float16)


def reset() -> None:
    """Drop everything buffered, e.g. between two drives of the same session."""
    with _lock:
        _A.clear()
        _B.clear()
        _A_done.clear()
        _B_done.clear()


def _blank_store(calls: int = 0) -> dict:
    """An empty store buffer. `calls` survives a flush: it is the window clock."""
    return {"calls": calls, "loc": [], "pos": [], "boundary": [], "step": [],
            "nope": [], "tail": []}


def _blank_attn(calls: int = 0) -> dict:
    return {"calls": calls, "step": [], "batch": [], "q": [], "idx": [], "lens": []}


# --- hook A: the store -------------------------------------------------------


def _rows_buffered(layer: int) -> int:
    return sum(t.shape[0] for t in _A[layer]["nope"])


def capture_store(layer, normed, tail, loc, pos, boundary) -> None:
    """Record one **decode** store call's rows.

    `normed` is the [n, HEAD_DIM] output of the vendor's norm (pre-quantization,
    pre-basis), `tail` the [n, ROPE_DIM] post-rope tail, `loc` the [n] c4 slots,
    `pos` the [n] rope positions, `boundary` the [n] bool of whether the row is a
    real compress boundary (decode stores non-boundary rows too, zeroed, onto
    slot 0 -- `stored_rows` in spectrum.py drops them, but they are kept here so
    the sample is not silently thinned).

    The caller filters out prefill and non-c4 stores; this records every decode
    row it is handed, tagged with the call index that produced it.
    """
    if not enabled() or layer is None:
        return
    with _lock:
        if layer in _A_done:
            return
        buf = _A.setdefault(layer, _blank_store())
        if buf["calls"] >= config.capture_steps() or _rows_buffered(layer) >= config.capture_rows():
            _flush_store_locked(layer)
            return
        n = normed.shape[0]
        step = buf["calls"]
        buf["calls"] += 1
        buf["step"].append(torch.full((n,), step, dtype=torch.int64))
        buf["loc"].append(_cpu_fp16(loc.to(torch.float32)).to(torch.int64).flatten())
        buf["pos"].append(pos.detach().to("cpu", copy=True).to(torch.int32).flatten())
        buf["boundary"].append(
            boundary.detach().to("cpu", copy=True).to(torch.bool).flatten()
        )
        buf["nope"].append(_cpu_fp16(normed[:, : config.NOPE_DIM]))
        buf["tail"].append(_cpu_fp16(tail))


# --- hook B: the decode attention -------------------------------------------


def capture_decode_attn(layer, q, indices, lengths) -> None:
    """Record one decode step's query and selected c4 indices.

    `q` is the backend's (b, 1, heads, HEAD_DIM) query -- only the first
    ``STARKV_CAPTURE_HEADS`` heads are kept, because the logit statistics are
    per-head and a full head block would be 64x the disk for no extra coverage.
    `indices` is (b, topk) selected c4 token ids and `lengths` the (b,) count of
    valid entries in each row (the tail of each index row is padding).

    Per-call batch sizes are recorded alongside, because a real serving step's
    batch varies: with steps concatenated into one tensor, the boundaries are
    otherwise unrecoverable and `spectrum.score_error` could not tell which
    queries belong to which step.
    """
    if not enabled() or layer is None:
        return
    # This runs on the decode path of a live engine, so anything the caller was
    # not able to hand over has to be a skipped sample, never an exception: a
    # raise here kills the server rather than losing one row of one measurement.
    # The call site only passes these once it has matched them to this query,
    # but a None reaching here means there is no Top-k entry to join the query
    # through, so the sample could not be scored anyway.
    if q is None or indices is None or lengths is None:
        return
    # The call site hands over the backend's *matched* index tensor, which the
    # attention path pads with a singleton query-position dim for the kernel
    # call -- (b, 1, topk), not the (b, topk) this hook records. Drop it here so
    # every consumer of the capture does not have to know about it.
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    elif indices.ndim != 2:
        return  # not a shape this hook can record
    with _lock:
        if layer in _B_done:
            return
        buf = _B.setdefault(layer, _blank_attn())
        if buf["calls"] >= config.capture_steps():
            _flush_attn_locked(layer)
            return
        step = buf["calls"]
        buf["calls"] += 1
        heads = config.capture_heads()
        q = q[:, :, :heads, :]
        buf["step"].append(step)
        buf["batch"].append(int(q.shape[0]))
        buf["q"].append(_cpu_fp16(q.reshape(q.shape[0], -1, q.shape[-1])))
        buf["idx"].append(indices.detach().to("cpu", copy=True).to(torch.int32))
        buf["lens"].append(
            lengths.detach().to("cpu", copy=True).to(torch.int32).flatten()
        )


# --- flush ------------------------------------------------------------------


def _write(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[capture] wrote {path}")


def _flush_store_locked(layer: int) -> None:
    buf = _A.get(layer)
    if not buf or not buf["nope"]:
        return
    os.makedirs(out_dir(), exist_ok=True)
    _write(
        _store_path(layer),
        {
            "layer": layer,
            "rank": config.RANK,
            "calls": buf["calls"],
            "loc": torch.cat(buf["loc"]),
            "pos": torch.cat(buf["pos"]),
            "step": torch.cat(buf["step"]),
            "boundary": torch.cat(buf["boundary"]),
            "nope": torch.cat(buf["nope"], dim=0).float(),
            "tail": torch.cat(buf["tail"], dim=0).float(),
        },
    )
    buf = _blank_store(buf["calls"])
    _A[layer] = buf
    _A_done.add(layer)  # one full buffer per layer is the sample

def _flush_attn_locked(layer: int) -> None:
    buf = _B.get(layer)
    if not buf or not buf["q"]:
        return
    os.makedirs(out_dir(), exist_ok=True)
    _write(
        _attn_path(layer),
        {
            "layer": layer,
            "calls": buf["calls"],
            "step": torch.tensor(buf["step"], dtype=torch.int64),
            "batch": torch.tensor(buf["batch"], dtype=torch.int64),
            "q": torch.cat(buf["q"], dim=0).float(),
            "idx": torch.cat(buf["idx"], dim=0),
            "lens": torch.cat(buf["lens"], dim=0),
        },
    )
    buf = _blank_attn(buf["calls"])
    _B[layer] = buf
    _B_done.add(layer)


def flush() -> int:
    """Write every buffered layer. Returns the number of files written."""
    written = 0
    with _lock:
        for layer in list(_A):
            if _A[layer]["nope"]:
                _flush_store_locked(layer)
                written += 1
        for layer in list(_B):
            if _B[layer]["q"]:
                _flush_attn_locked(layer)
                written += 1
    return written


def flush_at_exit() -> None:
    """Register a final flush, so a clean shutdown loses nothing.

    A SIGKILL still loses the partial buffer -- which is why the window closes on
    a call cap, so a layer is normally written well before shutdown.
    """
    if enabled():
        atexit.register(flush)


def summary() -> str:
    with _lock:
        rows = {k: _rows_buffered(k) for k in _A}
        steps = {k: v["calls"] for k, v in _B.items()}
    return f"store rows/layer={rows} attn calls/layer={steps}"


def alignment() -> dict:
    """Per-layer (store calls, attn calls) -- the join's precondition.

    Both hooks fire once per c4 layer per decode step, so these should match. A
    layer where they do not means one hook saw a step the other did not (a
    capture started mid-session, or a store declined), and the join for that
    layer covers only the smaller of the two windows.
    """
    with _lock:
        return {
            layer: (_A.get(layer, {}).get("calls", 0), _B.get(layer, {}).get("calls", 0))
            for layer in sorted(set(_A) | set(_B))
        }
