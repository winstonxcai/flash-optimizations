"""The feasibility measurement: does a per-layer rank-r basis hold?

CPU-only. Reads what `capture.py` wrote and answers three questions per layer
and per rank, none of which a Frobenius proxy answers honestly:

1. ``E_l(r)`` -- energy retained, in three flavours that fail differently:
   * ``selffit``  fit on the sampled rows, scored on the same rows: the ceiling.
   * ``heldout``  fit on one half, scored on the other: what a *frozen global*
     basis bought on latents it never saw.
   * ``drift``    fit on the lowest-position quartile, scored on the highest:
     the same question asked across a session, which is where a frozen basis
     actually dies -- the distribution moves during a long decode.
2. ``eps_score`` -- relative error of the attention logits, computed on the
   entries the model really selected (joined through the c4 slot), not on random
   pairs. Reported twice: ``eps_basis`` (rank truncation only) and ``eps_quant``
   (rank truncation plus the fp8 record), so the two costs stay separable.
3. ``r_l`` -- the smallest rank per layer that still meets a retention target,
   which is the per-layer heterogeneity the whole scheme is supposed to exploit.

Nothing here decides the study by itself; the numbers land in report.md, and the
decision rule there is stated before the run.
"""

import argparse
import glob
import json
import os

import torch

from .. import config, reference

# Position quantiles for the drift split. Fit on the low quartile, score on the
# high one: the widest within-session separation the capture offers.
DRIFT_FIT_Q = 0.25
DRIFT_EVAL_Q = 0.75

# Fraction of the model's actually-selected entries that must join to a sampled
# store row before `eps_score` is reported at all. The join is by c4 slot, and
# it only succeeds for selected slots the store hook also happened to capture,
# so coverage is a measurement in its own right and not a formality: the first
# full capture joined 255 of 5,525,504 entries.
MIN_COVERAGE = 0.01


def load_store(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def load_attn(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def _clamp_rank(rank: int, rows: int) -> int:
    """Largest usable rank at or below the request, floored to a scale tile.

    A rank that is not a multiple of the 64-wide tile has no defined record
    layout, and a rank above the row count is rank-deficient -- `fit_basis`
    refuses it. Both would otherwise surface as a crash mid-sweep or, worse, as
    a plausible-looking number from a short basis.
    """
    usable = min(rank, rows, config.NOPE_DIM)
    return (usable // config.TILE_SIZE) * config.TILE_SIZE


def stored_rows(store: dict) -> dict:
    """The rows the pool actually kept, as a capture dict.

    Decode zeroes non-boundary rows onto slot 0 (they are computed but never
    read back), so scoring them would let the measurement sample a distribution
    the inference path never reads. Prefill marks every row boundary, matching
    the vendor's "store all rows, no filtering" contract, so prefill rows all
    survive this filter.
    """
    mask = store["boundary"].bool()
    if bool(mask.all()):
        return store
    return {k: v[mask] if torch.is_tensor(v) and v.shape[:1] == mask.shape else v
            for k, v in store.items()}


def discover(capture_dir: str) -> dict:
    """Layer -> {"store": path, "attn": path|None} under a capture directory."""
    found = {}
    for path in sorted(glob.glob(os.path.join(capture_dir, "L*_store_r*.pt"))):
        layer = int(os.path.basename(path)[1:4])
        found.setdefault(layer, {})["store"] = path
    for path in sorted(glob.glob(os.path.join(capture_dir, "L*_attn_rank*.pt"))):
        layer = int(os.path.basename(path)[1:4])
        found.setdefault(layer, {})["attn"] = path
    return {k: v for k, v in sorted(found.items()) if "store" in v}


# --- energy retention -------------------------------------------------------


def _retention_stats(x, d) -> dict:
    r = reference.retention(x, d)
    return {"mean": float(r.mean()), "min": float(r.min()), "p05": float(r.quantile(0.05))}


def _drift_axis(store: dict):
    """The axis the drift split moves along, and its name.

    ``step`` is the decode-call index that produced the row, which is the axis a
    long session actually travels: does a basis fit in the first hundred decode
    steps still hold at the five-thousandth. ``pos`` (the rope position) is the
    fallback, but for a replay workload it is dominated by prompt length --
    concurrent requests sit at ~144k positions from the first step onward, so it
    barely moves within a session and would understate drift badly.
    """
    if "step" in store and store["step"].numel():
        return store["step"].float(), "decode_step"
    return store["pos"].float(), "rope_position"


def retention_table(store: dict, ranks) -> dict:
    x = store["nope"].float()
    axis, axis_name = _drift_axis(store)
    out = {}
    for requested in ranks:
        rank = _clamp_rank(requested, x.shape[0])
        if rank != requested:
            print(f"  rank {requested} -> {rank} ({x.shape[0]} rows available)")
        if rank < config.TILE_SIZE:
            continue
        row = {"rank": rank, "requested_rank": requested, "bytes": config.bytes_for_rank(rank)}

        d_self = reference.fit_basis(x, rank)
        row["selffit"] = _retention_stats(x, d_self)

        # Held-out: fit and score on disjoint halves, so the number is what a
        # basis frozen *before* seeing these latents would have achieved.
        half = x.shape[0] // 2
        if half >= rank:
            a = reference.fit_basis(x[:half], rank)
            b = reference.fit_basis(x[half:], rank)
            held_a = reference.retention(x[half:], a)
            held_b = reference.retention(x[:half], b)
            held = torch.cat([held_a, held_b])
            row["heldout"] = {
                "mean": float(held.mean()),
                "min": float(held.min()),
                "p05": float(held.quantile(0.05)),
            }
        else:
            row["heldout"] = None

        # Drift: the same question asked across the capture window.
        lo = torch.quantile(axis, DRIFT_FIT_Q).item()
        hi = torch.quantile(axis, DRIFT_EVAL_Q).item()
        early, late = x[axis <= lo], x[axis >= hi]
        if early.shape[0] >= rank and late.shape[0] > 0:
            d_early = reference.fit_basis(early, rank)
            row["drift"] = {
                "axis": axis_name,
                "fit_rows": int(early.shape[0]),
                "eval_rows": int(late.shape[0]),
                "fit_axis_max": float(lo),
                "eval_axis_min": float(hi),
                **_retention_stats(late, d_early),
            }
        else:
            row["drift"] = None
        out[rank] = row
    return out


# --- attention-logit error --------------------------------------------------


def _loc_index(store: dict) -> dict:
    """c4 slot -> row index in the store sample."""
    return {int(loc): i for i, loc in enumerate(store["loc"].tolist())}


def score_error(store: dict, attn: dict, d: torch.Tensor) -> dict:
    """eps on the selected entries, streaming one decode step at a time.

    `d` is the basis being scored (a frozen/held-out one, not a self-fit, or the
    number is an oracle). Keys come from the capture, joined by c4 slot; queries
    and the selected slots come from the attention hook. Entries whose slot was
    not sampled are dropped and counted, so the coverage is never implicit.

    Steps are split by the recorded per-step batch sizes rather than by index
    arithmetic: a serving step's batch varies, so `attn["q"]` is a concatenation
    with no recoverable row-per-step shape.
    """
    where = _loc_index(store)
    keys_nope = store["nope"].float()[:, : config.NOPE_DIM]
    q_all = attn["q"].float()
    idx_all = attn["idx"].long()
    # Captures written before the hook normalized the index tensor carry the
    # backend's singleton query-position dim: (rows, 1, topk). Unwrap it rather
    # than rejecting the capture -- the entries themselves are correct.
    if idx_all.ndim == 3 and idx_all.shape[1] == 1:
        idx_all = idx_all[:, 0]
    if idx_all.ndim != 2:
        raise ValueError(
            f"attention capture idx is {tuple(idx_all.shape)}; expected (rows, topk)"
        )
    lens_all = attn["lens"].long()
    if "batch" not in attn:
        raise ValueError(
            "attention capture has no per-step batch sizes; it predates the "
            "window fix and cannot be split into steps -- recapture"
        )
    sizes = [int(n) for n in attn["batch"].tolist()]
    if sum(sizes) != q_all.shape[0]:
        raise ValueError(
            f"batch sizes sum to {sum(sizes)} but q holds {q_all.shape[0]} rows"
        )

    num_basis = den = 0.0
    num_quant = 0.0
    used = missing = 0

    q_steps = torch.split(q_all, sizes)
    idx_steps = torch.split(idx_all, sizes)
    lens_steps = torch.split(lens_all, sizes)
    for q, rows, lengths in zip(q_steps, idx_steps, lens_steps):
        for b in range(rows.shape[0]):
            n = int(lengths[b])
            if n == 0:
                continue
            slots = rows[b, :n].tolist()
            keep = [i for i, s in enumerate(slots) if s in where]
            missing += len(slots) - len(keep)
            if not keep:
                continue
            k_true = keys_nope[[where[slots[i]] for i in keep]]  # [n_keep, 448]
            used += len(keep)
            # Logits are per (query, selected entry): every head of this batch
            # row scores the same key set, which is what the sparse kernel does.
            q_nope = q[b][:, : config.NOPE_DIM]  # [heads, 448]
            s_true = q_nope @ k_true.T           # [heads, n_keep]
            k_hat = reference.reconstruct(reference.encode(k_true, d), d)
            s_hat = q_nope @ k_hat.T
            num_basis += (s_true - s_hat).pow(2).sum().item()
            den += s_true.pow(2).sum().item()
            codes, scales = reference.quantize(reference.encode(k_true, d))
            k_q = reference.reconstruct(
                reference.dequantize(codes, scales, d.shape[-1]), d
            )
            num_quant += (s_true - q_nope @ k_q.T).pow(2).sum().item()
    cov = used / max(used + missing, 1)
    # A number computed on a sliver of the selected entries is not a logit error,
    # it is a logit error on a sliver -- and it is the kind of number that gets
    # quoted once it exists. Below the floor, report the coverage and refuse the
    # value, so an unusable join shows up as an unusable join.
    if den == 0.0 or cov < MIN_COVERAGE:
        return {
            "eps_basis": None,
            "eps_quant": None,
            "entries": used,
            "unmatched_entries": missing,
            "coverage": cov,
        }
    return {
        "eps_basis": (num_basis / den) ** 0.5,
        "eps_quant": (num_quant / den) ** 0.5,
        "entries": used,
        "unmatched_entries": missing,
        "coverage": cov,
    }


# --- rank profile -----------------------------------------------------------


def rank_profile(table: dict, target: float, key: str = "heldout") -> dict:
    """Smallest rank per layer meeting `target` retention under `key`."""
    for rank in sorted(table):
        row = table[rank]
        stat = row.get(key)
        if stat is not None and stat["mean"] >= target:
            return {"rank": rank, "bytes": row["bytes"], "retention": stat["mean"]}
    return None


# --- driver -----------------------------------------------------------------


def analyze_layer(store: dict, attn, ranks, target: float) -> dict:
    store = stored_rows(store)
    table = retention_table(store, ranks)
    out = {
        "rows": int(store["nope"].shape[0]),
        "retention": table,
        "rank_profile_heldout": rank_profile(table, target, "heldout"),
        "rank_profile_drift": rank_profile(table, target, "drift"),
    }
    if attn is not None:
        x = store["nope"].float()
        half = x.shape[0] // 2
        out["score"] = {}
        for rank in sorted(table):
            # Score a basis fit on the *first half only*: fitting on the very
            # rows being scored would make eps_basis an oracle bound and hide
            # the failure the logit metric exists to catch.
            if half < rank:
                out["score"][rank] = None
                continue
            d = reference.fit_basis(x[:half], rank)
            out["score"][rank] = score_error(store, attn, d)
    return out


def _profile_line(
    key: str, table: dict, target: float, prof: dict | None, split: str
) -> str:
    """One rank-profile verdict, keeping "short of target" apart from "unmeasured".

    Both used to print as `none`, which reads as "the latent is not low-rank
    enough" when it can equally mean "the basis could not be fit at this sample
    size" -- a statement about the capture, not about the model. The two call
    for opposite responses (raise the rank vs. widen the window), so they must
    not share a rendering.
    """
    measured = sorted(r for r, row in table.items() if row.get(key))
    head = f"smallest rank at {key} >= {target:.3f}: "
    if not measured:
        return head + (
            f"NOT MEASURABLE -- no sampled rank could be fit: this split holds "
            f"back only {split} of the sample, so it needs a rank well below "
            f"the row count"
        )
    span = f"r={measured[0]}..{measured[-1]}"
    if prof is None:
        return head + f"none -- every measured rank ({span}) fell short"
    return (
        head + f"{prof['rank']} ({prof['bytes']} B, retention "
        f"{prof['retention']:.4f}); measured over {span}"
    )


def format_markdown(results: dict, ranks, target: float, capture_dir: str) -> str:
    lines = [
        "# STAR-CSA spectrum",
        "",
        f"Capture: `{capture_dir}`",
        f"Retention target for the rank profile: **{target:.3f}**",
        "",
        "`selffit` fits and scores the same rows (the oracle ceiling). `heldout`",
        "fits on one half and scores the other (a basis frozen before these",
        "latents existed). `drift` fits on the earliest quarter of the capture",
        "window and scores the latest (what a longer session costs). `eps_*` is",
        "the relative logit error on the entries the model actually selected.",
        "",
    ]
    for layer, res in results.items():
        lines += [f"## Layer {layer}", "", f"rows sampled: {res['rows']}", ""]
        lines.append("| r | B | selffit | heldout | drift | eps_basis | eps_quant |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for rank in sorted(res["retention"]):
            row = res["retention"][rank]

            def fmt(stat):
                return "--" if stat is None else f"{stat['mean']:.4f}"

            sc = (res.get("score") or {}).get(rank) or {}
            eps_b = sc.get("eps_basis")
            eps_q = sc.get("eps_quant")
            lines.append(
                f"| {rank} | {row['bytes']} | {fmt(row['selffit'])} | "
                f"{fmt(row['heldout'])} | {fmt(row['drift'])} | "
                f"{'--' if eps_b is None else f'{eps_b:.4f}'} | "
                f"{'--' if eps_q is None else f'{eps_q:.4f}'} |"
            )
        prof = res["rank_profile_heldout"]
        drift_prof = res["rank_profile_drift"]
        lines += [
            "",
            _profile_line("heldout", res["retention"], target, prof, "half"),
            "",
            _profile_line("drift", res["retention"], target, drift_prof, "a quarter"),
            "",
        ]
    missing = [layer for layer, res in results.items() if "score" not in res]
    if missing:
        lines += ["## Layers with no attention capture", "", ", ".join(map(str, missing)), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture_dir", help="starkv/captures/<tag>")
    ap.add_argument("--ranks", default="128,192,256,320,384,448")
    ap.add_argument("--target", type=float, default=0.99)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    ranks = [int(r) for r in args.ranks.split(",") if r]
    found = discover(args.capture_dir)
    if not found:
        raise SystemExit(f"no store captures under {args.capture_dir}")

    results = {}
    for layer, paths in found.items():
        store = load_store(paths["store"])
        attn = load_attn(paths["attn"]) if paths.get("attn") else None
        results[layer] = analyze_layer(store, attn, ranks, args.target)
        print(f"layer {layer}: {results[layer]['rows']} rows")

    md = format_markdown(results, ranks, args.target, args.capture_dir)
    out_base = args.out or os.path.join(
        config.results_dir(), f"spectrum-{os.path.basename(args.capture_dir.rstrip('/'))}"
    )
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    with open(out_base + ".md", "w") as f:
        f.write(md)
    with open(out_base + ".json", "w") as f:
        json.dump({"capture": args.capture_dir, "target": args.target,
                   "ranks": ranks, "layers": results}, f, indent=2, default=str)
    print(md)
    print(f"\nwrote {out_base}.md and {out_base}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
