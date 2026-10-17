"""Exact scalar Packed evaluation; no checkpoint selection on fresh test data.

Author: Winston Cai.
"""

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import artifacts
from .data.dataset import load_cached_layer
from .losses import (
    RETENTION_COUNTS,
    forward_error,
    measure,
    packed_rows,
    simulated_packed_rows,
)
from .models import StaticScore, restored_model


# Comparison definitions and aggregation; independent of report presentation.
@dataclass(frozen=True)
class MaskMethod:
    key: str
    label: str
    retained: int
    group_size: int = 512


MASK_METHODS = (
    MaskMethod("topmag_256", "MUSTAFAR adaptation", 256),
    MaskMethod("structured_256", "HieraSparse-inspired, 2:4", 256, 4),
    MaskMethod("learned_256", "Learned", 256),
    MaskMethod("topmag_192", "MUSTAFAR adaptation", 192),
    MaskMethod("structured_192", "Structured extension, 3:8", 192, 8),
    MaskMethod("learned_192", "Learned", 192),
)
SWEEP_METHODS = tuple(
    MaskMethod(
        f"{family}_{retained}", label, retained, 8 if family == "structured" else 512
    )
    for retained in RETENTION_COUNTS
    for family, label in (
        ("topmag", "MUSTAFAR adaptation"),
        ("structured", "Structured N:8"),
        ("learned", "Learned"),
    )
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


def ranking_mask(
    scores: torch.Tensor, retained: int, *, group_size: int = 512
) -> torch.Tensor:
    """Keep global/group magnitudes or learned scores; preserve smallest-TopK ties.

    Inputs are finite CPU FP32 [rows,512]. Structured groups are consecutive
    coordinates in the original pre-RoPE basis, not reordered magnitude bins.
    """
    if (
        scores.ndim != 2
        or scores.shape[-1] != 512
        or scores.dtype != torch.float32
        or scores.device.type != "cpu"
        or not bool(torch.isfinite(scores).all())
        or retained not in RETENTION_COUNTS
        or group_size not in (4, 8, 512)
        or retained * group_size % 512
    ):
        raise ValueError(
            "finite CPU FP32 [rows,512] scores and integral supported mask budget required"
        )
    grouped = scores.reshape(scores.shape[0], 512 // group_size, group_size)
    mask = torch.ones_like(grouped, dtype=torch.bool)
    prune = group_size - retained * group_size // 512
    mask.scatter_(-1, grouped.topk(prune, dim=-1, largest=False).indices, False)
    return mask.reshape_as(scores)


def comparison_masks(
    raw: torch.Tensor,
    scores: torch.Tensor,
    *,
    methods: tuple[MaskMethod, ...] = MASK_METHODS,
) -> dict[str, torch.Tensor]:
    if scores.shape != raw.shape:
        raise ValueError("learned scores must match raw rows")
    return {
        method.key: ranking_mask(
            scores if method.key.startswith("learned") else raw.abs(),
            method.retained,
            group_size=method.group_size,
        )
        for method in methods
    }


def summarize_comparison(
    records: dict[str, list[dict]],
    expected: set[tuple[str, int, int]],
    *,
    methods: tuple[MaskMethod, ...] = MASK_METHODS,
) -> list[dict]:
    """Equal issue/layer means; pooled query-layer p95; paired issue bootstrap."""
    if not expected or set(records) != {m.key for m in methods}:
        raise ValueError(
            "all comparison methods and nonempty expected identities required"
        )
    issues = sorted({rid for rid, _, _ in expected})
    layers = sorted({layer for _, layer, _ in expected})
    expected_groups = {(rid, layer) for rid in issues for layer in layers}
    if {(rid, layer) for rid, layer, _ in expected} != expected_groups:
        raise ValueError("every issue must cover every requested layer")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(issues), size=(BOOTSTRAP_SAMPLES, len(issues))
    )
    summaries, vectors, samples = [], {}, {}
    for method in methods:
        rows = records[method.key]
        keys = [(r["rid"], r["layer"], r["ordinal"]) for r in rows]
        errors = np.asarray([r["error"] for r in rows], dtype=np.float64)
        if (
            len(keys) != len(set(keys))
            or set(keys) != expected
            or not np.isfinite(errors).all()
            or bool((errors < 0).any())
        ):
            raise ValueError(
                f"{method.key}: duplicate, missing, mismatched or nonfinite comparison records"
            )
        groups = defaultdict(list)
        for row in rows:
            groups[row["rid"], row["layer"]].append(row["error"])
        means = {key: float(np.mean(value)) for key, value in groups.items()}
        vector = np.asarray(
            [np.mean([means[rid, layer] for layer in layers]) for rid in issues]
        )
        vectors[method.key] = vector
        samples[method.key] = vector[draws].mean(axis=1)
        summaries.append(
            {
                "method": method.key,
                "label": method.label,
                "retained_features": method.retained,
                "count": len(rows),
                "mean": float(vector.mean()),
                "p95": float(np.quantile(errors, 0.95, method="linear")),
                "mean_ci95": np.quantile(samples[method.key], [0.025, 0.975]).tolist(),
                "issue_means": dict(zip(issues, vector.tolist())),
                "per_layer": {
                    str(layer): {
                        "mean": float(np.mean([means[rid, layer] for rid in issues])),
                        "p95": float(
                            np.quantile(
                                [r["error"] for r in rows if r["layer"] == layer],
                                0.95,
                                method="linear",
                            )
                        ),
                    }
                    for layer in layers
                },
                "scorer_trained_retained_features": (
                    256 if method.key.startswith("learned") else None
                ),
                "structured_pattern": (
                    f"{method.retained * method.group_size // 512}:{method.group_size}"
                    if method.key.startswith("structured")
                    else None
                ),
            }
        )
    for summary in summaries:
        key = summary["method"]
        baseline = f"topmag_{summary['retained_features']}"
        summary["paired_vs_magnitude"] = {
            "baseline": baseline,
            "mean_difference": float((vectors[key] - vectors[baseline]).mean()),
            "mean_difference_ci95": np.quantile(
                samples[key] - samples[baseline], [0.025, 0.975]
            ).tolist(),
        }
    return summaries


# Report presentation; no checkpoint selection or numerical replay here.
def _comparison_reports(output: Path, result: dict) -> None:
    """Only called after complete coverage for all methods/requested layers."""
    fields = ("method", "retained_features", "mean", "p95")
    with output.with_suffix(".csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in result["summary"])
    lines = [
        (
            "# Attention-error retention sweep"
            if result.get("retention_sweep")
            else "# Six-method attention-error comparison"
        ),
        "",
        f"**{result['status']} — reused development data, not a fresh held-out test.**",
        "",
        "| Method | Retained features | Mean relative output error ↓ | p95 relative output error ↓ |",
        "|---|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['label']} | {row['retained_features']} | {100*row['mean']:.4f}% | {100*row['p95']:.4f}% |"
        )
    lines += [
        "",
        f"Coverage: {result['coverage']['issues']} issues × {result['coverage']['layers']} layers; "
        f"{result['coverage']['query_layer_examples']} paired examples per method, all 64 heads.",
        "Mean: queries within each issue/layer, then layers and issues equally. "
        "p95: pooled individual query-layer errors (linear interpolation), not an uncertainty interval.",
        "JSON contains 95% mean and paired-difference intervals from 10,000 whole-issue bootstrap draws (seed 42).",
        (
            "Learned reuses the frozen 256-trained scorer at every budget; no budget-specific retraining."
            if result.get("retention_sweep")
            else "Learned at 192 reuses the frozen 256-trained scorer; it was not retrained at 192."
        ),
        "Named adaptations describe our magnitude masks, not full upstream algorithm reproductions. "
        "Non-256 budgets use an offline codec simulation, not a deployed variable-size record.",
        *(
            [
                "Structured masks always use N:8; the 256-feature point is 4:8, not the earlier 2:4 comparison."
            ]
            if result.get("retention_sweep")
            else []
        ),
        "Frozen attention replay does not establish full-model task quality or serving performance.",
        "",
    ]
    with output.with_suffix(".md").open("x") as stream:
        stream.write("\n".join(lines))


def plot_retention_sweep(result: dict, destination: Path) -> dict:
    """Plot exactly three measured curves; no interpolation or historical data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result.get("retention_sweep") is not True or result["status"] == "failed":
        raise ValueError("validated retention-sweep measurements required for plotting")
    rows = {row["method"]: row for row in result["summary"]}
    if set(rows) != {method.key for method in SWEEP_METHODS}:
        raise ValueError("complete seven-budget, three-method plot data required")
    files = [
        destination / f"attention-error-{metric}.png" for metric in ("mean", "p95")
    ]
    if any(path.exists() for path in files):
        raise ValueError("fresh retention-sweep figure paths required")
    destination.mkdir(parents=True, exist_ok=True)
    for metric, path in zip(("mean", "p95"), files):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        try:
            for family, label, color, marker in (
                ("topmag", "MUSTAFAR adaptation", "#1769aa", "o"),
                ("structured", "Structured N:8", "#d66122", "s"),
                ("learned", "Learned (trained at 256)", "#158158", "^"),
            ):
                ax.plot(
                    RETENTION_COUNTS,
                    [100 * rows[f"{family}_{k}"][metric] for k in RETENTION_COUNTS],
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=5,
                    label=label,
                )
            ax.set(
                xlabel="Retained latent coordinates (of 512)",
                ylabel=f'{"Mean" if metric == "mean" else "p95"} attention-output relative L2 error (%)',
                title=f'{"Mean" if metric == "mean" else "p95"} attention-output error',
                xlim=(48, 464),
                ylim=(0, None),
            )
            ax.set_xticks(RETENTION_COUNTS)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
            coverage = result["coverage"]
            fig.text(
                0.5,
                0.045,
                f'{coverage["issues"]} development issues · {coverage["layers"]} CSA layers · '
                f'{coverage["query_layer_examples"]:,} paired examples · 64 heads\n'
                "Structured: 1:8–7:8 (256 = 4:8, not 2:4). Frozen 256-trained learned scorer.\n"
                "Reused development data; quantized frozen-attention replay, not task-quality results.",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#4d5963",
            )
            fig.tight_layout(rect=(0, 0.16, 1, 1))
            fig.savefig(path, dpi=180, bbox_inches="tight")
        finally:
            plt.close(fig)
    return {
        "matplotlib": matplotlib.__version__,
        "files": {str(path.resolve()): artifacts.sha256(path) for path in files},
    }


def _publish_comparison(
    math, output: Path, result: dict, figure_dir: Path | None
) -> None:
    """Stage reports, publish JSON last, and roll back only our files on failure.

    Ancillary files are not a completed run without the canonical JSON marker.
    Exclusive links avoid overwriting a competing writer. Per-destination scratch
    files also support a figure directory on a different filesystem.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    owned: dict[Path, Path] = {}
    collisions: set[Path] = set()
    with tempfile.TemporaryDirectory(
        prefix=".evaluation-", dir=output.parent
    ) as scratch:
        stage = Path(scratch)
        staged_output = stage / output.name
        pairs = [
            (staged_output.with_suffix(suffix), output.with_suffix(suffix))
            for suffix in (".csv", ".md")
        ]
        try:
            if result.get("retention_sweep"):
                figures = plot_retention_sweep(result, stage / "figures")
                files = {}
                for source, digest in figures["files"].items():
                    source = Path(source)
                    target = figure_dir / source.name
                    files[str(target.resolve())] = digest
                    pairs.append((source, target))
                result["figures"] = {**figures, "files": files}
            _comparison_reports(staged_output, result)
            math.atomic_json(staged_output, result)
            pairs.append((staged_output, output))  # Completion marker is always last.
            for source, target in pairs:
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(
                    prefix=f".{target.name}.", dir=target.parent
                )
                temporary = Path(name)
                owned[target] = temporary
                with os.fdopen(fd, "wb") as stream, source.open("rb") as incoming:
                    shutil.copyfileobj(incoming, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    # Even an existing hard link to our temporary was created
                    # by somebody else, not by this failed publication call.
                    collisions.add(target)
                    raise
        except BaseException:
            for target, temporary in owned.items():
                if target in collisions:
                    continue
                try:
                    current = target.lstat()  # Never follow another writer's symlink.
                    ours = temporary.stat()
                    if (current.st_dev, current.st_ino) == (ours.st_dev, ours.st_ino):
                        target.unlink()
                except FileNotFoundError:
                    pass  # A competing deletion is already safe; preserve other entries.
            raise
        finally:
            for temporary in owned.values():
                temporary.unlink(missing_ok=True)


# Replay orchestration and public command.
def compare_masks(
    cache: Path,
    policy: Path,
    runner: Path,
    output: Path,
    *,
    layers: tuple[int, ...],
    threads: int,
    retention_sweep: bool = False,
    figure_dir: Path | None = None,
) -> dict:
    """Layer-at-a-time, inference-only comparison on the original development set."""
    outputs = [
        output,
        output.with_suffix(".csv"),
        output.with_suffix(".md"),
        output.with_suffix(".partial.json"),
    ]
    methods = SWEEP_METHODS if retention_sweep else MASK_METHODS
    if retention_sweep:
        figure_dir = figure_dir if figure_dir is not None else output.parent
        outputs.extend(
            figure_dir / f"attention-error-{metric}.png" for metric in ("mean", "p95")
        )
        # Missing plotting dependencies should fail before the numerical sweep.
        import matplotlib  # noqa: F401 -- fail before replay if plotting is unavailable
    if output.suffix != ".json" or any(p.exists() for p in outputs):
        raise ValueError("fresh .json/.csv/.md/.partial.json output paths required")
    if any(
        p.resolve().is_relative_to(Path(__file__).resolve().parents[1])
        for p in outputs[:4]
    ):
        raise ValueError("comparison outputs must be outside the repository")
    artifacts.validate_runner(cache, runner)
    config = json.loads((cache / "config.json").read_text())
    progress = json.loads((cache / "progress.json").read_text())
    if (
        progress["status"] != "complete"
        or artifacts.sha256(cache / "config.json") != progress["config_sha256"]
        or artifacts.sha256(cache / "index.json") != progress["index_sha256"]
        or not set(layers) <= set(config["layers"])
    ):
        raise ValueError(
            "complete, unchanged cache metadata and requested layer coverage required"
        )
    index = json.loads((cache / "index.json").read_text())
    # Check every requested checkpoint before doing attention work. Do not tune
    # checkpoints or inherit historical aggregate metrics from their manifests.
    provenance = artifacts.run_provenance(
        cache,
        policy,
        runner,
        device="cpu",
        threads=threads,
        torch_version=str(torch.__version__),
        layers=layers,
    )
    math = artifacts.load_runner(runner, sources=config["numerical_source"])
    ref = artifacts.load_reference(math, config["numerical_source"])
    records = {m.key: [] for m in methods}
    expected, completed = set(), []
    issues = set(config["split"]["development"])
    result = {
        "schema_version": 1,
        "status": "partial",
        "fresh_test_completed": False,
        "split": "development",
        "records": records,
        "completed_layers": completed,
        "definitions": {
            "error": "Frobenius relative L2 across all 64 heads after inverse output RoPE; saved native denominator",
            "mean": "mean queries within issue/layer, then equal layers and equal issues",
            "p95": "pooled individual query-layer errors; NumPy linear interpolation",
            "bootstrap": {
                "unit": "issue",
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "paired_baseline": "magnitude at the same survivor budget",
            },
            "mask_basis": (
                "raw pre-RoPE coordinates; consecutive groups for N:8"
                if retention_sweep
                else "raw pre-RoPE coordinates; consecutive groups for 2:4 and 3:8"
            ),
            "learned_192": "256-trained frozen scorer evaluated with 192 survivors",
            "scope": "Reused development data; numerical replay, not fresh-test significance or serving quality",
        },
        "source": artifacts.package_sources(),
        "provenance": {
            **provenance,
            "cache": str(cache.resolve()),
            "policy": str(policy.resolve()),
            "runner": str(runner.resolve()),
            "numerical_source_sha256": config["numerical_source"],
            "layers": {
                str(layer): {
                    "receipt_sha256": index[str(layer)]["record_sha256"],
                    "payload_sha256": json.loads(
                        (cache / f"layer-{layer:02d}.json").read_text()
                    )["states_sha256"],
                }
                for layer in layers
            },
            "cache_torch": config.get("torch"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "command": [
                sys.executable,
                "-m",
                "training.evaluate",
                "--compare-masks",
                "--cache",
                str(cache.resolve()),
                "--policy",
                str(policy.resolve()),
                "--runner",
                str(runner.resolve()),
                "--output",
                str(output.resolve()),
                "--layers",
                ",".join(map(str, layers)),
                "--threads",
                str(threads),
            ],
        },
    }
    if retention_sweep:
        result.update(retention_sweep=True, retained_features=list(RETENTION_COUNTS))
        result["provenance"]["command"][3] = "--retention-sweep"
        result["provenance"]["command"].extend(
            ["--figure-dir", str(figure_dir.resolve())]
        )
        result["definitions"][
            "learned_budget_transfer"
        ] = "Same 256-trained scorer at every budget; no retraining"
    started = time.monotonic()
    try:
        with torch.no_grad():
            for layer in layers:
                checkpoint = artifacts.read_checkpoint(
                    provenance["checkpoints"][str(layer)]
                )
                model = restored_model(
                    checkpoint["selected_parameters"], "cpu", 42 + layer
                ).eval()
                states = load_cached_layer(cache, layer)
                rows = states["development"]
                keys = {(s["rid"], layer, s["ordinal"]) for s in rows}
                if (
                    len(keys) != len(rows)
                    or {s["rid"] for s in rows} != issues
                    or any(sum(s["rid"] == rid for s in rows) != 10 for rid in issues)
                ):
                    raise ValueError(
                        f"layer {layer}: complete ten-position development coverage required"
                    )
                expected.update(keys)
                for state in rows:
                    identity = f"{state['rid']}/layer-{layer}/query-{state['ordinal']}"
                    denominator = float(state["reference_squared_norm"])
                    if not np.isfinite(denominator) or denominator <= 0:
                        raise ValueError(
                            f"{identity}: finite positive native denominator required"
                        )
                    masks = comparison_masks(
                        state["raw"], model(state["raw"], state["eps"]), methods=methods
                    )
                    # Admit every non-256 simulation only after checking all
                    # three 256 masks against the unchanged production codec.
                    for method in methods:
                        mask = masks[method.key]
                        if method.retained == 256:
                            args = (
                                math,
                                ref,
                                state["raw"],
                                mask,
                                state["weight"],
                                state["eps"],
                                state["freq"],
                            )
                            if not torch.equal(
                                packed_rows(*args), simulated_packed_rows(*args)
                            ):
                                raise ValueError(
                                    f"{identity}/{method.key}: dense codec differs from 256 reference"
                                )
                    for method in methods:
                        mask = masks[method.key]
                        error = float(
                            forward_error(
                                math,
                                ref,
                                state,
                                model,
                                keep_mask=mask,
                                retained_features=method.retained,
                            )
                        )
                        if not np.isfinite(error):
                            raise ValueError(
                                f"{identity}/{method.key}: nonfinite output error"
                            )
                        records[method.key].append(
                            {
                                "rid": state["rid"],
                                "layer": layer,
                                "ordinal": state["ordinal"],
                                "error": error,
                            }
                        )
                completed.append(layer)
                result["elapsed_seconds"] = time.monotonic() - started
                math.atomic_json(output.with_suffix(".partial.json"), result)
                print(
                    f"Mask comparison layer {layer}: {len(completed)}/{len(layers)}, "
                    f'{len(rows)} paired samples; {result["elapsed_seconds"]:.1f}s',
                    flush=True,
                )
                del states, rows, model, masks
        result["summary"] = summarize_comparison(records, expected, methods=methods)
        result["elapsed_seconds"] = time.monotonic() - started
        full = (
            set(layers) == set(artifacts.LAYERS)
            and len(issues) == 10
            and len(expected) == 2100
        )
        result["status"] = "complete" if full else "partial-layer diagnostic"
        result["coverage"] = {
            "issues": len(issues),
            "layers": len(layers),
            "heads": 64,
            "query_layer_examples": len(expected),
            "methods": len(methods),
            "codec_256_equivalence_checks": 3 * len(expected),
        }
        artifacts.assert_run_inputs(provenance, cache, policy, runner)
        _publish_comparison(math, output, result, figure_dir)
    except Exception as error:
        result.update(
            status="failed",
            failure=repr(error),
            elapsed_seconds=time.monotonic() - started,
        )
        math.atomic_json(output.with_suffix(".partial.json"), result)
        raise
    return result


def main(arguments=None):
    """Evaluate an explicitly supplied, hash-bound policy without changing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "policy", "runner", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layers", default=",".join(map(str, artifacts.LAYERS)))
    parser.add_argument("--threads", type=int, default=4)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--compare-masks",
        action="store_true",
        help="six-method development comparison, with frozen 256-trained scorer at both budgets",
    )
    modes.add_argument(
        "--retention-sweep",
        action="store_true",
        help="three-method curves at 64–448 survivors, using consistent N:8 structured groups",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="retention-sweep PNG destination (default: beside output)",
    )
    args = parser.parse_args(arguments)
    if args.threads < 1:
        parser.error("positive thread count required")
    if args.figure_dir is not None and not args.retention_sweep:
        parser.error("--figure-dir requires --retention-sweep")
    if args.output.exists() or args.output.resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ):
        parser.error("new external output file required")
    torch.set_num_threads(args.threads)
    if args.compare_masks or args.retention_sweep:
        return compare_masks(
            args.cache,
            args.policy,
            args.runner,
            args.output,
            layers=artifacts.parse_layers(args.layers),
            threads=args.threads,
            retention_sweep=args.retention_sweep,
            figure_dir=args.figure_dir,
        )
    artifacts.validate_runner(args.cache, args.runner)
    config = json.loads((args.cache / "config.json").read_text())
    math = artifacts.load_runner(args.runner, sources=config["numerical_source"])
    ref = artifacts.load_reference(math, config["numerical_source"])
    layers = artifacts.parse_layers(args.layers)
    provenance = artifacts.run_provenance(
        args.cache,
        args.policy,
        args.runner,
        device="cpu",
        threads=args.threads,
        torch_version=str(torch.__version__),
        layers=layers,
    )
    results = []
    for layer in layers:
        record = artifacts.read_checkpoint(provenance["checkpoints"][str(layer)])
        model = restored_model(record["selected_parameters"], "cpu", 42 + layer)
        states = load_cached_layer(args.cache, layer)
        roles = {}
        for role, rows in states.items():
            selected, pairs = measure(math, ref, rows, model)
            baseline, _ = measure(math, ref, rows, StaticScore(torch.zeros(512)))
            roles[role] = {"selected": selected, "topmag": baseline, "records": pairs}
        results.append({"layer": layer, "roles": roles})
    result = {
        "layers": results,
        "source": artifacts.package_sources(),
        "provenance": provenance,
        "policy_sha256": artifacts.sha256(args.policy),
        "runner_sha256": artifacts.sha256(args.runner),
        "fresh_test_completed": False,
    }
    artifacts.assert_run_inputs(provenance, args.cache, args.policy, args.runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".evaluation-", dir=args.output.parent
    ) as owned:
        staged = Path(owned) / args.output.name
        math.atomic_json(staged, result)
        os.link(staged, args.output)
    return result


if __name__ == "__main__":
    main()
