"""Accepted rank-64 training and continuation; model weights are never loaded.

Author: Winston Cai.
"""

import argparse
import copy
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import LAYERS, load_runner, parse_layers
from .data.batches import StateBatches, TrainingState
from .data.dataset import load_cached_layer
from .losses import (
    _select_training_objective,
    forward_error,
    measure,
    validation_acceptable,
)
from .models import ExpandedConditionalScore, StaticScore, extend_parameters, read_checkpoint, read_policy


def parameter_penalty(model):
    penalty = (
        0.001 * model.theta.square().mean() + 0.001 * model.decoder.square().mean()
    )
    penalty = penalty + 0.001 * model.decoder_even.square().mean()
    penalty = (
        penalty
        + 0.001 * model.extra_decoder.square().mean()
        + 0.001 * model.extra_decoder_even.square().mean()
    )
    return penalty


def expanded_parameter_groups(model):
    """Author: Winston Cai. Shared rank-64 optimizer geometry, before scaling."""
    return [
        {"params": [model.theta], "lr": 0.003},
        {"params": [model.encoder, model.extra_encoder], "lr": 0.001},
        {
            "params": [
                model.decoder,
                model.decoder_even,
                model.extra_decoder,
                model.extra_decoder_even,
            ],
            "lr": 0.003,
        },
    ]


def _checkpoint_snapshot(
    model,
    selected_state,
    *,
    mode,
    gradient_mode,
    initial_theta,
    warm_start,
    metrics,
    history,
):
    """Serialize the existing progress contract separately from optimization."""
    return {
        "mode": mode,
        "selected_parameters": (
            None
            if selected_state is None
            else {k: v.tolist() for k, v in selected_state.items()}
        ),
        "gradient_mode": gradient_mode,
        "current_parameters": {
            k: v.detach().tolist() for k, v in model.state_dict().items()
        },
        "warm_theta": initial_theta.tolist(),
        "warm_mode": mode if warm_start else "static",
        "optimizer_reinitialized": warm_start,
        "optimizer_step_guard": False,
        "development_selected": metrics,
        "history": history,
    }


def fit(
    math,
    ref,
    train: list[TrainingState],
    development: list[TrainingState],
    initial_theta: torch.Tensor,
    *,
    mode: str,
    epochs: int = 40,
    bound: float = 0.5,
    seed: int = 42,
    checkpoint=None,
    initial_parameters: dict | None = None,
    learning_rate_scale: float = 1.0,
    temperature_start: float = 0.03,
    temperature_end: float = 0.015,
    train_batch_size: int = 1,
    gradient_mode: str = "straight_through",
    device: torch.device | str | None = None,
    scratch_root: Path | None = None,
) -> dict:
    """Original rank-64 Adam fit, restricted to the three accepted gradient paths.

    Issue weighting, summation order, initialization, codec and scalar development
    selection are preserved. Inputs stay caller-owned; only the current batch
    is staged to device (default: the original training-state device). Anchored
    fits use temporary file-backed residuals under scratch_root or system temp.
    ``mode`` is a historical API/checkpoint field; only conditional_expanded is
    supported. Legacy guard fields remain in outputs for artifact compatibility.
    """
    if (
        mode not in {"conditional_expanded"}
        or not 0 < bound <= 1
        or (not 1 <= epochs <= 100)
    ):
        raise ValueError("explicit scoring mode and bounded trust region required")
    if gradient_mode not in {"straight_through", "relaxed", "anchored"}:
        raise ValueError("explicit supported training gradient mode required")
    if not isinstance(train_batch_size, int) or not 1 <= train_batch_size <= 16:
        raise ValueError("explicit bounded training batch size required")
    if (
        not 0 < learning_rate_scale <= 1
        or not 0 < temperature_end <= temperature_start <= 0.15
    ):
        raise ValueError(
            "bounded learning-rate scale and descending positive temperatures required"
        )
    if not train or not development or any((s.get("role") != "train" for s in train)):
        raise ValueError("nonempty labeled training and development sets required")
    if {s["rid"] for s in train} & {s["rid"] for s in development}:
        raise ValueError("issue leakage")
    device = torch.device(device) if device is not None else train[0]["raw"].device
    initial_theta = initial_theta.detach().to(device)
    if (
        initial_theta.shape != (512,)
        or not bool(torch.isfinite(initial_theta).all())
        or float(initial_theta.abs().max()) > bound + 1e-07
    ):
        raise ValueError("invalid initial weights")
    model = ExpandedConditionalScore(initial_theta, seed=seed)
    if initial_parameters is not None:
        expected = model.state_dict()
        if set(initial_parameters) != set(expected):
            raise ValueError("warm model parameter names changed")
        restored = {
            key: torch.as_tensor(value, device=device, dtype=expected[key].dtype)
            for key, value in initial_parameters.items()
        }
        if any(
            (
                restored[key].shape != expected[key].shape
                or not bool(torch.isfinite(restored[key]).all())
                for key in expected
            )
        ):
            raise ValueError("warm model parameter shapes/values invalid")
        if "theta" in restored and (
            not torch.equal(restored["theta"], initial_theta)
            or float(restored["theta"].abs().max()) > bound + 1e-07
        ):
            raise ValueError(
                "warm model theta disagrees with declared initialization/bound"
            )
        model.load_state_dict(restored)
    trainable_parameters = tuple((p for p in model.parameters() if p.requires_grad))
    warm = (
        StaticScore(initial_theta)
        if initial_parameters is None
        else copy.deepcopy(model)
    )
    baseline = StaticScore(torch.zeros_like(initial_theta))
    baseline_metrics, _ = measure(math, ref, development, baseline)
    warm_metrics, _ = measure(math, ref, development, warm)
    if initial_parameters is not None and (
        not validation_acceptable(warm_metrics, baseline_metrics)
    ):
        raise ValueError("warm candidate violates the TopMag mean/p95 guard")
    candidate_preflight = []
    if device.type == "cuda":
        cpu_model = copy.deepcopy(model).cpu()
        with torch.no_grad():
            samples = [train[0], development[0]]
            with StateBatches(samples, device, 1) as checks:
                for index, state in enumerate(samples):
                    cpu_state = {
                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in state.items()
                    }
                    cpu_error = forward_error(math, ref, cpu_state, cpu_model)
                    with checks.stage([index]) as batch:
                        gpu_error = forward_error(math, ref, batch.states[0], model)
                        torch.testing.assert_close(
                            gpu_error.cpu(), cpu_error, atol=2e-06, rtol=2e-05
                        )
                        candidate_preflight.append(
                            {
                                "rid": state["rid"],
                                "role": state["role"],
                                "cpu_error": float(cpu_error),
                                "cuda_error": float(gpu_error),
                            }
                        )
        del cpu_model
    with StateBatches(train, device, train_batch_size, scratch_root) as batches:
        objective = _select_training_objective(
            math, ref, model, batches, warm, gradient_mode
        )
        # Preserve the historical STE gradient check, independently of the recipe.
        preflight_temperature = (
            0.03 if gradient_mode == "straight_through" else temperature_start
        )
        with batches.stage([0]) as batch:
            exact = forward_error(math, ref, batch.states[0], model)
            surrogate = objective.scalar(batch, preflight_temperature)
            if gradient_mode == "straight_through":
                torch.testing.assert_close(
                    surrogate.detach(), exact.detach(), atol=0, rtol=0
                )
            gradients = torch.autograd.grad(surrogate, trainable_parameters)
            if not all((bool(torch.isfinite(g).all()) for g in gradients)):
                raise ValueError("candidate gradient preflight failed")
            del exact, surrogate, gradients
        batch_check = None
        if train_batch_size > 1:
            with batches.stage(range(min(train_batch_size, len(train)))) as batch:
                batch_check = objective.preflight(batch, temperature_start)
        groups = expanded_parameter_groups(model)
        for group in groups:
            group["lr"] *= learning_rate_scale
        optimizer = torch.optim.Adam(groups)

        best_metrics = warm_metrics
        best = None
        initial_metrics = (
            warm_metrics
            if initial_parameters is not None
            else measure(math, ref, development, model)[0]
        )
        if (
            validation_acceptable(initial_metrics, baseline_metrics)
            and initial_metrics["mean"] < best_metrics["mean"]
        ):
            best, best_metrics = (copy.deepcopy(model.state_dict()), initial_metrics)
        train_baseline = []
        with torch.no_grad():
            for index in range(len(train)):
                with batches.stage([index], with_anchors=False) as batch:
                    train_baseline.append(
                        float(forward_error(math, ref, batch.states[0], baseline))
                    )
        counts = Counter(state["rid"] for state in train)
        rng, history = (np.random.default_rng(seed), [])
        for epoch in range(epochs):
            temperature = temperature_start * (temperature_end / temperature_start) ** (
                epoch / max(epochs - 1, 1)
            )
            optimizer.zero_grad()
            losses = []
            order = rng.permutation(len(train))
            for start in range(0, len(train), train_batch_size):
                indices = order[start : start + train_batch_size]
                with batches.stage(indices) as batch:
                    if train_batch_size == 1:
                        index = indices[0]
                        error = objective.scalar(batch, temperature)
                        loss = (
                            error + 2 * torch.relu(error - train_baseline[index])
                        ) / (len(counts) * counts[train[index]["rid"]])
                    else:
                        error = objective.batch(batch, temperature)
                        base = error.new_tensor(
                            [train_baseline[index] for index in indices]
                        )
                        weights = error.new_tensor(
                            [
                                1 / (len(counts) * counts[train[index]["rid"]])
                                for index in indices
                            ]
                        )
                        loss = ((error + 2 * torch.relu(error - base)) * weights).sum()
                    loss.backward()
                    losses.append(loss.detach())
                    del error, loss
            # Preserve the original Python summation order and FP64 rounding.
            total = sum(torch.stack(losses).cpu().tolist())
            regularizer = parameter_penalty(model)
            regularizer.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                model.theta.clamp_(-bound, bound)
            saturation = float((model.theta.abs() >= bound - 1e-06).float().mean())
            metrics, _ = measure(math, ref, development, model)
            accepted = (
                validation_acceptable(metrics, baseline_metrics)
                and metrics["mean"] < best_metrics["mean"]
            )
            if accepted:
                best, best_metrics = (copy.deepcopy(model.state_dict()), metrics)
            history.append(
                {
                    "epoch": epoch + 1,
                    "temperature": temperature,
                    "train_objective": total,
                    "development": metrics,
                    "checkpoint_selected": accepted,
                    "bound_fraction": saturation,
                    "step_guard": None,
                }
            )
            selected_state = (
                (None if initial_parameters is None else warm.state_dict())
                if best is None
                else best
            )
            if checkpoint is not None or epoch + 1 == epochs:
                snapshot = _checkpoint_snapshot(
                    model,
                    selected_state,
                    mode=mode,
                    gradient_mode=gradient_mode,
                    initial_theta=initial_theta,
                    warm_start=initial_parameters is not None,
                    metrics=best_metrics,
                    history=history,
                )
                if checkpoint is not None:
                    checkpoint(snapshot)
            print(
                f"{mode} epoch {epoch + 1}/{epochs}: {metrics['mean']:.8f}, selected={accepted}",
                flush=True,
            )
        if best is None:
            selected_model = warm
        else:
            model.load_state_dict(best)
            selected_model = model
        selected_metrics, records = measure(math, ref, development, selected_model)
        result = {
            **snapshot,
            "development_baseline": baseline_metrics,
            "development_warm": warm_metrics,
            "candidate_preflight": candidate_preflight,
            "batch_preflight": batch_check,
            "train_batch_size": train_batch_size,
            "development_selected": selected_metrics,
            "selected_records": records,
            "fallback_to_warm": best is None,
            "reduction_percent": 100
            * (1 - selected_metrics["mean"] / baseline_metrics["mean"]),
        }
        if objective.anchors is not None:
            result.update(
                codec_anchor_policy="frozen_warm",
                codec_anchor_queries=len(train),
                codec_anchor_uses_development=False,
                codec_anchor_bytes=batches.anchor_bytes,
            )
        return result


def run(
    cache: Path,
    policy: Path,
    runner: Path,
    output: Path,
    *,
    recipe: str,
    layers: tuple[int, ...],
    device: str = "cpu",
    threads: int = 4,
) -> dict:
    """Fit requested layers and publish their complete inherited policy."""
    recipes_path = Path(__file__).parent / "configs/recipes.json"
    recipes = json.loads(recipes_path.read_text())
    if recipe not in recipes or threads < 1 or device not in ("cpu", "cuda"):
        raise ValueError(
            "known recipe, positive thread count and explicit cpu/cuda device required"
        )
    parse_layers(",".join(map(str, layers)))
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("training outputs must be outside the repository")
    if output.exists():
        raise ValueError(
            "new output directory required; historical runs are never overwritten"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no silent CPU fallback")
    math = load_runner(runner)
    ref = math.reference()
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    selected = read_policy(policy)
    missing_layers = sorted(set(layers) - selected.keys())
    if missing_layers:
        raise ValueError(f"policy missing requested layers: {missing_layers}")
    records = {}
    for layer, entry in selected.items():
        record = read_checkpoint(entry)
        if layer in layers:
            records[layer] = record
    configuration = {
        "recipe": recipe,
        "settings": recipes[recipe],
        "layers": list(layers),
        "cache": str(cache.resolve()),
        "policy": str(policy.resolve()),
        "runner": str(runner.resolve()),
        "torch": str(torch.__version__),
        "device": device,
        "threads": threads,
        "fresh_test_completed": False,
    }
    output.mkdir(parents=True)
    math.atomic_json(output / "config.json", configuration)
    completed = []
    for layer in layers:
        states = load_cached_layer(cache, layer)
        parameters = records[layer]["selected_parameters"]
        if recipe == "rank64":
            parameters = extend_parameters(parameters, 42 + layer)
        result = fit(
            math,
            ref,
            states["train"],
            states["development"],
            torch.tensor(parameters["theta"], device=device),
            mode="conditional_expanded",
            seed=42 + layer,
            initial_parameters=parameters,
            device=device,
            scratch_root=output,
            **recipes[recipe],
            checkpoint=lambda partial: math.atomic_json(
                output / f"partial-layer-{layer:02d}.json", partial
            ),
        )
        result["layer"] = layer
        checkpoint = output / f"selected-layer-{layer:02d}.json"
        math.atomic_json(checkpoint, result)
        selected[layer] = {
            "layer": layer,
            "path": str(checkpoint.resolve()),
        }
        completed.append(layer)
        math.atomic_json(
            output / "progress.json",
            {"layers": completed, "complete": False},
        )
    published = []
    for layer, entry in sorted(selected.items()):
        read_checkpoint(entry)
        published.append(
            {**entry, "path": Path(entry["path"]).name} if layer in layers else entry
        )
    policy_path = output / "policy.json"
    # The policy is the final handoff marker, not merely the last completed fit.
    # A failed fit or publication must not advertise a complete run.
    owned_policy = None
    try:
        with tempfile.TemporaryDirectory(prefix=".policy-", dir=output) as temporary:
            staged = Path(temporary) / "policy.json"
            math.atomic_json(
                staged,
                {
                    "selected": published,
                    "trained_layers": list(layers),
                    "fresh_test_completed": False,
                },
            )
            owned_policy = staged.stat()
            math.atomic_json(
                output / "progress.json", {"layers": completed, "complete": True}
            )
            try:
                os.link(staged, policy_path)
            except FileExistsError:
                # Even a hard link to our staged inode is an unowned collision.
                owned_policy = None
                raise
    except BaseException:
        # Cover signals after link() and scratch cleanup failures after success.
        # The staged file may already be gone; retain its inode identity above.
        try:
            if owned_policy is not None and os.path.samestat(
                owned_policy, policy_path.lstat()
            ):
                policy_path.unlink()
        except FileNotFoundError:
            pass
        math.atomic_json(
            output / "progress.json", {"layers": completed, "complete": False}
        )
        raise
    return {"layers": completed, "output": str(output), "policy": str(policy_path)}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "policy", "runner", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--recipe",
        choices=("rank64", "continuation", "relaxed", "anchored"),
        required=True,
    )
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(arguments)
    return run(
        args.cache,
        args.policy,
        args.runner,
        args.output,
        recipe=args.recipe,
        layers=parse_layers(args.layers),
        device=args.device,
        threads=args.threads,
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
