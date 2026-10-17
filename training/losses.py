"""Unchanged scalar/batched Packed losses and training-only relaxations.

Native targets, sink, codec rounding and RoPE retain the historical contracts.
Scalar Packed replay remains the checkpoint judge.
Author: Winston Cai.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import torch

from .data.batches import StagedBatch, StateBatches, TrainingState

RETENTION_COUNTS = tuple(range(64, 449, 64))


class BudgetGate(torch.autograd.Function):
    """Sigmoid gate with implicit threshold enforcing sum(gates)=256.

    Its Jacobian is for the smooth relaxation only, not discrete Top-K.
    Subtracting the weighted mean accounts for the threshold's dependence
    on the scores and makes gradients invariant to a common score shift.
    """

    @staticmethod
    def forward(ctx, scores, temperature):
        if temperature <= 0 or scores.shape[-1] != 512:
            raise ValueError("positive temperature and 512 scores required")
        low = scores.amin(-1, keepdim=True) - 40 * temperature
        high = scores.amax(-1, keepdim=True) + 40 * temperature
        for _ in range(48):
            threshold = (low + high) / 2
            mass = torch.sigmoid((scores - threshold) / temperature).sum(
                -1, keepdim=True
            )
            low = torch.where(mass > 256, threshold, low)
            high = torch.where(mass > 256, high, threshold)
        gate = torch.sigmoid((scores - (low + high) / 2) / temperature)
        ctx.save_for_backward(gate)
        ctx.temperature = temperature
        return gate

    @staticmethod
    def backward(ctx, gradient):
        (gate,) = ctx.saved_tensors
        w = gate * (1 - gate)
        center = (w * gradient).sum(-1, keepdim=True) / w.sum(
            -1, keepdim=True
        ).clamp_min(1e-30)
        return w * (gradient - center) / ctx.temperature, None


def _validate_mask(
    raw: torch.Tensor, mask: torch.Tensor, retained_features: int
) -> None:
    if retained_features not in RETENTION_COUNTS:
        raise ValueError("retained features must be 64 through 448 in multiples of 64")
    if (
        mask.dtype != torch.bool
        or mask.shape != raw.shape
        or mask.device != raw.device
        or raw.ndim != 2
        or raw.shape[-1] != 512
        or not bool((mask.sum(-1) == retained_features).all())
    ):
        raise ValueError(
            f"bool mask matching raw rows with exactly {retained_features} survivors required"
        )


def simulated_packed_rows(
    math, ref, raw, mask, weight, eps, frequencies, *, retained_features: int = 256
) -> torch.Tensor:
    """Dense offline codec simulation, not a variable-size production record.

    Use the same masked FP32 RMSNorm, BF16 rounding and per-64-coordinate
    UE8M0/E4M3 quantization as the frozen 256-value reference. The evaluator
    requires bit-exact 256-value equivalence before using other budgets.
    """
    _validate_mask(raw, mask, retained_features)
    masked = raw.float().masked_fill(~mask, 0)
    normalized = (
        masked
        * torch.rsqrt(masked.square().mean(-1, keepdim=True) + eps)
        * weight.float()
    ).to(torch.bfloat16)
    quantized, codes = ref._ue8m0_quantize(normalized)
    scales = torch.exp2(codes.to(torch.int32).float() - 127).repeat_interleave(64, -1)
    decoded = (
        (quantized.float() * scales).masked_fill(~mask, 0).to(torch.bfloat16).float()
    )
    return math.rotate(decoded, frequencies).to(torch.bfloat16).float()


def packed_rows(
    math, ref, raw, mask, weight, eps, frequencies, *, retained_features: int = 256
) -> torch.Tensor:
    _validate_mask(raw, mask, retained_features)
    if retained_features != 256:
        return simulated_packed_rows(
            math,
            ref,
            raw,
            mask,
            weight,
            eps,
            frequencies,
            retained_features=retained_features,
        )
    codes, bits, scales = ref.pack_rows_ref(raw, mask, weight, eps)
    return (
        math.rotate(ref.unpack_rows_ref(codes, bits, scales).float(), frequencies)
        .to(torch.bfloat16)
        .float()
    )


def rounded_output(math, value, frequencies):
    return (
        math.rotate(value.to(torch.bfloat16).float(), frequencies, True)
        .to(torch.bfloat16)
        .float()
    )


def aggregate(records):
    groups = defaultdict(list)
    for x in records:
        groups[x["rid"]].append(x)
    return {
        "normalized_mse": float(
            np.mean([np.mean([x["error"] ** 2 for x in v]) for v in groups.values()])
        ),
        "mean": float(
            np.mean([np.mean([x["error"] for x in v]) for v in groups.values()])
        ),
        "p95": float(np.quantile([x["error"] for x in records], 0.95)),
        "issue_means": {
            rid: float(np.mean([x["error"] for x in v]))
            for rid, v in sorted(groups.items())
        },
    }


def forward_error(
    math,
    ref,
    state: TrainingState,
    model,
    temperature=None,
    *,
    retained_features: int = 256,
    keep_mask: torch.Tensor | None = None,
):
    """Exact packed attention loss, differentiating a smooth-mask surrogate."""
    if temperature is not None and (retained_features != 256 or keep_mask is not None):
        raise ValueError("comparison masks and non-256 budgets are inference-only")
    if retained_features not in RETENTION_COUNTS:
        raise ValueError("retained features must be 64 through 448 in multiples of 64")
    if keep_mask is not None:
        mask = keep_mask
    else:
        scores = model(state["raw"], state["eps"])
        if retained_features == 256 and hasattr(model, "hard_mask"):
            mask = model.hard_mask(ref, state["raw"], state["eps"])
        else:
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask.scatter_(
                -1,
                scores.detach()
                .topk(512 - retained_features, dim=-1, largest=False)
                .indices,
                False,
            )
    with torch.no_grad():
        exact = packed_rows(
            math,
            ref,
            state["raw"],
            mask,
            state["weight"],
            state["eps"],
            state["freq"],
            retained_features=retained_features,
        )
    rows = exact
    if temperature is not None:
        if state.get("role") != "train":
            raise ValueError("only training issues may produce gradients")
        soft = BudgetGate.apply(scores, temperature)
        gate = mask.float() + (soft - soft.detach())
        x = state["raw"] * gate
        continuous = math.rotate(
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + state["eps"])
            * state["weight"],
            state["freq"],
        )
        rows = exact + (continuous - continuous.detach())
    output = math.attention(
        state["q"], torch.cat((state["swa"], rows)), state["sink"], state["scale"]
    )[0]
    exact_output = rounded_output(math, output.detach(), state["query_freq"])
    if temperature is None:
        output = exact_output
    else:
        continuous = math.rotate(output, state["query_freq"], True)
        output = exact_output + (continuous - continuous.detach())
    squared = (output - state["target"]).square().sum().to(torch.float64)
    return (squared / state["reference_squared_norm"]).clamp_min(1e-30).sqrt()


def relaxed_forward_error(
    math, state: TrainingState, model, temperature: float
) -> torch.Tensor:
    """Author: Winston Cai. True differentiable fixed-mass training relaxation.

    Fractional gates enter RMSNorm and attention in both forward and backward.
    Unlike the STE, autograd differentiates the value actually returned here.
    No BF16/FP8 rounding is included in this *training-only* proxy; native
    targets/norms are unchanged. As temperature vanishes it approaches the
    continuous hard-mask path (away from ties), NOT the packed codec. Only
    original exact packed replay may judge a candidate's quality.
    """
    if state.get("role") != "train":
        raise ValueError("only training issues may use the continuous relaxation")
    if not 0 < temperature <= 0.15:
        raise ValueError("positive bounded relaxation temperature required")
    gate = BudgetGate.apply(model(state["raw"], state["eps"]), temperature)
    masked = state["raw"] * gate
    rows = math.rotate(
        masked
        * torch.rsqrt(masked.square().mean(-1, keepdim=True) + state["eps"])
        * state["weight"],
        state["freq"],
    )
    output = math.attention(
        state["q"], torch.cat((state["swa"], rows)), state["sink"], state["scale"]
    )[0]
    output = math.rotate(output, state["query_freq"], True)
    squared = (output - state["target"]).square().sum().to(torch.float64)
    return (squared / state["reference_squared_norm"]).clamp_min(1e-30).sqrt()


def codec_residual_anchor(math, ref, state: TrainingState, model) -> dict:
    """Author: Winston Cai. Fixed warm codec offsets for a local training proxy.

    Rows retain the producer/reconstruction codec residual at the warm hard
    mask; output retains final BF16/inverse-RoPE rounding residual. These
    constants are held fixed when differentiating or testing a proposal.
    They are not an exact derivative of quantization and not serving inputs.
    """
    if state.get("role") != "train":
        raise ValueError("only training states may construct codec anchors")
    with torch.no_grad():
        scores = model(state["raw"], state["eps"])
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(-1, scores.topk(256, dim=-1, largest=False).indices, False)
        x = state["raw"] * mask
        continuous = math.rotate(
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + state["eps"])
            * state["weight"],
            state["freq"],
        )
        packed = packed_rows(
            math, ref, state["raw"], mask, state["weight"], state["eps"], state["freq"]
        )
        output = math.attention(
            state["q"], torch.cat((state["swa"], packed)), state["sink"], state["scale"]
        )[0]
        return {
            "state": state,
            "rows": (packed - continuous).detach(),
            "output": (
                rounded_output(math, output, state["query_freq"])
                - math.rotate(output, state["query_freq"], True)
            ).detach(),
        }


def validate_codec_anchor(state, anchor):
    """Author: Winston Cai. Reject reassociated, trainable or malformed offsets."""
    if state.get("role") != "train" or anchor.get("state") is not state:
        raise ValueError("same original training state required for codec anchor")
    for key, expected in (("rows", state["raw"]), ("output", state["target"])):
        value = anchor[key]
        if (
            value.shape != expected.shape
            or value.device != expected.device
            or value.dtype != expected.dtype
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("finite fixed codec residual geometry required")


def anchored_relaxed_forward_error(math, state, model, temperature, anchor):
    """Author: Winston Cai. Differentiate a real fixed-anchor continuous loss.

    Warm residual constants correct the proxy's forward codec bias. The
    sigmoid gate, RMSNorm, joint attention and sink remain differentiable.
    Anchors are training-only, query-bound and never recomputed inside this
    function; original exact packed replay remains the proposal judge.
    """
    validate_codec_anchor(state, anchor)
    if not 0 < temperature <= 0.15:
        raise ValueError("positive bounded relaxation temperature required")
    gate = BudgetGate.apply(model(state["raw"], state["eps"]), temperature)
    x = state["raw"] * gate
    rows = (
        math.rotate(
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + state["eps"])
            * state["weight"],
            state["freq"],
        )
        + anchor["rows"]
    )
    output = math.attention(
        state["q"], torch.cat((state["swa"], rows)), state["sink"], state["scale"]
    )[0]
    output = math.rotate(output, state["query_freq"], True) + anchor["output"]
    squared = (output - state["target"]).square().sum().to(torch.float64)
    return (squared / state["reference_squared_norm"]).clamp_min(1e-30).sqrt()


def stack_states(states, anchors=None):
    if not states:
        raise ValueError("nonempty batch required")
    first = states[0]
    keys = ("raw", "weight", "freq", "q", "swa", "sink", "target", "query_freq")
    for state in states:
        if state["eps"] != first["eps"] or state["scale"] != first["scale"]:
            raise ValueError("same layer epsilon and scale required")
        if any(state[key].shape != first[key].shape for key in keys):
            raise ValueError("uniform fully-admitted shapes required")
    tensors = {key: torch.stack([state[key] for state in states]) for key in keys}
    tensors.update(
        eps=first["eps"],
        scale=first["scale"],
        roles=[state["role"] for state in states],
        reference_squared_norm=torch.tensor(
            [s["reference_squared_norm"] for s in states],
            dtype=torch.float64,
            device=first["raw"].device,
        ),
    )
    if anchors is not None:
        if len(anchors) != len(states):
            raise ValueError(
                "one fixed codec anchor per original training state required"
            )
        for state, anchor in zip(states, anchors):
            validate_codec_anchor(state, anchor)
        tensors["codec_rows"] = torch.stack([a["rows"] for a in anchors])
        tensors["codec_output"] = torch.stack([a["output"] for a in anchors])
    return tensors


def forward_errors(math, ref, batch, model, temperature=None):
    count, selected, width = batch["raw"].shape
    raw = batch["raw"].reshape(-1, width)
    frequencies = batch["freq"].reshape(-1, 32)
    weight = (
        batch["weight"][:, None, :].expand(count, selected, width).reshape(-1, width)
    )
    scores = model(raw, batch["eps"])
    if hasattr(model, "hard_mask"):
        mask = model.hard_mask(ref, raw, batch["eps"])
    else:
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(
            -1, scores.detach().topk(256, dim=-1, largest=False).indices, False
        )
    with torch.no_grad():
        exact = packed_rows(math, ref, raw, mask, weight, batch["eps"], frequencies)
    rows = exact
    if temperature is not None:
        if any(role != "train" for role in batch["roles"]):
            raise ValueError("only training issues may produce gradients")
        soft = BudgetGate.apply(scores, temperature)
        gate = mask.float() + (soft - soft.detach())
        x = raw * gate
        continuous = math.rotate(
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + batch["eps"]) * weight,
            frequencies,
        )
        rows = exact + (continuous - continuous.detach())
    rows = torch.cat((batch["swa"], rows.reshape(count, selected, width)), dim=1)
    logits = torch.bmm(batch["q"], rows.transpose(1, 2)) * batch["scale"]
    probabilities = torch.softmax(
        torch.cat((logits, batch["sink"][..., None]), dim=-1), dim=-1
    )[..., :-1]
    output = torch.bmm(probabilities, rows)
    frequencies = batch["query_freq"][:, None, :]
    exact_output = rounded_output(math, output.detach(), frequencies)
    if temperature is None:
        output = exact_output
    else:
        continuous = math.rotate(output, frequencies, True)
        output = exact_output + (continuous - continuous.detach())
    squared = (output - batch["target"]).square().sum(dim=(-2, -1)).to(torch.float64)
    return (squared / batch["reference_squared_norm"]).clamp_min(1e-30).sqrt()


def relaxed_forward_errors(math, batch, model, temperature, *, anchored=False):
    """Author: Winston Cai. Batched true continuous loss, training only."""
    if any(role != "train" for role in batch["roles"]):
        raise ValueError("only training issues may use the continuous relaxation")
    if not 0 < temperature <= 0.15:
        raise ValueError("positive bounded relaxation temperature required")
    count, selected, width = batch["raw"].shape
    raw = batch["raw"].reshape(-1, width)
    gate = BudgetGate.apply(model(raw, batch["eps"]), temperature)
    masked = raw * gate
    weight = (
        batch["weight"][:, None, :].expand(count, selected, width).reshape(-1, width)
    )
    rows = math.rotate(
        masked
        * torch.rsqrt(masked.square().mean(-1, keepdim=True) + batch["eps"])
        * weight,
        batch["freq"].reshape(-1, 32),
    )
    if anchored:
        rows = rows + batch["codec_rows"].reshape(-1, width)
    rows = torch.cat((batch["swa"], rows.reshape(count, selected, width)), dim=1)
    logits = torch.bmm(batch["q"], rows.transpose(1, 2)) * batch["scale"]
    probabilities = torch.softmax(
        torch.cat((logits, batch["sink"][..., None]), dim=-1), dim=-1
    )[..., :-1]
    output = math.rotate(
        torch.bmm(probabilities, rows), batch["query_freq"][:, None, :], True
    )
    if anchored:
        output = output + batch["codec_output"]
    squared = (output - batch["target"]).square().sum(dim=(-2, -1)).to(torch.float64)
    return (squared / batch["reference_squared_norm"]).clamp_min(1e-30).sqrt()


def relaxed_preflight(math, states, model, temperature, anchors=None):
    """Author: Winston Cai. Independent scalar/batch values and derivatives."""
    scalar = (
        torch.stack(
            [relaxed_forward_error(math, s, model, temperature) for s in states]
        )
        if anchors is None
        else torch.stack(
            [
                anchored_relaxed_forward_error(math, s, model, temperature, a)
                for s, a in zip(states, anchors)
            ]
        )
    )
    vector = relaxed_forward_errors(
        math,
        stack_states(states, anchors),
        model,
        temperature,
        anchored=anchors is not None,
    )
    torch.testing.assert_close(vector, scalar, atol=2e-6, rtol=2e-5)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    scalar_grad = torch.autograd.grad(scalar.sum(), parameters)
    vector_grad = torch.autograd.grad(vector.sum(), parameters)
    for actual, expected in zip(vector_grad, scalar_grad):
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("nonfinite relaxed batch gradient")
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    return {
        "states": len(states),
        "forward_max_abs": float((vector - scalar).detach().abs().max()),
        "gradient_max_abs": max(
            float((a - b).abs().max()) for a, b in zip(vector_grad, scalar_grad)
        ),
        "selection": "Unchanged scalar packed development replay",
    }


def preflight(math, ref, states, model, temperature):
    """Same candidate/input on scalar and batch paths, including gradients."""
    batch = stack_states(states)
    with torch.no_grad():
        scalar = torch.stack([forward_error(math, ref, s, model) for s in states])
        vector = forward_errors(math, ref, batch, model)
    torch.testing.assert_close(vector, scalar, atol=2e-6, rtol=2e-5)
    scalar_loss = sum(forward_error(math, ref, s, model, temperature) for s in states)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    scalar_grad = torch.autograd.grad(scalar_loss, parameters)
    vector_values = forward_errors(math, ref, batch, model, temperature)
    torch.testing.assert_close(vector_values.detach(), vector, atol=0, rtol=0)
    vector_grad = torch.autograd.grad(vector_values.sum(), parameters)
    for actual, expected in zip(vector_grad, scalar_grad):
        if not bool(torch.isfinite(actual).all()):
            raise ValueError("nonfinite batch gradient")
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    return {
        "states": len(states),
        "forward_max_abs": float((vector - scalar).abs().max()),
        "gradient_max_abs": max(
            float((a - b).abs().max()) for a, b in zip(vector_grad, scalar_grad)
        ),
        "selection": "Unchanged scalar packed development replay",
    }


def validation_acceptable(metrics, baseline):
    return metrics["mean"] <= baseline["mean"] and metrics["p95"] <= baseline["p95"]


def measure(math, ref, states: list[TrainingState], model) -> tuple[dict, list[dict]]:
    if not states:
        raise ValueError("nonempty scalar measurement states required")
    with torch.no_grad(), StateBatches(states, model.theta.device, 1) as batches:
        # Scalar arithmetic stays unchanged; transfer the small vector once.
        errors = []
        for index in range(len(states)):
            with batches.stage([index]) as batch:
                errors.append(forward_error(math, ref, batch.states[0], model))
        errors = torch.stack(errors).cpu().tolist()
        records = [
            {"rid": s["rid"], "ordinal": s["ordinal"], "error": error}
            for s, error in zip(states, errors)
        ]
    return aggregate(records), records


class _TrainingObjective(NamedTuple):
    """Losses use the live model and the caller's scoped, ordered batch."""

    scalar: Callable[[StagedBatch, float], torch.Tensor]
    batch: Callable[[StagedBatch, float], torch.Tensor]
    preflight: Callable[[StagedBatch, float], dict]
    anchors: StateBatches | None


def _select_training_objective(
    math,
    ref,
    model: torch.nn.Module,
    states: StateBatches,
    warm: torch.nn.Module,
    gradient_mode: str,
) -> _TrainingObjective:
    """Select once per fit; retain the independent scalar/batched references.

    Callbacks keep the live model, not a parameter snapshot. Only anchored
    training computes residuals, once at the frozen warm model. The batch owner
    preserves original identity and ordering while keeping transfers out of math.
    """
    anchors = None
    if gradient_mode == "straight_through":

        def scalar(staged, temperature):
            return forward_error(math, ref, staged.states[0], model, temperature)

        def batch(staged, temperature):
            return forward_errors(
                math, ref, stack_states(staged.states), model, temperature
            )

        def check(staged, temperature):
            return preflight(math, ref, staged.states, model, temperature)

    elif gradient_mode == "relaxed":

        def scalar(staged, temperature):
            return relaxed_forward_error(math, staged.states[0], model, temperature)

        def batch(staged, temperature):
            return relaxed_forward_errors(
                math, stack_states(staged.states), model, temperature
            )

        def check(staged, temperature):
            return relaxed_preflight(math, staged.states, model, temperature)

    elif gradient_mode == "anchored":
        states.prepare_anchors(
            lambda state: codec_residual_anchor(math, ref, state, warm)
        )
        anchors = states

        def scalar(staged, temperature):
            return anchored_relaxed_forward_error(
                math, staged.states[0], model, temperature, staged.anchors[0]
            )

        def batch(staged, temperature):
            return relaxed_forward_errors(
                math,
                stack_states(staged.states, staged.anchors),
                model,
                temperature,
                anchored=True,
            )

        def check(staged, temperature):
            return relaxed_preflight(
                math, staged.states, model, temperature, staged.anchors
            )

    else:
        raise ValueError("explicit supported training gradient mode required")
    return _TrainingObjective(scalar, batch, check, anchors)
