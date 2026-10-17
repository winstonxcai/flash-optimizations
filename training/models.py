"""Accepted sensitivity scorers; checkpoint names and arithmetic are preserved.

Inputs are raw FP32 rows [..., 512] and RMSNorm epsilon. Only row-local inputs
enter the scorer; future queries and native targets are training data only.
Author: Winston Cai.
"""

import torch


def restored_model(parameters: dict, device: str, seed: int) -> torch.nn.Module:
    """Restore the accepted evaluation model with historical trainability flags."""
    theta = torch.tensor(parameters["theta"], device=device)
    if (
        theta.shape != (512,)
        or not bool(torch.isfinite(theta).all())
        or float(theta.abs().max()) > 0.5 + 1e-7
    ):
        raise ValueError("finite bounded 512 warm weights required")
    model = ExpandedConditionalScore(theta, seed=seed).to(device)
    expected = model.state_dict()
    if set(parameters) != set(expected):
        raise ValueError("complete rank-64 conditional warm model required")
    tensors = {
        name: torch.as_tensor(value, device=device, dtype=expected[name].dtype)
        for name, value in parameters.items()
    }
    if any(
        tensors[name].shape != expected[name].shape
        or not bool(torch.isfinite(tensors[name]).all())
        for name in expected
    ):
        raise ValueError("warm parameter shape or finiteness changed")
    model.load_state_dict(tensors)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "theta")
    return model


class StaticScore(torch.nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = torch.nn.Parameter(theta.detach().clone())

    def forward(self, raw, eps):
        return raw.abs().clamp_min(1e-30).log() + self.theta

    def hard_mask(self, ref, raw, eps):
        # Preserve the existing positive-score TopK/tie behavior exactly.
        return ref.topmag_keep_mask(raw.abs() * self.theta.detach().exp(), 0.5)


def conditional_projected_diagonal(
    raw: torch.Tensor,
    log_diagonal: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """diag(P.T H(x) P), treating H(x) as fixed at the reference row.

    P is the unpruned RMSNorm Jacobian with its row-uniform scale omitted.
    This local deletion formula does not guarantee the best joint hard mask.
    """
    if raw.shape != log_diagonal.shape or eps <= 0:
        raise ValueError("one positive diagonal per row and positive epsilon required")
    diagonal = log_diagonal.exp()
    weighted_raw = raw * diagonal
    normalizer = raw.square().sum(-1, keepdim=True) + raw.shape[-1] * eps
    result = (
        diagonal
        - 2 * raw * weighted_raw / normalizer
        + raw.square()
        * (raw * weighted_raw).sum(-1, keepdim=True)
        / normalizer.square()
    )
    return result.clamp_min(1e-12)


class ConditionalScore(torch.nn.Module):
    def __init__(self, theta, rank=16, seed=42, residual_bound=0.5):
        super().__init__()
        if theta.shape != (512,) or rank < 1 or not 0 < residual_bound <= 1:
            raise ValueError("512 weights, positive rank and bounded residual required")
        self.theta = torch.nn.Parameter(theta.detach().clone())
        generator = torch.Generator(device=theta.device).manual_seed(seed)
        self.encoder = torch.nn.Parameter(
            torch.randn(
                512, rank, device=theta.device, dtype=theta.dtype, generator=generator
            )
            / 512**0.5
        )
        self.decoder = torch.nn.Parameter(theta.new_zeros(rank, 512))
        self.residual_bound = residual_bound

    def log_weights(self, raw, eps):
        normalized = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + eps)
        feature = torch.tanh(normalized @ self.encoder)
        residual = self.residual_bound * torch.tanh(feature @ self.decoder)
        return self.theta + residual

    def forward(self, raw, eps):
        log_diagonal = 2 * self.log_weights(raw, eps)
        marginal = conditional_projected_diagonal(raw, log_diagonal, eps)
        return raw.abs().clamp_min(1e-30).log() + 0.5 * marginal.log()


class EvenConditionalScore(ConditionalScore):
    """Include quadratic row-direction features without changing the codec.

    (N(x).T a)^2 = a.T N(x) N(x).T a is an even second-moment feature.
    The former bias-free odd network cannot represent such an adjustment.
    Zero-initializing the extra decoder preserves the old scorer exactly;
    this adds capacity, not a claim that attention sensitivity is even.
    """

    def __init__(self, theta, rank=16, seed=42, residual_bound=0.5):
        super().__init__(theta, rank, seed, residual_bound)
        self.decoder_even = torch.nn.Parameter(theta.new_zeros(rank, 512))

    def log_weights(self, raw, eps):
        normalized = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + eps)
        projection = normalized @ self.encoder
        odd = torch.tanh(projection) @ self.decoder
        even = torch.tanh(projection.square() - 1) @ self.decoder_even
        return self.theta + self.residual_bound * torch.tanh(odd + even)


class ExpandedConditionalScore(EvenConditionalScore):
    """Rank-16 warm features plus 48 new features; preserve checkpoint names."""

    def __init__(self, theta, seed=42):
        super().__init__(theta, rank=16, seed=seed)
        generator = torch.Generator(device=theta.device).manual_seed(seed + 1)
        self.extra_encoder = torch.nn.Parameter(
            torch.randn(
                512, 48, device=theta.device, dtype=theta.dtype, generator=generator
            )
            / 512**0.5
        )
        self.extra_decoder = torch.nn.Parameter(theta.new_zeros(48, 512))
        self.extra_decoder_even = torch.nn.Parameter(theta.new_zeros(48, 512))

    def log_weights(self, raw, eps):
        normalized = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + eps)
        projection = normalized @ self.encoder
        original = (
            torch.tanh(projection) @ self.decoder
            + torch.tanh(projection.square() - 1) @ self.decoder_even
        )
        extra = normalized @ self.extra_encoder
        residual = (
            torch.tanh(extra) @ self.extra_decoder
            + torch.tanh(extra.square() - 1) @ self.extra_decoder_even
        )
        return self.theta + self.residual_bound * torch.tanh(original + residual)


def extend_parameters(parameters, seed):
    if set(parameters) != {"theta", "encoder", "decoder", "decoder_even"}:
        raise ValueError("complete rank-16 quadratic warm scorer required")
    model = ExpandedConditionalScore(torch.tensor(parameters["theta"]), seed=seed)
    if any(
        torch.tensor(value).shape != model.state_dict()[name].shape
        for name, value in parameters.items()
    ):
        raise ValueError("warm parameter geometry changed")
    return {
        **parameters,
        **{
            name: value.tolist()
            for name, value in model.state_dict().items()
            if name.startswith("extra_")
        },
    }
