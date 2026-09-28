"""Torch reference for STAR-CSA: encode, reconstruct, latent attention, records.

Everything here is pure torch and runs on CPU, so the maths is testable without
a GPU or a server. ``ops.py`` is the runtime wrapper around it.

The three identities this module exists to pin down:

1. encode/reconstruct   z = D^T x,  x_hat = D z            (round-trip)
2. score identity       q_N^T (D z) == (D^T q_N)^T z       (reconstruct == latent)
3. output absorption    W_o,a (D z_bar) == (W_o,a D) z_bar (z_bar = sum_i p_i z_i)
"""

import os
from typing import Optional, Tuple

import torch

from . import config

fp8_dtype = getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn)

# Per-layer bases, keyed by (layer_id, device). Populated by load_basis() or
# fit_basis()/set_basis(); kept out of the module-level API so tests can build
# bases without touching disk.
_basis: dict = {}
_basis_dir = ""


# --- basis providers --------------------------------------------------------


def fit_basis(x: torch.Tensor, rank: int, center: bool = False) -> torch.Tensor:
    """Rank-`rank` orthonormal basis for the columns of ``x`` ([M, NOPE_DIM]).

    ``D = V[:, :rank]`` from ``x = U S V^T``, i.e. the top-`rank` right singular
    vectors. This is the *self-fit* basis when ``x`` is the latents being
    compressed, and the calibration basis when ``x`` is a held-out capture.
    """
    if x.shape[-1] != config.NOPE_DIM:
        raise ValueError(f"expected last dim {config.NOPE_DIM}, got {x.shape[-1]}")
    x = x.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    # An SVD of M rows yields at most M right singular vectors, so a rank above
    # the row count would silently return a short basis and every downstream
    # shape would be wrong but plausible. Refuse instead.
    if x.shape[0] < rank:
        raise ValueError(
            f"cannot fit rank {rank} from {x.shape[0]} rows; "
            "the basis would be rank-deficient"
        )
    # [M, d] -> V is [d, d]; compute only the leading `rank` columns.
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[:rank].T.contiguous()  # [NOPE_DIM, rank]


def set_basis(layer_id: int, d: torch.Tensor) -> None:
    _basis[layer_id] = d.detach().float().contiguous()


def basis_path(layer_id: int, rank: Optional[int] = None) -> str:
    rank = config.RANK if rank is None else rank
    return os.path.join(_basis_dir, f"D_{layer_id:03d}_r{rank}.pt")


def set_basis_dir(path: str) -> None:
    global _basis_dir
    _basis_dir = path


def load_basis(layer_id: int, rank: Optional[int] = None) -> Optional[torch.Tensor]:
    """Load a frozen global basis from ``basis_dir()``; None if not present."""
    if layer_id in _basis:
        return _basis[layer_id]
    path = basis_path(layer_id, rank)
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu").float()
    _basis[layer_id] = d.contiguous()
    return _basis[layer_id]


def basis_for(layer_id: int, device, rank: Optional[int] = None) -> Optional[torch.Tensor]:
    d = load_basis(layer_id, rank)
    return None if d is None else d.to(device)


def save_basis(layer_id: int, d: torch.Tensor, rank: Optional[int] = None) -> str:
    os.makedirs(_basis_dir or config.basis_dir(), exist_ok=True)
    path = basis_path(layer_id, rank)
    torch.save(d.detach().cpu().float(), path)
    return path


# --- encode / reconstruct ---------------------------------------------------


def encode(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """z = D^T x, mapping [..., NOPE_DIM] -> [..., rank]."""
    return x.float() @ d


def reconstruct(z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """x_hat = D z, mapping [..., rank] -> [..., NOPE_DIM]."""
    return z.float() @ d.T


def retention(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Fraction of per-row energy kept: ||D D^T x||^2 / ||x||^2, shape [M]."""
    x = x.float()
    num = reconstruct(encode(x, d), d).pow(2).sum(-1)
    den = x.pow(2).sum(-1).clamp_min(1e-12)
    return num / den


# --- fp8 record payload -----------------------------------------------------


def quantize(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-64-tile fp8 e4m3 codes + uint8 scale exponents, same scheme as the
    native store's ``quantize_block_size=64`` tiles."""
    n, rank = z.shape
    if rank % config.TILE_SIZE:
        raise ValueError(f"rank {rank} must be a multiple of {config.TILE_SIZE}")
    tiles = rank // config.TILE_SIZE
    info = torch.finfo(fp8_dtype)
    zt = z.float().view(n, tiles, config.TILE_SIZE)
    maxabs = zt.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    exponent = torch.ceil(torch.log2(maxabs / info.max)).clamp(-127.0, 128.0)
    scaled = (zt / (2.0**exponent)).clamp(info.min, info.max)
    codes = scaled.to(fp8_dtype).reshape(n, rank)
    scales = (exponent + 127.0).to(torch.uint8).reshape(n, tiles)
    return codes, scales


def dequantize(codes: torch.Tensor, scales: torch.Tensor, rank: int) -> torch.Tensor:
    n = codes.shape[0]
    tiles = (rank + config.TILE_SIZE - 1) // config.TILE_SIZE
    c = codes.float().view(n, tiles, config.TILE_SIZE)
    s = (2.0 ** (scales.float() - 127.0)).unsqueeze(-1)
    return (c * s).reshape(n, rank)


def pack_record(z: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """Build the ``[z fp8][scales u8][tail bf16][pad]`` byte record per row.

    ``z`` is [n, rank] float, ``tail`` is [n, ROPE_DIM] float (already rotated).
    Returns a [n, BYTES_PER_TOKEN] uint8 tensor.
    """
    n, rank = z.shape
    if rank != config.RANK:
        raise ValueError(f"rank {rank} != config.RANK {config.RANK}")
    codes, scales = quantize(z)
    out = torch.zeros(n, config.BYTES_PER_TOKEN, dtype=torch.uint8, device=z.device)
    out[:, config.Z_OFFSET : config.Z_OFFSET + rank] = codes.view(torch.uint8).view(
        n, rank
    )
    out[:, config.SCALE_OFFSET : config.SCALE_OFFSET + config.SCALE_TILES] = scales
    tail_bf16 = tail.to(torch.bfloat16).view(torch.uint8).view(n, config.ROPE_BYTES)
    out[:, config.ROPE_OFFSET : config.ROPE_OFFSET + config.ROPE_BYTES] = tail_bf16
    return out


def unpack_record(rec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_record: returns (z [n, rank] float32, tail [n, ROPE_DIM])."""
    n = rec.shape[0]
    rank = config.RANK
    codes = rec[:, config.Z_OFFSET : config.Z_OFFSET + rank].contiguous()
    codes = codes.view(fp8_dtype).view(n, rank)
    scales = rec[:, config.SCALE_OFFSET : config.SCALE_OFFSET + config.SCALE_TILES]
    z = dequantize(codes, scales, rank)
    raw = rec[:, config.ROPE_OFFSET : config.ROPE_OFFSET + config.ROPE_BYTES].contiguous()
    tail = raw.view(torch.bfloat16).view(n, config.ROPE_DIM).float()
    return z, tail


def assemble(z: torch.Tensor, tail: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Rebuild the 512-dim cache entry: [D z (448) | tail (64)]."""
    return torch.cat([reconstruct(z, d), tail.float()], dim=-1)


# --- attention paths --------------------------------------------------------


def prewarm(layer_ids, device) -> int:
    """Fault every layer's basis into `device` before the first decode step.

    `basis_for` lazily loads from disk on first use, and the first use is a
    decode step running inside a cuda graph, where a host-side load is a
    capture-time allocation. Returns the number of layers that resolved.
    """
    return sum(1 for lid in layer_ids if basis_for(lid, device) is not None)


# --- the vendor's norm+rope, in torch ---------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim, matching ``fused_norm_rope_inplace_triton``.

    The kernel accumulates sum-of-squares in fp32 over the full 512 dims and
    scales by ``rsqrt(sum_sq / head_dim + eps)``; so does this. The basis and
    the fp8 record are far coarser than any reduction-order difference here.
    """
    x = x.float()
    rms_inv = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / x.shape[-1] + eps)
    return x * rms_inv * weight.float()


def apply_rope(
    x: torch.Tensor, freqs_cis: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    """Rotate the last ROPE_DIM dims by ``freqs_cis[pos]``, interleaved pairs.

    Pair ``j`` is elements ``(2j, 2j+1)`` of the rotary segment, and the
    rotation is ``(a+bi)(c+di)`` -- the same convention as the vendor kernel:
    ``real' = a c - b d``, ``imag' = a d + b c``. Returns only the rotated
    segment, shape [n, ROPE_DIM].
    """
    n = x.shape[0]
    fc = torch.view_as_real(freqs_cis[pos.long()])  # [n, ROPE_DIM//2, 2]
    pairs = x[:, config.NOPE_DIM:].float().view(n, config.ROPE_DIM // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    out = torch.empty_like(pairs)
    out[..., 0] = real * fc[..., 0] - imag * fc[..., 1]
    out[..., 1] = real * fc[..., 1] + imag * fc[..., 0]
    return out.reshape(n, config.ROPE_DIM)


# --- pool traffic -----------------------------------------------------------


def _page_offsets(buf: torch.Tensor, loc: torch.Tensor, page_size: int) -> torch.Tensor:
    """Byte offset of each token in a flat uint8 pool page."""
    page_bytes = buf.shape[-1]
    return (
        (loc.long() // page_size) * page_bytes
        + (loc.long() % page_size) * config.BYTES_PER_TOKEN
    )


def store_records(
    buf: torch.Tensor,
    loc: torch.Tensor,
    records: torch.Tensor,
    page_size: int,
) -> None:
    """Scatter [n, BYTES_PER_TOKEN] records into the pool at `loc`.

    No masking and no compaction: every row is written, which is what the
    vendor's own store does on decode (non-boundary rows are zeroed upstream
    and land on loc 0). Keeps the op fixed-shape and capturable.
    """
    base = _page_offsets(buf, loc, page_size)
    flat = buf.reshape(-1)
    cols = torch.arange(config.BYTES_PER_TOKEN, device=records.device)
    flat[(base[:, None] + cols[None, :]).reshape(-1)] = records.reshape(-1)


def gather_records(
    buf: torch.Tensor, flat_token_ids: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Gather [n, BYTES_PER_TOKEN] records for a flat list of token ids."""
    base = _page_offsets(buf, flat_token_ids, page_size)
    flat = buf.reshape(-1)
    cols = torch.arange(config.BYTES_PER_TOKEN, device=flat_token_ids.device)
    return flat[base[:, None] + cols[None, :]]


def reconstruct_records(records: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Records -> [n, HEAD_DIM] float: ``[D z | tail]``."""
    z, tail = unpack_record(records)
    return torch.cat([reconstruct(z, d), tail], dim=-1)


def latent_scores(
    q_nope: torch.Tensor,
    z: torch.Tensor,
    q_rope: torch.Tensor,
    tail: torch.Tensor,
    d: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """s_i = (D^T q_N)^T z_i + q_R^T c_R,i without materialising D z_i.

    ``q_nope`` [..., NOPE_DIM], ``z`` [..., rank], ``q_rope`` [..., ROPE_DIM],
    ``tail`` [..., ROPE_DIM]. Broadcasting matches the reconstruct path.
    """
    q_latent = q_nope.float() @ d  # [..., rank]
    return scale * (
        q_latent @ z.float().transpose(-1, -2)
        + q_rope.float() @ tail.float().transpose(-1, -2)
    )


def reconstruct_scores(
    q_nope: torch.Tensor,
    z: torch.Tensor,
    q_rope: torch.Tensor,
    tail: torch.Tensor,
    d: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reference score: rebuild D z, then the ordinary q^T k. Same value as
    latent_scores up to fp accumulation order."""
    k = assemble(z, tail, d)
    q = torch.cat([q_nope.float(), q_rope.float()], dim=-1)
    return scale * (q @ k.transpose(-1, -2))


def latent_values(
    probs: torch.Tensor, z: torch.Tensor, tail: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """o = D (sum_i p_i z_i) summed over the NoPE part, with the RoPE part
    aggregated directly. ``probs`` [..., n]; returns [..., HEAD_DIM]."""
    z_bar = probs.float() @ z.float()
    return torch.cat([reconstruct(z_bar, d), probs.float() @ tail.float()], dim=-1)


def absorb_output_projection(w_oa: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Fold D into V4's first output projection: W'_o,a = W_o,a[:, :NOPE] D.

    After this the 448 NoPE dims never need to exist on the read path -- the
    kernel goes straight from the aggregated latent z_bar to the group
    projection. ``w_oa`` consumes the full HEAD_DIM head output, so only its
    NoPE block meets D; the RoPE block is stored exact and needs no folding.
    Returns [..., rank].
    """
    return w_oa.float()[..., : config.NOPE_DIM] @ d.float()
