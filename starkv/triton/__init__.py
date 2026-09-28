"""The fused latent-attention kernel: contract only, not implemented.

The reconstruct MVE in ``ops.py`` stores a rank-r code and rebuilds the full
448 NoPE dims on read, so it pays the same read bandwidth as the native 512-dim
record minus the record's own size. The reason STAR-CSA is interesting is the
step after that: the 448 dims never have to exist at read time at all.

The maths (``reference.latent_scores`` / ``latent_values`` assert both halves):

    score(q, c) = q_N^T (D z) + q_R^T c_R
                = (D^T q_N)^T z   + q_R^T c_R
    out         = (p @ z) @ (W_o D)^T + (p @ c_R) @ W_o[:, 448:]^T

Both factorisations are exact, so a kernel implementing them is numerically
equivalent to the reconstruct path up to accumulation order -- the identities
are pinned by tests, not by this docstring.

What the kernel has to do that the MVE does not
------------------------------------------------
1. **Pre-project the query once per layer.** ``q_hat = D_l^T q_N`` is
   ``[b, h, NOPE_DIM] -> [b, h, r]``, computed outside the block loop and shared
   by every attended entry. This is the single largest saving: the score is then
   ``q_hat @ z^T`` over rank-r codes instead of ``q_N @ k_N^T`` over 448-dim
   keys.
2. **Dequantize in-register, never to memory.** The fp8 code and its per-64-tile
   scale are consumed inside the tile that loaded them; there is no 448-wide
   staging buffer, which is exactly the traffic the MVE still pays.
3. **Keep the two score terms separate.** The RoPE term is a plain bf16 dot
   product against the stored tail; it needs no basis and no dequantization, so
   folding it into the same accumulator is a real saving rather than an
   incidental one.
4. **Aggregate values in rank space.** ``p @ z`` is rank-r and can be lifted
   straight into ``W_o``'s input space through the precomputed ``W_o D``
   (``reference.absorb_output_projection``), so the 448-dim value never lands in
   memory either.

Layout assumptions
------------------
The kernel is written against the same record the MVE stores
(``config.Z_OFFSET`` / ``SCALE_OFFSET`` / ``ROPE_OFFSET`` / ``BYTES_PER_TOKEN``)
and the same sparse-descriptor convention as ``flash_mla_sparse_fwd``: one row
per attended slot in a flat workspace, c4 region first, then SWA, with ``-1``
padding beyond ``topk_length``. Whatever this module ends up doing, that
descriptor and those offsets are the interface it has to keep, because they are
also what ``ops.decode_lowrank`` already produces and what the tests pin.

Why it is not written yet
-------------------------
The rank regime is not chosen. A fused kernel has to fix its tile shapes, its
per-layer rank, and whether the basis is per-layer or shared -- all three are
outputs of the measurement, and writing the kernel first would mean rewriting it
after. ``ops.py`` raises ``NotImplementedError`` on the latent read path for the
same reason.
"""

from .. import config

# No entry point here is callable; this exists so the tree documents the
# contract and so `from .triton import fused_latent_sparse_attention` fails
# loudly rather than silently doing something approximate.
IMPLEMENTED = False

# Tile shapes a first implementation would start from. TILE_SIZE matches the
# native fp8 scale tile, so a scale fits one dimension of the same block.
SCORE_BLOCK_M = 64   # queries per program
SCORE_BLOCK_N = 64   # attended entries per program
RANK_BLOCK = config.TILE_SIZE
NUM_WARPS = 4
NUM_STAGES = 2


def is_available() -> bool:
    return False


def fused_latent_sparse_attention(*args, **kwargs):
    """Not implemented. See the module docstring for the contract."""
    raise NotImplementedError(
        "the latent read path is unimplemented; see starkv/triton/__init__.py "
        "for the contract and use STARKV_RECON=1 for the reconstruct MVE"
    )
