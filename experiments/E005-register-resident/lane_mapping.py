"""CPU model for the Remnant register-resident lane ownership experiment.

Author: Winston Cai.

This is a mapping preflight, not a CUDA-kernel validity test.  It models the
four feature-group lanes that cooperate on one token in FlashMLA's producer
warp and checks that a requested survivor word comes from the lane that owns
the requester's rank, rather than from a source lane's unrelated rank query.
"""

from __future__ import annotations

import random


HEAD_DIM = 512
SURVIVOR_COUNT = 256
SURVIVOR_SEGMENT = 64
LANE_GROUPS = 4
COORDS_PER_GROUP = 16
COORDS_PER_FRAGMENT = 4


def _prefix_ranks(mask: list[bool]) -> list[int]:
    ranks: list[int] = []
    count = 0
    for kept in mask:
        ranks.append(count)
        count += int(kept)
    return ranks


def _source_lane_response(
    survivors: list[int],
    lane_group: int,
    lane_query_rank: int,
    word_slot: int,
) -> list[int]:
    """Model the flawed implementation's source-local register selection."""
    lane_aligned = lane_query_rank & ~3
    lane_local = lane_aligned % SURVIVOR_SEGMENT
    local_start = lane_local + 4 * word_slot
    result = []
    for byte in range(4):
        local_index = local_start + byte
        index = lane_group * SURVIVOR_SEGMENT + local_index
        result.append(
            survivors[index]
            if local_index < SURVIVOR_SEGMENT and index < len(survivors)
            else 0
        )
    return result


def _requester_rank_response(
    survivors: list[int], request_rank: int, word_slot: int
) -> list[int]:
    """Model the required response: select the word using the requester's rank."""
    aligned = (request_rank & ~3) + 4 * word_slot
    if aligned >= SURVIVOR_COUNT:
        return [0, 0, 0, 0]
    owner_group = aligned // SURVIVOR_SEGMENT
    owner_local = aligned - owner_group * SURVIVOR_SEGMENT
    assert 0 <= owner_group < LANE_GROUPS
    assert owner_local % 4 == 0
    owner_segment_start = owner_group * SURVIVOR_SEGMENT
    return [
        survivors[owner_segment_start + owner_local + byte]
        if owner_segment_start + owner_local + byte < len(survivors)
        else 0
        for byte in range(4)
    ]


def _verify_mask(mask: list[bool]) -> tuple[int, int]:
    assert len(mask) == HEAD_DIM
    assert sum(mask) == SURVIVOR_COUNT

    ranks = _prefix_ranks(mask)
    survivors = [((rank * 73 + 19) & 0xFF) for rank in range(SURVIVOR_COUNT)]
    fragments = 0
    flawed_mismatches = 0

    for tile_start in range(0, HEAD_DIM, 64):
        for fragment_start in range(tile_start, tile_start + 64, COORDS_PER_FRAGMENT):
            keep = mask[fragment_start : fragment_start + COORDS_PER_FRAGMENT]
            if not any(keep):
                continue

            request_rank = ranks[fragment_start]
            aligned = request_rank & ~3
            rank_pack = [
                request_rank + sum(keep[:offset])
                for offset, is_kept in enumerate(keep)
                if is_kept
            ]

            # The corrected ownership rule fetches each aligned word from the
            # 64-survivor segment containing that word's absolute rank.
            first = _requester_rank_response(survivors, request_rank, 0)
            second = _requester_rank_response(survivors, request_rank, 1)
            actual = first + second
            for rank in rank_pack:
                byte_offset = rank - aligned
                assert actual[byte_offset] == survivors[rank]

            # Reproduce the failed shuffle idea: the selected source lane uses
            # its own prefix/rank to choose a word, although the requester
            # needs a word selected by request_rank.
            owner = aligned // SURVIVOR_SEGMENT
            owner_fragment = tile_start + owner * COORDS_PER_GROUP
            owner_rank = ranks[owner_fragment]
            flawed_first = _source_lane_response(survivors, owner, owner_rank, 0)
            flawed_second = _source_lane_response(survivors, owner, owner_rank, 1)
            flawed = flawed_first + flawed_second
            if any(flawed[rank - aligned] != survivors[rank] for rank in rank_pack):
                flawed_mismatches += 1
            fragments += 1

    return fragments, flawed_mismatches


def _make_masks() -> list[list[bool]]:
    patterns = [
        [coordinate % 2 == 0 for coordinate in range(HEAD_DIM)],
        [coordinate % 2 == 1 for coordinate in range(HEAD_DIM)],
        [coordinate < 256 for coordinate in range(HEAD_DIM)],
        [coordinate >= 256 for coordinate in range(HEAD_DIM)],
        [coordinate % 4 in (0, 1) for coordinate in range(HEAD_DIM)],
        [coordinate % 8 < 4 for coordinate in range(HEAD_DIM)],
        [coordinate % 16 < 8 for coordinate in range(HEAD_DIM)],
    ]
    rng = random.Random(20260924)
    for _ in range(1_000):
        selected = set(rng.sample(range(HEAD_DIM), SURVIVOR_COUNT))
        patterns.append([coordinate in selected for coordinate in range(HEAD_DIM)])
    return patterns


def main() -> None:
    total_fragments = 0
    total_flawed_mismatches = 0
    masks = _make_masks()
    for mask in masks:
        fragments, mismatches = _verify_mask(mask)
        total_fragments += fragments
        total_flawed_mismatches += mismatches

    # Explicitly retain the minimal all-kept counterexample: logical group 1
    # requests ranks 16..19, while source group 0's own query starts at rank 0.
    all_kept = [True] * HEAD_DIM
    ranks = _prefix_ranks(all_kept)
    assert ranks[16] == 16 and ranks[0] == 0
    assert _source_lane_response(list(range(256)), 0, ranks[0], 0) != list(range(16, 20))

    assert total_flawed_mismatches > 0
    print(f"masks_checked={len(masks)}")
    print(f"fragments_checked={total_fragments}")
    print(f"flawed_lane_local_rank_mismatches={total_flawed_mismatches}")
    print("requester-rank ownership: PASS")


if __name__ == "__main__":
    main()
