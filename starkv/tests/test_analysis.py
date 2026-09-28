"""Analysis checks: the spectrum maths and the stability instrumentation.

Runs on CPU: `python -m starkv selftest`. Each test is also a pytest function.

Both modules would happily produce plausible-looking tables from garbage input,
so the fixtures here are built to have a *known* right answer: a capture whose
latents genuinely live in a low-rank subspace, a basis that is exactly exact,
and a stub server whose drift is a linear function of position.
"""

import json
import os

import torch

from .. import config, reference
from ..analysis import capture, spectrum, stability

# --- synthetic capture -------------------------------------------------------


def _concentrated(m=256, r_true=64, seed=0):
    """Latents near an `r_true`-dim subspace, rows first."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(config.NOPE_DIM, r_true, generator=g))[0]
    return torch.randn(m, r_true, generator=g) @ basis.T


def _capture(m=1024, r_true=64, steps=4, batch=2, heads=2, topk=6, seed=0):
    """A store + attn capture pair with a mostly-known join.

    Boundary rows are every other row on decode, so the fixture deliberately
    contains non-boundary rows: if `stored_rows` stopped filtering, the row
    count would double and every downstream number would drift with it.

    `m` is generous because the boundary filter halves it and the drift split
    takes a quarter of that again: the smallest slice any rank in these tests
    gets fitted on is m/8, which has to stay above the rank being fitted or
    `fit_basis` refuses it.
    """
    g = torch.Generator().manual_seed(seed + 1)
    nope = _concentrated(m, r_true, seed)
    loc = torch.arange(1, m + 1, dtype=torch.int64)
    pos = torch.arange(m, dtype=torch.int32)
    # Rows arrive step-major: a decode step writes one row per batch element, so
    # consecutive rows share a step index. `spectrum` splits drift on this axis.
    step = torch.arange(m, dtype=torch.int64) // max(1, batch)
    boundary = torch.zeros(m, dtype=torch.bool)
    boundary[::2] = True
    store = {"layer": 2, "loc": loc, "pos": pos, "step": step, "boundary": boundary,
             "nope": nope, "tail": torch.randn(m, config.ROPE_DIM, generator=g)}

    # Every selected slot is a stored, boundary slot -- except one per step, so
    # the coverage accounting has something to report.
    valid_locs = loc[boundary].tolist()
    idx = torch.zeros(steps * batch, topk, dtype=torch.int32)
    for s in range(steps):
        for b in range(batch):
            row = s * batch + b
            idx[row, : topk - 1] = torch.tensor(valid_locs[: topk - 1])
            idx[row, topk - 1] = 10**6  # never a stored slot
    # Shaped exactly like `capture._flush_attn_locked` writes it: steps
    # concatenated, with the per-step batch sizes beside them.
    attn = {
        "layer": 2,
        "step": torch.arange(steps),
        "batch": torch.full((steps,), batch, dtype=torch.int64),
        "q": torch.randn(steps * batch, heads, config.HEAD_DIM, generator=g),
        "idx": idx,
        "lens": torch.full((steps * batch,), topk, dtype=torch.int32),
    }
    return store, attn


# --- spectrum: row selection -------------------------------------------------


def test_stored_rows_drops_non_boundary_decode_rows():
    store, _ = _capture(m=64)
    kept = spectrum.stored_rows(store)
    assert kept["nope"].shape[0] == 32
    assert kept["loc"].shape[0] == 32 and kept["pos"].shape[0] == 32
    assert bool(kept["boundary"].all())
    # The surviving rows are the even ones, in order.
    assert torch.equal(kept["loc"], store["loc"][::2])


def test_stored_rows_leaves_a_prefill_capture_alone():
    """Prefill stores every row, so the filter must be a no-op there."""
    store, _ = _capture(m=32)
    store["boundary"] = torch.ones(32, dtype=torch.bool)
    kept = spectrum.stored_rows(store)
    assert torch.equal(kept["nope"], store["nope"])
    assert torch.equal(kept["loc"], store["loc"])


def test_clamp_rank_floors_to_a_tile_and_to_the_row_count():
    assert spectrum._clamp_rank(320, 4096) == 320
    assert spectrum._clamp_rank(500, 4096) == 448  # cannot exceed NOPE_DIM
    assert spectrum._clamp_rank(320, 100) == 64    # cannot exceed the rows
    assert spectrum._clamp_rank(30, 4096) == 0     # below one tile
    for rows in (100, 256, 1000):
        for rank in (64, 128, 448, 999):
            got = spectrum._clamp_rank(rank, rows)
            assert got % config.TILE_SIZE == 0 and got <= min(rank, rows)


# --- spectrum: retention -----------------------------------------------------


def test_the_three_retention_flavours_agree_when_the_rank_matches_the_subspace():
    """The split must measure what it says, so on a clean fixture it agrees.

    `_concentrated` puts all the energy in `r_true` directions, so a rank-64
    basis recovers essentially all of it however it was fit -- self-fit,
    held-out, or fit on an early window. All three landing at ~1.0 is the check
    that the three code paths compute the same quantity on a case where the
    right answer is known.
    """
    store, _ = _capture(r_true=64)
    table = spectrum.retention_table(store, [64])
    row = table[64]
    assert row["selffit"]["mean"] > 0.999
    assert row["heldout"]["mean"] > 0.999
    assert row["drift"]["mean"] > 0.999


def test_retention_falls_when_the_rank_is_below_the_true_dimension():
    """Below the intrinsic dimension the fit cannot be exact, and says so.

    Latents built from 128 directions have nothing left over for a rank-64
    basis to keep, so the drop is the truncation and not a fixture artefact.
    """
    store, _ = _capture(r_true=128)
    table = spectrum.retention_table(store, [64, 128])
    assert table[64]["selffit"]["mean"] < 0.95
    assert table[64]["selffit"]["mean"] > 0.2
    assert table[128]["selffit"]["mean"] > 0.99
    assert table[64]["selffit"]["mean"] < table[128]["selffit"]["mean"]


def test_retention_stats_report_min_and_p05_not_just_the_mean():
    """A mean hides the rows a basis fails on, which are the ones that babble."""
    store, _ = _capture(r_true=64)
    row = spectrum.retention_table(store, [64])[64]
    for key in ("selffit", "heldout", "drift"):
        assert set(row[key]) >= {"mean", "min", "p05"}
        assert row[key]["min"] <= row[key]["p05"] + 1e-9
        assert row[key]["p05"] <= row[key]["mean"] + 1e-9
    assert row["drift"]["fit_rows"] > 0 and row["drift"]["eval_rows"] > 0
    assert row["drift"]["axis"] == "decode_step"
    assert row["drift"]["fit_axis_max"] < row["drift"]["eval_axis_min"]


def test_rank_profile_picks_the_smallest_qualifying_rank():
    table = {
        64: {"bytes": 68, "heldout": {"mean": 0.90}, "drift": {"mean": 0.90}},
        128: {"bytes": 132, "heldout": {"mean": 0.995}, "drift": {"mean": 0.99}},
        256: {"bytes": 260, "heldout": {"mean": 0.999}, "drift": {"mean": 0.999}},
    }
    assert spectrum.rank_profile(table, 0.99)["rank"] == 128
    assert spectrum.rank_profile(table, 0.999)["rank"] == 256
    assert spectrum.rank_profile(table, 0.9999) is None


# --- spectrum: logit error ---------------------------------------------------


def test_score_error_separates_the_basis_cost_from_the_quantization_cost():
    """Two costs, two numbers, and a fixture where each is known.

    With the identity basis nothing is truncated, so `eps_basis` must be exactly
    zero and whatever `eps_quant` reports is the fp8 record's cost alone. With a
    rank-64 basis on rank-128 structure the truncation is the dominant term.
    """
    store, attn = _capture(m=64, r_true=32)
    got = spectrum.score_error(store, attn, torch.eye(config.NOPE_DIM))
    assert got["eps_basis"] == 0.0
    assert got["eps_quant"] > 0.0

    store_u, attn_u = _capture(r_true=128)
    part = spectrum.score_error(
        store_u, attn_u, reference.fit_basis(store_u["nope"], 64)
    )
    assert part["eps_basis"] > 0.1
    assert part["eps_quant"] > 0.1


def test_score_error_counts_the_entries_it_could_not_join():
    """Coverage is reported, never assumed: unjoined slots are dropped loudly."""
    store, attn = _capture(m=64, topk=6, steps=2, batch=2)
    got = spectrum.score_error(store, attn, torch.eye(config.NOPE_DIM))
    per_step = 2 * 2 * 5  # steps * batch * (topk - 1) joined slots
    assert got["entries"] == per_step
    assert got["unmatched_entries"] == 2 * 2 * 1
    assert abs(got["coverage"] - 5 / 6) < 1e-9


def test_analyze_layer_end_to_end_produces_every_table():
    store, attn = _capture(r_true=64)
    res = spectrum.analyze_layer(store, attn, [64, 128], 0.99)
    assert res["rows"] == 512  # boundary rows only
    assert set(res["retention"]) == {64, 128}
    assert set(res["score"]) == {64, 128}
    assert res["rank_profile_heldout"]["rank"] == 64
    md = spectrum.format_markdown({2: res}, [64, 128], 0.99, "captures/synthetic")
    assert "| r | B | selffit | heldout | drift |" in md
    assert "smallest rank at heldout" in md


def test_analyze_layer_without_an_attention_capture_still_produces_retention():
    store, _ = _capture(r_true=64)
    res = spectrum.analyze_layer(store, None, [64], 0.99)
    assert "score" not in res
    md = spectrum.format_markdown({2: res}, [64], 0.99, "captures/synthetic")
    assert "| 64 |" in md


# --- stability: pure helpers -------------------------------------------------


def test_position_buckets_cover_the_axis_without_gaps_or_overlap():
    edges = [0, 2047, 2048, 8191, 8192, 16383, 16384, 32767, 32768, 400000]
    got = [stability.bucket_of(p) for p in edges]
    assert got == ["0-2048", "0-2048", "2048-8192", "2048-8192", "8192-16384",
                   "8192-16384", "16384-32768", "16384-32768", "32768+", "32768+"]


def test_drift_table_buckets_by_position_and_averages_only_what_it_saw():
    deltas = [0.0] * 2048 + [0.1] * 6144 + [0.5] * 4096
    agree = [True] * 12288
    agree[2048:] = [False] * (12288 - 2048)
    table = stability.drift_table(deltas, agree)
    assert set(table) == {"0-2048", "2048-8192", "8192-16384"}
    assert table["0-2048"]["n"] == 2048
    assert table["0-2048"]["mean_abs_dlogprob"] == 0.0
    assert table["0-2048"]["top1_agreement"] == 1.0
    assert abs(table["2048-8192"]["mean_abs_dlogprob"] - 0.1) < 1e-12
    assert table["2048-8192"]["top1_agreement"] == 0.0
    assert abs(table["8192-16384"]["max_abs_dlogprob"] - 0.5) < 1e-12


def test_drift_table_omits_top1_agreement_when_it_was_not_measured():
    table = stability.drift_table([0.1, 0.2], None)
    assert table["0-2048"]["top1_agreement"] is None


def test_longest_repeated_ngram_finds_the_repeated_span():
    assert stability.longest_repeated_ngram(list("abcdefabcdef")) == 6
    assert stability.longest_repeated_ngram(list("abcdefghij")) == 0
    assert stability.longest_repeated_ngram([]) == 0
    assert stability.longest_repeated_ngram(["x"]) == 0
    # The babble signature: one token repeated is a repeated n-gram of n-1.
    assert stability.longest_repeated_ngram(["x"] * 50) == 49


def test_degeneration_flags_a_repeated_stream_against_a_varied_one():
    varied = stability.degeneration(" ".join(f"w{i}" for i in range(200)))
    stuck = stability.degeneration(" ".join(["w7"] * 200))
    # Nothing at all repeats in the varied stream, down to a single token.
    assert varied["distinct_1"] == 1.0
    assert varied["longest_repeated_ngram"] == 0
    # One token repeated collapses distinct-n and stretches the repeat to n-1.
    assert stuck["distinct_1"] == 1 / 200
    assert stuck["distinct_2"] == 1 / 199
    assert stuck["longest_repeated_ngram"] == 199


# --- stability: the ladder, against a stub server ----------------------------


class _StubServers:
    """A `/generate` endpoint whose drift is a known function of position.

    The text carries its own token ids (whitespace-separated integers), so the
    stub needs no tokenizer and `score` can be checked against arithmetic rather
    than against itself.
    """

    DRIFT_PER_1K = 0.01

    def __init__(self, vocab=1000):
        self.vocab = vocab

    def _ids(self, text):
        return [int(t) for t in text.split()]

    def _logprob(self, server, position):
        base = -0.001 * (position % self.vocab)
        if "starkv" in server:
            return base - self.DRIFT_PER_1K * (position // 1000)
        return base

    def _next_ids(self, prompt_ids, n):
        return [(prompt_ids[-1] + 1 + i) % self.vocab for i in range(n)]

    def __call__(self, server, path, payload, timeout=3600.0):
        assert path == "/generate"
        ids = self._ids(payload["text"])
        n_new = payload["sampling_params"]["max_new_tokens"]
        start = payload.get("logprob_start_len", 0)
        if n_new:
            gen = self._next_ids(ids, n_new)
            # Only the continuation, as a real `/generate` returns it -- the
            # caller concatenates it onto the prompt, so a full-ids reply would
            # silently double the prompt and desync the token positions.
            return {
                "text": " " + " ".join(str(i) for i in gen),
                "meta_info": {
                    "input_token_logprobs": [
                        [self._logprob(server, i), ids[i], str(ids[i])]
                        for i in range(len(ids))
                    ],
                    "output_token_logprobs": [
                        [self._logprob(server, len(ids) + j), gen[j], str(gen[j])]
                        for j in range(n_new)
                    ],
                    "finish_reason": {"type": "length"},
                },
            }
        return {
            "text": payload["text"],
            "meta_info": {
                "input_token_logprobs": [
                    [self._logprob(server, i), ids[i], str(ids[i])]
                    for i in range(start, len(ids))
                ],
                "input_top_logprobs": [
                    [[self._logprob(server, i), ids[i]]] for i in range(start, len(ids))
                ],
            },
        }


def _two_phase_with_stub(stability_mod, lengths, tag="stub"):
    """Drive run_reference then run_compare, with both legs stubbed.

    Passing the document through `json.dumps`/`loads` between the phases is the
    point: that round trip is what lets the two legs be booted one at a time,
    so anything the compare phase needs must survive serialisation.
    """
    stub = _StubServers()
    saved = stability_mod._post
    stability_mod._post = stub
    try:
        doc = stability_mod.run_reference(
            "http://stub-native", "1 2 3 4", lengths, tag
        )
        doc = json.loads(json.dumps(doc))
        return stability_mod.run_compare("http://stub-starkv", doc, tag)
    finally:
        stability_mod._post = saved


def test_two_phase_scores_every_rung_and_keeps_the_reference_shared():
    res = _two_phase_with_stub(stability, [10, 40])
    assert res["prompt_tokens"] == 4
    assert res["reference"]["n_output_tokens"] == 40
    # One reference generated once, then a prefix per rung -- not one
    # generation per rung, which would compare different sequences.
    assert res["rungs"][10]["scored_positions"] == 10
    assert res["rungs"][40]["scored_positions"] == 40
    assert res["rungs"][10]["max_abs_dlogprob"] == 0.0  # drift starts at 1000
    assert res["rungs"][40]["max_abs_dlogprob"] == 0.0
    assert res["rungs"][40]["by_position"]["0-2048"]["top1_agreement"] == 1.0


def test_the_reference_phase_alone_carries_everything_the_compare_phase_needs():
    """A reference document is a complete handoff, not a partial result."""
    stub = _StubServers()
    saved = stability._post
    stability._post = stub
    try:
        doc = stability.run_reference("http://stub-native", "1 2 3 4", [10, 40], "handoff")
    finally:
        stability._post = saved
    # Everything run_compare reads must be present and JSON-safe.
    rt = json.loads(json.dumps(doc))
    assert rt["span"].startswith("1 2 3 4")
    assert rt["prompt_tokens"] == 4
    assert rt["lengths"] == [10, 40]
    assert len(rt["legs"]["native"]["logprobs"]) == 40
    assert "starkv" not in rt["legs"]  # filled in by the compare phase


def test_two_phase_reports_drift_that_grows_with_position():
    res = _two_phase_with_stub(stability, [2000, 4000], tag="drift")
    table = res["rungs"][4000]["by_position"]
    # positions 0-999 see no drift, 1000-1999 one step, 2000-2999 two, ...
    assert table["0-2048"]["mean_abs_dlogprob"] > 0
    assert table["0-2048"]["max_abs_dlogprob"] > table["0-2048"]["mean_abs_dlogprob"] / 2
    assert res["rungs"][2000]["mean_abs_dlogprob"] < res["rungs"][4000]["mean_abs_dlogprob"]


def test_two_phase_formats_without_a_bucket_being_dropped():
    res = _two_phase_with_stub(stability, [10, 40])
    md = stability.format_markdown(res)
    assert "| leg | tokens | distinct-1 |" in md
    assert "native" in md and "starkv" in md
    for bucket in res["rungs"][40]["by_position"]:
        assert bucket in md


def test_load_prompt_reads_jsonl_and_raw_text():
    import tempfile

    d = tempfile.mkdtemp()
    jsonl = os.path.join(d, "replay.jsonl")
    with open(jsonl, "w") as f:
        f.write(json.dumps({"prompt": "hello from jsonl"}) + "\n")
        f.write(json.dumps({"prompt": "second record"}) + "\n")
    assert stability.load_prompt(jsonl) == "hello from jsonl"
    assert stability.load_prompt(jsonl, 1) == "second record"

    txt = os.path.join(d, "prompt.txt")
    with open(txt, "w") as f:
        f.write("raw text prompt")
    assert stability.load_prompt(txt) == "raw text prompt"


def test_load_prompt_renders_a_replay_request():
    """A replay record is a chat request; the ladder scores raw text."""
    import tempfile

    d = tempfile.mkdtemp()
    path = os.path.join(d, "request.json")
    rec = {
        "id": "abc",
        "usage": {"prompt_tokens": 137904},
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]},
            {"role": "assistant", "content": "ok"},
        ],
    }
    with open(path, "w") as f:
        json.dump(rec, f)
    out = stability.load_prompt(path)
    # Role headers survive (the model still sees turn structure), and a
    # multi-part content block is joined in order.
    assert out == "<|system|>\nbe terse\n<|user|>\nfirst\nsecond\n<|assistant|>\nok"
    # The recorded length is for *ranking* prompts, not a substitute for
    # measuring the rendered text -- the two are different quantities.
    assert "137904" not in out


def test_render_messages_tolerates_empty_and_missing_content():
    assert stability.render_messages([]) == ""
    assert stability.render_messages([{"role": "user"}]) == "<|user|>\n"
    assert stability.render_messages(
        [{"role": "user", "content": [{"type": "text"}]}]
    ) == "<|user|>\n"


def test_capture_hooks_share_one_window_and_tag_rows_with_their_call():
    """The join's precondition, asserted on the written files.

    Both hooks fire once per c4 layer per decode step, and their call indices are
    what put the store sample and the attention sample in the same window. If the
    store hook buffered prefill rows while the attention hook buffered decode
    steps -- the bug this window exists to prevent -- the two would close at
    unrelated points and the join would come back empty. So drive both hooks for
    the same number of calls and check the files line up.
    """
    import tempfile

    d = tempfile.mkdtemp()
    keys = ("STARKV_CAPTURE", "STARKV_CAPTURES", "STARKV_CAPTURE_STEPS",
            "STARKV_CAPTURE_HEADS")
    saved = {k: os.environ.get(k) for k in keys}
    steps, batch, heads, topk, n_heads = 4, 3, 5, 3, 2
    slots = torch.tensor([1, 2, 3], dtype=torch.int64)
    try:
        os.environ.update({
            "STARKV_CAPTURE": "unit",
            "STARKV_CAPTURES": d,
            "STARKV_CAPTURE_STEPS": str(steps),
            "STARKV_CAPTURE_HEADS": str(n_heads),
        })
        capture.reset()
        g = torch.Generator().manual_seed(3)
        for layer in (2, 4):
            for _ in range(steps):
                capture.capture_store(
                    layer,
                    torch.randn(batch, config.HEAD_DIM, generator=g),
                    torch.randn(batch, config.ROPE_DIM, generator=g),
                    loc=slots,
                    pos=torch.arange(batch, dtype=torch.int32),
                    boundary=torch.ones(batch, dtype=torch.bool),
                )
                capture.capture_decode_attn(
                    layer,
                    q=torch.randn(batch, 1, heads, config.HEAD_DIM, generator=g),
                    indices=slots.to(torch.int32).repeat(batch, 1),
                    lengths=torch.full((batch,), topk, dtype=torch.int32),
                )
        assert capture.flush() == 4  # one store + one attn file per layer

        for layer in (2, 4):
            store = torch.load(capture._store_path(layer))
            attn = torch.load(capture._attn_path(layer))
            assert store["calls"] == steps and attn["calls"] == steps
            assert attn["step"].tolist() == list(range(steps))
            assert sorted(set(store["step"].tolist())) == list(range(steps))
            # The join itself: every selected slot is one the store recorded.
            stored = set(store["loc"].tolist())
            selected = set(attn["idx"].reshape(-1).tolist())
            assert selected and selected <= stored
            # Head limiting happens where the tensor is still on-device. `q` is
            # stored [step*batch, heads, HEAD_DIM]: the (b, 1, h, d) query has
            # its singleton slot folded into the head axis by the reshape.
            assert attn["q"].shape[1] == n_heads
            assert attn["q"].shape[0] == steps * batch
            assert attn["q"].shape[2] == config.HEAD_DIM
            # Steps are only recoverable through the recorded batch sizes.
            assert int(attn["batch"].sum()) == attn["q"].shape[0]
            assert attn["lens"].shape[0] == attn["idx"].shape[0] == attn["q"].shape[0]

            # The whole point of the window: what capture.py writes is what
            # spectrum.py can consume. An identity basis makes eps_basis exactly
            # zero, so any shape or split error here shows up as a mismatch.
            got = spectrum.score_error(store, attn, torch.eye(config.NOPE_DIM))
            assert got["entries"] == steps * batch * topk  # one per selected slot
            assert got["coverage"] == 1.0
            assert got["eps_basis"] == 0.0
            assert got["eps_quant"] > 0.0

        assert all(v == (steps, steps) for v in capture.alignment().values())

        # The window closes: further calls are neither buffered nor written.
        capture.capture_store(2, torch.randn(batch, config.HEAD_DIM, generator=g),
                              torch.randn(batch, config.ROPE_DIM, generator=g),
                              loc=slots, pos=torch.zeros(batch, dtype=torch.int32),
                              boundary=torch.ones(batch, dtype=torch.bool))
        assert capture.flush() == 0
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        capture.reset()


TESTS = (
    test_stored_rows_drops_non_boundary_decode_rows,
    test_stored_rows_leaves_a_prefill_capture_alone,
    test_clamp_rank_floors_to_a_tile_and_to_the_row_count,
    test_the_three_retention_flavours_agree_when_the_rank_matches_the_subspace,
    test_retention_falls_when_the_rank_is_below_the_true_dimension,
    test_retention_stats_report_min_and_p05_not_just_the_mean,
    test_rank_profile_picks_the_smallest_qualifying_rank,
    test_score_error_separates_the_basis_cost_from_the_quantization_cost,
    test_score_error_counts_the_entries_it_could_not_join,
    test_analyze_layer_end_to_end_produces_every_table,
    test_analyze_layer_without_an_attention_capture_still_produces_retention,
    test_position_buckets_cover_the_axis_without_gaps_or_overlap,
    test_drift_table_buckets_by_position_and_averages_only_what_it_saw,
    test_drift_table_omits_top1_agreement_when_it_was_not_measured,
    test_longest_repeated_ngram_finds_the_repeated_span,
    test_degeneration_flags_a_repeated_stream_against_a_varied_one,
    test_two_phase_scores_every_rung_and_keeps_the_reference_shared,
    test_the_reference_phase_alone_carries_everything_the_compare_phase_needs,
    test_two_phase_reports_drift_that_grows_with_position,
    test_two_phase_formats_without_a_bucket_being_dropped,
    test_load_prompt_reads_jsonl_and_raw_text,
    test_load_prompt_renders_a_replay_request,
    test_render_messages_tolerates_empty_and_missing_content,
    test_capture_hooks_share_one_window_and_tag_rows_with_their_call,
)


def run() -> int:
    failures = 0
    for t in TESTS:
        try:
            t()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {t.__name__}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
