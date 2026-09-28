"""Long-decode collapse instrumentation.

Retention loss does not announce itself as a wrong answer. It announces itself
as babble after a few thousand decode steps, and by then a single end-of-run
score cannot tell you at which position it started. So this module measures the
*shape* of the divergence rather than its endpoint:

* **teacher-forced logprob drift** -- take a continuation the ``native`` leg
  generated, then score that exact token sequence on both legs (teacher-forced,
  so the token sequence is identical and only the model differs). Report
  ``|dlp|`` and top-1 agreement bucketed by decode position. The failure
  signature is a curve that grows with position; a flat curve means the basis is
  holding no matter how long the session runs.
* **greedy degeneration** -- temperature 0 continuation on both legs, then
  distinct-1/2/3 and the longest repeated n-gram over the generated span.
  Babble shows up as distinct-n collapsing and the repeated-n-gram length
  exploding.

Both are reported at a ladder of decode lengths (default 2k / 8k / 32k), because
a check that passes at 2k says nothing about 32k -- which is the regime the
failure lives in.

The run is split into two CLI phases, ``reference`` then ``compare``, because at
the memory fraction this model needs a single leg already claims most of its
GPUs and two resident legs cannot share a GPU set. Phase ``reference`` runs
against the ``native`` leg, generates the continuation once and saves the text
plus that leg's per-position logprobs; phase ``compare`` boots the other leg and
scores the byte-identical text. Which basis mode is under test is therefore
decided by which leg is booted, not by a flag here.

Stdlib only (no torch), so it runs from the host as easily as from the
container.
"""

import argparse
import json
import os
import urllib.error
import urllib.request
from collections import Counter

# Decode-position buckets, in tokens generated or scored. The last bucket is
# open-ended; the ladder chooses how far into it the run actually reaches.
POSITION_BUCKETS = [(0, 2048), (2048, 8192), (8192, 16384), (16384, 32768), (32768, None)]


# --- transport --------------------------------------------------------------


def _post(server: str, path: str, payload: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(
        server.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"{server}{path} -> HTTP {e.code}: {body}") from None


def health(server: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/health", timeout=timeout):
            return True
    except Exception:
        return False


# --- server calls -----------------------------------------------------------


def generate(server: str, text: str, max_new_tokens: int) -> dict:
    """Temperature-0 continuation. Returns {text, n_input_tokens}."""
    out = _post(server, "/generate", {
        "text": text,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    })
    meta = out.get("meta_info", {})
    return {
        "text": out.get("text", ""),
        "n_input_tokens": len(meta.get("input_token_logprobs") or []),
        "n_output_tokens": len(meta.get("output_token_logprobs") or []),
        "finish_reason": (meta.get("finish_reason") or {}).get("type"),
    }


def token_count(server: str, text: str) -> int:
    """Prompt length in tokens, obtained without a tokenizer.

    A zero-length generation still returns one logprob entry per input token,
    which is exactly the count -- and using the server's own tokenizer is the
    only way the index stays valid for `logprob_start_len`.
    """
    out = _post(server, "/generate", {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
    })
    return len(out.get("meta_info", {}).get("input_token_logprobs") or [])


def score(server: str, text: str, start: int) -> dict:
    """Teacher-forced logprobs for every token at index >= `start`.

    Returns {"logprobs": [...], "top1": [...]} aligned to those positions;
    `top1` is empty when the server does not report top logprobs.
    """
    out = _post(server, "/generate", {
        "text": text,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 0,
            "top_logprobs_num": 1,
        },
        "return_logprob": True,
        "logprob_start_len": start,
    })
    meta = out.get("meta_info", {})
    lps = [entry[0] for entry in (meta.get("input_token_logprobs") or [])]
    tops = meta.get("input_top_logprobs") or []
    top1 = [entry[0][1] if entry else None for entry in tops]
    return {"logprobs": lps, "top1": top1}


# --- comparison -------------------------------------------------------------


def bucket_of(position: int) -> str:
    """Position -> bucket label. Every non-negative position lands in a bucket."""
    for lo, hi in POSITION_BUCKETS:
        if position >= lo and (hi is None or position < hi):
            return f"{lo}-{hi}" if hi is not None else f"{lo}+"
    raise ValueError(f"negative decode position {position}")


def drift_table(deltas, top1_agree) -> dict:
    """Bucket |dlp| and top-1 agreement by position. `deltas[i]` is position i."""
    acc: dict = {}
    for pos, d in enumerate(deltas):
        if d is None:
            continue
        b = acc.setdefault(bucket_of(pos), {"n": 0, "sum": 0.0, "max": 0.0, "agree": 0})
        b["n"] += 1
        b["sum"] += d
        b["max"] = max(b["max"], d)
        if top1_agree is not None and top1_agree[pos]:
            b["agree"] += 1
    out = {}
    for key in sorted(acc, key=lambda k: int(k.split("-")[0].rstrip("+"))):
        b = acc[key]
        out[key] = {
            "n": b["n"],
            "mean_abs_dlogprob": b["sum"] / b["n"],
            "max_abs_dlogprob": b["max"],
            "top1_agreement": (b["agree"] / b["n"]) if top1_agree is not None else None,
        }
    return out


def longest_repeated_ngram(tokens, cap: int = 4096) -> int:
    """Longest n with some n-gram appearing twice. Binary search over n.

    Counting n-grams directly over a 32k-token span at every n would be O(n^2);
    repeat length is monotone in n (if an n-gram repeats, so does its prefix),
    so the search is binary and each probe is one Counter pass.
    """
    n_tok = len(tokens)
    if n_tok < 2:
        return 0
    lo, hi = 1, min(cap, n_tok - 1)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        grams = Counter(tuple(tokens[i : i + mid]) for i in range(n_tok - mid + 1))
        if any(c > 1 for c in grams.values()):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def distinct_n(text: str, n: int) -> float:
    toks = text.split()
    if len(toks) < n:
        return 1.0 if toks else 0.0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def degeneration(text: str) -> dict:
    toks = text.split()
    return {
        "tokens": len(toks),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "longest_repeated_ngram": longest_repeated_ngram(toks),
    }


# --- driver -----------------------------------------------------------------


def render_messages(messages) -> str:
    """A replay request's messages as one plain-text prompt.

    The replay corpus stores OpenAI chat requests, but this module scores with
    `/generate`, which takes raw text -- and its `logprob_start_len` is the only
    reliable way to get per-position logprobs for a span that long. So the
    messages are rendered rather than chat-templated.

    That is an approximation of the original prompt, and deliberately an
    acceptable one: every measurement here is *relative* between the two legs on
    byte-identical text, so the prompt only has to be long, realistic agentic
    context -- not a faithful reconstruction of the request that produced it.
    Role headers are kept so the model still sees the turn structure.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content if isinstance(p, dict)
                and p.get("type", "text") == "text"
            )
        elif content is None:
            content = ""
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def load_prompt(path: str, index: int = 0) -> str:
    """A prompt from a LongSWE-Bench replay record, or a raw text file.

    Accepts a replay request `.json` (rendered via `render_messages`), a
    `.jsonl` of prompts, or plain text.
    """
    if path.endswith(".jsonl"):
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        rec = records[index]
        for key in ("prompt", "text", "input", "request"):
            if key in rec:
                return rec[key] if isinstance(rec[key], str) else json.dumps(rec[key])
        raise SystemExit(f"no prompt field in record {index} of {path}")
    if path.endswith(".json"):
        with open(path) as f:
            rec = json.load(f)
        if "messages" in rec:
            return render_messages(rec["messages"])
        for key in ("prompt", "text", "input"):
            if key in rec and isinstance(rec[key], str):
                return rec[key]
        raise SystemExit(f"no messages or prompt field in {path}")
    with open(path) as f:
        return f.read()


def run_reference(server: str, prompt: str, lengths, tag: str) -> dict:
    """Phase 1: generate the reference continuation and score it on this leg.

    Run against the ``native`` server. The result is a document -- the prompt,
    the full spanned text, and this leg's per-position logprobs -- so phase 2 can
    score the byte-identical text on the other leg without both servers needing
    to be resident at once. That is not just convenience: two servers at the
    memory fraction this model needs cannot share a GPU set, so a single-process
    comparison would demand twice the hardware.

    One generation serves every rung: it runs at temperature 0 for the longest
    rung, and a shorter rung scores a prefix of it, which is the same sequence a
    shorter run would have produced.
    """
    lengths = sorted(lengths)
    n_prompt = token_count(server, prompt)
    ref = generate(server, prompt, max(lengths))
    span = prompt + ref["text"]
    scored = score(server, span, n_prompt)
    return {
        "tag": tag,
        "lengths": list(lengths),
        "prompt": prompt,
        "span": span,
        "prompt_tokens": n_prompt,
        "reference": {
            "n_output_tokens": ref["n_output_tokens"],
            "finish_reason": ref["finish_reason"],
        },
        "legs": {"native": scored},
        "degeneration": {
            "native": {
                **degeneration(ref["text"]),
                "n_output_tokens": ref["n_output_tokens"],
                "finish_reason": ref["finish_reason"],
            }
        },
    }


def run_compare(server: str, doc: dict, tag: str) -> dict:
    """Phase 2: score the reference text on the other leg and tabulate drift."""
    lengths = sorted(doc["lengths"])
    n_prompt = doc["prompt_tokens"]
    native = doc["legs"]["native"]
    other = score(server, doc["span"], n_prompt)
    doc["legs"]["starkv"] = other

    available = min(len(native["logprobs"]), len(other["logprobs"]))
    deltas_all = [abs(a - b) for a, b in
                  zip(other["logprobs"][:available], native["logprobs"][:available])]
    agree_all = None
    if native["top1"] and other["top1"]:
        agree_all = [
            (a is not None and a == b)
            for a, b in zip(other["top1"][:available], native["top1"][:available])
        ]

    doc["rungs"] = {}
    for length in lengths:
        n = min(length, available)
        if n == 0:
            doc["rungs"][length] = {"error": "no positions scored"}
            continue
        deltas = deltas_all[:n]
        agree = None if agree_all is None else agree_all[:n]
        doc["rungs"][length] = {
            "scored_positions": n,
            "mean_abs_dlogprob": sum(deltas) / n,
            "max_abs_dlogprob": max(deltas),
            "by_position": drift_table(deltas, agree),
        }
        print(f"  {length}: scored {n} positions, "
              f"mean |dlp| {doc['rungs'][length]['mean_abs_dlogprob']:.4f}")

    out = generate(server, doc["prompt"], max(lengths))
    doc["degeneration"]["starkv"] = {
        **degeneration(out["text"]),
        "n_output_tokens": out["n_output_tokens"],
        "finish_reason": out["finish_reason"],
    }
    print(f"  greedy starkv: {doc['degeneration']['starkv']}")
    return doc


def format_markdown(res: dict) -> str:
    lines = [
        f"# Long-decode stability: {res['tag']}",
        "",
        f"Prompt: {res.get('prompt_tokens', '?')} tokens.",
        f"Reference continuation: {res.get('reference', {}).get('n_output_tokens', '?')} "
        "tokens, generated on the native leg and then scored identically on both "
        "(teacher-forced, so the token sequence is shared).",
        "",
        "## Teacher-forced logprob drift",
        "",
        "| scored | mean \\|dlp\\| | max \\|dlp\\| | position bucket | n | mean \\|dlp\\| | max \\|dlp\\| | top-1 agree |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for length, rung in sorted(res["rungs"].items(), key=lambda kv: int(kv[0])):
        if "error" in rung:
            lines.append(f"| {length} | -- | -- | error: {rung['error']} | | | | |")
            continue
        mean = f"{rung['mean_abs_dlogprob']:.4f}"
        peak = f"{rung['max_abs_dlogprob']:.4f}"
        first = True
        for bucket, b in rung["by_position"].items():
            agree = b["top1_agreement"]
            agree_s = "--" if agree is None else f"{agree:.3f}"
            lines.append(
                f"| {length if first else ''} | {mean if first else ''} | "
                f"{peak if first else ''} | {bucket} | {b['n']} | "
                f"{b['mean_abs_dlogprob']:.4f} | {b['max_abs_dlogprob']:.4f} | {agree_s} |"
            )
            first = False
    lines += ["", "## Greedy degeneration (temperature 0)", "",
              "| leg | tokens | distinct-1 | distinct-2 | distinct-3 | longest repeated n-gram |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, d in (res.get("degeneration") or {}).items():
        lines.append(
            f"| {name} | {d['tokens']} | {d['distinct_1']:.3f} | {d['distinct_2']:.3f} | "
            f"{d['distinct_3']:.3f} | {d['longest_repeated_ngram']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("reference", "compare"),
                    help="reference: generate + score on this leg and save. "
                         "compare: score the saved text on this leg and tabulate.")
    ap.add_argument("--server", required=True, help="base URL of this leg")
    ap.add_argument("--prompt", help="replay prompt file (jsonl or text); reference phase")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--lengths", default="2048,8192")
    ap.add_argument("--tag", default="stability")
    ap.add_argument("--out", default="", help="result base path (no extension)")
    args = ap.parse_args(argv)

    from .. import config

    if not health(args.server):
        raise SystemExit(f"leg at {args.server} is not healthy")
    out_base = args.out or os.path.join(config.results_dir(), f"stability-{args.tag}")
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    if args.phase == "reference":
        if not args.prompt:
            raise SystemExit("--prompt is required for the reference phase")
        lengths = [int(x) for x in args.lengths.split(",") if x]
        doc = run_reference(args.server, load_prompt(args.prompt, args.index),
                            lengths, args.tag)
        # Written before the compare phase so the two legs need never be up at
        # the same time.
        with open(out_base + ".json", "w") as f:
            json.dump(doc, f, indent=2)
        print(f"reference: {doc['reference']['n_output_tokens']} tokens generated "
              f"from a {doc['prompt_tokens']}-token prompt")
        print(f"\nwrote {out_base}.json -- now boot the starkv leg and run the "
              f"compare phase against the same --out")
        return 0

    if not os.path.exists(out_base + ".json"):
        raise SystemExit(f"{out_base}.json not found; run the reference phase first")
    with open(out_base + ".json") as f:
        doc = json.load(f)
    doc = run_compare(args.server, doc, args.tag)

    md = format_markdown(doc)
    with open(out_base + ".json", "w") as f:
        json.dump(doc, f, indent=2)
    with open(out_base + ".md", "w") as f:
        f.write(md)
    print(md)
    print(f"\nwrote {out_base}.md and {out_base}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
