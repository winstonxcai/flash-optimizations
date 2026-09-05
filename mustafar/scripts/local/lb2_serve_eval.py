#!/usr/bin/env python3
"""LongBench v2 full eval over an sglang OpenAI-compatible server.

Server-mode twin of transferibility/sg_capture.py::cmd_run_lb2 (the in-process
engine path that produced the report's earlier n=100 rows): same data, same
prompt, same answer extraction -- but the prompt is delivered as a role=user
chat-completions message so the server's DeepSeek chat template supplies the
<|User|>/<|Assistant|> markers (exactly what the in-process harness emulated by
hand-wrapping the bare prompt). The server's deepseek-v4 reasoning parser
handles the <think> block; extraction still splits on </think> defensively.

Scope: "the entire LongBench v2" = all feasible samples. Samples whose
tot_tokens exceed the server context cap (1,048,576) cannot be served as-is and
are DROPPED (the dataset's longest reach 4.6M tokens; truncating would put the
answer -- which may lie anywhere in the context -- out of reach and bias the
run). 473 of 503 are feasible.

Usage (host python3, host-networked server):
  python3 lb2_serve_eval.py \
    --server http://127.0.0.1:30212 --model deepseek-v4-flash \
    --tag packed-0731 \
    --data /home/jovyan/winstonxcai/transferibility/data/longbench/lb2_data.json \
    --tokens /home/jovyan/winstonxcai/transferibility/data/longbench/lb2_tokens.json \
    --out /home/jovyan/winstonxcai/flash-optimizations/mustafar/results/lb2-full/<leg>/<ts>/results.json \
    [--max-concurrency 6] [--max-tokens 512] [--ctx-cap 1048576]
Resume: --out json is read at start; already-done ids are skipped.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time

import requests

_LB2_PROMPT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)

_ANS_RE = (r"The correct answer is \(([A-D])\)", r"The correct answer is ([A-D])")


def lb2_extract(text):
    """Official LongBench v2 answer extraction, mirroring sg_capture._lb2_extract."""
    if not text:
        return None, False
    t = text.replace("*", "")
    parts = t.rsplit("</think>", 1)
    if len(parts) == 2:
        t = parts[1]
    for pat in _ANS_RE:
        m = re.search(pat, t)
        if m:
            return m.group(1), True
    m = re.search(r"\b([A-D])\b", t)
    return (m.group(1) if m else None), False


def build_prompt(d):
    return _LB2_PROMPT.format(context=d["context"], question=d["question"],
                              a=d["choice_A"], b=d["choice_B"],
                              c=d["choice_C"], d=d["choice_D"])


def call_once(args, prompt, sid, timeout_s):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    t0 = time.time()
    r = requests.post(f"{args.server}/v1/chat/completions",
                      json=payload, timeout=timeout_s)
    dt = time.time() - t0
    if r.status_code != 200:
        return None, dt, f"HTTP {r.status_code}: {r.text[:300]}"
    try:
        content = r.json()["choices"][0]["message"].get("content")
    except Exception as e:  # noqa: BLE001
        return None, dt, f"parse: {e} :: {r.text[:300]}"
    if not content:
        return None, dt, "empty content"
    return content, dt, None


def call_with_retry(args, prompt, sid):
    # generous timeout: a >500k sample takes ~40-90 s prefill + queue wait
    timeout_s = args.request_timeout
    for attempt in range(args.retries + 1):
        try:
            return call_once(args, prompt, sid, timeout_s)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last = f"conn/timeout: {e}"
            if attempt < args.retries:
                time.sleep(5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            return None, time.time(), f"fatal: {e}"
    return None, 0.0, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:30212")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--tag", required=True, help="leg label, e.g. packed-0731")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="mirror cmd_run_lb2 --max-new (thinking model budget)")
    ap.add_argument("--ctx-cap", type=int, default=1048576)
    ap.add_argument("--request-timeout", type=int, default=3600)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--drop-over-cap", action="store_true", default=True,
                    help="drop samples with tot_tokens>ctx-cap (dataset max 4.6M)")
    args = ap.parse_args()

    data = {d["_id"]: d for d in json.load(open(args.data))}
    toks = {t["_id"]: t for t in json.load(open(args.tokens))}
    assert len(data) == len(toks) == 503, (len(data), len(toks))

    # ordered, deterministic; feasible only
    ids = []
    dropped = []
    for d in json.load(open(args.data)):
        qid = d["_id"]
        tot = toks[qid]["tot_tokens"]
        if tot > args.ctx_cap:
            dropped.append((qid, tot))
            continue
        ids.append(qid)
    print(f"[lb2] {args.tag}: {len(ids)} feasible / 503 "
          f"(dropped {len(dropped)} over {args.ctx_cap}: "
          f"max {max(t for _, t in dropped) if dropped else '-'})", flush=True)
    if not ids:
        print("[lb2] nothing to run", flush=True)
        return 1

    results = {"tag": args.tag, "server": args.server, "model": args.model,
               "max_tokens": args.max_tokens, "ctx_cap": args.ctx_cap,
               "by_id": {}, "summary": {}, "dropped": dropped}
    if os.path.exists(args.out):
        try:
            old = json.load(open(args.out))
            results["by_id"] = old.get("by_id", {})
        except Exception:  # noqa: BLE001
            results["by_id"] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pending = [qid for qid in ids if not results["by_id"].get(qid, {}).get("done")]
    print(f"[lb2] resume: {len(ids) - len(pending)} done, {len(pending)} pending", flush=True)

    total_tok = sum(toks[qid]["ctx_tokens"] for qid in pending)
    t_all = time.time()

    def work(qid):
        d = data[qid]
        prompt = build_prompt(d)
        content, dt, err = call_with_retry(args, prompt, qid)
        ans, followed = lb2_extract(content)
        entry = {
            "done": True, "answer": d["answer"], "domain": d["domain"],
            "length": toks[qid]["length"], "ctx_tokens": toks[qid]["ctx_tokens"],
            "tot_tokens": toks[qid]["tot_tokens"],
            "correct": 1.0 if (ans is not None and ans == d["answer"]) else 0.0,
            "pred_ans": ans, "followed_format": followed,
            "pred_head": (content or "")[:160],
            "secs": round(dt, 1),
        }
        if err:
            entry["error"] = err
        return qid, entry

    done_ct = len(results["by_id"]) - sum(1 for v in results["by_id"].values() if not v.get("done"))
    n_correct = sum(1 for v in results["by_id"].values() if v.get("correct"))
    with cf.ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        futs = {ex.submit(work, qid): qid for qid in pending}
        it = cf.as_completed(futs)
        for i, fut in enumerate(it, 1):
            qid, entry = fut.result()
            results["by_id"][qid] = entry
            if entry.get("error"):
                print(f"[lb2] {qid[:12]} ERROR {entry['error'][:80]}", flush=True)
            done_ct += 1
            n_correct += 1 if entry.get("correct") else 0
            if i % 20 == 0 or i == len(pending):
                el = (time.time() - t_all) / 60
                print(f"[lb2] {i}/{len(pending)} done | {done_ct}/{len(ids)} total | "
                      f"correct {n_correct}/{done_ct} ({n_correct/done_ct*100:.1f}%) | "
                      f"elapsed {el:.0f}min | est prefill "
                      f"{total_tok/12000/60:.0f}min @12k tok/s", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1, default=str)

    # final summary
    by_id = results["by_id"]
    n_done = sum(1 for v in by_id.values() if v.get("done"))
    n_ok = sum(1 for v in by_id.values() if v.get("correct"))
    errs = sum(1 for v in by_id.values() if v.get("error"))
    results["summary"] = {
        "feasible": len(ids), "total_attempted": n_done, "dropped_over_cap": len(dropped),
        "correct": n_ok, "accuracy": n_ok / n_done if n_done else 0.0,
        "errors": errs, "elapsed_min": round((time.time() - t_all) / 60, 1),
        "prompt_tokens_run": sum(toks[qid]["ctx_tokens"] for qid in ids),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"[lb2] DONE {args.tag}: {n_ok}/{n_done} ({n_ok/n_done*100:.1f}%) "
          f"errors={errs} elapsed={(time.time()-t_all)/60:.0f}min "
          f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
