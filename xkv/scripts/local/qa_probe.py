#!/usr/bin/env python3
"""Fixed-prompt QA probe: one file per serving leg, so two boots can be diffed.

Runs *inside* the xkv container against whatever is live on --port. It exists to
answer the two questions the bench protocol cannot: does the leg produce text at
all (the pre-fix ue8m0 bug rendered every output NaN), and does the low-rank leg
still answer the prompt the same way the native leg does.

Stdlib only (urllib), greedy sampling, so the only difference between legs is the
KV path under test -- not a sampling seed.

    qa_probe.py --port 30213 --out /path/qa_xkv.jsonl [--tag xkv]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# Short, checkable prompts. Each has an answer that is obviously right or
# obviously wrong to a human reading the report, which is the point: this is a
# liveness/parity probe, not a benchmark.
PROMPTS = [
    ("arithmetic",
     "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
     "How far did it travel in total? Show the two legs and the sum."),
    ("recall",
     "List the first five prime numbers, then state which of them is the only "
     "even one and why that is necessarily true."),
    ("instruct",
     "Write exactly three sentences about why memory bandwidth, not FLOPs, "
     "usually limits large-language-model decoding. Do not use bullet points."),
]

MAX_NEW_TOKENS = 128


def probe(port, prompt, *, timeout=600):
    payload = json.dumps({
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "skip_special_tokens": True,
        },
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    records = []
    for name, prompt in PROMPTS:
        try:
            body = probe(args.port, prompt)
            text = body.get("text", "")
            record = {"prompt": name, "text": text,
                      "meta": body.get("meta_info", {})}
        except (urllib.error.URLError, TimeoutError) as exc:
            # A failed probe is a finding, not a crash: record it and keep going
            # so the file still says what happened to the other prompts.
            record = {"prompt": name, "error": f"{type(exc).__name__}: {exc}"}
        record["tag"] = args.tag
        records.append(record)
        status = "ERR" if "error" in record else f"{len(record['text'])} chars"
        print(f"  qa[{args.tag or '?'}] {name}: {status}", flush=True)

    with open(args.out, "w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return 0 if all("error" not in r for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
