# E005 preflight — register-resident lane ownership

Status: CPU mapping preflight passed; **no CUDA candidate implemented or run**.

## Failure being addressed

The discarded register-resident draft assigned each warp lane a 64-survivor
segment, then used the source lane's own rank query to select the word returned
by a shuffle. A consumer instead needs the word selected by its own rank. Those
queries differ across the four feature-group lanes for a token, so the source
can return a valid but unrelated word. The all-kept counterexample is minimal:
feature group 1 requests ranks 16–19, while source group 0's local query starts
at rank 0.

## Local check

`lane_mapping.py` models the four feature-group lanes, bitmap prefix ranks,
64-survivor register segments, and four-coordinate output fragments. It checks
that requester-rank ownership returns the same bytes as scalar rank
reconstruction, and counts fragments where the source-local selection fails.

```text
masks_checked=1007
fragments_checked=120825
flawed_lane_local_rank_mismatches=103650
requester-rank ownership: PASS
```

The masks include balanced patterns and 1,000 deterministic random
256-of-512 selections. This proves the ownership rule in a host-side model; it
does not compile or validate CUDA, synchronization, or performance.

## Decision / next gate

No live FlashMLA kernel was changed: the faulty draft had already been
reverted, and the current `remnant/flashmla-v0518` source remains at E004. A
literal repair would require each producer lane to receive a requester's rank
before selecting a word from its registers. Since the four feature-group lanes
have independent ranks, a naïve implementation repeats warp exchanges for
each fragment; that is too much unmeasured overhead to present as a plausible
performance candidate under the strict per-shape E001 gate.

Before writing CUDA, redesign the producer ownership/layout so survivor words
are loaded by the lanes that consume them (or otherwise make the rank request
common), then repeat this CPU mapping check. No Modal task was launched.
