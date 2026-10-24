"""Reproduce four-run Packed store/decode ablation summaries. Author: Winston Cai.

Consumes retained metadata only. Historical runs lack a verified client/server
join; instrumented runs can use the explicit first-output pipeline diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    'received_to_forward_s': ('forward_entry_time', 'request_received_ts'),
    'prefill_span_s': ('prefill_finished_time', 'forward_entry_time'),
    'prefill_to_first_output_s': ('first_output_ts', 'prefill_finished_time'),
    'first_output_to_response_sent_s': ('response_sent_to_client_ts', 'first_output_ts'),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(x) for x in ordered):
        raise ValueError('Missing or nonfinite measurement')
    def quantile(q: float) -> float:
        index = (len(ordered) - 1) * q
        lo = math.floor(index)
        hi = math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {'count': len(ordered), 'mean': statistics.mean(ordered),
            'p50': quantile(.5), 'p90': quantile(.9), 'p95': quantile(.95)}


def server_stats(records: list[dict]) -> dict:
    result = {}
    for stage, (end, start) in STAGES.items():
        values = [r[end] - r[start] for r in records]
        if min(values) < 0:
            raise ValueError(f'Negative {stage}')
        result[stage] = stats(values)
    result['scheduler_queue_s'] = stats([r['queue_time'] for r in records])
    return result


def client_stats(records: list[dict]) -> dict:
    """Use the replay harness's nearest-rank convention for client percentiles."""
    result = {}
    for metric in ('ttft_ms', 'tpot_ms', 'latency_ms'):
        values = sorted(r[metric] for r in records)
        result[metric] = {
            'count': len(values), 'mean': statistics.mean(values),
            'p50': values[math.ceil(.5 * len(values)) - 1],
            'p90': values[math.ceil(.9 * len(values)) - 1],
            'p95': values[math.ceil(.95 * len(values)) - 1],
        }
    result['ttft_gt_10s_percent'] = 100 * sum(r['ttft_ms'] > 10000 for r in records) / len(records)
    return result


def diagnostic_associations(client: list[dict], server: list[dict]) -> dict:
    """Associate unique token-count signatures for exploratory tail analysis.

    No request-ID join is available. Exclude ambiguous signatures and check
    cache counts, finish reasons and timing consistency. These checks increase
    confidence but do not turn the association into an ID-verified join.
    """
    client_groups, server_groups = defaultdict(list), defaultdict(list)
    for record in client:
        client_groups[(record['prompt_tokens'], record['completion_tokens'])].append(record)
    for record in server:
        server_groups[(record['prompt_tokens'], record['completion_tokens'])].append(record)
    matches = []
    for key in sorted(client_groups.keys() & server_groups.keys()):
        if len(client_groups[key]) != 1 or len(server_groups[key]) != 1:
            continue
        c, s = client_groups[key][0], server_groups[key][0]
        if not c['valid']:
            continue
        cache = c['cached_tokens_details'] or {}
        cached_tokens = sum(cache.get(tier, 0) for tier in ('device', 'host', 'storage'))
        # OpenAI maps a successful tool call to tool_calls; scheduler reports stop.
        finish_reason = 'stop' if c['finish_reason'] == 'tool_calls' else c['finish_reason']
        residual = c['ttft_ms'] / 1000 - (s['first_output_ts'] - s['request_received_ts'])
        if (cached_tokens != s['cached_tokens'] or finish_reason != s['finish_reason']['type']
                or c['latency_ms'] / 1000 < s['e2e_latency'] or residual < 0):
            raise ValueError(f"Inconsistent metadata association for {c['request_id']}")
        matches.append((c, s))
    result = {'id_verified': False, 'associated_successful_requests': len(matches),
              'basis': 'Unique prompt/completion token counts; cache/finish/timing consistency checked',
              'cohorts': {}}
    for name, pairs in (
        ('all', matches),
        ('after_180s', [(c, s) for c, s in matches if c['dispatch_offset_s'] >= 180]),
    ):
        groups = {}
        for label, subset in (
            ('ttft_le_10s', [(c, s) for c, s in pairs if c['ttft_ms'] <= 10000]),
            ('ttft_gt_10s', [(c, s) for c, s in pairs if c['ttft_ms'] > 10000]),
        ):
            groups[label] = {
                'count': len(subset), 'server_stages': server_stats([s for c, s in subset]),
                'queue_ge_1s_percent': 100 * sum(s['queue_time'] >= 1 for c, s in subset) / len(subset),
                'client_ttft_minus_server_first_output_s': stats([
                    c['ttft_ms'] / 1000 - (s['first_output_ts'] - s['request_received_ts'])
                    for c, s in subset
                ]),
            }
        result['cohorts'][name] = groups
    return result


def first_output_pipeline_diagnostics(
    client: list[dict],
    server: list[dict],
    replay_start: float,
    *,
    include_preparation: bool = False,
) -> dict:
    """Join captured SSE IDs to server metrics and validate first-output spans.

    Every successful client response requires a unique, unambiguous server ID
    and all pipeline boundaries. Failed-response identity diagnostics remain
    visible but do not substitute for successful-response coverage. Unmatched
    server records may belong to failed responses and are counted separately.
    The after-180s cohort uses server receive time, matching server diagnostics.
    No missing timestamp or ID is inferred from token counts or nearby events.
    """
    finite = lambda value: (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
    )
    if not finite(replay_start) or replay_start <= 0:
        raise ValueError('First-output pipeline requires a finite positive replay_start')

    errors: list[str] = []
    identity_errors: list[str] = []
    client_identity_diagnostics = []
    client_groups, server_groups = defaultdict(list), defaultdict(list)
    successful = [record for record in client if record.get('valid') is True]
    for index, record in enumerate(client):
        request_id = record.get('server_request_id')
        diagnostic = record.get('server_request_id_error')
        if diagnostic is not None or not isinstance(request_id, str) or not request_id.strip():
            message = f'Client row {index} ({record.get("request_id")!r}): ' + (
                str(diagnostic) if diagnostic is not None else 'missing_or_invalid_server_request_id'
            )
            client_identity_diagnostics.append(message)
            if record.get('valid') is True:
                identity_errors.append(message)
        if isinstance(request_id, str) and request_id.strip():
            client_groups[request_id].append(record)
    for index, record in enumerate(server):
        request_id = record.get('id')
        if not isinstance(request_id, str) or not request_id.strip():
            identity_errors.append(f'Server row {index}: missing_or_invalid_id')
        else:
            server_groups[request_id].append(record)
    for source, groups in (('client', client_groups), ('server', server_groups)):
        for request_id, rows in groups.items():
            if len(rows) != 1:
                identity_errors.append(f'Duplicate {source} server ID {request_id!r}: {len(rows)} rows')

    matched = []
    for index, record in enumerate(client):
        if record.get('valid') is not True:
            continue
        request_id = record.get('server_request_id')
        if (record.get('server_request_id_error') is not None
                or not isinstance(request_id, str) or not request_id.strip()
                or len(client_groups[request_id]) != 1):
            continue
        rows = server_groups.get(request_id, [])
        if not rows:
            identity_errors.append(f'Successful client row {index}: no server record for {request_id!r}')
        elif len(rows) == 1:
            matched.append(rows[0])
    errors.extend(identity_errors)
    if not successful:
        errors.append('No successful client responses for first-output pipeline diagnostics')

    boundaries = (
        'prefill_finished_time', 'first_output_send_ts', 'detokenizer_recv_ts',
        'detokenizer_send_ts', 'frontend_recv_ts', 'frontend_handle_ts', 'first_output_ts',
    )
    stage_names = (
        'prefill_to_send_s', 'send_to_detokenizer_recv_s', 'detokenizer_recv_to_send_s',
        'detokenizer_send_to_frontend_recv_s', 'frontend_recv_to_handle_s',
        'frontend_handle_to_first_output_s',
    )
    if include_preparation:
        boundaries = (
            "request_received_ts",
            "input_preparation_queued_ts",
            "input_preparation_start_ts",
            "input_preparation_finish_ts",
            "input_preparation_resume_ts",
            "api_server_dispatch_finish_ts",
            "forward_entry_time",
        ) + boundaries
        stage_names = (
            "initial_api_handling_s",
            "preparation_wait_s",
            "preparation_execution_s",
            "preparation_resume_s",
            "post_preparation_dispatch_s",
            "dispatch_to_forward_s",
            "prefill_span_s",
        ) + stage_names
    valid_rows = []
    for record in matched:
        request_id = record['id']
        required = ('request_received_ts',) + boundaries
        invalid = [name for name in required
                   if not finite(record.get(name)) or record[name] <= 0]
        if invalid:
            errors.append(f'Server {request_id!r}: missing/nonfinite/nonpositive boundaries {invalid}')
            continue
        times = [record[name] for name in required]
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            errors.append(f'Server {request_id!r}: out-of-order first-output pipeline boundaries')
            continue
        valid_rows.append(record)

    cohorts = {}
    for name, rows in (
        ('all', valid_rows),
        ('after_180s', [record for record in valid_rows
                        if record['request_received_ts'] >= replay_start + 180]),
    ):
        stages = {}
        if rows:
            for stage, earlier, later in zip(stage_names, boundaries, boundaries[1:]):
                stages[stage] = stats([record[later] - record[earlier] for record in rows])
            stages['post_prefill_to_first_output_s'] = stats([
                record['first_output_ts'] - record['prefill_finished_time'] for record in rows
            ])
        cohorts[name] = {'count': len(rows), 'stages': stages}
    return {
        'valid': not errors,
        'id_verified': bool(successful) and not identity_errors and len(matched) == len(successful),
        'basis': 'Exact captured SSE server_request_id equals server metrics id; no inferred joins',
        'cohort_basis': 'Successful exact-ID joins; after_180s uses server request_received_ts',
        'errors': errors,
        'client_identity_diagnostics': client_identity_diagnostics,
        'coverage': {
            'client_records': len(client),
            'successful_client_records': len(successful),
            'client_records_with_server_id': sum(map(len, client_groups.values())),
            'server_records': len(server),
            'matched_successful': len(matched),
            'timing_valid_successful': len(valid_rows),
            'unmatched_successful': len(successful) - len(matched),
            'unmatched_server_records': len(server) - len(matched),
        },
        'cohorts': cohorts,
    }


def fixed_store_diagnostics(responses: dict[str, dict]) -> dict:
    """Compare decode modes with store fixed, including workload sensitivity."""
    result = {}
    for store in ('triton', 'kernel'):
        triton = responses[f'{store}-store-triton-decode']
        kernel = responses[f'{store}-store-kernel-decode']
        shared = sorted(triton.keys() & kernel.keys())
        identical = [k for k in shared
                     if triton[k]['completion_tokens'] == kernel[k]['completion_tokens']
                     and triton[k]['cached_tokens_details'] == kernel[k]['cached_tokens_details']]
        cohorts = {}
        for name, keys in (
            ('all_matched', shared),
            ('same_completion_and_cache', identical),
            ('same_completion_and_cache_after_180s', [k for k in identical
                if min(triton[k]['dispatch_offset_s'], kernel[k]['dispatch_offset_s']) >= 180]),
        ):
            diffs = [kernel[k]['ttft_ms'] - triton[k]['ttft_ms'] for k in keys]
            users = defaultdict(list)
            for key in keys:
                users[triton[key]['user_id']].append(kernel[key]['ttft_ms'] - triton[key]['ttft_ms'])
            cohorts[name] = {
                'kernel_minus_triton_ttft_ms': stats(diffs),
                'kernel_slower_request_percent': 100 * sum(x > 0 for x in diffs) / len(diffs),
                'mean_kernel_minus_triton_tpot_ms': statistics.mean(
                    kernel[k]['tpot_ms'] - triton[k]['tpot_ms'] for k in keys),
                'users': len(users), 'kernel_slower_users': sum(statistics.mean(v) > 0 for v in users.values()),
                'equal_user_weight_mean_difference_ms': statistics.mean(statistics.mean(v) for v in users.values()),
            }
        bins = []
        for low, high in ((0, 180), (180, 360), (360, 600), (600, 900), (900, 1200)):
            keys = [k for k in shared if low <= triton[k]['dispatch_offset_s'] < high
                    and low <= kernel[k]['dispatch_offset_s'] < high]
            bins.append({'start_s': low, 'end_s': high,
                         'kernel_minus_triton_ttft_ms': stats([kernel[k]['ttft_ms'] - triton[k]['ttft_ms'] for k in keys])})
        result[store] = {'cohorts': cohorts, 'both_dispatches_in_same_time_bin': bins}
    return result


def plot_diagnostics(output: dict, responses: dict[str, dict], root: Path = ROOT, concurrency: int = 24) -> None:
    """Optional static figure; JSON/CSV generation requires only the stdlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = {
        'triton-store-triton-decode': 'Triton store / Triton decode',
        'kernel-store-triton-decode': 'Kernel store / Triton decode',
        'triton-store-kernel-decode': 'Triton store / Direct decode',
        'kernel-store-kernel-decode': 'Kernel store / Direct decode',
    }
    colors = ['#2878b5', '#24946e', '#e07b22', '#b44360']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for (label, name), color in zip(names.items(), colors):
        values = sorted(r['ttft_ms'] / 1000 for r in responses[label].values()
                        if r['dispatch_offset_s'] >= 180)
        axes[0].step(values, [100 * (i + 1) / len(values) for i in range(len(values))],
                     where='post', label=name, color=color)
    axes[0].axvline(10, color='#444444', linestyle=':', label='10s SLO threshold')
    axes[0].axhline(90, color='#bbbbbb', linestyle=':')
    axes[0].set(xlim=(3, 14), ylim=(30, 100), xlabel='Client TTFT (seconds)',
                ylabel='Successful requests at or below TTFT (%)',
                title='TTFT distribution after the first 180 seconds')
    axes[0].legend(loc='lower right', fontsize=8)
    for store, color in (('triton', '#2878b5'), ('kernel', '#24946e')):
        bins = output['fixed_store_diagnostics'][store]['both_dispatches_in_same_time_bin'][1:]
        axes[1].plot([(b['start_s'] + b['end_s']) / 2 for b in bins],
                     [b['kernel_minus_triton_ttft_ms']['mean'] for b in bins],
                     marker='o', label=f'{store.title()} store held fixed', color=color)
    axes[1].axhline(0, color='#444444', linestyle=':')
    axes[1].set(xlabel='Replay time (seconds)', ylabel='Mean Direct − Triton TTFT (ms)',
                title='Same-body requests dispatched in the same time window')
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=.15)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Packed C{concurrency} store/decode ablation — one sequential run per cell', fontsize=14)
    fig.savefig(root / 'comparison.png', dpi=180)
    plt.close(fig)



def frontend_delay_diagnostics(records: list[dict], replay_start: float) -> dict:
    """Compare post-prefill wall time with other requests' API preparation.

    These intervals bracket OpenAI conversion and tokenizer dispatch, not a CPU
    sampling profile. Use their union to avoid double counting, and never infer
    a verified client/server request join from temporal overlap.
    """
    intervals = sorted((r['request_received_ts'], r['api_server_dispatch_finish_ts'])
                       for r in records)
    merged: list[list[float]] = []
    overlapping_intervals = 0
    for start, end in intervals:
        if end < start:
            raise ValueError('Negative API receive-to-dispatch interval')
        if merged and start < merged[-1][1]:
            overlapping_intervals += 1
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    steady = [r for r in records if r['request_received_ts'] >= replay_start + 180]
    groups = {}
    for name, rows in (
        ('after_180s', steady),
        ('after_180s_queue_lt_10ms_post_prefill_gt_1s', [r for r in steady
            if r['queue_time'] < .01
            and r['first_output_ts'] - r['prefill_finished_time'] > 1]),
    ):
        spans, covered, counts = [], [], []
        for r in rows:
            start, end = r['prefill_finished_time'], r['first_output_ts']
            if end < start:
                raise ValueError('Negative prefill-to-first-output interval')
            spans.append(end - start)
            covered.append(sum(max(0, min(end, b) - max(start, a)) for a, b in merged))
            counts.append(sum(a < end and b > start for a, b in intervals))
        groups[name] = {
            'count': len(rows),
            'post_prefill_s': stats(spans),
            'overlapping_api_preparation_s': stats(covered),
            'remaining_s': stats([a - b for a, b in zip(spans, covered)]),
            'intersecting_api_preparations': stats(counts),
            'covered_fraction_of_summed_post_prefill_time': sum(covered) / sum(spans),
        }
    return {
        'basis': 'Server request timestamps only; temporal overlap, not a CPU profile or causal attribution',
        'overlapping_api_intervals': overlapping_intervals,
        'first_outputs_inside_api_preparation_with_100us_margin': sum(
            any(a + .0001 < r['first_output_ts'] < b - .0001 for a, b in merged)
            for r in records),
        'steady_api_preparation_s': stats([
            r['api_server_dispatch_finish_ts'] - r['request_received_ts'] for r in steady]),
        'cohorts': groups,
    }

def main(make_plot: bool = False, root: Path = ROOT, concurrency: int = 24) -> None:
    runs, responses, hashes = {}, {}, {}
    for directory in sorted(root.glob('*-store-*-decode/*')):
        label = directory.parent.name
        summary = read(directory / 'client/summary.json')
        validation = read(directory / 'request-timing-validation.json')
        server = read(directory / 'server-info.json')
        manifest = read(directory / 'client/manifest.json')
        window = read(directory / 'replay-window.json')
        records = lines(directory / 'request-timings.jsonl')
        client = lines(directory / 'client/responses.jsonl')
        successful = [r for r in client if r['valid'] and r['status'] == 200]
        assert summary['valid'] and validation['complete'] and not validation['errors']
        assert not validation['partial'] and len(records) == validation['records']
        assert server['random_seed'] == validation['server_seed'] == 20260919
        assert len(successful) == summary['counts']['successful']
        assert manifest['replay']['max_concurrency'] == concurrency
        assert manifest['replay']['max_duration_s'] == 1200
        decode = server['dsv4_c4_cache_implementation']
        store = server.get('dsv4_c4_cache_store_implementation') or decode
        assert label == f'{store}-store-{decode}-decode'
        responses[label] = {r['request_id']: r for r in successful}
        stages = server_stats(records)
        gpu_rows = [list(map(int, line.split(','))) for line in (directory / 'gpu-samples.csv').read_text().splitlines()]
        gpu_usage = {str(index): stats([row[2] for row in gpu_rows if row[0] == index]) for index in range(4, 8)}
        steady = [r for r in records if r['request_received_ts'] >= window['started_at'] + 180]
        runs[label] = {
            'run_id': directory.name, 'store': store, 'decode': decode,
            'client_summary': summary, 'validation': validation,
            'frontend_delay_diagnostics': frontend_delay_diagnostics(records, window['started_at']),
            'server_stages': stages, 'gpu_utilization_percent': gpu_usage, 'server_stages_after_180s': server_stats(steady),
            'api_receive_to_dispatch_s': stats([
                r['api_server_dispatch_finish_ts'] - r['request_received_ts'] for r in records]),
            'requests_with_retractions': sum(r['num_retractions'] > 0 for r in records),
            'first_output_to_request_finish_s': stats([
                r['request_finished_ts'] - r['first_output_ts'] for r in records]),
            'requests_finished_within_200ms_of_first_output': sum(
                r['request_finished_ts'] - r['first_output_ts'] < .2 for r in records),
            'client_after_180s': client_stats([r for r in successful if r['dispatch_offset_s'] >= 180]),
            'server_queue_cohorts': {
                'queue_lt_10ms': server_stats([r for r in records if r['queue_time'] < .01]),
                'queue_ge_1s': server_stats([r for r in records if r['queue_time'] >= 1]),
            },
            'diagnostic_client_server_associations': diagnostic_associations(client, records),
            'ttft_gt_10s_count': sum(r['ttft_ms'] > 10000 for r in successful),
            'ttft_gt_10s_percent': 100 * sum(r['ttft_ms'] > 10000 for r in successful) / len(successful),
            'dataset_sha256': manifest['dataset']['sha256'],
            'harness_sources': manifest['audit']['sources'],
            'cache_hit_rate': summary['cache']['device_hit_rate'],
            'replay': manifest['replay'],
        }
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                hashes[str(path.relative_to(root))] = {
                    'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    assert len(runs) == 4
    assert len({r['dataset_sha256'] for r in runs.values()}) == 1
    for source in ('adapter', 'runner'):
        assert len({r['harness_sources'][source]['sha256'] for r in runs.values()}) == 1
    pairs = {}
    for a, b in itertools.combinations(runs, 2):
        shared = sorted(responses[a].keys() & responses[b].keys())
        mismatches = [k for k in shared if responses[a][k]['request_body_sha256'] != responses[b][k]['request_body_sha256']]
        if mismatches:
            raise ValueError(f'{a} versus {b}: body hash mismatch')
        pairs[f'{b}_minus_{a}'] = {
            'matched_successful_requests': len(shared), 'body_hash_mismatches': 0,
            'mean_difference_ms': {
                key: statistics.mean(responses[b][k][key] - responses[a][k][key] for k in shared)
                for key in ('ttft_ms', 'tpot_ms', 'latency_ms')
            },
            'matched_after_180s': sum(min(responses[a][k]['dispatch_offset_s'], responses[b][k]['dispatch_offset_s']) >= 180 for k in shared),
            'mean_difference_ms_after_180s': {
                key: statistics.mean(responses[b][k][key] - responses[a][k][key] for k in shared
                                     if min(responses[a][k]['dispatch_offset_s'], responses[b][k]['dispatch_offset_s']) >= 180)
                for key in ('ttft_ms', 'tpot_ms', 'latency_ms')
            },
        }
    common = sorted(set.intersection(*(set(r) for r in responses.values())))
    common_stats = {}
    for label, client in responses.items():
        common_stats[label] = {
            key: stats([client[k][key] for k in common])
            for key in ('ttft_ms', 'tpot_ms', 'latency_ms')
        }
    output = {
        'schema_version': 1, 'runs': runs, 'common_all_four_count': len(common), 'common_all_four_client_stats': common_stats, 'matched_client_comparisons': pairs,
        'fixed_store_diagnostics': fixed_store_diagnostics(responses),
        'limitations': [
            'One sequential run per combination; workload order is fixed but arrival, cache and scheduling histories differ.',
            'Server/client request IDs differ; server stages are aggregate, not per-request joined client TTFT.',
            'Prefill timestamps are scheduler wall-clock spans, not isolated GPU durations.',
            'Exported artifacts lack runtime source/binary hashes; Git pins are launch provenance, not proof of loaded binary identity.',
            'Supplementary tail associations use unique token-count signatures and consistency checks; they are not request-ID-verified joins.',
        ],
    }
    cell_means = {k: r['client_summary']['ttft_ms']['mean'] for k, r in runs.items()}
    tt, kt, tk, kk = [cell_means[k] for k in (
        'triton-store-triton-decode', 'kernel-store-triton-decode',
        'triton-store-kernel-decode', 'kernel-store-kernel-decode')]
    output['descriptive_unpaired_factor_contrasts_ms'] = {
        'average_decode_difference': ((tk - tt) + (kk - kt)) / 2,
        'average_store_difference': ((kt - tt) + (kk - tk)) / 2,
        'interaction': (kk - kt) - (tk - tt),
        'limitation': 'One sequential run per cell; run order and scheduling histories can confound these descriptive contrasts.',
    }
    (root / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    (root / 'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'files': hashes,
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    with (root / 'summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['store', 'decode', 'run_id', 'successful', 'failed', 'ttft_mean_ms', 'ttft_p90_ms', 'tpot_mean_ms', 'ttft_gt_10s_percent', *STAGES])
        for r in runs.values():
            s = r['client_summary']
            writer.writerow([r['store'], r['decode'], r['run_id'], s['counts']['successful'], s['counts']['failed'], s['ttft_ms']['mean'], s['ttft_ms']['p90'], s['tpot_ms']['mean'], r['ttft_gt_10s_percent'], *[r['server_stages'][k]['mean'] for k in STAGES]])
    if make_plot:
        plot_diagnostics(output, responses, root, concurrency)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true', help='Also render comparison.png (requires matplotlib)')
    parser.add_argument('--root', type=Path, default=ROOT, help='Directory containing the four configuration/run directories')
    parser.add_argument('--concurrency', type=int, choices=(24, 28), default=24)
    args = parser.parse_args()
    main(args.plot, args.root.resolve(), args.concurrency)
