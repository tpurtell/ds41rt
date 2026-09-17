#!/usr/bin/env python3
"""Render one native v6 table set for both README and the performance report."""
import argparse
import csv
import io
import json
from pathlib import Path

CASES = {'code': 'Code', 'code-reasoning': 'Code with reasoning', 'math': 'Math',
         'fable': 'Fable', 'hello': 'Hello', 'topic': 'Topic',
         'structured-json': 'Natural JSON', 'structured-json-schema': 'Schema JSON',
         'multilingual': 'Multilingual', 'counting': 'Counting 1–200'}
LAYOUTS = ('single', 'dual')


def number(value):
    return f'{value:,.2f}'


def change(new, old):
    return f'{100 * (new / old - 1):+.1f}%'


def render(report, official):
    assert report['release'] == 'v6' and report['performance_matrix_passed']
    assert set(report['layouts']) == set(LAYOUTS)
    assert set(report['weighted_case_ids']) == set(CASES) - {'counting'}
    assert report['controls']['repeats'] == 3 and report['controls']['rtx_power_limit_watts'] == 400
    layouts = report['layouts']
    lines = ['RTX measurements use **400 W per card and standard memory speed, without a memory overclock**. '
             'Each performance cell has three samples. Reasoning code uses high-effort thinking; '
             'other throughput cases disable thinking. All new TP2 switches are off.', '']

    def table(title, description, headers, rows):
        lines.extend([f'**{title}.** {description}', '', '| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] + ['---:'] * (len(headers) - 1)) + '|'])
        for row in rows:
            assert len(row) == len(headers)
            lines.append('| ' + ' | '.join(map(str, row)) + ' |')
        lines.append('')

    def concurrency(result, case, count):
        rows = result['concurrency'][case]['summaries']
        return next(row['median_aggregate_tps'] for row in rows if row['concurrency'] == count)

    metrics = [
        ('Best median prefill', lambda r: max(c['median_effective_prefill_tokens_per_second'] for c in r['prefill']['cells'])),
        ('Counting target-only decode', lambda r: r['target_decode']['cases']['counting']['median_tps']),
        ('Counting dSpark decode', lambda r: r['decode']['cases']['counting']['median_tps']),
        ('Weighted nine-category target-only decode', lambda r: r['target_decode']['median_weighted_tps']),
        ('Weighted nine-category dSpark decode', lambda r: r['decode']['median_weighted_tps']),
        *[(f'C16 {case} aggregate', lambda r, case=case: concurrency(r, case, 16)) for case in ['code', 'topic', 'counting']],
        ('C16 mixed aggregate', lambda r: next(c['median_aggregate_tps'] for c in r['mixed']['summaries'] if c['concurrency'] == 16)),
    ]
    table('Headlines', 'Tokens/s. Changes compare two RTX cards with one. Counting is outside the weighted score.',
          ['Measurement', '1 RTX', '2 RTX', 'Change'],
          [[title, number(fn(layouts['single'])), number(fn(layouts['dual'])), change(fn(layouts['dual']), fn(layouts['single']))]
           for title, fn in metrics])
    table('Content-type decode', 'Median tokens/s. Official Flash values are the historical one-shot reference, '
          'including its prior fable wording; they were not rerun.',
          ['Case', '1 RTX target', '1 RTX dSpark', '2 RTX target', '2 RTX dSpark', 'Historical official Flash'],
          [[title, *[number(layouts[layout][mode]['cases'][case]['median_tps'])
                      for layout in LAYOUTS for mode in ['target_decode', 'decode']],
            ('—' if case not in official else 'HTTP 400' if official[case] is None else number(official[case]))]
           for case, title in CASES.items()])
    for layout, label in [('single', '1 RTX'), ('dual', '2 RTX')]:
        prefill = layouts[layout]['prefill']
        cells = {(row['base_context_tokens'], row['suffix_tokens']): row['median_effective_prefill_tokens_per_second']
                 for row in prefill['cells']}
        table(f'{label} prefill matrix', 'Median effective tokens/s after shape warmup and verified parent reuse.',
              ['Retained base', *[f'+{suffix // 1024}K' for suffix in prefill['suffixes']]],
              [[f'{base // 1024}K', *[f'{cells[base, suffix]:,.0f}' for suffix in prefill['suffixes']]] for base in prefill['bases']])
    retained = {layout: {row['context_tokens']: row['weighted_observed_decode_tokens_per_second']
                        for key in ['retained_decode', 'retained_decode_2k']
                        for row in layouts[layout][key]['context_summaries']} for layout in LAYOUTS}
    assert set(retained['single']) == set(retained['dual']) == {0, 2048, 32768, 65536, 131072, 262144}
    table('Decode over retained context', 'Weighted nine-category dSpark tokens/s with verified prefix reuse.',
          ['Retained base', '1 RTX', '2 RTX', 'Change'],
          [[f'{base // 1024}K', number(retained['single'][base]), number(retained['dual'][base]),
            change(retained['dual'][base], retained['single'][base])] for base in sorted(retained['single'])])
    table('Concurrency scaling', 'Median aggregate tokens/s from earliest first output to final completion, including admission gaps.',
          ['Concurrency', *[f'{label} {case}' for case in ['counting', 'code', 'topic'] for label in ['1 RTX', '2 RTX']]],
          [[count, *[number(concurrency(layouts[layout], case, count))
                     for case in ['counting', 'code', 'topic'] for layout in LAYOUTS]] for count in [1, 2, 4, 8, 16]])
    mixed = {layout: {row['concurrency']: row for row in layouts[layout]['mixed']['summaries']} for layout in LAYOUTS}
    def mixed_cell(layout, count):
        row = mixed[layout][count]
        return f"{number(row['median_aggregate_tps'])} ({number(row['min_aggregate_tps'])}–{number(row['max_aggregate_tps'])})"
    table('Mixed traffic', 'Code/fable/topic mix; aggregate tokens/s median and range across three sweeps.',
          ['Concurrency', '1 RTX', '2 RTX'],
          [[count, *[mixed_cell(layout, count) for layout in LAYOUTS]] for count in [1, 2, 4, 8, 16]])
    deployments = []
    for layout, label in [('single', '1 RTX'), ('dual', '2 RTX')]:
        launch = report['launches'][layout]
        d, ram = launch['deployment'], launch['ram_capacity']
        deployments.append([label, f"0–{d['rtx_expert_layers'] - 1}, TP{d['rtx_expert_tp']}",
                            f"{d['spark_layers']} / {d['spark_active_layers']} / {d['spark_budget_bytes_each'] / 2**30:g} GiB each",
                            f"{d['global_pool_bytes'] / 1e9:.3f} GB / {ram['device_tokens']:,} usable + {d['private_tail_tokens']:,} private-tail tokens",
                            f"{ram['pinned_bytes'] / 2**30:g} GiB pinned / {ram['host_tokens']:,} logical tokens",
                            f"{ram['combined_tokens']:,}",
                            f"{d['prompt_retention_entries']} / {d['completed_turn_retention_entries']}"])
    table('Deployment and cache capacity', 'RAM bytes include staging and snapshot overhead. Combined tokens count each logical source once; active requests must fit the GPU pool.',
          ['Configuration', 'RTX expert layers', 'Spark resident / active layers / budget', 'Global FP4 source pool', 'RAM cache', 'Combined usable tokens', 'Prompt / turn retention'], deployments)
    table('Startup', 'Standard launcher to API readiness, including orchestration; one observation per layout.',
          ['Configuration', 'Seconds'], [[label, number(report['launches'][layout]['startup_seconds'])]
                                        for layout, label in [('single', '1 RTX'), ('dual', '2 RTX')]])
    memory = []
    for layout, label in [('single', '1 RTX'), ('dual', '2 RTX')]:
        launch = report['launches'][layout]
        rows = {row['uuid']: row for row in csv.DictReader(io.StringIO(launch['gpu_memory']), skipinitialspace=True)}
        for rank, uuid in enumerate(launch['gpu_uuids']):
            row = rows[uuid]
            memory.append([label, rank, number(float(row['memory.used [MiB]'].split()[0])),
                           number(float(row['memory.free [MiB]'].split()[0])),
                           number(launch['deployment']['runtime_headroom_bytes_per_gpu'] / 2**20)])
    table('Memory after readiness', 'GPU allocations include weights, KV and workspaces; later graph capture can consume additional memory.',
          ['Configuration', 'Logical RTX', 'Loaded MiB', 'Free MiB', 'Planned runtime reserve MiB'], memory)
    lines.append('Historical EXL3 performance, acceptance and quantization analysis are preserved in the '
                 '[v5 performance report](https://github.com/tpurtell/ds41rt/blob/v5/docs/release-v5-performance.md); '
                 'they are not v6 measurements.\n')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    historical = json.loads((root / 'docs/release-v1-performance.json').read_text())['eight_type_and_counting']['official_flash']
    official = {row['case']: row['observed_decode_tokens_per_second'] for row in historical['cases']}
    official['counting'] = historical['counting']['observed_decode_tokens_per_second']
    args.output.write_text(render(json.loads(args.summary.read_text()), official))


if __name__ == '__main__':
    main()
