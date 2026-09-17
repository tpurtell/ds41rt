#!/usr/bin/env python3
"""Render matched full/EXL3 release measurements; never substitute old results."""
import argparse
import json
import hashlib
import runpy
from pathlib import Path

MODELS = ('full', 'exl3')
LAYOUTS = ('single', 'dual')
LABELS = {'full': 'Full', 'exl3': 'EXL3'}
CASES = {'code': 'Code', 'code-reasoning': 'Code with reasoning', 'math': 'Math',
         'fable': 'Fable', 'hello': 'Hello', 'topic': 'Topic',
         'structured-json': 'Natural JSON', 'structured-json-schema': 'Schema JSON',
         'multilingual': 'Multilingual', 'counting': 'Counting 1–200'}
CONCURRENCY = (1, 2, 4, 8, 16)


def fmt(value):
    return f'{value:,.2f}'


def change(new, old):
    return f'{100*(new/old-1):+.1f}%'


def indexed(rows, field):
    result = {row[field]: row for row in rows}
    assert len(result) == len(rows), f'duplicate {field}'
    return result


def validate(reports):
    assert set(reports) == set(MODELS)
    expected = set(CASES) - {'counting'}
    for model, report in reports.items():
        assert report['performance_matrix_passed'] and report['model'] == model
        assert set(report['weighted_case_ids']) == expected
        assert report['controls']['rtx_power_limit_watts'] == 400
        assert report['controls']['repeats'] == 3
        reason = report['controls']['case_controls']['code-reasoning']
        assert reason['thinking'] == 'enabled' and reason['reasoning_effort'] == 'high'
        for layout in LAYOUTS:
            result = report['layouts'][layout]
            for mode in ('decode', 'target_decode'):
                assert result[mode]['passed'] and result[mode]['repeats'] == 3
                assert set(result[mode]['cases']) == set(CASES)
                assert all(row['samples'] == 3 for row in result[mode]['cases'].values())
            for case in ('counting', 'code', 'topic'):
                assert set(indexed(result['concurrency'][case]['summaries'], 'concurrency')) == set(CONCURRENCY)
            assert set(indexed(result['mixed']['summaries'], 'concurrency')) == set(CONCURRENCY)
            assert len(report['readiness_memory'][layout]['gpus']) == (1 if layout == 'single' else 2)
    # Both checkpoints must use the same qualified engine and measured corpus.
    for field in ('binary_sha256', 'source_manifest_sha256', 'corpus_sha256'):
        assert reports['full'][field] == reports['exl3'][field], f'mismatched {field}'
    baseline = reports['full']['layouts']['single']
    for report in reports.values():
        for result in report['layouts'].values():
            for field in ('bases', 'suffixes', 'context_sha256'):
                assert result['prefill'][field] == baseline['prefill'][field], f'mismatched prefill {field}'
            for case in ('counting', 'code', 'topic'):
                assert result['concurrency'][case]['prompt'] == baseline['concurrency'][case]['prompt'], \
                    f'mismatched concurrency prompt: {case}'


def load_acceptance(directory, reports):
    """Reproduce rates from saved traces and check the measured engine/corpus."""
    load = lambda path: json.loads(path.read_text())
    complete = load(directory / 'complete.json')
    assert len(complete['results']) == 4 and all(row['passed'] for row in complete['results'])
    assert not load(directory / 'restoration.json')['errors']
    here = Path(__file__).parent
    parse = runpy.run_path(str(here / 'summarize-ds41-native-policy.py'))['parse']
    summarize = runpy.run_path(str(here / 'collect-ds41-content-acceptance.py'))['summarize_acceptance']
    result = {}
    for model in MODELS:
        for layout, count in [('single', 1), ('dual', 2)]:
            path = directory / f'{model}-rtx{count}'
            deployment = load(path / 'deployment.json')
            summary = load(path / 'content/summary.json')
            assert deployment['binary_sha256'] == reports[model]['binary_sha256']
            assert deployment['coordinator']['Config']['Labels']['org.opencontainers.image.revision'] == reports[model]['engine_commit']
            assert summary['corpus_sha256'] == reports[model]['corpus_sha256']
            assert summary['passed'] and summary['repeats'] == 3 and summary['concurrency'] == 1
            assert summary['rtx_gpus'] == count and summary['draft_policy'] == 'adaptive'
            assert set(summary['cases']) == set(CASES) - {'counting'}
            for case, row in summary['cases'].items():
                raw = (path / f'content/{case}-trace.log').read_bytes()
                assert hashlib.sha256(raw).hexdigest() == row['trace_sha256']
                observations, _ = parse(raw.decode())
                assert len({o['request_id'] for o in observations}) == 3
                assert summarize(observations) == row['acceptance']
                client = load(path / f'content/{case}.json')
                assert client['passed'] and len(client['samples']) == 3
                assert {sample['repeat'] for sample in client['samples']} == {1, 2, 3}
                for sample in client['samples']:
                    assert sample['case'] == case and sample['passed']
                    assert sample['request']['thinking']['type'] == row['thinking']
                    assert sample['request'].get('reasoning_effort') == row['reasoning_effort']
                if case == 'code-reasoning':
                    assert row['thinking'] == 'enabled' and row['reasoning_effort'] == 'high'
            result[model, layout] = summary
    return result


def render(reports, official=None, tools=None, acceptance=None):
    validate(reports)
    lines = ['RTX measurements use **400 W per card** and **standard memory speed, without a memory overclock**. '
             'All performance tables use FP8 PLE. Reasoning code uses thinking enabled at high effort; '
             'its throughput includes reasoning and final-answer tokens.', '']

    def table(title, description, headers, rows):
        lines.extend([f'**{title}.** {description}', '', '| '+' | '.join(headers)+' |',
                      '|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|'])
        lines.extend('| '+' | '.join(map(str, row))+' |' for row in rows)
        lines.append('')

    def layout(model, name):
        return reports[model]['layouts'][name]

    def triple(fn, model):
        one, two = [fn(layout(model, name)) for name in LAYOUTS]
        return [fmt(one), fmt(two), change(two, one)]

    def concurrent(result, case, c):
        return indexed(result['concurrency'][case]['summaries'], 'concurrency')[c]['median_aggregate_tps']

    metrics = [
        ('Best median prefill', lambda q: max(c['median_effective_prefill_tokens_per_second'] for c in q['prefill']['cells'])),
        ('Counting target-only decode', lambda q: q['target_decode']['cases']['counting']['median_tps']),
        ('Counting dSpark decode', lambda q: q['decode']['cases']['counting']['median_tps']),
        ('Weighted nine-category target-only decode', lambda q: q['target_decode']['median_weighted_tps']),
        ('Weighted nine-category dSpark decode', lambda q: q['decode']['median_weighted_tps']),
        *[(f'C16 {case} aggregate', lambda q, case=case: concurrent(q, case, 16)) for case in ('code', 'topic', 'counting')],
        ('C16 mixed aggregate', lambda q: indexed(q['mixed']['summaries'], 'concurrency')[16]['median_aggregate_tps'])]
    columns = ['Measurement', 'Full 1 RTX', 'Full 2 RTX', 'Change', 'EXL3 1 RTX', 'EXL3 2 RTX', 'Change']
    table('Headlines', 'Median tokens/s. Changes compare two RTX cards with one for the same checkpoint. '
          'Counting is outside the weighted content score.', columns,
          [[label, *triple(fn, 'full'), *triple(fn, 'exl3')] for label, fn in metrics])
    for model in MODELS:
        headers = ['Case', '1 RTX target', '1 RTX dSpark', '2 RTX target', '2 RTX dSpark']
        if official is not None:
            headers.append('Historical official Flash')
        rows = []
        for case, label in CASES.items():
            row = [label, *[fmt(layout(model, name)[mode]['cases'][case]['median_tps'])
                           for name in LAYOUTS for mode in ('target_decode', 'decode')]]
            if official is not None:
                row.append(fmt(official[case]) if official.get(case) is not None
                           else ('HTTP 400' if case in official else '—'))
            rows.append(row)
        table(f'{LABELS[model]} content-type decode', 'Three samples per case. The nine-category score retains '
              'non-thinking code separately from reasoning code. Historical official results, when shown, '
              'were not rerun and include different prior fable wording; no reasoning-code reference exists.', headers, rows)
    for model in MODELS:
        for name, label in [('single', '1 RTX'), ('dual', '2 RTX')]:
            q = layout(model, name)['prefill']
            cells = {(c['base_context_tokens'], c['suffix_tokens']): c['median_effective_prefill_tokens_per_second'] for c in q['cells']}
            assert len(cells) == len(q['cells']) == len(q['bases'])*len(q['suffixes'])
            rows = [[f'{base//1024}K', *[f'{cells[base, suffix]:,.0f}' for suffix in q['suffixes']]] for base in q['bases']]
            table(f'{LABELS[model]} {label} prefill matrix', 'Median effective tokens/s, three samples per cell '
                  'after shape warmup and verified parent reuse.', ['Retained base', *[f'+{s//1024}K' for s in q['suffixes']]], rows)
    retained = {(model, name): indexed([r for key in ('retained_decode', 'retained_decode_2k')
                                       for r in layout(model, name)[key]['context_summaries']], 'context_tokens')
                for model in MODELS for name in LAYOUTS}
    contexts = set(retained['full', 'single'])
    assert all(set(value) == contexts for value in retained.values())
    rows = []
    for context in sorted(contexts):
        row = [f'{context//1024}K']
        for model in MODELS:
            one, two = [retained[model, name][context]['weighted_observed_decode_tokens_per_second'] for name in LAYOUTS]
            row.extend([fmt(one), fmt(two), change(two, one)])
        rows.append(row)
    table('Decode over retained context', 'Weighted nine-category dSpark tokens/s with verified retained-prefix reuse; '
          'three samples per case and base.', ['Retained base', *columns[1:]], rows)
    for model in MODELS:
        rows = [[c, *[fmt(concurrent(layout(model, name), case, c)) for case in ('counting', 'code', 'topic') for name in LAYOUTS]]
                for c in CONCURRENCY]
        table(f'{LABELS[model]} concurrency scaling', 'Median aggregate tokens/s across three runs, from earliest '
              'first output to final completion, including admission gaps.',
              ['Concurrency', '1 RTX counting', '2 RTX counting', '1 RTX code', '2 RTX code', '1 RTX topic', '2 RTX topic'], rows)
    rows = []
    for c in CONCURRENCY:
        row = [c]
        for model in MODELS:
            for name in LAYOUTS:
                q = indexed(layout(model, name)['mixed']['summaries'], 'concurrency')[c]
                row.append(f"{fmt(q['median_aggregate_tps'])} ({fmt(q['min_aggregate_tps'])}–{fmt(q['max_aggregate_tps'])})")
        rows.append(row)
    table('Mixed traffic', 'Matched code/fable/topic mix; median and range across three sweeps.',
          ['Concurrency', 'Full 1 RTX', 'Full 2 RTX', 'EXL3 1 RTX', 'EXL3 2 RTX'], rows)
    if acceptance is not None:
        assert set(acceptance) == {(model, layout) for model in MODELS for layout in LAYOUTS}
        rows = []
        for case, label in CASES.items():
            if case == 'counting':
                continue
            row = [label + (' (grammar-constrained)' if case == 'structured-json-schema' else '')]
            group = 'grammar_constrained' if case == 'structured-json-schema' else 'unconstrained'
            for model in MODELS:
                for name in LAYOUTS:
                    q = acceptance[model, name]['cases'][case]['acceptance'][group]
                    assert q['nonterminal_observations'] > 0 and q['verified_drafts'] > 0
                    assert 0 <= q['accepted_drafts'] <= q['verified_drafts']
                    assert q['acceptance'] == q['accepted_drafts'] / q['verified_drafts']
                    row.append(f"{100*q['acceptance']:.2f}% ({q['mean_emitted_tokens']:.2f})")
            rows.append(row)
        table('Adaptive draft acceptance by content',
              'C1, three requests per category. Each cell shows accepted/verified draft percentage '
              'and mean emitted tokens per observed nonterminal verification cycle in parentheses. '
              'Terminal cycles are excluded; schema JSON uses grammar-constrained targets. '
              'Reasoning code includes both reasoning and final-answer generation. Adaptive selection '
              'censors unverified drafts, and model continuations differ: these are serving acceptance '
              'measurements, not teacher-forced quant agreement. Instrumented timings are excluded from TPS tables.',
              ['Content', 'Full 1 RTX', 'Full 2 RTX', 'EXL3 1 RTX', 'EXL3 2 RTX'], rows)
    deployments, startups, memory = [], [], []
    for model in MODELS:
        for name, count in [('single', 1), ('dual', 2)]:
            label = f'{LABELS[model]} {count} RTX'
            q = reports[model]['deployment'][name]
            deployments.append([label, f"0–{q['rtx_expert_layers']-1}, TP{q['rtx_expert_tp']}",
                f"{q['spark_layers']} resident / {q.get('spark_active_layers', 40-q['rtx_expert_layers'])} active / {q['spark_budget_bytes_each']/2**30:g} GiB each",
                f"{q['global_pool_bytes']/1e9:.3f} GB / {q['logical_pool_tokens']:,} logical + {q['private_tail_tokens']:,} private-tail tokens",
                f"{q['prompt_retention_entries']} / {q['completed_turn_retention_entries']}"])
            startups.append([label, fmt(reports[model]['startup_seconds'][name])])
            for index, gpu in enumerate(reports[model]['readiness_memory'][name]['gpus']):
                memory.append([label, index, fmt(gpu['used_mib']), fmt(gpu['free_mib']), fmt(q['runtime_headroom_bytes_per_gpu']/2**20)])
    table('Deployment and cache capacity', 'Measured placement and pool reservation. Spark resident layers can include unused weights below the active range; retention values count cache entries.',
          ['Configuration', 'RTX expert layers', 'Spark resident / active layers / budget', 'Global FP4 source pool', 'Prompt / completed retention'], deployments)
    table('Startup', 'Standard-script launch to API readiness, including orchestration; one launch per configuration.',
          ['Configuration', 'Seconds'], startups)
    table('Memory after readiness', 'All GPU allocations, including weights, KV and workspace.',
          ['Configuration', 'GPU', 'Loaded MiB', 'Free MiB', 'Runtime reserve MiB'], memory)
    if tools is not None:
        assert set(tools) == {'full', 'exl3', 'exl3-fp4ple'}
        rows = []
        for model, label in [('full', 'Full / FP8 PLE'), ('exl3', 'EXL3 / FP8 PLE'), ('exl3-fp4ple', 'EXL3 / FP4 PLE')]:
            runs = tools[model]['runs']
            assert len(runs) == len({run['run_id'] for run in runs}) == 3
            for run in runs:
                assert (run['thinking'] is True or run['thinking'] == 'enabled') and run['reasoning_effort'] == 'high'
                rows.append([label, run['run_id'], *[f"{run[key+'_points']}/{run[key+'_max']}" for key in ('basic', 'hard', 'total')]])
        table('Tool calling', 'Three campaigns per checkpoint variant with high-effort thinking. Failures remain in '
              'the scores; see each campaign report for its engine/image provenance and response cap.',
              ['Checkpoint', 'Run', 'Basic', 'Hard', 'Total'], rows)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('full', 'exl3', 'full-tools', 'exl3-tools', 'fp4ple-tools', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--acceptance-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    load = lambda path: json.loads(path.read_text())
    root = Path(__file__).resolve().parents[1]
    reference = load(root/'docs/release-v1-performance.json')['eight_type_and_counting']['official_flash']
    official = {row['case']: row['observed_decode_tokens_per_second'] for row in reference['cases']}
    official['counting'] = reference['counting']['observed_decode_tokens_per_second']
    reports = {'full': load(args.full), 'exl3': load(args.exl3)}
    acceptance = load_acceptance(args.acceptance_dir, reports)
    text = render(reports, official,
                  {'full': load(args.full_tools), 'exl3': load(args.exl3_tools), 'exl3-fp4ple': load(args.fp4ple_tools)},
                  acceptance)
    args.output.write_text(text)


if __name__ == '__main__':
    main()
