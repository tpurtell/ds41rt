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


def validate_quant_analysis(report):
    assert report['schema'] == 1 and report['passed']
    snapshots = report['snapshots']
    assert set(snapshots) == {'full', 'exl3', 'exl3_fp4ple'}
    assert snapshots['full']['revision'] == report['exl3']['metadata']['provenance']['source_revision']
    assert report['exl3']['nominal_average_bpw'] == 3.25
    assert 3.25 < report['exl3']['packed_effective_bpw'] < 3.27
    assert report['exl3']['tensor_entries'] == 41 * 384 * 3
    assert report['hardlink_clone']['shared_shards'] == 48
    assert len(report['hardlink_clone']['changed_shards']) == 4
    for category in ('routed_expert', 'shared_expert', 'embedding', 'head', 'other'):
        assert snapshots['exl3']['categories'][category] == snapshots['exl3_fp4ple']['categories'][category]
    assert snapshots['exl3_fp4ple']['categories']['ple']['payload_bytes'] < snapshots['exl3']['categories']['ple']['payload_bytes']
    tier_counts = {}
    for row in report['exl3']['tiers']:
        key = row['projection'], row['bits']
        tier_counts[key] = tier_counts.get(key, 0) + row['tensors']
    assert sum(tier_counts.values()) == report['exl3']['tensor_entries']
    assert set(bits for projection, bits in tier_counts) == {3, 4}
    return report


def load_top1(directory):
    load = lambda path: json.loads(path.read_text())
    complete = load(directory/'complete.json')
    assert not load(directory/'restoration.json')['errors']
    assert [row['case'] for row in complete['results']] == ['full', 'exl3', 'fp4ple']
    assert all(row['passed'] for row in complete['results'])
    collector = runpy.run_path(str(Path(__file__).with_name('collect-ds41-top1-agreement.py')))
    reports = {}
    results = {}
    for result in complete['results']:
        case = result['case']; path = directory/case/'top1.json'; report = load(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == result['output_sha256']
        assert report['passed'] and len(report['samples']) == 132
        assert report['binary_sha256'] == complete['diagnostic']['binary_sha256']
        assert report['corpus_sha256'] == complete['corpus_sha256']
        assert report['protocol'] == {
            'target_only': True, 'draft_model': 'disabled', 'concurrency': 1,
            'temperature': 0, 'top_p': 1, 'max_tokens': 1, 'thinking': 'disabled',
            'token_source': 'instrumented target-model scores.select(mask) after fixed-prompt prefill',
            'constrained': False, 'comparison': 'exact token ID on byte-identical OpenAI messages'}
        trace = (directory/case/'trace.log').read_bytes()
        ids = set()
        for row in report['samples']:
            raw = trace[row['trace_start']:row['trace_end']]
            assert hashlib.sha256(raw).hexdigest() == row['trace_sha256']
            parsed = collector['parse_top1'](raw)
            assert len(parsed) == 1 and parsed[0]['token_id'] == row['token_id']
            assert parsed[0]['request_id'] == row['request_id'] and parsed[0]['prompt_tokens'] == row['prompt_tokens']
            assert row['id'] not in ids; ids.add(row['id'])
        reports[case] = report; results[case] = result
    baseline = reports['full']; baseline_rows = {row['id']:row for row in baseline['samples']}
    assert baseline['role'] == 'baseline' and baseline['overall'] is None
    baseline_sha = hashlib.sha256((directory/'full/top1.json').read_bytes()).hexdigest()
    for case in ('exl3', 'fp4ple'):
        report = reports[case]
        assert report['role'] == 'candidate' and report['reference']['sha256'] == baseline_sha
        assert collector['summary'](report['samples']) == report['overall']
        for category, value in report['categories'].items():
            assert collector['summary']([row for row in report['samples'] if row['category'] == category]) == value
        for row in report['samples']:
            assert row['reference_token_id'] == baseline_rows[row['id']]['token_id']
            assert row['matches_reference'] == (row['token_id'] == row['reference_token_id'])
    assert reports['exl3']['checkpoint_revision'] != reports['fp4ple']['checkpoint_revision']
    return dict(complete=complete, reports=reports, results=results)


def render(reports, official=None, tools=None, acceptance=None, quant=None, top1=None):
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
            source = tools[model]
            runs = source['runs'] if isinstance(source, dict) else source
            assert isinstance(runs, list)
            assert len(runs) == len({run['run_id'] for run in runs}) == 3
            for run in runs:
                assert (run['thinking'] is True or run['thinking'] == 'enabled') and run['reasoning_effort'] == 'high'
                rows.append([label, run['run_id'], *[f"{run[key+'_points']}/{run[key+'_max']}" for key in ('basic', 'hard', 'total')]])
        table('Tool calling', 'Three campaigns per checkpoint variant with high-effort thinking. Full/FP8-PLE '
              'rows preserve the clean v4 campaigns for the unchanged tool and schema path; both EXL3 variants '
              'are fresh v5 campaigns. Failures remain in the scores; see each campaign report for its '
              'engine/image provenance and response cap.',
              ['Checkpoint', 'Run', 'Basic', 'Hard', 'Total'], rows)
    if quant is not None:
        quant = validate_quant_analysis(quant)
        lines.extend(['### Quant analysis', ''])
        snapshots = quant['snapshots']; full_bytes = snapshots['full']['snapshot_bytes']
        labels = [('full', 'Full / FP8 PLE'), ('exl3', 'EXL3 / FP8 PLE'),
                  ('exl3_fp4ple', 'EXL3 / FP4 PLE')]
        rows = []
        for key, label in labels:
            snapshot = snapshots[key]
            delta = '—' if key == 'full' else f"{-100*(full_bytes-snapshot['snapshot_bytes'])/full_bytes:.1f}%"
            rows.append([label, snapshot['shard_count'], f"{snapshot['snapshot_bytes']/2**30:,.2f}", delta,
                f"{snapshot['categories']['routed_expert']['payload_bytes']/2**30:,.2f}",
                f"{snapshot['categories']['ple']['payload_bytes']/2**30:,.2f}"])
        table('Checkpoint and tensor payload sizes', 'GiB uses 2^30 bytes. Tensor columns exclude safetensors headers. '
              'Both EXL3 checkpoints share identical routed experts; FP4 PLE changes only the lookup tables.',
              ['Checkpoint', 'Shards', 'Checkpoint GiB', 'Size vs full', 'Routed-expert GiB', 'PLE GiB'], rows)
        tier_counts = {}
        tier_shapes = {}
        for row in quant['exl3']['tiers']:
            tier_counts[row['projection'], row['bits']] = tier_counts.get((row['projection'], row['bits']), 0) + row['tensors']
        for row in quant['exl3']['shapes']:
            tier_shapes.setdefault(row['projection'], (row['suh'], row['svh']))
            assert tier_shapes[row['projection']] == (row['suh'], row['svh'])
        rows = []
        for projection in ('w1', 'w2', 'w3'):
            shape = tier_shapes[projection]
            low, high = tier_counts[projection, 3], tier_counts[projection, 4]
            rows.append([projection, f"{shape[0][0]:,} × {shape[1][0]:,}", f'{low:,}', f'{high:,}', f'{100*high/(low+high):.2f}%'])
        table('EXL3 routed projection tiers', f"Target layers plus one MTP layer: 384 experts each. The aggregate is "
              f"{quant['exl3']['nominal_average_bpw']:.2f} nominal bpw and "
              f"{quant['exl3']['packed_effective_bpw']:.4f} bpw including scales and metadata.",
              ['Projection', 'Logical shape', '3-bit tensors', '4-bit tensors', '4-bit share'], rows)
        def ple_geometry(snapshot):
            shapes = snapshot['categories']['ple']['shapes']
            return '; '.join(f"{row['dtype']} " + ('×'.join(f'{value:,}' for value in row['shape']) if row['shape'] else 'scalar')
                            + f" ({row['tensors']}×)" for row in shapes)
        table('PLE table geometry', f"The FP4-PLE clone hard-links {quant['hardlink_clone']['shared_shards']} unchanged shards "
              f"and replaces {len(quant['hardlink_clone']['changed_shards'])} shards.",
              ['Variant', 'Stored tensor geometry', 'Payload GiB'], [
                  ['FP8 PLE', ple_geometry(snapshots['exl3']), f"{snapshots['exl3']['categories']['ple']['payload_bytes']/2**30:,.2f}"],
                  ['FP4 PLE', ple_geometry(snapshots['exl3_fp4ple']), f"{snapshots['exl3_fp4ple']['categories']['ple']['payload_bytes']/2**30:,.2f}"]])
    if top1 is not None:
        reports1, results1 = top1['reports'], top1['results']
        assert set(reports1) == {'full', 'exl3', 'fp4ple'}
        rows = [['Full baseline', 132, 'Reference', '—', f"{results1['full']['launch_seconds']+results1['full']['collection_seconds']:.1f}"]]
        for key, label in [('exl3', 'EXL3 / FP8 PLE'), ('fp4ple', 'EXL3 / FP4 PLE')]:
            result = reports1[key]['overall']; low, high = result['wilson_95']
            rows.append([label, result['samples'], f"{result['matches']}/{result['samples']} ({100*result['agreement']:.2f}%)",
                         f"{100*low:.2f}–{100*high:.2f}%", f"{results1[key]['launch_seconds']+results1[key]['collection_seconds']:.1f}"])
        table('Fixed-history top-1 agreement', 'Exact unconstrained target argmax IDs at C1 with dSpark disabled. '
              'All checkpoints use the same 20-layer TP2 placement, byte-identical teacher-forced assistant prefixes, '
              f"and corpus SHA-256 `{top1['complete']['corpus_sha256']}`. Runtime includes launch and collection. "
              'This isolates next-token argmax preservation; it is not a long-form generation-quality score.',
              ['Checkpoint', 'Samples', 'Matches', 'Wilson 95% CI', 'Seconds'], rows)
        categories = sorted(reports1['exl3']['categories'])
        rows = []
        for category in categories:
            row = [category.replace('-', ' ').title()]
            for key in ('exl3', 'fp4ple'):
                value = reports1[key]['categories'][category]
                row.append(f"{value['matches']}/{value['samples']} ({100*value['agreement']:.1f}%)")
            rows.append(row)
        table('Top-1 agreement by category', 'Each category uses twelve fixed continuations; percentages retain exact token-ID denominators.',
              ['Category', 'EXL3 / FP8 PLE', 'EXL3 / FP4 PLE'], rows)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('full', 'exl3', 'full-tools', 'exl3-tools', 'fp4ple-tools', 'quant-analysis', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--acceptance-dir', type=Path, required=True)
    parser.add_argument('--top1-dir', type=Path, required=True)
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
                  acceptance, load(args.quant_analysis), load_top1(args.top1_dir))
    args.output.write_text(text)


if __name__ == '__main__':
    main()
