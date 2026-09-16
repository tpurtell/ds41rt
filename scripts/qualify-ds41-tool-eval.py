#!/usr/bin/env python3
"""Run native tool-eval qualification with thinking enabled at high effort.

Runs are sequential; the benchmark uses the requested concurrency internally.
The vllm label selects its OpenAI-compatible adapter, not the serving engine.
The benchmark caps each response, including reasoning, at 4096 tokens by default.
Use --max-tokens to select another cap explicitly.
Use --runs 3 only for the final qualified release artifact. Preserve the whole
output directory, including failed cases, commands, logs and raw tool traces.
"""
import argparse
import collections
import json
from pathlib import Path
import sqlite3
import subprocess


def collect(directory):
    result = json.loads((directory / 'tool-eval.json').read_text())
    assert result['status'] == 'completed', result['status']
    scenarios = result['scores']['scenario_results']
    assert len(scenarios) == 88 and len({r['scenario_id'] for r in scenarios}) == 88
    config = result['config']
    assert config['extra_params']['thinking']['type'] == 'enabled'
    assert config['extra_params']['reasoning_effort'] == 'high'
    connection = sqlite3.connect(directory / 'data/benchmarks.sqlite')
    traces = [dict(scenario_id=identifier, raw_log=raw) for identifier, raw in connection.execute(
        'select scenario_id,raw_log from scenario_traces where run_id=? order by scenario_id',
        (result['run_id'],))]
    connection.close()
    assert len(traces) == 88
    (directory / 'tool-eval-traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n')
    basic = [r for r in scenarios if int(r['scenario_id'][3:]) <= 69]
    hard = [r for r in scenarios if int(r['scenario_id'][3:]) > 69]
    summary = dict(run_id=result['run_id'], thinking=True, reasoning_effort='high',
                   concurrency=config['concurrency'], basic_points=sum(r['points'] for r in basic),
                   basic_max=len(basic) * 2, hard_points=sum(r['points'] for r in hard),
                   hard_max=len(hard) * 2, total_points=result['scores']['total_points'],
                   total_max=result['scores']['max_points'],
                   statuses=dict(collections.Counter(r['status'] for r in scenarios)),
                   failures=[{k: r[k] for k in ['scenario_id', 'summary', 'points']}
                             for r in scenarios if r['status'] == 'fail'],
                   output_cap=config['extra_params'].get('max_tokens', 4096),
                   output_cap_override=config['extra_params'].get('max_tokens'),
                   output_cap_source='explicit override' if 'max_tokens' in config['extra_params'] else 'benchmark default',
                   backend_note='vllm is the compatibility adapter label; the server is ds41rt serve-native.')
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--max-turns', type=int, default=12)
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Per-response output cap including reasoning (benchmark default: 4096).')
    parser.add_argument('--reference-date', required=True)
    parser.add_argument('--label', default='ds41-native-thinking-high')
    parser.add_argument('--collect-only', action='store_true',
                        help='Export an existing completed tool-eval.json and its SQLite traces.')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.collect_only:
        print(json.dumps(collect(args.output_dir), ensure_ascii=False))
        return
    assert args.base_url and args.runs > 0 and 1 <= args.parallel <= 16
    extra = dict(thinking=dict(type='enabled'), reasoning_effort='high')
    if args.max_tokens is not None:
        assert args.max_tokens > 0
        extra['max_tokens'] = args.max_tokens
    summaries = []
    for index in range(args.runs):
        directory = args.output_dir / f'run-{index + 1:02}'
        directory.mkdir(parents=True, exist_ok=False)
        command = ['tool-eval-bench', '--model', 'deepseek-ai/DeepSeek-V4.1-Flash',
                   '--backend', 'vllm', '--base-url', args.base_url, '--api-key', 'local',
                   '--format', 'openai', '--temperature', '0', '--backend-kwargs', json.dumps(extra),
                   '--hardmode', '--parallel', str(args.parallel), '--timeout', str(args.timeout),
                   '--max-turns', str(args.max_turns), '--reference-date', args.reference_date,
                   '--no-live', '--no-probe-engine', '--label', f'{args.label}-{index + 1}',
                   '--json-file', str(directory / 'tool-eval.json'), '--output-dir', str(directory / 'report')]
        (directory / 'tool-eval-command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (directory / 'tool-eval.log').open('w') as log:
            subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
        summaries.append(collect(directory))
        (args.output_dir / 'summaries.json').write_text(json.dumps(summaries, indent=2) + '\n')
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
