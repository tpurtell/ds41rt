#!/usr/bin/env python3
"""Summarize opt-in serving traces; elapsed stages include scheduler/transport waits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


def stats(values):
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    results = []
    for window in json.loads(args.windows.read_text()):
        lines = raw[window['start_byte']:window['end_byte']].decode().splitlines()
        forecasts, layers, pending, rounds = {}, defaultdict(list), {}, []
        for line in lines:
            fields = {k:float(v) if '.' in v else int(v)
                for k,v in re.findall(r'\b(\w+)=([0-9]+(?:\.[0-9]+)?)',line)}
            if 'verification cost forecast ' in line:
                forecasts[fields['batch']] = fields
            elif 'verification layer cost ' in line:
                backend = re.search(r'routed_backend="([^"]+)"',line)[1]
                layers[fields['batch']].append(dict(fields, backend=backend))
            elif 'verification round cost ' in line:
                assert fields['lane'] not in pending, 'previous lane round missing completion'
                pending[fields['lane']] = fields
            elif 'native independent lane round ' in line:
                cost = pending.pop(fields['lane'])
                batch = cost['batch']
                entries = layers.pop(batch, [])
                assert len(entries)==40 and sorted(e['layer'] for e in entries)==list(range(40)), (batch,len(entries))
                forecast = forecasts.pop(batch, None)
                rounds.append(dict(fields, batch=batch, verifier_rows=cost['rows'],
                    predicted_verify_us=None if forecast is None else forecast['predicted_verify_us'], layers=entries))
        assert rounds and not pending, 'incomplete measured round window'
        proposed = sum(r['proposed'] for r in rounds)
        accepted = sum(r['accepted'] for r in rounds)
        assert 0 <= accepted <= proposed
        backend_samples = defaultdict(list)
        for r in rounds:
            for layer in r['layers']:
                backend_samples[layer['backend']].append(layer)
        ratios = [r['predicted_verify_us']/r['verify_us'] for r in rounds if r['predicted_verify_us'] is not None]
        result = dict(case=window['case'], concurrency=window['concurrency'], rounds=len(rounds),
            verified_draft_tokens=proposed, accepted_draft_tokens=accepted,
            accepted_fraction=accepted/proposed if proposed else None,
            request_cycles=sum(r['requests'] for r in rounds),
            emitted_tokens=sum(r['emitted'] for r in rounds),
            zero_acceptance_lane_rounds=sum(r['accepted']==0 for r in rounds),
            stages_us={key:stats([r[key] for r in rounds]) for key in ['draft_us','verify_us','total_us']},
            predicted_over_observed_verify=stats(ratios),
            backends={backend:dict(layer_samples=len(rows),
                stages_us={key:stats([r[key] for r in rows]) for key in ['produced_us','index_us','attention_us','experts_us','finish_us']},
                rows=stats([r['rows'] for r in rows]), distinct_experts=stats([r['distinct_experts'] for r in rows]))
                for backend,rows in backend_samples.items()},
            raw_rounds=rounds)
        result['emitted_per_request_cycle']=result['emitted_tokens']/result['request_cycles']
        result['verified_draft_per_request_cycle']=proposed/result['request_cycles']
        results.append(result)
    report=dict(scope=__doc__, caveats=['Includes benchmark warmup and all three measured samples.',
        'Concurrent lanes overlap; layer times must not be added as end-to-end wall time.',
        'Zero acceptance counts lane rounds, not individual request cycles.',
        'Tracing perturbs throughput; these are optimization diagnostics, not release tables.'],results=results)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for r in results:
        print(json.dumps({k:r[k] for k in ['case','concurrency','rounds','accepted_fraction',
            'emitted_per_request_cycle','verified_draft_per_request_cycle','predicted_over_observed_verify']}))


if __name__ == '__main__':
    main()
