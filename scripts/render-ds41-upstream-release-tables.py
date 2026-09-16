#!/usr/bin/env python3
"""Render measured throughput tables once the complete summary is available."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads(a.summary.read_text());assert d['performance_matrix_passed']
root=Path(__file__).resolve().parents[1];old=json.loads((root/'docs/phase2-release-performance.json').read_text())
official=json.loads((root/'docs/release-v1-performance.json').read_text())['eight_type_and_counting']['official_flash']
ref={x['case']:x['observed_decode_tokens_per_second'] for x in official['cases']};ref['counting']=official['counting']['observed_decode_tokens_per_second']
x=d['layouts'];s=x['single'];t=x['dual'];lines=['RTX power limit: **400 W per card**, with **standard memory speed** (no memory overclock). Loaded memory clock is typically 13,365 MHz; monitored maximum is at most 14,001 MHz.', '']
def f(v):return f'{v:,.2f}'
def pct(v,b):return f'{100*(v/b-1):+.1f}%'
def section(title,description,headers,rows):
 lines.extend([f'**{title}.** {description}','', '| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|'])
 lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows);lines.append('')
def rate(l,c):return x[l]['concurrency'][c]['summaries'][-1]['median_aggregate_tps']
metrics=[('Best median prefill',lambda q:max(c['median_effective_prefill_tokens_per_second'] for c in q['prefill']['cells'])),('Counting target-only decode',lambda q:q['target_decode']['cases']['counting']['median_tps']),('Counting dSpark decode',lambda q:q['decode']['cases']['counting']['median_tps']),('Weighted eight-type target-only decode',lambda q:q['target_decode']['median_weighted_tps']),('Weighted eight-type dSpark decode',lambda q:q['decode']['median_weighted_tps']),*[(f'C16 {c} aggregate',lambda q,c=c:q['concurrency'][c]['summaries'][-1]['median_aggregate_tps']) for c in ['code','topic','counting']],('C16 mixed aggregate',lambda q:q['mixed']['summaries'][-1]['median_aggregate_tps'])]
rows=[]
for title,fn in metrics:rows.append([title,f(fn(s)),f(fn(t)),pct(fn(t),fn(s))])
section('Decode headlines','Median tokens/s; counting is outside the weighted real-content score.',['Measurement','1 RTX','2 RTX','2 RTX change'],rows)
rows=[]
for title,fn in metrics:rows.append([title,f(fn(old['layouts']['single'])),f(fn(s)),pct(fn(s),fn(old['layouts']['single'])),f(fn(old['layouts']['dual'])),f(fn(t)),pct(fn(t),fn(old['layouts']['dual']))])
section('Change from v3','Matched prompts and three measured samples per mode. Numerical changes may change generated text and draft acceptance.',['Measurement','v3 1 RTX','v4 1 RTX','Change','v3 2 RTX','v4 2 RTX','Change'],rows)
cases={'code':'Code','math':'Math','fable':'Fable','hello':'Hello','topic':'Topic','structured-json':'Natural JSON','structured-json-schema':'Schema JSON','multilingual':'Multilingual','counting':'Counting 1–200'}
rows=[]
for c,label in cases.items():
 rows.append([label,*[f(x[l][mode]['cases'][c]['median_tps']) for l in ['single','dual'] for mode in ['target_decode','decode']],f(ref[c]) if ref[c] is not None else 'HTTP 400'])
section('Eight content types and counting','Local results have three samples. Official Flash is the preserved one-shot prior reference, including its prior fable wording; it was not rerun. Code, math and JSON have objective checks; open prose is unscored.',['Case','1 RTX target','1 RTX dSpark','2 RTX target','2 RTX dSpark','Official Flash'],rows)
for l,label in [('single','One-RTX'),('dual','Two-RTX')]:
 q=x[l]['prefill'];lookup={(c['base_context_tokens'],c['suffix_tokens']):c['median_effective_prefill_tokens_per_second'] for c in q['cells']}
 rows=[[str(b//1024)+'K' if b else '0',*[f'{lookup[b,z]:,.0f}' for z in q['suffixes']]] for b in q['bases']]
 section(f'{label} prefill matrix','Median effective tokens/s after one shape warmup, with three samples per cell and verified parent reuse.',['Retained base',*[f'+{z//1024}K' for z in q['suffixes']]],rows)
rows=[]
retained={l:{r['context_tokens']:r for key in ['retained_decode','retained_decode_2k'] for r in x[l][key]['context_summaries']} for l in ['single','dual']}
for n in sorted(retained['single']):
 rows.append([str(n//1024)+'K' if n else '0',*[f(retained[l][n]['weighted_observed_decode_tokens_per_second']) for l in ['single','dual']],*[f"{retained[l][n]['serving_completed']}/{retained[l][n]['cache_valid']}" for l in ['single','dual']]])
section('Decode over retained context','Three samples for each of eight content types per base. The 2K row is a separate realistic-context measurement without a matching v3 baseline; it is not substituted for the original zero-context protocol.',['Retained base','1 RTX weighted dSpark','2 RTX weighted dSpark','1 RTX completed/cache-valid','2 RTX completed/cache-valid'],rows)
rows=[]
for i,c in enumerate([1,2,4,8,16]):rows.append([c,*[f(x[l]['concurrency'][case]['summaries'][i]['median_aggregate_tps']) for case in ['counting','code','topic'] for l in ['single','dual']]])
section('Concurrency scaling','Median aggregate tokens/s across three runs, timed from earliest first content to last completion, including admission gaps.',['Concurrency','1 RTX counting','2 RTX counting','1 RTX code','2 RTX code','1 RTX topic','2 RTX topic'],rows)
rows=[]
for i,c in enumerate([1,2,4,8,16]):
 row=[c]
 for l in ['single','dual']:
  z=x[l]['mixed']['summaries'][i];row.append(f"{f(z['median_aggregate_tps'])} ({f(z['min_aggregate_tps'])}–{f(z['max_aggregate_tps'])})")
 rows.append(row)
section('Mixed traffic','Fixed code/fable/topic mix with simultaneous admission and nonce seed 56001. Three-sweep ranges preserve workload and scheduling variation.',['Concurrency','1 RTX median (range)','2 RTX median (range)'],rows)
section('Startup', 'Measured standard-script startup for the qualified image.',
 ['Layout', 'Startup seconds'], [[label, f(d['startup_seconds'][layout])] for layout,label in [('single','1 RTX'),('dual','2 RTX')]])
rows=[]
for phase,gpus in sorted(d['gpu_telemetry'].items()):
 for uuid,gpu in sorted(gpus.items()):
  rows.append([phase,uuid, f(gpu['peak_memory_mib']/1024),f(gpu['peak_power_watts']),f(gpu['peak_memory_clock_mhz'])])
section('Observed GPU peaks', 'Sampled device usage across each performance phase, including idle monitored cards. Used memory includes weights, KV and workspaces; it is not KV capacity. Peaks need not occur simultaneously.',
 ['Phase','GPU UUID','Used GiB','Power W','Memory MHz'],rows)
tools=json.loads((root/'docs/sparkinfer-upstream-tool-eval-20260916.json').read_text())
rows=[[run['run_id'],f"{run['basic_points']}/{run['basic_max']}",f"{run['hard_points']}/{run['hard_max']}",f"{run['total_points']}/{run['total_max']}"] for run in tools['runs']]
section('Tool calling', 'Exactly three completed campaigns, thinking enabled at high effort, C16, and a 4,096-token response cap including reasoning. These runs used the earlier clean integration image; later graph-lifetime changes receive focused checks. See the tool-eval report for image provenance and failures.',
 ['Run','Basic','Hard','Total'],rows)
a.output.write_text('\n'.join(lines))
