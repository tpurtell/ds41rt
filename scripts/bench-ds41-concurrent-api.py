#!/usr/bin/env python3
"""Concurrent warm counting, code, or topic decode through C16, including admission gaps."""
import argparse,concurrent.futures,hashlib,json,runpy,time,statistics,uuid
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:18042')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--concurrency', type=int, nargs='+', default=[1,2,4,8,16])
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--label', default='release-concurrency')
parser.add_argument('--case', choices=['counting','code','code-reasoning','topic'], default='counting')
parser.add_argument('--nonce', help='Use the same prompt nonce for controlled comparisons; defaults to a fresh UUID')
args=parser.parse_args()
if args.repeats < 1 or any(c < 1 or c > 16 for c in args.concurrency):
 parser.error('repeats must be positive and concurrency must be 1..16')
if args.output.exists() or len(set(args.concurrency)) != len(args.concurrency):
 parser.error('output must be new and concurrency values unique')
api=runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')));records=[]
corpus_path=Path(__file__).with_name('fixtures')/'release-semantic-corpus.json'
corpus=json.loads(corpus_path.read_text())
checks=runpy.run_path(str(Path(__file__).with_name('release_throughput_checks.py')))
definition={'prompt':'Count from 1 to 200, separated by commas. Output only the sequence.','max_tokens':640} if args.case=='counting' else corpus['cases'][args.case]
prompt=f"{args.label} {args.nonce if args.nonce is not None else uuid.uuid4().hex}. {definition['prompt']}"
def validate(result):
 if args.case=='counting':
  assert [x.strip() for x in result['text'].split(',')]==[str(x) for x in range(1,201)],'counting sequence was incorrect'
  return {'response_nonempty':True,'objective_checks':{'counting_sequence':{'passed':True}},'objective_checks_passed':True,'prose_quality_assessed':False}
 checked=checks['check_output'](args.case,result['text'])
 assert checked['response_nonempty'] and checked['objective_checks_passed'] is not False,checked
 return checked
def run(i):
 b=api['payload'](prompt,True);b['max_tokens']=definition['max_tokens']
 b['thinking']={'type':definition.get('thinking','disabled')}
 if definition.get('reasoning_effort'):b['reasoning_effort']=definition['reasoning_effort']
 start=time.perf_counter();r=api['stream_case'](args.base_url,b)
 if definition.get('thinking')=='enabled':assert r['reasoning'].strip(),'missing requested reasoning'
 r.pop('events',None)
 return dict(start=start,result=r,output_checks=validate(r))
warmup=run(0)['result'];reference=[warmup['text'],{k:warmup['usage'][k] for k in ['prompt_tokens','completion_tokens','total_tokens']}]
warmup_checks=validate(warmup)
for c in args.concurrency:
 for repeat in range(args.repeats):
  with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:rows=list(pool.map(run,range(c)))
  for row in rows:
   usage=row['result']['usage'];pair=[row['result']['text'],{k:usage[k] for k in ['prompt_tokens','completion_tokens','total_tokens']}]
   if args.case=='counting':assert pair==reference,(c,repeat,'counting output changed')
   assert usage['prompt_cache_hit_tokens']==usage['prompt_tokens'],(c,repeat,'prompt was not warm')
  # Inclusive span from earliest reasoning/answer delta to finish, including admission gaps.
  begin=min(r['start']+r['result']['first_output_seconds'] for r in rows)
  end=max(r['start']+r['result']['finish_seconds'] for r in rows)
  agg=sum(r['result']['usage']['completion_tokens']-1 for r in rows)/(end-begin)
  record=dict(concurrency=c,repeat=repeat+1,aggregate_tps=agg,per_stream_mean=statistics.mean(r['result']['observed_decode_tokens_per_second'] for r in rows),rows=rows)
  records.append(record);args.output.write_text(json.dumps(records,indent=2))
  print(c,repeat+1,round(agg,2),round(record['per_stream_mean'],2),flush=True)
summaries=[]
for c in args.concurrency:
 values=[row['aggregate_tps'] for row in records if row['concurrency']==c]
 summaries.append(dict(concurrency=c,samples=len(values),median_aggregate_tps=statistics.median(values),min_aggregate_tps=min(values),max_aggregate_tps=max(values)))
args.output.write_text(json.dumps(dict(scope=__doc__,base_url=args.base_url,label=args.label,case=args.case,corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),prompt=prompt,max_tokens=definition['max_tokens'],thinking=definition.get('thinking','disabled'),reasoning_effort=definition.get('reasoning_effort'),warmup=warmup,warmup_checks=warmup_checks,concurrency=args.concurrency,repeats=args.repeats,records=records,summaries=summaries,passed=True),indent=2)+'\n')
