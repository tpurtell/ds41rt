#!/usr/bin/env python3
import argparse,json,runpy,hashlib
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description='Audit raw versus fitted conditional confidence across explicit workload segments.')
parser.add_argument('directories',nargs='+',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():parser.error('output must be new')
m=runpy.run_path(str(Path(__file__).with_name('summarize-ds41-native-policy.py')))
records=[];hashes={}
for directory in args.directories:
 raw=(directory/'server.log').read_bytes();hashes[directory.name]=hashlib.sha256(raw).hexdigest()
 width=int(directory.name[-1])
 for seg in json.loads((directory/'segments.json').read_text()):
  obs,_=m['parse'](raw[seg['begin']:seg['end']].decode())
  for row in obs:
   if row['terminal'] or row['constrained'] or row['generated']<=7:continue
   row.update(width=width,workload=seg['workload'],source=directory.name)
   records.append(row)
training=[x for x in records if x['width']%2 and x['workload'] in ['code','mixed']]
def labels(rows):
 return np.array([(x['confidence'][j],int(x['matched']>j)) for x in rows
  for j in range(min(x['rows']-1,x['matched']+1))],dtype=float)
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))
values=labels(training);X=np.column_stack([np.ones(len(values)),values[:,0]]);y=values[:,1]
beta=np.array([0.,1.])
for _ in range(30):
 p=sigmoid(X@beta);w=p*(1-p)
 step=np.linalg.solve((X.T*w)@X+np.eye(2)*.01,X.T@(p-y)+.01*beta)
 beta-=step
 if np.linalg.norm(step)<1e-9:break
assert np.isfinite(beta).all() and beta[1]>0
report=dict(scope='Offline conditional-confidence audit; odd code/mixed train, even code/mixed and all topic/reasoning held out. No policy changes.',trace_sha256=hashes,training_observations=len(training),training_conditional_labels=len(y),logit_affine=beta.tolist(),cohorts=[])
for case in sorted({x['workload'] for x in records}):
 for parity in [0,1]:
  rows=[x for x in records if x['workload']==case and x['width']%2==parity]
  if not rows:continue
  labels_=labels(rows);rawp=sigmoid(labels_[:,0]);fitp=sigmoid(beta[0]+beta[1]*labels_[:,0]);target=labels_[:,1]
  def expected(x,b):return float(1+np.cumprod(sigmoid(b[0]+b[1]*np.array(x['confidence'][:x['rows']-1]))).sum())
  actual=np.array([1+x['matched'] for x in rows]);raw=np.array([expected(x,[0,1]) for x in rows]);fit=np.array([expected(x,beta) for x in rows])
  item=dict(workload=case,parity='odd' if parity else 'even',training=case in ['code','mixed'] and parity==1,observations=len(rows),conditional_labels=len(target),raw_brier=float(np.mean((rawp-target)**2)),fitted_brier=float(np.mean((fitp-target)**2)),mean_actual_emitted=float(actual.mean()),mean_raw_expected_emitted=float(raw.mean()),mean_fitted_expected_emitted=float(fit.mean()))
  report['cohorts'].append(item)
report['limitations']=['Fixed short-context corpus; correlations among repeated requests remain.','Conditional labels stop at first mismatch; terminal/constrained/first seven generated tokens excluded.','This audit does not establish serving benefit or justify changing confidence defaults.']
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
