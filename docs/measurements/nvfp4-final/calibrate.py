import json,subprocess,time,sys
from pathlib import Path
root=Path.cwd();out=root/'runs/nvfp4-opt';base=json.loads(subprocess.check_output(['docker','inspect','ds41rt-nvfp4-pad-ab']))[0]
def run(args,**kw):return subprocess.run([str(x) for x in args],check=True,**kw)
def logs(name):return subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT)
run(['docker','stop','ds41rt-nvfp4-pad-ab'])
for width in [1,3,5,7,4]:
 d=out/f'calibration-rtx1-k{width}';d.mkdir(exist_ok=False)
 name=f'ds41rt-nvfp4-tune-k{width}'
 cmd=base['Config']['Cmd']+['--dspark-draft-limit',str(width),'--dspark-fixed']
 create=['docker','create','--name',name,'--gpus','device=0','--network','host','--ipc','host','--device','/dev/infiniband:/dev/infiniband','--cap-add','IPC_LOCK','--ulimit','memlock=-1:-1','--security-opt','seccomp=unconfined','--security-opt','label=disable','-e','RUST_LOG=info,ds41rt::cost_model=debug','-v','/home/tj/.cache/huggingface:/root/.cache/huggingface:ro','-v',str(out/'adaptive-artifacts/coordinator.so')+':/opt/ds41rt/lib/libds41rt_native.so:ro','-v',str(out/'coordinator/ds41rt')+':/opt/ds41rt/bin/ds41rt:ro',base['Config']['Image'],*cmd]
 (d/'launch.json').write_text(json.dumps(create,indent=2)+'\n');run(create,stdout=subprocess.DEVNULL);run(['docker','start',name],stdout=subprocess.DEVNULL)
 try:
  with (d/'ready.log').open('w') as f:run(['python3',out/'wait-api.py'],stdout=f,stderr=subprocess.STDOUT)
  begin=len(logs(name));segments=[dict(begin=0,end=begin,workload='warmup')]
  with (d/'client.log').open('w') as f:
   run([root/'.venv/bin/python',root/'scripts/bench-ds41-concurrent-api.py','--base-url','http://127.0.0.1:8000','--output',d/'code.json','--concurrency','1','4','--repeats','1','--case','code','--max-tokens','512','--label','nvfp4-profile','--nonce','fixed-calibration'],stdout=f,stderr=subprocess.STDOUT)
   end=len(logs(name));segments.append(dict(begin=begin,end=end,workload='code'));begin=end
   run([root/'.venv/bin/python',root/'scripts/bench-ds41-adaptive-mixed.py','--base-url','http://127.0.0.1:8000','--tokenizer','/home/tj/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079/tokenizer.json','--output',d/'mixed.json','--concurrency','4','8','--nonce-seed','198475001','--skip-lifecycle'],stdout=f,stderr=subprocess.STDOUT)
  raw=logs(name);segments.append(dict(begin=begin,end=len(raw),workload='mixed'));(d/'server.log').write_bytes(raw);(d/'segments.json').write_text(json.dumps(segments,indent=2)+'\n');print('COMPLETE',width,flush=True)
 finally:
  (d/'server-final.log').write_bytes(logs(name));run(['docker','stop',name],stdout=subprocess.DEVNULL)
