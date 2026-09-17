#!/usr/bin/env python3
"""Exercise a two-GPU graph with SM peer exchange and BF16 arithmetic.

Run with exactly the intended two GPUs in CUDA_VISIBLE_DEVICES. This standalone
process owns every allocation; an error exits the process after recording the
failing CUDA call. It never attaches to a serving process.
"""
import argparse
import ctypes as c
import json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-lib',type=Path,required=True)
parser.add_argument('--cudart',default='libcudart.so')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
r=c.CDLL(args.cudart)
p=c.c_void_p
for name,parameters in {
 'cudaSetDevice':[c.c_int], 'cudaMalloc':[c.POINTER(p),c.c_size_t],
 'cudaStreamCreateWithFlags':[c.POINTER(p),c.c_uint], 'cudaEventCreateWithFlags':[c.POINTER(p),c.c_uint],
 'cudaMemsetAsync':[p,c.c_int,c.c_size_t,p], 'cudaEventRecord':[p,p],
 'cudaStreamWaitEvent':[p,p,c.c_uint], 'cudaStreamBeginCapture':[p,c.c_int],
 'cudaStreamEndCapture':[p,c.POINTER(p)], 'cudaGraphInstantiateWithFlags':[c.POINTER(p),p,c.c_ulonglong],
 'cudaGraphLaunch':[p,p], 'cudaStreamSynchronize':[p],
 'cudaMemcpy':[p,p,c.c_size_t,c.c_int], 'cudaDeviceEnablePeerAccess':[c.c_int,c.c_uint], 'cudaGraphExecDestroy':[p], 'cudaGraphDestroy':[p],
 'cudaStreamDestroy':[p], 'cudaEventDestroy':[p], 'cudaFree':[p],
}.items():
 f=getattr(r,name);f.argtypes=parameters;f.restype=c.c_int
r.cudaGetErrorString.argtypes=[c.c_int];r.cudaGetErrorString.restype=c.c_char_p
native=c.CDLL(str(args.native_lib))
add=native.ds41rt_v41_add_tp2_shared_async
add.argtypes=[p,p,p,c.c_size_t,p];add.restype=c.c_int
def kernel(a,b,out,stream):
 status=add(a,b,out,32,stream)
 if status:raise RuntimeError('add kernel: '+r.cudaGetErrorString(status).decode())
copy=native.ds41rt_v41_peer_copy_async
copy.argtypes=[p,p,c.c_ulonglong,p];copy.restype=c.c_int
native.ds41rt_v41_peer_copy_initialize.argtypes=[]
native.ds41rt_v41_peer_copy_initialize.restype=c.c_int
def peer(dst,src,stream):
 status=copy(dst,src,64,stream)
 if status:raise RuntimeError('SM peer copy: '+r.cudaGetErrorString(status).decode())
steps=[]
def call(name,*args):
 status=getattr(r,name)(*args);steps.append({'call':name,'status':status})
 if status: raise RuntimeError(name+': '+r.cudaGetErrorString(status).decode())
try:
 streams=[p(),p()];events=[p(),p()];buffers=[p(),p()];inputs=[p(),p()];final=p()
 for i in range(2):
  call('cudaSetDevice',i);call('cudaDeviceEnablePeerAccess',1-i,0);call('cudaMalloc',c.byref(buffers[i]),64);call('cudaMalloc',c.byref(inputs[i]),64)
  call('cudaStreamCreateWithFlags',c.byref(streams[i]),1)
  call('cudaEventCreateWithFlags',c.byref(events[i]),2)
 ones=(c.c_ushort*32)(*([0x3f80]*32))
 for i in range(2):
  call('cudaSetDevice',i);assert native.ds41rt_v41_peer_copy_initialize()==0;call('cudaMemcpy',inputs[i],ones,64,1);kernel(inputs[i],inputs[i],buffers[i],streams[i]);call('cudaStreamSynchronize',streams[i])
 call('cudaSetDevice',0);call('cudaMalloc',c.byref(final),64);call('cudaStreamBeginCapture',streams[0],1)
 kernel(inputs[0],inputs[0],buffers[0],streams[0]);call('cudaEventRecord',events[0],streams[0])
 call('cudaSetDevice',1);call('cudaStreamWaitEvent',streams[1],events[0],0)
 peer(inputs[1],buffers[0],streams[1]);kernel(inputs[1],inputs[1],buffers[1],streams[1]);call('cudaEventRecord',events[1],streams[1])
 call('cudaSetDevice',0);call('cudaStreamWaitEvent',streams[0],events[1],0)
 peer(inputs[0],buffers[1],streams[0]);kernel(inputs[0],buffers[0],final,streams[0])
 graph=p();executable=p();call('cudaStreamEndCapture',streams[0],c.byref(graph))
 call('cudaGraphInstantiateWithFlags',c.byref(executable),graph,0)
 for repeat in range(3):
  ones=(c.c_ushort*32)(*([0x3f80+repeat*0x80]*32))
  for i in range(2):
   call('cudaSetDevice',i);call('cudaMemsetAsync',buffers[i],0,64,streams[i]);call('cudaStreamSynchronize',streams[i])
  call('cudaSetDevice',0);call('cudaMemcpy',inputs[0],ones,64,1);call('cudaGraphLaunch',executable,streams[0]);call('cudaStreamSynchronize',streams[0])
  for i,buffer,expected in [(0,final,0x40c0+repeat*0x80),(1,buffers[1],0x4080+repeat*0x80)]:
   call('cudaSetDevice',i);data=(c.c_ushort*32)();call('cudaMemcpy',data,buffer,64,2)
   assert list(data)==[expected]*32,(i,list(data))
 call('cudaSetDevice',0);call('cudaGraphExecDestroy',executable);call('cudaGraphDestroy',graph);call('cudaFree',final)
 for i in range(2):
  call('cudaSetDevice',i);call('cudaStreamDestroy',streams[i]);call('cudaEventDestroy',events[i]);call('cudaFree',buffers[i]);call('cudaFree',inputs[i])
 result={'passed':True,'replays':3,'steps':steps}
except Exception as e:result={'passed':False,'error':str(e),'steps':steps}
if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='steps'}))
raise SystemExit(0 if result['passed'] else 1)
