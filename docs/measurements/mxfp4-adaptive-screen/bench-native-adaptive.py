import sys,json,statistics
sys.path.insert(0,'/wip/native-adaptive-source')
import torch,cutlass,cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel
from tests.moe.test_v41_grouped_slices import _check_grouped_slices
N=int(sys.argv[1]) if len(sys.argv)>1 else 576
WIDTH=int(sys.argv[2]) if len(sys.argv)>2 else 192
import tests.moe.test_v41_grouped_slices as fixture
original_cases=fixture._routing_cases
fixture._routing_cases=lambda: original_cases()+[("shared8",torch.arange(6).expand(8,6).clone())]
state={}
def measure(**case):
 args=case['args'];meta=case['meta_view'];m=case['rows'];routes=case['routes']
 if not state:
  out=torch.empty_like(case['out']);alt=list(args);alt[-1]=from_dlpack(out,assumed_align=16)
  count=torch.zeros(1,dtype=torch.int32,device='cuda');cv=from_dlpack(count,assumed_align=4)
  kernel=cute.compile(V41FusedSliceKernel(WIDTH,grouped=True,intermediate=N,adaptive_sms=torch.cuda.get_device_properties(0).multi_processor_count),*alt,cutlass.Int32(80),current_cuda_stream(),meta,cutlass.Int32(64),cv)
  state.update(out=out,args=alt,count=count,cv=cv,kernel=kernel)
 state['count'].fill_(case['groups']);state['out'].fill_(12345)
 state['kernel'](*state['args'],m,current_cuda_stream(),meta,64,state['cv'])
 torch.cuda.synchronize()
 torch.testing.assert_close(state['out'][:,:routes],case['out'][:,:routes],rtol=0,atol=0)
 assert (state['out'][:,routes:]==12345).all()
 graphs=[]
 for adaptive in (False,True):
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   if adaptive:state['kernel'](*state['args'],m,current_cuda_stream(),meta,64,state['cv'])
   else:case['baseline_compiled'](*args,m,current_cuda_stream(),meta,64)
  for _ in range(3):graph.replay()
  graphs.append(graph)
 samples=[[],[]]
 for rep in range(9):
  for i in ([0,1] if rep%2==0 else [1,0]):
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   start.record()
   for _ in range(50):graphs[i].replay()
   end.record();end.synchronize();samples[i].append(start.elapsed_time(end)*1000/50)
 torch.testing.assert_close(state['out'][:,:routes],case['out'][:,:routes],rtol=0,atol=0)
 print(json.dumps({'case':case['case'],'groups':case['groups'],'exact':True,'medians_us':[statistics.median(s) for s in samples],'samples':samples}),flush=True)
_check_grouped_slices(WIDTH,measure,n=N)
