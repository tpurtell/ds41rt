#!/usr/bin/env python3
"""Export runtime-row BF16 V4.1 router projections and validate their C ABI."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import _pinned_sparkinfer  # noqa: F401


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    os.environ['B12X_COMPILE_DISK_CACHE']='0'
    os.environ['B12X_COMPILE_MEMORY_CACHE']='0'
    from b12x.moe._shared.v41_router import compile_v41_router_scores_aot, v41_router_gemm_min_rows
    a.output_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=a.output_dir/'v41_router.json'
    manifest_path.unlink(missing_ok=True)
    variants=[]
    normalize=lambda value:re.sub(r'\s+','',value)
    for experts in (128,384):
        label=f'v41_router_e{experts}'
        compile_v41_router_scores_aot(experts=experts).export_to_c(str(a.output_dir),label,'ds41rt_'+label)
        h=(a.output_dir/(label+'.h')).read_text()
        expected=[f'ds41rt_{label}_Kernel_Module_t *module','void *x','void *w','void *logits','int32_t live_rows','cudaStream_t stream']
        signature=re.search(r'static inline int32_t cute_dsl_ds41rt_'+label+r'_wrapper\(([^)]*)\)',h)
        if not signature or normalize(signature[1])!=normalize(','.join(expected)):
            raise ValueError(f'unexpected router C signature: {label}')
        args=re.search(r'void \*args\[6\] = \{([^}]*)\}',h)
        symbols=re.findall(r'void (_mlir_\w+)\(void \*\*args, int32_t num_args\);',h)
        if not args or normalize(args[1])!='&x,&w,&logits,&live_rows,&stream,&ret' or len(symbols)!=1 or f'{symbols[0]}(args, 6);' not in h:
            raise ValueError(f'unexpected router dispatch ABI: {label}')
        variants.append(dict(experts=experts,min_rows=v41_router_gemm_min_rows(experts=experts),input_width=5120,output_dtype='FP32',tile_m=64,tile_n=16,tile_k=64,
                             symbol=symbols[0],label=label,artifacts={label+suffix:hashlib.sha256((a.output_dir/(label+suffix)).read_bytes()).hexdigest() for suffix in ('.h','.o')}))
        print(f'exported {label}',flush=True)
    lock=_pinned_sparkinfer.LOCK_DATA
    dispatch=a.output_dir/'v41_router_dispatch.h'
    dispatch.write_text('#pragma once\n'+''.join(f"#define DS41RT_V41_ROUTER_E{v['experts']}_MIN_ROWS {v['min_rows']}\n" for v in variants))
    manifest_path.write_text(json.dumps(dict(schema=1,sparkinfer=lock,variants=variants,dispatch_sha256=hashlib.sha256(dispatch.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    main()
