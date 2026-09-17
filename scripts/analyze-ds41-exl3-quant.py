#!/usr/bin/env python3
"""Summarize v5 full/EXL3 checkpoint storage, tensor shapes, and EXL3 tiers."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct

TORCH_DTYPE = {'int16':'I16','float16':'F16','int32':'I32'}
DTYPE_BYTES = {'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E5M2':1,'F8_E8M0':1,
               'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,
               'I64':8,'U64':8,'F64':8}
EXPERT = re.compile(r'^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])$')


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
    return digest.hexdigest()


def tensor_category(name):
    if '.ffn.experts.' in name: return 'routed_expert'
    if '.engram.embed.' in name: return 'ple'
    if '.ffn.shared_experts.' in name: return 'shared_expert'
    if 'embed' in name: return 'embedding'
    if 'head' in name: return 'head'
    return 'other'


def read_header(path):
    with path.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        assert 1 <= size <= 1<<30
        value=json.loads(stream.read(size))
    return {key:row for key,row in value.items() if key!='__metadata__'}


def inspect_snapshot(root):
    index_path=root/'model.safetensors.index.json'
    index=json.loads(index_path.read_text())['weight_map']
    files=sorted(root.glob('*.safetensors'))
    assert files and set(index.values())=={p.name for p in files}
    categories=defaultdict(lambda:{'tensors':0,'payload_bytes':0,'dtypes':Counter(),'shapes':Counter()})
    seen={}
    for path in files:
        for name,row in read_header(path).items():
            assert name not in seen and index[name]==path.name
            start,end=row['data_offsets']; assert 0 <= start <= end
            elements=math.prod(row['shape']); dtype=row['dtype']
            assert dtype in DTYPE_BYTES and end-start==elements*DTYPE_BYTES[dtype]
            seen[name]=path.name
            group=categories[tensor_category(name)]
            group['tensors']+=1;group['payload_bytes']+=end-start
            group['dtypes'][dtype]+=end-start;group['shapes'][(dtype,tuple(row['shape']))]+=1
    assert seen==index
    def normalize(group):
        return dict(tensors=group['tensors'],payload_bytes=group['payload_bytes'],
                    dtypes={key:value for key,value in sorted(group['dtypes'].items())},
                    shapes=[dict(dtype=dtype,shape=list(shape),tensors=count)
                            for (dtype,shape),count in sorted(group['shapes'].items())])
    return dict(path=str(root), revision=root.name, shard_count=len(files),
                snapshot_bytes=sum(path.stat().st_size for path in files),
                tensor_payload_bytes=sum(row['payload_bytes'] for row in categories.values()),
                index_sha256=sha256(index_path), tensor_count=len(seen),
                categories={key:normalize(value) for key,value in sorted(categories.items())},
                shards={path.name:dict(bytes=path.stat().st_size,device=path.stat().st_dev,
                                      inode=path.stat().st_ino,links=path.stat().st_nlink) for path in files})


def inspect_tiers(root):
    path=root/'quantize_config.json'; config=json.loads(path.read_text())
    storage=config['tensor_storage']; counts=Counter(); logical=Counter(); stored=Counter(); shapes=Counter()
    for name,row in storage.items():
        match=EXPERT.fullmatch(name); assert match,name
        scope,layer,expert,projection=match.groups(); bits=row['bits_per_weight']
        assert 2 <= bits <= 5 and row['quant_format']=='exl3'
        tensors=row['stored_tensors']; assert set(tensors)=={name+'.mcg',name+'.suh',name+'.svh',name+'.trellis'}
        suh=tensors[name+'.suh']['shape'];svh=tensors[name+'.svh']['shape']
        weights=math.prod(suh)*math.prod(svh)
        size=sum((math.prod(spec['shape']) if spec['shape'] else 1)
                 * DTYPE_BYTES[TORCH_DTYPE[spec['torch_dtype']]] for spec in tensors.values())
        key=(scope,projection,bits);counts[key]+=1;logical[key]+=weights;stored[key]+=size
        shapes[(projection,bits,tuple(suh),tuple(svh),tuple(tensors[name+'.trellis']['shape']))]+=1
    total_weights=sum(logical.values());total_stored=sum(stored.values())
    raw_bits=sum(bits*value for (scope,projection,bits),value in logical.items())
    return dict(quantize_config_sha256=sha256(path), tensor_entries=len(storage),
                nominal_average_bpw=raw_bits/total_weights,
                packed_effective_bpw=total_stored*8/total_weights,
                logical_weights=total_weights,stored_bytes=total_stored,
                tiers=[dict(scope=scope,projection=projection,bits=bits,tensors=counts[key],
                            logical_weights=logical[key],stored_bytes=stored[key])
                       for key in sorted(counts) for scope,projection,bits in [key]],
                shapes=[dict(projection=p,bits=b,suh=list(a),svh=list(c),trellis=list(t),tensors=count)
                        for (p,b,a,c,t),count in sorted(shapes.items())],
                metadata=config['meta']['ds41rt'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',type=Path,required=True)
    parser.add_argument('--exl3',type=Path,required=True)
    parser.add_argument('--fp4ple',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output exists')
    snapshots={key:inspect_snapshot(path) for key,path in
               [('full',args.full),('exl3',args.exl3),('exl3_fp4ple',args.fp4ple)]}
    tiers=inspect_tiers(args.exl3); fp4_tiers=inspect_tiers(args.fp4ple)
    assert {k:v for k,v in tiers.items() if k not in ('quantize_config_sha256','metadata')} == \
           {k:v for k,v in fp4_tiers.items() if k not in ('quantize_config_sha256','metadata')}
    assert tiers['nominal_average_bpw']==3.25
    assert abs(tiers['packed_effective_bpw']-3.2600721571180555)<1e-12
    first,second=snapshots['exl3']['shards'],snapshots['exl3_fp4ple']['shards']
    assert set(first)==set(second)
    shared=[name for name in first if (first[name]['device'],first[name]['inode'])==(second[name]['device'],second[name]['inode'])]
    changed=sorted(set(first)-set(shared))
    assert len(shared)==48 and len(changed)==4
    for category in ('routed_expert','shared_expert','embedding','head','other'):
        assert snapshots['exl3']['categories'][category]==snapshots['exl3_fp4ple']['categories'][category]
    comparisons={}
    for key in ('exl3','exl3_fp4ple'):
        before=snapshots['full']['snapshot_bytes'];after=snapshots[key]['snapshot_bytes']
        comparisons[key]=dict(bytes_saved=before-after,fraction_saved=(before-after)/before)
    report=dict(schema=1,passed=True,scope=__doc__,snapshots=snapshots,exl3=tiers,
                fp4ple_metadata=fp4_tiers['metadata']['ple_quantization'],
                hardlink_clone=dict(shared_shards=len(shared),changed_shards=changed,
                    additional_safetensor_bytes=sum(second[n]['bytes'] for n in changed)),
                comparisons_to_full=comparisons)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
