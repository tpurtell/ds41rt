#!/usr/bin/env python3
"""Check Rust residency descriptors and staged hashes against B12x/safetensors."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import _pinned_sparkinfer


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from safetensors import safe_open
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import build_projection_tiered_maps
    data=json.loads(args.plan.read_text()); plan=data['plan']
    experts=plan['experts']; world=plan['world'];rank=plan['rank'];width=plan['intermediate']
    # Independently partition the 18 complete H128 blocks in global order.
    partitions=torch.tensor_split(torch.arange(18),world)
    start=int(partitions[rank][0])*128;stop=(int(partitions[rank][-1])+1)*128
    assert stop-start==width
    layer=plan['layer']
    prefix=(f"layers.{layer['Backbone']}" if 'Backbone' in layer else f"mtp.{layer['Dspark']}")+'.ffn.experts'
    index=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    files={}
    def reader(name):
        shard=index[name]
        if shard not in files:
            files[shard]=safe_open(args.snapshot/shard,framework='pt',device='cpu')
        return files[shard]
    tier_ids=[]
    counts=[]
    for projection in ['w1','w3','w2']:
        row=[]
        for expert in range(experts):
            name=f'{prefix}.{expert}.{projection}.trellis'
            bits=reader(name).get_slice(name).get_shape()[-1]//16
            row.append(plan['tiers'].index(bits))
        tier_ids.append(row)
        counts.append([row.count(tier) for tier in range(len(plan['tiers']))])
    assert [list(v) for v in zip(*counts)]==plan['projection_counts']
    route,descriptor=build_projection_tiered_maps(*tier_ids,
        tier_slots=[experts]*len(plan['tiers']),device=torch.device('cpu'))
    buffers={b['name']:b for b in plan['buffers']}
    assert route.tolist()==buffers['global_to_combined']['initial_words']
    assert descriptor.tolist()==buffers['descriptor_map']['initial_words']
    hashes={v['tensor']:v['sha256'] for v in data['read_hashes']}
    assert len(hashes)==len(plan['loads'])==experts*9
    for job in plan['loads']:
        name=job['tensor'];projection,suffix=name.split('.')[-2:]
        value=reader(name).get_tensor(name)
        if suffix=='trellis':
            value=value[start//16:stop//16] if projection=='w2' else value[:,start//16:stop//16]
        elif (projection=='w2' and suffix=='suh') or (projection!='w2' and suffix=='svh'):
            value=value[start:stop]
        raw=value.contiguous().numpy().tobytes()
        assert len(raw)==job['bytes'] and hashlib.sha256(raw).hexdigest()==hashes[name],name
    report=dict(passed=True,scope='CPU residency descriptor and compressed staging checks; no device residency, expert execution or serving claim.',
        sparkinfer_revision=_pinned_sparkinfer.REVISION,
        snapshot_revision=args.snapshot.name,plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        layer=layer,world=world,rank=rank,intermediate=width,
        b12x_descriptor_and_route_maps_equal=True,projection_counts_equal=True,
        independently_sliced_tensor_hashes_matched=len(hashes),resident_payload_bytes=data['resident_bytes'],
        staging_bytes=data['staging_bytes'],read_scratch_bytes=data['scratch_bytes'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
