"""Focused parser tests for v5 checkpoint quant analysis."""
import json
from pathlib import Path
import runpy
import struct

ROOT = Path(__file__).resolve().parents[2]
ANALYZE = runpy.run_path(str(ROOT/'scripts/analyze-ds41-exl3-quant.py'))


def write_safetensors(path, tensors):
    offset=0;header={}
    payload=b''
    sizes={'I16':2,'F16':2,'I32':4,'U8':1}
    for name,(dtype,shape) in tensors.items():
        count=1
        for value in shape:count*=value
        size=count*sizes[dtype]
        header[name]={'dtype':dtype,'shape':shape,'data_offsets':[offset,offset+size]}
        payload+=bytes(size);offset+=size
    encoded=json.dumps(header,separators=(',',':')).encode()
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)


def test_snapshot_parser_validates_index_offsets_and_categories(tmp_path):
    shard=tmp_path/'model-00001-of-00001.safetensors'
    tensors={'layers.0.ffn.experts.0.w1.trellis':('I16',[2,3]),
             'layers.1.engram.embed.weight':('U8',[4,5]),
             'model.head.weight':('F16',[2,2])}
    write_safetensors(shard,tensors)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map':{name:shard.name for name in tensors}}))
    result=ANALYZE['inspect_snapshot'](tmp_path)
    assert result['tensor_count']==3 and result['shard_count']==1
    assert result['categories']['routed_expert']['payload_bytes']==12
    assert result['categories']['ple']['payload_bytes']==20
    assert result['categories']['head']['payload_bytes']==8


def test_tier_parser_accounts_for_scale_overhead(tmp_path):
    name='layers.0.ffn.experts.0.w1'
    stored={name+'.mcg':{'shape':[],'torch_dtype':'int32'},
            name+'.suh':{'shape':[4],'torch_dtype':'float16'},
            name+'.svh':{'shape':[8],'torch_dtype':'float16'},
            name+'.trellis':{'shape':[1,1,6],'torch_dtype':'int16'}}
    config={'tensor_storage':{name:{'bits_per_weight':3,'quant_format':'exl3','stored_tensors':stored}},
            'meta':{'ds41rt':{'schema':'test'}}}
    (tmp_path/'quantize_config.json').write_text(json.dumps(config))
    result=ANALYZE['inspect_tiers'](tmp_path)
    assert result['logical_weights']==32
    assert result['nominal_average_bpw']==3
    assert result['stored_bytes']==4+8+16+12
    assert result['packed_effective_bpw']==10
