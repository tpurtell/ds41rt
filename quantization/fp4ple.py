"""Detached CPU-only PLE variation: bounded conversion and cross-repo publication."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import numpy as np

from export_assets import asset_target
from stream_shard import read_header
from write_export import _fingerprint, _publish_json, _sync_dir

BASE_REPO = 'wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1'
BASE_COMMIT = 'cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88'
REPO = 'wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1'
HF = Path('/home/tj/.cache/huggingface')
BASE = HF / 'ds41rt-exports/DeepSeek-V4.1-EXL3-K3.25-v1-numbered'
BASE_STATE = Path('/home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1/export-state/numbered-shards-v1')
ROOT = Path('/home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-fp4ple-v1')
OUTPUT = HF / 'ds41rt-exports/DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1'
CHUNK_ROWS = 262144  # 64 MiB source weights; total chunk storage < 128 MiB.


def event(kind, **fields):
    item = dict(event=kind, time=time.time(), **fields)
    print(json.dumps(item), flush=True)
    with (ROOT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(item) + '\n')
        stream.flush()


def atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.progress-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, separators=(',', ':'))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def pointer(array):
    return array.ctypes.data_as(C.c_void_p)


def library():
    source = Path(__file__).with_name('ple_nvfp4_cpu.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = ROOT / ('cpu-' + digest[:16] + '.so')
    if not target.exists():
        temporary = target.with_suffix('.partial.so')
        subprocess.run(['g++', '-std=c++17', '-O3', '-fPIC', '-shared', '-fopenmp',
                        '-ffp-contract=off', str(source), '-o', str(temporary)], check=True)
        os.replace(temporary, target)
    lib = C.CDLL(str(target))
    lib.ple_stats.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p]
    lib.ple_amax.argtypes = [C.c_void_p]
    lib.ple_amax.restype = C.c_float
    lib.ple_lut.argtypes = [C.c_float, C.c_void_p, C.c_void_p]
    lib.ple_convert.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t] + [C.c_void_p] * 4
    return lib


def read_chunk(stream, offset, rows, width):
    stream.seek(offset)
    data = np.frombuffer(stream.read(rows * width), dtype=np.uint8)
    if data.size != rows * width:
        raise ValueError('truncated source chunk')
    # Sequential pread/read does not retain a full tensor or borrowed mapping.
    return data


def reclaim(stream, start, length):
    if hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(stream.fileno(), start, length, os.POSIX_FADV_DONTNEED)


def make_output(path, tensors):
    header, cursor = {}, 0
    for name, dtype, shape, size in tensors:
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[cursor, cursor + size])
        cursor += size
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    prefix = struct.pack('<Q', len(encoded)) + encoded
    if not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
        with os.fdopen(fd, 'r+b') as stream:
            stream.write(prefix)
            stream.truncate(len(prefix) + cursor)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        with path.open('rb') as stream:
            if stream.read(len(prefix)) != prefix or path.stat().st_size != len(prefix) + cursor:
                raise ValueError('partial output header/size differs')
    return len(prefix)


def check_row(lib, w, s, global_scale, codes, scales):
    """Only one row: independent Torch FP8 casting and NumPy E2M1 arithmetic."""
    import torch
    torch.set_num_threads(1)
    packed, block_scales = np.empty(128, np.uint8), np.empty(16, np.uint8)
    lib.ple_convert(pointer(w), pointer(s), 1, pointer(codes), pointer(scales),
                    pointer(packed), pointer(block_scales))
    values = torch.from_numpy(w.copy()).view(torch.float8_e4m3fn).float().numpy()
    values *= np.repeat(np.exp2(s.astype(np.float32) - 127), 32)
    blocks = values.reshape(16, 16)
    ideal = np.max(np.abs(blocks), axis=1) / np.float32(6 * np.float32(global_scale))
    ideal[ideal == 0] = 1
    expected_scales = torch.from_numpy(np.clip(ideal, 2**-9, 448)).to(torch.float8_e4m3fn)
    scale_float = expected_scales.float().numpy()
    normalized = blocks / (scale_float * np.float32(global_scale))[:, None]
    boundaries = np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], np.float32)
    absolute = np.abs(normalized)
    ordinals = np.searchsorted(boundaries, absolute).astype(np.uint8)
    ordinals += ((absolute == .75) | (absolute == 1.75) | (absolute == 3.5)).astype(np.uint8)
    ordinals |= ((normalized < 0).astype(np.uint8) << 3)
    flat = ordinals.ravel()
    expected_packed = flat[::2] | (flat[1::2] << 4)
    if not np.array_equal(packed, expected_packed) or not np.array_equal(block_scales, expected_scales.view(torch.uint8).numpy()):
        raise ValueError('single-row CPU reference packing/scale mismatch')
    table = np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], np.float32)
    decoded = (table[ordinals] * (scale_float * np.float32(global_scale))[:, None]).ravel()
    error = decoded.astype(np.float64) - values
    return dict(values=256, packed_and_scales_exact=True, max_abs_error=float(np.abs(error).max()),
        relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(values.astype(np.float64)), 1e-30)))


def convert_table(lib, layer, scale_shard, weight_shard, base_receipt, index):
    prefix = f'layers.{layer}.engram.embed'
    paths = [BASE / index[prefix + '.' + suffix] for suffix in ('weight', 'scale')]
    descriptors = [read_header(path) for path in paths]
    wdesc, sdesc = [h[prefix + '.' + suffix] for (h, _), suffix in zip(descriptors, ('weight', 'scale'))]
    rows, width = wdesc['shape']
    if width != 256 or sdesc['shape'] != [rows, 8] or wdesc['dtype'] != 'F8_E4M3' or sdesc['dtype'] != 'F8_E8M0':
        raise ValueError('unexpected source PLE geometry')
    offsets = [base + desc['data_offsets'][0] for (_, base), desc in zip(descriptors, (wdesc, sdesc))]
    state = ROOT / f'ple-{layer}'
    state.mkdir(exist_ok=True)
    completed = load(state / 'complete.json')
    if completed:
        for name, record in completed['files'].items():
            if _fingerprint(OUTPUT / name) != record:
                raise ValueError('completed converted PLE changed')
        return completed
    stats = load(state / 'stats.json', dict(rows=0, maxima=[0]*256))
    maxima = np.array(stats['maxima'], dtype=np.uint8)
    with paths[0].open('rb') as ws, paths[1].open('rb') as ss:
        for start in range(stats['rows'], rows, CHUNK_ROWS):
            n = min(CHUNK_ROWS, rows - start)
            w = read_chunk(ws, offsets[0] + start*256, n, 256)
            s = read_chunk(ss, offsets[1] + start*8, n, 8)
            lib.ple_stats(pointer(w), pointer(s), n, pointer(maxima))
            if maxima[255]:
                raise ValueError('nonfinite source PLE data')
            atomic_json(state / 'stats.json', dict(rows=start+n, maxima=maxima.tolist()))
            reclaim(ws, offsets[0]+start*256, n*256)
            reclaim(ss, offsets[1]+start*8, n*8)
            if start % (CHUNK_ROWS*64) == 0 or start+n == rows:
                event('ple_weight_maximum', layer=layer, rows=start+n, total_rows=rows)
        amax = lib.ple_amax(pointer(maxima))
        global_scale = float(np.float32(amax / 2688)) if amax else 1.0
        if not math.isfinite(global_scale) or global_scale <= 0:
            raise ValueError('invalid NVFP4 global scale')
        codes, scales = np.zeros(256*128*256, np.uint8), np.zeros(256*128, np.uint8)
        lib.ple_lut(global_scale, pointer(codes), pointer(scales))
        row = rows // 2
        check = check_row(lib, read_chunk(ws, offsets[0]+row*256, 1, 256),
            read_chunk(ss, offsets[1]+row*8, 1, 8), global_scale, codes, scales)
        _publish_json(state / 'row-check.json', dict(row=row, **check))
        event('ple_row_check', layer=layer, row=row, **check)
        wp, sp = OUTPUT / (weight_shard+'.partial'), OUTPUT / (scale_shard+'.partial')
        wf, sf = OUTPUT / weight_shard, OUTPUT / scale_shard
        # A crash between final renames and completion can reuse the exact
        # header-checked final file; committed progress binds all payload rows.
        wpath, spath = (wf if wf.exists() else wp), (sf if sf.exists() else sp)
        wb = make_output(wpath, [(prefix+'.weight','U8',[rows,128],rows*128)])
        sb = make_output(spath, [(prefix+'.weight_scale','F8_E4M3',[rows,16],rows*16),
                               (prefix+'.weight_scale_2','F32',[],4)])
        progress = load(state / 'conversion.json', dict(rows=0))
        with wpath.open('r+b') as outw, spath.open('r+b') as outs:
            for start in range(progress['rows'], rows, CHUNK_ROWS):
                n = min(CHUNK_ROWS, rows-start)
                w = read_chunk(ws, offsets[0]+start*256, n, 256)
                s = read_chunk(ss, offsets[1]+start*8, n, 8)
                packed, block_scales = np.empty(n*128,np.uint8), np.empty(n*16,np.uint8)
                lib.ple_convert(pointer(w),pointer(s),n,pointer(codes),pointer(scales),pointer(packed),pointer(block_scales))
                outw.seek(wb+start*128); outw.write(packed)
                outs.seek(sb+start*16); outs.write(block_scales)
                outw.flush(); outs.flush()
                os.fsync(outw.fileno()); os.fsync(outs.fileno())
                atomic_json(state / 'conversion.json', dict(rows=start+n))
                reclaim(ws, offsets[0]+start*256, n*256); reclaim(ss, offsets[1]+start*8, n*8)
                reclaim(outw, wb+start*128,n*128); reclaim(outs,sb+start*16,n*16)
                if start % (CHUNK_ROWS*64) == 0 or start+n == rows:
                    event('ple_converted', layer=layer, rows=start+n,total_rows=rows)
            outs.seek(sb+rows*16); outs.write(struct.pack('<f',global_scale))
            outs.flush(); os.fsync(outs.fileno())
        for partial, final in ((wp,wf),(sp,sf)):
            if partial.exists():
                os.link(partial,final); partial.unlink(); _sync_dir(OUTPUT)
        result = dict(layer=layer, rows=rows, amax=amax, global_scale=global_scale,
                      files={p.name:_fingerprint(p) for p in (wf,sf)}, row_check=check)
        _publish_json(state/'complete.json',result)
        return result


def assemble(receipt, tables):
    changed = {'README.md','config.json','quantize_config.json','model.safetensors.index.json'}
    ple_files = {f'model-{i:05d}-of-00052.safetensors' for i in range(49,53)}
    reused = set(receipt['files']) - changed - ple_files
    for name in reused:
        src, dst = asset_target(BASE,name), asset_target(OUTPUT,name)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            os.link(src,dst)
        if not os.path.samefile(src,dst):
            raise ValueError('unchanged artifact must be original hardlink')
    index = load(BASE/'model.safetensors.index.json')
    for layer in (1,14):
        stem = f'layers.{layer}.engram.embed'
        del index['weight_map'][stem+'.scale']
    total = 0
    for filename in sorted({*index['weight_map'].values(), *ple_files}):
        header, _ = read_header(OUTPUT/filename)
        for name,item in header.items():
            if name == '__metadata__':
                continue
            if filename in ple_files:
                index['weight_map'][name] = filename
            elif index['weight_map'].get(name) != filename:
                raise ValueError('unchanged header/index mismatch')
            total += item['data_offsets'][1]-item['data_offsets'][0]
    index['metadata']['total_size'] = total
    _publish_json(OUTPUT/'model.safetensors.index.json',index)
    description = dict(schema='ds41rt.nvfp4-ple.v1',format='nvfp4',block_size=16,
        packing='even-element-low-nibble',scale_layout='row-major',
        reconstruction='E2M1(weight) * FP8_E4M3(weight_scale) * FP32(weight_scale_2)',
        source_repo=BASE_REPO,source_revision=BASE_COMMIT,
        algorithm='weight-amax-rne-no-activation-calibration',
        tensors={f'layers.{t["layer"]}.engram.embed':dict(logical_shape=[t['rows'],256],
            weight_dtype='uint8',weight_scale_dtype='float8_e4m3fn',weight_scale_2_dtype='float32',
            global_scale=t['global_scale']) for t in tables},
        numerical_validation='one-row-per-table-only; whole-model validation deferred')
    config=load(BASE/'config.json'); config['ds41rt_ple_quantization']=description
    external=load(BASE/'quantize_config.json'); external['meta']['ds41rt']['ple_quantization']=description
    _publish_json(OUTPUT/'config.json',config); _publish_json(OUTPUT/'quantize_config.json',external)
    card = f'''---
license: mit
base_model:
- {BASE_REPO}
tags:
- exl3
- nvfp4
- deepseek_v41
---
# DeepSeek V4.1 EXL3 K3.25 with NVFP4 PLE

PLE-only variant of [{BASE_REPO}](https://huggingface.co/{BASE_REPO}/tree/{BASE_COMMIT}).
The routed EXL3 expert quantization (40 main and 3 dSpark blocks) is unchanged.
Only the two PLE embedding tables at layers 1 and 14 were converted from the
source FP8/E8M0 representation to weight-only NVFP4 using CPU conversion.

## Storage

One standard indexed safetensors checkpoint, 52 globally numbered shards.
Shards 1–48 are byte-identical to the base repository and reused server-side.
Shards 49/50 hold PLE 1 scales/weights; shards 51/52 hold PLE 14 scales/weights.
Each PLE has packed E2M1 `weight` [rows,128] uint8 (even element low nibble),
`weight_scale` [rows,16] FP8 E4M3, and scalar FP32 `weight_scale_2`.
The original logical shape is [rows,256], with contiguous 16-element blocks.
Decode as E2M1 times the block scale times the global scale. Scales are stored
row-major, not GPU-kernel-swizzled. Config `ds41rt_ple_quantization` and
`quantize_config.json` metadata describe this representation explicitly.
No other non-routed tensors were changed. Source code/license/tokenizer assets
are retained from the base model; `README.source.md` is the original source card.

## Recipe and checks

Global scale = table weight absolute maximum / 2688. Block scales use each
16-value absolute maximum divided by (6 * global scale), clamp to [2^-9,448],
and E4M3 round-to-nearest-even. Values use E2M1 round-to-nearest-even.
No activation calibration, expert requantization, GPU work, or whole-model
validation/replay was performed. One actual row per table was checked against
an independent arithmetic reference for exact packed bytes/scales, with
reconstruction error recorded in the conversion report. This is not a quality
evaluation. Inference-engine integration must support this PLE representation
alongside the base EXL3/native formats; standard safetensors is not a claim of
stock-engine execution support.

Conversion implementation and reproducible process:
https://github.com/tpurtell/ds41rt/tree/main/quantization
'''
    path=OUTPUT/'README.md'
    if path.exists() and path.read_text()!=card:
        raise ValueError('existing variant card differs')
    if not path.exists():
        with path.open('x') as stream:
            stream.write(card); stream.flush(); os.fsync(stream.fileno())
    actual={str(p.relative_to(OUTPUT)) for p in OUTPUT.rglob('*') if p.is_file()}
    if actual != reused|changed|ple_files:
        raise ValueError('variant file inventory differs')
    _publish_json(ROOT/'artifact.json',dict(files=sorted(actual),reused=sorted(reused),
        changed=sorted(changed|ple_files),payload_bytes=total,ple_quantization=description))
    event('artifact_ready',files=len(actual),payload_bytes=total,reused_files=len(reused))
    return reused, changed|ple_files


def publish(base_receipt,reused,changed):
    from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError
    from upload_model import _remote, _restored_operation
    from materialize_cache import materialize_cache
    api=HfApi()
    base_remote={n:{k:r[k] for k in ('bytes','blob')} for n,r in base_receipt['files'].items()}
    if _remote(api,BASE_REPO,BASE_COMMIT)!=base_remote:
        raise ValueError('pinned public base differs from local receipts')
    parent=load(ROOT/'upload-parent.json')
    if parent is None:
        try:
            info=api.repo_info(REPO,repo_type='model')
        except RepositoryNotFoundError:
            api.create_repo(REPO,repo_type='model',private=False)
            info=api.repo_info(REPO,repo_type='model')
        if info.private or set(_remote(api,REPO,info.sha)) - {'.gitattributes'}:
            raise ValueError('destination must be new and public')
        parent=dict(commit=info.sha)
        _publish_json(ROOT/'upload-parent.json',parent)
    operations=[CommitOperationCopy(src_path_in_repo=n,path_in_repo=n,src_revision=BASE_COMMIT,
        src_repo_id=BASE_REPO,src_repo_type='model') for n in sorted(reused)]
    prepared={n:dict(**base_remote[n],local=_fingerprint(OUTPUT/n)) for n in reused}
    uploads=ROOT/'preuploaded'; uploads.mkdir(exist_ok=True)
    for name in sorted(changed):
        path=OUTPUT/name; local=_fingerprint(path)
        record_path=uploads/(name+'.json'); record=load(record_path)
        if record is None:
            event('upload_started',file=name,bytes=local['bytes'])
            op=CommitOperationAdd(name,path)
            api.preupload_lfs_files(REPO,[op],repo_type='model',revision='main',
                                   num_threads=1,free_memory=False,gitignore_content='')
            if op._should_ignore or op._upload_mode not in ('lfs','regular'):
                raise ValueError('unexpected upload classification')
            if op._upload_mode=='lfs':
                if not op._is_uploaded or not op.upload_info.is_hashed:
                    raise ValueError('new payload upload not acknowledged')
                blob=op.upload_info.sha256.hex()
            else:
                if local['bytes']>64*1024*1024:
                    raise ValueError('unexpected large regular upload')
                payload=path.read_bytes()
                blob=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
            record=dict(mode=op._upload_mode,bytes=local['bytes'],blob=blob,local=local)
            _publish_json(record_path,record)
            event('upload_prepared',file=name,bytes=local['bytes'])
        else:
            if record['local']!=local:
                raise ValueError('changed prepared upload')
            op=_restored_operation(name,path,record)
        if _fingerprint(path)!=local:
            raise ValueError('variant changed during upload')
        prepared[name]={k:record[k] for k in ('bytes','blob','local')}
        operations.append(op)
    expected={n:{k:r[k] for k in ('bytes','blob')} for n,r in prepared.items()}
    info=api.repo_info(REPO,repo_type='model')
    if info.private:
        raise ValueError('destination became private')
    if info.sha!=parent['commit']:
        if _remote(api,REPO,info.sha)!=expected:
            raise ValueError('destination head changed unexpectedly')
        commit=info.sha
    else:
        commit=api.create_commit(REPO,operations,repo_type='model',revision='main',
            parent_commit=parent['commit'],commit_message='Publish CPU-quantized NVFP4 PLE variant; reuse base weights server-side').oid
    if _remote(api,REPO,commit)!=expected:
        raise ValueError('published variant differs')
    receipt=dict(schema='ds41rt-upload-receipt-v1',status='uploaded',repo_id=REPO,commit=commit,files=prepared)
    _publish_json(ROOT/'upload-complete.json',receipt)
    owner=ROOT.stat()
    cached=materialize_cache(OUTPUT,HF/'hub',receipt,owner=(owner.st_uid,owner.st_gid))
    _publish_json(ROOT/'cache-complete.json',cached)
    result=dict(status='complete',repo_id=REPO,commit=commit,snapshot=cached['snapshot'],
        base_weight_upload_bytes=0,new_upload_bytes=sum(prepared[n]['bytes'] for n in changed),
        numerical_checks='one row per PLE; no whole-model validation')
    _publish_json(ROOT/'publication-complete.json',result)
    for path in (ROOT,*ROOT.rglob('*')):
        os.chown(path,owner.st_uid,owner.st_gid,follow_symlinks=False)
    event('publication_complete',**result)


def main(stage):
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True); OUTPUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=load(BASE_STATE/'upload-complete.json')
        if receipt['repo_id']!=BASE_REPO or receipt['commit']!=BASE_COMMIT:
            raise ValueError('base receipt differs')
        for name,r in receipt['files'].items():
            if _fingerprint(asset_target(BASE,name))!=r['local']:
                raise ValueError('base artifact changed')
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__),Path(__file__).with_name('ple_nvfp4_cpu.cpp'))}
        _publish_json(ROOT/'identity.json',dict(repo=REPO,base_repo=BASE_REPO,base_commit=BASE_COMMIT,
            source_files=receipt['files'],code=code,chunk_rows=CHUNK_ROWS,device='cpu'))
        event('started',stage=stage,device='cpu')
        lib=library(); index=load(BASE/'model.safetensors.index.json')['weight_map']
        tables=[convert_table(lib,layer,f'model-{a:05d}-of-00052.safetensors',
            f'model-{b:05d}-of-00052.safetensors',receipt,index) for layer,a,b in ((1,49,50),(14,51,52))]
        reused,changed=assemble(receipt,tables)
        if stage=='all':
            publish(receipt,reused,changed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['convert','all'],default='all')
    args=parser.parse_args()
    main(args.stage)
