"""Offline file-identity audit only; no tensor computation or weight hashing."""
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from write_export import _fingerprint, _publish_json


def audit():
    root = Path('/home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-fp4ple-v1')
    base_state = root.parent / 'deepseek-v41-exl3-k325-v1/export-state/numbered-shards-v1'
    hf = Path('/home/tj/.cache/huggingface')
    output = hf / 'ds41rt-exports/DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1'
    base_output = hf / 'ds41rt-exports/DeepSeek-V4.1-EXL3-K3.25-v1-numbered'
    pub = json.loads((root/'publication-complete.json').read_text())
    receipt = json.loads((root/'upload-complete.json').read_text())
    base = json.loads((base_state/'upload-complete.json').read_text())
    artifact = json.loads((root/'artifact.json').read_text())
    snapshot = Path(pub['snapshot'])
    if os.getuid() != 1000 or pub['commit'] != receipt['commit']:
        raise ValueError('must audit completed revision as the model owner')
    actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file()}
    if actual != set(receipt['files']) or len(actual) != 94:
        raise ValueError('incomplete snapshot')
    for name, record in receipt['files'].items():
        path = Path(hf_hub_download(pub['repo_id'],name,cache_dir=hf/'hub',local_files_only=True))
        if path != snapshot/name or not os.path.samefile(path,output/name) or _fingerprint(path)!=record['local']:
            raise ValueError('offline snapshot identity differs: '+name)
        with path.open('rb') as stream:
            stream.read(8)
    for name, record in base['files'].items():
        if _fingerprint(base_output/name)!=record['local']:
            raise ValueError('base export changed: '+name)
    for name in artifact['reused']:
        if receipt['files'][name]['blob']!=base['files'][name]['blob'] or not os.path.samefile(output/name,base_output/name):
            raise ValueError('unchanged base file was not reused: '+name)
    result = dict(status='passed',uid=os.getuid(),files=len(actual),reused_files=len(artifact['reused']),
        all_files_readable=True,all_cache_files_hardlinked=True,base_export_unchanged=True,
        unchanged_remote_blob_ids=True,commit=pub['commit'],weight_hashes=False,
        tensor_computation=False)
    _publish_json(root/'offline-cache-audit.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    audit()
