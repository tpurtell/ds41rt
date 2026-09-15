"""Publish a numbered-shard revision using server-side copies, never weight uploads.

Default is local preparation/validation only. --publish authorizes a single
parent-guarded Hub commit and cache ref advance. Old exports/receipts survive.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi
from export_assets import asset_target
from materialize_cache import materialize_cache
from upload_model import _remote
from validate_export import validate_export
from write_export import _fingerprint, _publish_json, _sync_dir


def numbered_mapping(files):
    shards = [name for name in files if name.endswith('.safetensors')]
    def order(name):
        match = re.fullmatch(r'(model|ple-1|ple-14)-(\d{5})-of-(\d{5})\.safetensors', name)
        if not match:
            raise ValueError('unexpected source shard name')
        return ({'model': 0, 'ple-1': 1, 'ple-14': 2}[match[1]], int(match[2]))
    return {name: f'model-{i:05d}-of-{len(shards):05d}.safetensors'
            for i, name in enumerate(sorted(shards, key=order), 1)}


def run(manifest, source_state, state, output, parent, *, publish=False):
    repo = manifest['publication']['repo_id']
    if repo != 'wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1':
        raise ValueError('unexpected publication repository')
    old = Path(manifest['output']).resolve(strict=True)
    output = output.resolve()
    if output == old or old in output.parents or output in old.parents:
        raise ValueError('new export must be separate from original')
    receipt = json.loads((Path(manifest['export_state']) / 'upload-complete.json').read_text())
    if receipt['commit'] != parent or receipt['repo_id'] != repo:
        raise ValueError('original publication receipt differs')
    plan = json.loads((source_state / 'plan.json').read_text())
    if plan['output'] != str(old):
        raise ValueError('source plan output differs')
    mapping = numbered_mapping(receipt['files'])
    if len(mapping) != 52 or set(mapping) != set(plan['layout']['files']):
        raise ValueError('expected all 52 original shards')
    for name, record in receipt['files'].items():
        if _fingerprint(asset_target(old, name)) != record['local']:
            raise ValueError(f'original export changed: {name}')
    revised = copy.deepcopy(plan)
    revised['output'] = str(output)
    revised['layout']['files'] = {mapping[name]: members for name, members in plan['layout']['files'].items()}
    revised['layout']['index']['weight_map'] = {name: mapping[filename]
        for name, filename in plan['layout']['index']['weight_map'].items()}
    state.mkdir(parents=True, exist_ok=True)
    _publish_json(state / 'authorization.json', dict(repo_id=repo, parent=parent,
        source_state=str(source_state), output=str(output), mapping=mapping))
    _publish_json(state / 'plan.json', revised)
    output.mkdir(parents=True, exist_ok=True)
    index_name = 'model.safetensors.index.json'
    for name in receipt['files']:
        if name == index_name:
            continue
        target = asset_target(output, mapping.get(name, name))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(target, old / name):
                raise ValueError('new export entry is not original hardlink')
        else:
            os.link(old / name, target)
            _sync_dir(target.parent)
    _publish_json(output / index_name, revised['layout']['index'])
    validation = validate_export(output, state, Path(manifest['snapshot']))
    _publish_json(state / 'structure-validation.json', validation)
    api = HfApi()
    expected_old = {n: {k: r[k] for k in ('bytes', 'blob')} for n, r in receipt['files'].items()}
    if _remote(api, repo, parent) != expected_old:
        raise ValueError('original remote revision differs')
    expected = {mapping.get(n, n): r for n, r in expected_old.items()}
    payload = (output / index_name).read_bytes()
    expected[index_name] = dict(bytes=len(payload), blob=hashlib.sha256(payload).hexdigest())
    if len(expected_old[index_name]['blob']) != 64:
        raise ValueError('expected existing LFS index')
    info = api.repo_info(repo, repo_type='model')
    if info.private:
        raise ValueError('repository must be public')
    if info.sha != parent and _remote(api, repo, info.sha) != expected:
        raise ValueError('remote head changed unexpectedly')
    print(json.dumps(dict(status='prepared', files=len(expected), renamed_shards=len(mapping),
                         weight_upload_bytes=0, metadata_bytes=len(payload))), flush=True)
    if not publish:
        return
    if info.sha == parent:
        operations = [CommitOperationCopy(src_path_in_repo=src, path_in_repo=dst, src_revision=parent)
                      for src, dst in mapping.items()]
        operations += [CommitOperationDelete(path_in_repo=src) for src in mapping if src not in mapping.values()]
        operations.append(CommitOperationAdd(index_name, payload))
        commit = api.create_commit(repo, operations, repo_type='model', revision='main',
            parent_commit=parent, commit_message='Normalize all 52 shard filenames; preserve isolated PLE boundaries').oid
    else:
        commit = info.sha  # Recover a lost response only after exact inventory comparison.
    if _remote(api, repo, commit) != expected:
        raise ValueError('renamed remote inventory differs')
    updated = dict(schema='ds41rt-upload-receipt-v1', status='uploaded', repo_id=repo, commit=commit,
        files={name: dict(**record, local=_fingerprint(output / name)) for name, record in expected.items()})
    _publish_json(state / 'upload-complete.json', updated)
    owner = Path(manifest['run_root']).stat()
    cached = materialize_cache(output, manifest['publication']['cache_root'], updated,
        owner=(owner.st_uid, owner.st_gid), previous_commit=parent)
    _publish_json(state / 'cache-complete.json', cached)
    result = dict(status='complete', repo_id=repo, commit=commit, previous_commit=parent,
        snapshot=cached['snapshot'], output=str(output), weight_upload_bytes=0,
        numerical_validation='deferred-to-inference-engine-integration')
    _publish_json(state / 'publication-complete.json', result)
    for path in (state, *state.rglob('*')):
        os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    run(json.loads(args.manifest.read_text()), args.source_state, args.state, args.output,
        args.parent, publish=args.publish)
