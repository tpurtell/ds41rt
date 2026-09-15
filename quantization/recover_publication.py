"""Explicit publication-only recovery for Hub-added JSON LFS attributes.

No GPU work, weight hashing, retransmission, or remote mutation. Preserve the
original export/upload plans; validate an explicitly recorded metadata revision.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

from huggingface_hub import HfApi
from export_assets import asset_target, JSON_LFS_RULES
from materialize_cache import materialize_cache
from upload_model import _remote
from validate_export import validate_export
from write_export import _fingerprint, _publish_json, _sync_dir


RULES = JSON_LFS_RULES


def check_attributes(original, remote):
    if remote != original + RULES.encode():
        raise ValueError('remote attributes are not the reviewed two appended JSON LFS rules')


def recover(manifest, commit):
    output, state = Path(manifest['output']), Path(manifest['export_state'])
    repo = manifest['publication']['repo_id']
    if repo != 'wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1' or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('recovery requires the exact requested repository and pinned commit')
    api = HfApi()
    info = api.repo_info(repo, repo_type='model')
    if info.private or info.sha != commit:
        raise ValueError('public repository head differs from reviewed commit')
    remote = _remote(api, repo, commit)
    records = [json.loads(p.read_text()) for p in sorted((state / 'preuploaded').glob('*.json'))]
    prepared = {r['name']: {k: r[k] for k in ('local', 'bytes', 'blob')} for r in records}
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() or p.is_symlink()}
    if len(prepared) != len(records) or set(prepared) != actual or set(remote) != actual:
        raise ValueError('publication file inventories differ')
    for name, record in prepared.items():
        if name != '.gitattributes' and (remote[name] != {k: record[k] for k in ('bytes', 'blob')}
                or _fingerprint(asset_target(output, name)) != record['local']):
            raise ValueError(f'unreviewed artifact difference: {name}')
    plan = json.loads((state / 'plan.json').read_text())
    original_record = plan['assets']['.gitattributes']
    original = (original_record['content'].encode() if 'content' in original_record
                else Path(original_record['path']).read_bytes())
    if hashlib.sha256(original).hexdigest() != original_record['sha256']:
        raise ValueError('original attributes differ from frozen plan')
    with urlopen(f'https://huggingface.co/{repo}/resolve/{commit}/.gitattributes', timeout=60) as response:
        attributes = response.read(65537)
    if len(attributes) > 65536:
        raise ValueError('oversized attributes')
    check_attributes(original, attributes)
    blob = hashlib.sha1(f'blob {len(attributes)}\0'.encode() + attributes).hexdigest()
    if remote['.gitattributes'] != dict(bytes=len(attributes), blob=blob):
        raise ValueError('downloaded attributes differ from committed blob')
    recovery = state / 'hub-json-lfs-recovery-v1'
    recovery.mkdir(exist_ok=True)
    _publish_json(recovery / 'authorization.json', dict(schema='ds41rt-hub-json-lfs-recovery-v1',
        commit=commit, repo_id=repo, original_attributes=original.decode(),
        committed_attributes=attributes.decode(), unchanged_files=len(actual)-1,
        original_upload_attributes=prepared['.gitattributes']))
    target = asset_target(output, '.gitattributes')
    current = target.read_bytes()
    if current not in (original, attributes) or target.stat().st_nlink != 1 and current != attributes:
        raise ValueError('local attributes changed or have unexpected hard links')
    owner = Path(manifest['run_root']).stat()
    if current != attributes:
        if _fingerprint(target) != prepared['.gitattributes']['local']:
            raise ValueError('original local attributes fingerprint differs')
        fd, temporary = tempfile.mkstemp(dir=output, prefix='.attributes-recovery-')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(attributes)
                os.fchmod(stream.fileno(), 0o644)
                os.fchown(stream.fileno(), owner.st_uid, owner.st_gid)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            _sync_dir(output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    plan['assets']['.gitattributes'] = dict(content=attributes.decode(), bytes=len(attributes),
                                           sha256=hashlib.sha256(attributes).hexdigest())
    _publish_json(recovery / 'plan.json', plan)
    validation = validate_export(output, recovery, Path(manifest['snapshot']))
    _publish_json(recovery / 'structure-validation.json', validation)
    prepared['.gitattributes'] = dict(local=_fingerprint(target), bytes=len(attributes), blob=blob)
    receipt = dict(schema='ds41rt-upload-receipt-v1', status='uploaded', repo_id=repo,
                   commit=commit, files=prepared)
    _publish_json(state / 'upload-complete.json', receipt)
    cached = materialize_cache(output, manifest['publication']['cache_root'], receipt)
    repo_root = Path(cached['snapshot']).parent.parent
    # Payloads already belong to the user; chown metadata/symlinks without
    # following cache links or changing payload data or fingerprints.
    for root in (repo_root, recovery):
        for path in (root, *root.rglob('*')):
            os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
    _publish_json(state / 'cache-complete.json', cached)
    result = dict(status='complete', repo_id=repo, commit=commit, snapshot=cached['snapshot'],
        numerical_validation='deferred-to-inference-engine-integration',
        publication_recovery=str(recovery / 'authorization.json'))
    _publish_json(state / 'publication-complete.json', result)
    for name in ('upload-complete.json', 'cache-complete.json', 'publication-complete.json'):
        os.chown(state / name, owner.st_uid, owner.st_gid)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    recover(json.loads(args.manifest.read_text()), args.commit)
