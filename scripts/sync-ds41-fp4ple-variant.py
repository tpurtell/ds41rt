#!/usr/bin/env python3
"""Materialize the FP4-PLE snapshot using shared base blobs on each Spark.

Only replacement safetensors and small snapshot metadata traverse SSH. Existing
base blobs are never written. Interrupted rsync transfers can be resumed by
rerunning this command; refs/main is set only after replacement hash checks.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


BASE = "models--wrldsuksgo2mars--DeepSeek-V4.1-EXL3-K3.25-v1"
VARIANT = "models--wrldsuksgo2mars--DeepSeek-V4.1-EXL3-K3.25-FP4PLE-v1"
BASE_REV = "cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88"
VARIANT_REV = "04cada4d3f38584f069e0a7debc53720832d738e"


def remote(host, code, payload):
    program = "payload = " + repr(payload) + "\n" + code
    return subprocess.check_output(
        ["ssh", host, "python3 -"], input=program, text=True
    )


PREPARE = r'''
import json, os, pathlib, shutil
p = payload
root = pathlib.Path(p['hub'])
base = root / p['base']
variant = root / p['variant']
assert (base / 'refs/main').read_text().strip() == p['base_rev']
snapshot = variant / 'snapshots' / p['variant_rev']
snapshot.mkdir(parents=True, exist_ok=True)
(variant / 'blobs').mkdir(exist_ok=True)
assert shutil.disk_usage(root).free > sum(s['bytes'] for s in p['changed']), 'Insufficient conservative transfer space'
for s in p['shared']:
    source = (base / 'snapshots' / p['base_rev'] / s['name']).resolve(strict=True)
    assert source.name == s['sha256'] and source.stat().st_size == s['bytes']
    dest = variant / 'blobs' / s['sha256']
    if dest.exists():
        assert os.path.samefile(source, dest), 'Existing blob is not the expected hard link'
    else:
        os.link(source, dest)
    link = snapshot / s['name']
    if link.is_symlink() or link.exists():
        assert os.path.samefile(link, source)
    else:
        link.symlink_to('../../blobs/' + s['sha256'])
print(json.dumps({'shared_shards': len(p['shared']), 'shared_bytes': sum(s['bytes'] for s in p['shared'])}))
'''

VERIFY = r'''
import hashlib, json, os, pathlib
p = payload
root = pathlib.Path(p['hub'])
base = root / p['base'] / 'snapshots' / p['base_rev']
variant = root / p['variant']
snapshot = variant / 'snapshots' / p['variant_rev']
for s in p['shared']:
    assert os.path.samefile(base / s['name'], snapshot / s['name'])
for s in p['changed']:
    blob = variant / 'blobs' / s['sha256']
    assert blob.stat().st_size == s['bytes']
    with blob.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    assert digest == s['sha256'], (s['name'], digest)
    link = snapshot / s['name']
    if link.exists() or link.is_symlink():
        assert os.path.samefile(link, blob)
    else:
        link.symlink_to('../../blobs/' + s['sha256'])
index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    assert (snapshot / name).is_file(), name
(variant / 'refs').mkdir(exist_ok=True)
tmp = variant / 'refs/main.next'
tmp.write_text(p['variant_rev'])
tmp.replace(variant / 'refs/main')
print(json.dumps({'verified': True, 'shared_shards': len(p['shared']), 'replacement_shards': len(p['changed']), 'revision': p['variant_rev']}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", default=["ostrich", "dodo", "emu", "kiwi"])
    parser.add_argument("--hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--remote-hub", default="/home/tj/.cache/huggingface/hub")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = args.hub / BASE / "snapshots" / BASE_REV
    variant = args.hub / VARIANT / "snapshots" / VARIANT_REV
    shared, changed = [], []
    for f in sorted(variant.glob("*.safetensors")):
        source = f.resolve(strict=True)
        digest = source.name
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        same = (base / f.name).resolve(strict=True).name == digest
        entry = {"name": f.name, "sha256": digest, "bytes": f.stat().st_size}
        (shared if same else changed).append(entry)
    assert len(shared) == 48
    assert [s["name"] for s in changed] == [f"model-{n:05d}-of-00052.safetensors" for n in range(49, 53)]
    payload = dict(hub=args.remote_hub, base=BASE, variant=VARIANT,
                   base_rev=BASE_REV, variant_rev=VARIANT_REV,
                   shared=shared, changed=changed)
    record = {"started_at": time.time(), "manifest": payload, "hosts": {}}
    args.record.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.record.write_text(json.dumps(record, indent=2) + "\n")

    save()
    print(f"Share {len(shared)} shards; transfer {sum(s['bytes'] for s in changed):,} shard bytes per host", flush=True)
    if args.dry_run:
        return
    for host in args.hosts:
        record["hosts"][host] = {"started_at": time.time()}
        save()
        print(host, remote(host, PREPARE, payload).strip(), flush=True)
        dest = f"{args.remote_hub}/{VARIANT}"
        subprocess.run(["rsync", "-rLt", "--exclude=*.safetensors",
                        str(variant) + "/", f"{host}:{shlex.quote(dest + '/snapshots/' + VARIANT_REV + '/')}"] , check=True)
        for s in changed:
            print(host, "transfer", s["name"], flush=True)
            subprocess.run(["rsync", "-Lt", "--partial-dir=.rsync-partial",
                            str((variant / s["name"]).resolve()),
                            f"{host}:{shlex.quote(dest + '/blobs/' + s['sha256'])}"], check=True)
        result = json.loads(remote(host, VERIFY, payload))
        record["hosts"][host].update(result, completed_at=time.time())
        save()
        print(host, "verified", flush=True)
    record["completed_at"] = time.time()
    save()


if __name__ == "__main__":
    main()
