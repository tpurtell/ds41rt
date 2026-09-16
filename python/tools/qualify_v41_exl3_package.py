#!/usr/bin/env python3
"""Check relocated package integrity and native ABI loading, without Python CUDA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from package_v41_exl3_aot import verify, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = verify(args.package, runtime=args.runtime)
    assert 'torch' not in sys.modules and 'b12x' not in sys.modules
    with tempfile.TemporaryDirectory(prefix='ds41rt-exl3-relocation-') as temporary:
        relocated = Path(temporary) / 'lib' / 'exl3'
        shutil.copytree(args.package, relocated)
        verify(relocated, manifest['sparkinfer_revision'], args.runtime, manifest['role'])
        libraries = [relocated / v['directory'] / 'libds41rt_exl3.so' for v in manifest['variants']]
        probe = subprocess.run([str(args.probe.resolve()), *map(str, libraries)],
                               check=True, text=True, capture_output=True)
        for variant, library in zip(manifest['variants'], libraries):
            line = next(line for line in probe.stdout.splitlines() if line.startswith(str(library) + ':'))
            for key, value in {'hidden': 5120, 'intermediate': variant['intermediate'],
                'capacity': variant['capacity'], 'experts': variant['experts'], 'topk': variant['top_k'],
                'tier_count': len(variant['bits']),
                'output_element_bytes': 2 if variant['output_dtype'] == 'bf16' else 4}.items():
                assert int(re.search(r'\b' + key + r': (\d+)', line)[1]) == value, (key, line)
            bits = [int(v) for v in re.search(r'bits: \[([^]]+)\]', line)[1].split(',')]
            assert bits == variant['bits'] + [0] * (4 - len(variant['bits']))
        assert f'{len(libraries)} modules loaded concurrently' in probe.stdout
        rejections = []
        def reject(name, **kwargs):
            try:
                verify(relocated, **kwargs)
            except (ValueError, FileNotFoundError):
                rejections.append(name)
            else:
                raise AssertionError(f'invalid package accepted: {name}')
        victim = libraries[0]
        original = victim.read_bytes()
        victim.unlink()
        reject('missing native module')
        victim.write_bytes(b'broken')
        reject('damaged native module')
        victim.write_bytes(original)
        extra = relocated / 'unlisted.bin'
        extra.write_bytes(b'extra')
        reject('unlisted file')
        extra.unlink()
        reject('wrong source revision', revision='wrong')
        reject('wrong serving role', role='spark' if manifest['role'] == 'coordinator' else 'coordinator')
        fake_runtime = Path(temporary) / 'wrong-runtime.so'
        fake_runtime.write_bytes(b'wrong')
        reject('wrong CuTe runtime', runtime=fake_runtime)
        verify(relocated, runtime=args.runtime)
        report = {'passed': True, 'scope': 'Package relocation, integrity and native ABI initialization; not kernel numerics or serving performance.',
                  'role': manifest['role'], 'compute': manifest['compute'],
                  'package_manifest_sha256': digest(args.package / 'manifest.json'),
                  'runtime_sha256': digest(args.runtime), 'variants': manifest['variants'],
                  'python_cuda_imported': False, 'native_probe_stdout': probe.stdout,
                  'native_probe_stderr': probe.stderr, 'rejected': rejections}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'role': report['role'], 'variants': len(report['variants']), 'rejections': len(rejections)}))


if __name__ == '__main__':
    main()
