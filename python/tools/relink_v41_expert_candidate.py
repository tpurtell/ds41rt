#!/usr/bin/env python3
"""Controlled expert-only relink for serving A/B tests, not a clean release build."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--aot-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', choices=('spark', 'rtx_tp2'), required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
    args = parser.parse_args()
    build, aot, output = args.build_dir.resolve(), args.aot_dir.resolve(), args.output.resolve()
    if output.exists():
        parser.error('output must be new; preserve prior comparison artifacts')
    manifest = json.loads((aot / 'v41_experts.json').read_text())
    assert manifest['role'] == args.role
    for name, expected in manifest['artifact_sha256'].items():
        assert digest(aot / name) == expected, name
    tp2 = args.role == 'rtx_tp2'
    if tp2:
        shutil.copyfile(aot / 'v41_expert_variants.h', aot / 'v41_tp2_expert_variants.h')
    source_name = 'v41_tp2_experts.cc' if tp2 else 'v41_experts.cc'
    obj = aot / (source_name + '.o')
    compile_command = ['g++', '-std=c++17', '-O3', '-DNDEBUG', '-fPIC', '-c',
        '-I', str(args.source_root / 'native/include'), '-I', str(args.cuda_root / 'include'),
        '-I', str(aot), str(args.source_root / 'native/src' / source_name), '-o', str(obj)]
    subprocess.run(compile_command, check=True)
    raw = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands',
                                  'libds41rt_native.so'], text=True).splitlines()[-1]
    assert raw.startswith(': && ') and raw.endswith(' && :'), raw
    command = shlex.split(raw[5:-5])
    command[command.index('-o') + 1] = str(output)
    command = [v for v in command if not v.startswith('-Wl,--dependency-file=')]
    directory = 'v41_tp2_experts' if tp2 else 'v41_experts'
    stem = 'v41_rtx_tp2_m' if tp2 else 'v41_spark_m'
    replacements = []
    for index, value in enumerate(command):
        path = Path(value)
        if path.parent.name == directory and path.name.startswith(stem) and path.suffix == '.o':
            replacement = aot / path.name
        elif value == f'CMakeFiles/ds41rt_native.dir/src/{source_name}.o':
            replacement = obj
        else:
            continue
        assert replacement.is_file(), replacement
        command[index] = str(replacement)
        replacements.append([value, str(replacement)])
    assert len(replacements) == 7, replacements
    inputs = []
    for value in command:
        if value.endswith(('.o', '.a')):
            path = Path(value)
            path = path if path.is_absolute() else build / path
            inputs.append({'path': str(path), 'sha256': digest(path)})
    baseline = build / 'libds41rt_native.so'
    baseline_sha = digest(baseline)
    subprocess.run(command, cwd=build, check=True)
    assert digest(baseline) == baseline_sha
    for entry in inputs:
        assert digest(Path(entry['path'])) == entry['sha256']
    record = {'scope': __doc__, 'role': args.role, 'baseline_library_sha256': baseline_sha,
        'compile_command': compile_command, 'link_command': command, 'cwd': str(build),
        'replacements': replacements, 'link_inputs': inputs, 'aot_manifest': manifest,
        'artifact': str(output), 'artifact_sha256': digest(output)}
    output.with_suffix('.link.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'artifact': str(output), 'sha256': record['artifact_sha256']}))


if __name__ == '__main__':
    main()
