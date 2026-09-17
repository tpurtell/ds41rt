#!/usr/bin/env python3
"""Build relocatable native EXL3 packages; verify them without importing CUDA."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residency_overrides(values: list[str], capacities: list[int], paired: bool) -> dict[int, int]:
    """Explicit offline build choices; B12X validates kernel resource limits."""
    if values and not paired:
        raise ValueError('residency overrides require a paired TP4 package')
    result = {}
    for value in values:
        try:
            capacity, blocks = map(int, value.split('='))
        except (ValueError, AttributeError):
            raise ValueError('residency override must be CAPACITY=BLOCKS') from None
        if capacity not in capacities or blocks not in (1, 2):
            raise ValueError('residency override requires a selected capacity and one or two blocks/SM')
        if capacity in result:
            raise ValueError('duplicate residency override capacity')
        result[capacity] = blocks
    return result


def verify(package: Path, revision: str | None = None, runtime: Path | None = None, role: str | None = None) -> dict:
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest['schema'] != 'ds41rt.exl3-package.v1':
        raise ValueError('unsupported EXL3 package schema')
    if revision is not None and manifest['sparkinfer_revision'] != revision:
        raise ValueError('EXL3 package/source revision mismatch')
    if role is not None and manifest['role'] != ('spark' if role == 'expert' else role):
        raise ValueError('EXL3 package/serving role mismatch')
    paired = manifest.get('paired_tp4', False)
    if not isinstance(paired, bool) or (paired and manifest['role'] != 'spark'):
        raise ValueError('paired EXL3 package requires Spark role')
    overrides = residency_overrides(manifest.get('residency_overrides', []),
                                    [v['capacity'] for v in manifest['variants']], paired)
    expected = manifest['files']
    actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    if actual != set(expected) | {'manifest.json'}:
        raise ValueError('EXL3 package contains missing or unlisted files')
    for name, spec in expected.items():
        path = package / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError(f'unsafe EXL3 package path: {name}')
        if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise ValueError(f'EXL3 package file mismatch: {name}')
    if runtime is not None and digest(runtime) != manifest['runtime']['sha256']:
        raise ValueError('EXL3 CuTe runtime mismatch')
    required = set()
    seen = set()
    for variant in manifest['variants']:
        directory = variant['directory']
        if Path(directory).is_absolute() or '..' in Path(directory).parts:
            raise ValueError('unsafe EXL3 variant directory')
        if directory in seen:
            raise ValueError('duplicate EXL3 package variant')
        seen.add(directory)
        meta = json.loads((package / directory / 'v41_exl3.json').read_text())
        for key in ('capacity', 'intermediate', 'experts', 'top_k', 'output_dtype', 'bits'):
            if meta[key] != variant[key]:
                raise ValueError(f'EXL3 variant metadata mismatch: {directory}/{key}')
        if 'blocks_per_sm' in variant and variant['blocks_per_sm'] != meta.get('blocks_per_sm'):
            raise ValueError('EXL3 variant residency metadata mismatch')
        if meta['capacity'] in overrides and meta.get('blocks_per_sm') != overrides[meta['capacity']]:
            raise ValueError('EXL3 compiled residency differs from requested override')
        boundary = meta.get('paired_boundary')
        if paired:
            rank_name = directory.split('/')[0]
            boundaries = {'tp4-rank0': 'last', 'tp4-rank1': 'first', 'tp4-rank2': 'last', 'tp4-rank3': 'first'}
            if (boundary != boundaries.get(rank_name) or boundary is None
                    or variant.get('paired_boundary') != boundary
                    or meta.get('descriptor_rows') != 4 or meta.get('native_info_version') != 3
                    or meta['intermediate'] != 640 or len(meta['bits']) != 2 or meta['top_k'] != 6):
                raise ValueError('paired EXL3 package boundary/contract mismatch')
        elif boundary is not None or variant.get('paired_boundary') is not None:
            raise ValueError('paired artifact in disjoint EXL3 package')
        if meta['sparkinfer_revision'] != manifest['sparkinfer_revision']:
            raise ValueError('EXL3 variant/source revision mismatch')
        required.update(f'{directory}/{name}' for name in
                        ('v41_exl3.json', 'trellis_lut.bin', 'libds41rt_exl3.so'))
        if meta['requires_route_preparation']:
            required.update(f'{directory}/routes/{name}' for name in
                            ('v41_exl3_routes.json', 'libv41_exl3_routes.so'))
            if digest(package / directory / meta['route_preparation']['manifest']) != meta['route_preparation']['sha256']:
                raise ValueError('EXL3 route manifest mismatch')
    if not seen or not required.issubset(expected):
        raise ValueError('incomplete EXL3 package')
    return manifest


def validate_destination(output: Path) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError('EXL3 output must be a package directory')
    # Ninja creates the parent directories of declared BYPRODUCTS before the
    # command runs. An empty directory tree is still a fresh destination.
    # Symlinks are never treated as empty scaffolding, including dangling ones.
    if output.exists() and any(p.is_symlink() or not p.is_dir() for p in output.rglob('*')):
        marker = output / 'manifest.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('schema') != 'ds41rt.exl3-package.v1':
            raise ValueError('refusing to replace a non-package directory')


def install_package(source: Path, output: Path) -> None:
    validate_destination(output)
    src, dst = source.resolve(), output.resolve()
    if src == dst or src.is_relative_to(dst) or dst.is_relative_to(src):
        raise ValueError('EXL3 source and destination must not overlap')
    verify(source)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.name == 'manifest.json':
            continue
        if path.is_dir():
            shutil.copytree(path, output / path.name)
        else:
            shutil.copy2(path, output / path.name)
    # Completion marker is installed only after all payload files.
    shutil.copy2(source / 'manifest.json', output / 'manifest.json')
    verify(output)


def build(args: argparse.Namespace) -> None:
    validate_destination(args.output)
    paired = getattr(args, 'paired_tp4', False)
    if paired and (args.role != 'spark' or len(args.bits) != 2):
        raise ValueError('paired TP4 package requires Spark role and two tiers')
    capacities = sorted(set(int(v) for v in args.capacities.split(',')))
    if not capacities or any(v < 1 or v > 4096 for v in capacities):
        raise ValueError('EXL3 capacities must be in 1..4096')
    overrides = residency_overrides(getattr(args, 'residency', []), capacities, paired)
    # Import the source-pinned compiler only for builds, never package checks.
    import _pinned_sparkinfer
    from export_b12x_v41_exl3_aot import export
    import torch

    props = torch.cuda.get_device_properties(0)
    expected_compute = (12, 1) if args.role == 'spark' else (12, 0)
    if (props.major, props.minor) != expected_compute:
        raise ValueError(f'{args.role} package requires GPU {expected_compute}')
    profiles = (
        [('tp4-width640', 640, 384, 6, 'bf16', ['tp4-rank0', 'tp4-rank1']),
         ('tp4-width512', 512, 384, 6, 'bf16', ['tp4-rank2', 'tp4-rank3'])]
        if args.role == 'spark' else
        [('rtx-tp1', 2304, 384, 6, 'fp32', ['rtx-tp1']),
         ('rtx-tp2', 1152, 384, 6, 'fp32', ['rtx-tp2']),
         ('dspark', 2304, 128, 3, 'bf16', ['dspark'])]
    )
    if paired:
        profiles = [('paired-last', 640, 384, 6, 'bf16', ['tp4-rank0', 'tp4-rank2']),
                    ('paired-first', 640, 384, 6, 'bf16', ['tp4-rank1', 'tp4-rank3'])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.exl3-package-', dir=args.output.parent) as temporary:
        stage = Path(temporary)
        variants = []
        for profile, width, experts, topk, dtype, destinations in profiles:
            for capacity in capacities:
                raw = args.build_dir / profile / f'm{capacity}'
                options = {'paired_boundary': profile.removeprefix('paired-')} if paired else {}
                if capacity in overrides:
                    options['blocks_per_sm'] = overrides[capacity]
                meta = export(raw, width, experts, capacity, tuple(args.bits), 'auto', topk, dtype, **options)
                core = raw / 'libds41rt_exl3.so'
                subprocess.run([args.cxx, '-shared', '-fPIC', '-std=c++17',
                    f'-I{args.cuda_include}', str(raw / 'v41_exl3_bridge.cc'),
                    str(raw / 'v41_exl3_core.o'), str(raw / 'v41_exl3_sum.o'),
                    f'-L{args.cuda_libdir}', '-lcudart', f'-L{args.runtime.parent}',
                    '-lcute_dsl_runtime', '-Wl,-z,defs',
                    '-o', str(core)], check=True)
                runtime_files = ['v41_exl3.json', 'trellis_lut.bin', 'libds41rt_exl3.so']
                if meta['requires_route_preparation']:
                    routes = raw / 'routes'
                    subprocess.run([args.cxx, '-shared', '-fPIC', '-std=c++17',
                        f'-I{args.cuda_include}', str(routes / 'v41_exl3_routes.cc'),
                        str(args.cuda_driver), '-Wl,-z,defs',
                        '-o', str(routes / 'libv41_exl3_routes.so')], check=True)
                    runtime_files += ['routes/v41_exl3_routes.json', 'routes/libv41_exl3_routes.so']
                for destination in destinations:
                    directory = f'{destination}/m{capacity}'
                    for name in runtime_files:
                        target = stage / directory / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(raw / name, target)
                    variants.append({'directory': directory, **{key: meta[key] for key in
                        ('capacity', 'intermediate', 'experts', 'top_k', 'output_dtype', 'bits', 'blocks_per_sm')}})
                    if paired:
                        variants[-1]['paired_boundary'] = meta['paired_boundary']
                # Large prefill exports must not retain another capacity's arenas.
                gc.collect()
                torch.cuda.empty_cache()
        files = {str(p.relative_to(stage)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest = {'schema': 'ds41rt.exl3-package.v1', 'role': args.role,
                    'sparkinfer_revision': _pinned_sparkinfer.REVISION,
                    'compute': [props.major, props.minor], 'sms': props.multi_processor_count,
                    'variants': variants, 'files': files,
                    'runtime': {'library': 'libcute_dsl_runtime.so', 'sha256': digest(args.runtime),
                                'provider': 'installed nvidia-cutlass-dsl CUDA runtime; release entrypoint sets its library path'}}
        if paired:
            manifest['paired_tp4'] = True
        if overrides:
            manifest['residency_overrides'] = [f'{capacity}={blocks}' for capacity, blocks in sorted(overrides.items())]
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify(stage, _pinned_sparkinfer.REVISION)
        install_package(stage, args.output)
    print(json.dumps({'package': str(args.output), 'role': args.role, 'variants': len(variants)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('build')
    create.add_argument('--role', choices=('spark', 'coordinator'), required=True)
    create.add_argument('--paired-tp4', action='store_true', help='Export explicit paired H128 ownership kernels for all four Spark ranks')
    create.add_argument('--residency', action='append', default=[], metavar='CAPACITY=BLOCKS',
                        help='Explicit paired-package blocks/SM override; repeat per capacity (for example 80=2). B12X validates resources.')
    create.add_argument('--capacities', default='1,16,80,256,1024,4096')
    create.add_argument('--bits', type=int, nargs='+', default=[3, 4])
    create.add_argument('--build-dir', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--cxx', required=True)
    create.add_argument('--cuda-include', type=Path, required=True)
    create.add_argument('--cuda-libdir', type=Path, required=True)
    create.add_argument('--cuda-driver', type=Path, required=True)
    create.add_argument('--runtime', type=Path, required=True)
    install = commands.add_parser('install')
    install.add_argument('--package', type=Path, required=True)
    install.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('verify')
    check.add_argument('--package', type=Path, required=True)
    check.add_argument('--sparkinfer-revision')
    check.add_argument('--runtime', type=Path)
    check.add_argument('--role', choices=('spark', 'expert', 'coordinator'))
    args = parser.parse_args()
    if args.command == 'build':
        build(args)
    elif args.command == 'install':
        install_package(args.package, args.output)
    else:
        manifest = verify(args.package, args.sparkinfer_revision, args.runtime, args.role)
        print(json.dumps({'verified': True, 'role': manifest['role'], 'variants': len(manifest['variants'])}))


if __name__ == '__main__':
    main()
