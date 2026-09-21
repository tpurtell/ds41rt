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
        # A variant's tile is a claim about the geometry that was compiled, not a
        # build-argument echo, so it is checked against the export's own record.
        # Conditional on the variant carrying the key: v9 manifests predate it and
        # must keep verifying byte-for-byte unchanged.
        for field in ('tile', 'tile_requested'):
            value = variant.get(field)
            if field in variant and (not isinstance(value, list) or len(value) != 4
                    # bool is an int subclass; a True/False tile is a broken record.
                    or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0
                               for n in value)):
                raise ValueError(f'EXL3 variant {field} must be four positive integers: {directory}')
        if 'tile' in variant and variant['tile'] != meta.get('tile'):
            raise ValueError(f'EXL3 variant tile mismatch: {directory}')
        if 'tile_requested' in variant and variant['tile_requested'] != meta.get('tile'):
            raise ValueError(f'EXL3 tile override was not applied as requested: {directory}')
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
    # `requested_layouts` records what the build was asked to contain, so a partial
    # Spark export cannot be published as a full contract. Packages built before
    # this existed (v9 and earlier) carry no key and verify exactly as before. The
    # check is a minimum, not an exact set: a package may legitimately carry more
    # layouts than one consumer asked about.
    # The capacity set comes from the contract, not from the variants that happen to
    # exist: otherwise dropping one capacity for every rank would still verify.
    # Older packages record no capacities, so they keep the previous behaviour.
    requested_capacities = manifest.get('requested_capacities') or sorted(
        {v['capacity'] for v in manifest['variants']})
    for layout in manifest.get('requested_layouts') or []:
        for capacity in requested_capacities:
            if not any(v['directory'] == f'{layout}/m{capacity}' for v in manifest['variants']):
                raise ValueError(f'EXL3 package is missing requested layout {layout}/m{capacity}')
    return manifest


def verify_root(root: Path, revision: str | None = None, runtime: Path | None = None,
                role: str | None = None) -> list[dict]:
    """Verify an EXL3 root holding either one flat package or family packages.

    Release images keep a single well-known ``exl3/`` root. A multi-family build
    nests ``exl3-kXX`` packages beneath it; a direct ``manifest.json`` in the
    root is the legacy single-family layout. Both are accepted, and this is
    strict by design:

    * a family directory without ``manifest.json`` is an error, not a silently
      skipped entry, so one valid package can never mask a broken sibling;
    * a root carrying both a flat manifest and family packages is rejected as
      ambiguous rather than verifying only one of them;
    * every discovered package goes through the unchanged strict single-package
      ``verify``, and a root yielding no package at all is an error.
    """
    flat_manifest = root / 'manifest.json'
    children = sorted(child for child in root.iterdir() if child.is_dir()) if root.is_dir() else []
    families = [child for child in children if child.name.startswith('exl3-')]
    stray = [
        child for child in children
        if not child.name.startswith('exl3-') and (child / 'manifest.json').is_file()
    ]
    if flat_manifest.is_file() and families:
        raise ValueError(
            f'ambiguous EXL3 root has both a flat manifest and family packages: {root}'
        )
    if stray:
        raise ValueError(f'unexpected EXL3 package directory (expected exl3-<bits>): {stray[0]}')
    if flat_manifest.is_file():
        return [verify(root, revision, runtime, role)]
    if not families:
        raise ValueError(f'no EXL3 package or family manifest under {root}')
    missing = [family for family in families if not (family / 'manifest.json').is_file()]
    if missing:
        raise ValueError(f'EXL3 family is missing its manifest.json: {missing[0]}')
    return [verify(family, revision, runtime, role) for family in families]


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


def profiles_for_role(role: str) -> list[tuple]:
    """(profile, width, experts, top-k, dtype, [layout destinations]) per role.

    TP4 keeps the padded 640/512 pair-split because I=2304/4 is not a whole number
    of 128-wide Trellis blocks; TP2 (9+9) and TP3 (6+6+6) split exactly, so each of
    those compiles one export per capacity that its ranks share byte-for-byte.
    """
    if role == 'spark':
        return [('tp4-width640', 640, 384, 6, 'bf16', ['tp4-rank0', 'tp4-rank1']),
                ('tp4-width512', 512, 384, 6, 'bf16', ['tp4-rank2', 'tp4-rank3']),
                ('tp2-width1152', 1152, 384, 6, 'bf16', ['tp2-rank0', 'tp2-rank1']),
                ('tp3-width768', 768, 384, 6, 'bf16',
                 ['tp3-rank0', 'tp3-rank1', 'tp3-rank2'])]
    return [('rtx-tp1', 2304, 384, 6, 'fp32', ['rtx-tp1']),
            ('rtx-tp2', 1152, 384, 6, 'fp32', ['rtx-tp2']),
            ('dspark', 2304, 128, 3, 'bf16', ['dspark'])]


def parse_requested_layouts(values: list[str], role: str) -> list[str]:
    """`--require-layout tp3-rank0,tp3-rank1` (repeatable): layouts the package needs.

    Scoped to the role being built: a coordinator cannot produce Spark ranks, so
    asking for one must fail during argument validation instead of after a full
    GPU compile.
    """
    requested: list[str] = []
    known = {layout for _, _, _, _, _, destinations in profiles_for_role(role)
             for layout in destinations}
    for value in values:
        for name in (part.strip() for part in value.split(',')):
            if not name:
                raise ValueError('EXL3 requested layout list has an empty entry')
            if name not in known:
                raise ValueError(f'unknown EXL3 layout requested: {name}')
            if name in requested:
                raise ValueError(f'duplicate EXL3 requested layout: {name}')
            requested.append(name)
    return sorted(requested)


def tile_overrides(values: list[str], capacities: list[int], role: str, paired: bool) -> dict[str, dict[int, tuple]]:
    """Parse repeatable `--tile PROFILE=CAPACITIES:FC1_K,FC1_N,FC2_K,FC2_N`.

    Capacity scoping is the point: the pinned B12x policy already picks a wider
    tile only at m16 (verified against `_projection_mixed_tile_config`), so an A/B
    must be able to retarget one capacity without silently flattening the others.
    `CAPACITIES` is `all` or a `+`-joined list of already-selected capacities. The
    tile itself is b12x's own `(fc1_k, fc1_n, fc2_k, fc2_n)` vocabulary and is only
    shape-checked (four integers); the pinned planner decides whether a geometry is
    legal, so its rules are not restated here.
    """
    if values and paired:
        raise ValueError('EXL3 tile overrides are not supported for paired TP4 packages')
    # Role-scoped for the same reason as --require-layout: an override naming a
    # profile this role does not build would otherwise be accepted and silently
    # ignored, which is a knob that ships looking effective.
    widths = {profile: width for profile, width, *_ in profiles_for_role(role)}
    result: dict[str, dict[int, tuple]] = {}
    for value in values:
        profile, sep, rest = value.partition('=')
        targets, sep2, tiles = rest.partition(':')
        if not sep or not sep2 or profile not in widths:
            raise ValueError(
                f'EXL3 tile override must name a {role} profile as '
                f'PROFILE=CAPACITIES:FC1_K,FC1_N,FC2_K,FC2_N, got: {value} '
                f'(available: {sorted(widths)})')
        if targets == 'all':
            selected = list(capacities)
        else:
            try:
                selected = sorted({int(part) for part in targets.split('+')})
            except ValueError:
                raise ValueError(f'EXL3 tile override capacities must be integers or all: {targets}') from None
            unknown = [c for c in selected if c not in capacities]
            if unknown:
                raise ValueError(
                    f'EXL3 tile override targets capacities that are not packaged: {unknown} '
                    f'(selected: {capacities})')
        try:
            tile = tuple(int(part) for part in tiles.split(','))
        except ValueError:
            raise ValueError(f'EXL3 tile override needs four integers: {tiles}') from None
        if len(tile) != 4 or any(part < 1 for part in tile):
            raise ValueError(f'EXL3 tile override needs four positive tiles: {tiles}')
        overlap = set(result.get(profile, {})) & set(selected)
        if overlap:
            raise ValueError(f'duplicate EXL3 tile override for {profile} at {sorted(overlap)}')
        per_capacity = result.setdefault(profile, {})
        per_capacity.update({capacity: tile for capacity in selected})
    return result


def build(args: argparse.Namespace) -> None:
    validate_destination(args.output)
    paired = getattr(args, 'paired_tp4', False)
    if paired and (args.role != 'spark' or len(args.bits) != 2):
        raise ValueError('paired TP4 package requires Spark role and two tiers')
    capacities = sorted(set(int(v) for v in args.capacities.split(',')))
    if not capacities or any(v < 1 or v > 4096 for v in capacities):
        raise ValueError('EXL3 capacities must be in 1..4096')
    overrides = residency_overrides(getattr(args, 'residency', []), capacities, paired)
    profiles = profiles_for_role(args.role)
    requested_layouts = parse_requested_layouts(getattr(args, 'require_layout', []), args.role)
    tiles = tile_overrides(getattr(args, 'tile', []), capacities, args.role, paired)
    # Import the source-pinned compiler only for builds, never package checks.
    import _pinned_sparkinfer
    from export_b12x_v41_exl3_aot import export
    import torch

    props = torch.cuda.get_device_properties(0)
    expected_compute = (12, 1) if args.role == 'spark' else (12, 0)
    if (props.major, props.minor) != expected_compute:
        raise ValueError(f'{args.role} package requires GPU {expected_compute}')
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
                tile = tiles.get(profile, {}).get(capacity)
                # Content-keyed: an override changes the compiled geometry, so it
                # must never land in (or be read from) the policy-keyed export.
                # Without an override the path is what every existing build used.
                profile_dir = (args.build_dir / profile if tile is None else
                               args.build_dir / f"{profile}+tile{'-'.join(map(str, tile))}")
                raw = profile_dir / f'm{capacity}'  # one capacity, one directory
                options = {'paired_boundary': profile.removeprefix('paired-')} if paired else {}
                if capacity in overrides:
                    options['blocks_per_sm'] = overrides[capacity]
                if tile is not None:
                    options['tile'] = tile
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
                    variant = {'directory': directory, **{key: meta[key] for key in
                        ('capacity', 'intermediate', 'experts', 'top_k', 'output_dtype', 'bits', 'blocks_per_sm')}}
                    if 'tile' in meta:
                        # What the compiler actually resolved, always: the pinned
                        # policy varies per capacity (m16 is the known special case),
                        # so a build without an override still has a real tile.
                        variant['tile'] = meta['tile']
                    if tile is not None:
                        variant['tile_requested'] = list(tile)
                    variants.append(variant)
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
        if requested_layouts:
            present = {v['directory'].split('/')[0] for v in variants}
            missing = sorted(set(requested_layouts) - present)
            if missing:
                raise ValueError(
                    f'EXL3 package did not produce requested layouts: {missing}')
            manifest['requested_layouts'] = requested_layouts
            manifest['requested_capacities'] = capacities
        # Per-variant `tile` already records what each capacity compiled with; the
        # B12x policy legitimately picks a wider tile only at m16, so a package may
        # mix tile values across capacities and that is not an error.
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
    create.add_argument('--require-layout', action='append', default=[],
                        help='Layouts this package must contain (comma list, repeatable). '
                             'Recorded in the manifest and re-checked on verify, so a '
                             'partial build cannot be published as a full contract.')
    create.add_argument('--tile', action='append', default=[],
                        help='Opt-in tile override PROFILE=CAPACITIES:FC1_K,FC1_N,FC2_K,FC2_N '
                             '(CAPACITIES is all or 16+80) for a controlled A/B, for example '
                             'tp3-width768=16:64,256,64,256; default is the B12x per-capacity policy')
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
        manifests = verify_root(args.package, args.sparkinfer_revision, args.runtime, args.role)
        print(json.dumps({
            'verified': True,
            'packages': [
                {'role': manifest['role'], 'variants': len(manifest['variants'])}
                for manifest in manifests
            ],
        }))


if __name__ == '__main__':
    main()
