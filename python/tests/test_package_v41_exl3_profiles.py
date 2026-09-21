"""Exercise package contents with a fake compiler; no CUDA, no device context.

The tile assertions ask the pinned B12x planner what its own policy resolves, so
this file does not restate numbers it cannot own. That import is a pure arithmetic
resolver and creates no CUDA context; it is not fully isolated, because unloading
torch's C extensions corrupts later imports in the same process (see
`_load_planner_tile`), so the pin may stay cached for the session. Where the pin is
absent the dependent tests skip rather than error.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]

def _load_planner_tile(source=None):
    """Import the pinned planner's tile resolver without leaking into this process.

    `_projection_mixed_tile_config` is pure arithmetic, so no CUDA context is created.
    On success only `sys.path` is restored: the import drags in torch's C extensions,
    and deleting those from `sys.modules` afterwards leaves them half-registered, so a
    later `import torch` in the same process fails with
    `SystemError: ... bad call flags`. Anything unavailable yields None so the
    dependent tests skip instead of erroring. `source` is a parameter so the failure
    path is testable.
    """
    source = str(source or ROOT / 'third_party' / 'sparkinfer')
    path_snapshot, modules_before = list(sys.path), set(sys.modules)
    try:
        sys.path.insert(0, source)
        from b12x.moe.fused_moe._impl import _projection_mixed_tile_config
        return _projection_mixed_tile_config
    except Exception:
        # A failed import may leave half-initialized b12x modules behind. Drop only
        # the newly added b12x ones: torch and the rest of the dependency graph are
        # shared with the rest of the session and unloading them corrupts later
        # imports (SystemError: bad call flags).
        for name in set(sys.modules) - modules_before:
            if name == 'b12x' or name.startswith('b12x.'):
                sys.modules.pop(name, None)
        return None
    finally:
        sys.path[:] = path_snapshot


POLICY_TILE = _load_planner_tile()


def _load_direct_route_policy():
    """The same helper production uses to choose direct vs packed routing.

    Imported the way the tile policy is, so the fixture follows the pin's rule
    instead of freezing a constant that happens to be right today.
    """
    path_snapshot = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT / 'third_party' / 'sparkinfer'))
        from b12x.moe.fused_moe._impl import _projection_mixed_direct_topk_routes
        return _projection_mixed_direct_topk_routes
    except Exception:  # pragma: no cover - absence only relaxes the fixture
        return None
    finally:
        sys.path[:] = path_snapshot


DIRECT_ROUTES = _load_direct_route_policy()
# Positive placeholder used when the pinned planner is unavailable. Never compared
# against the planner; it exists so non-policy tests still carry a valid tile shape.
NEUTRAL_POLICY_TILE = [16, 32, 16, 32]

spec = importlib.util.spec_from_file_location('exl3_package_profiles',
    Path(__file__).resolve().parents[1] / 'tools/package_v41_exl3_aot.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageProfileTests(unittest.TestCase):
    @staticmethod
    def expected_policy_tile(variant):
        """Ask the pinned planner; never restate its numbers in a test."""
        # Two-tier fixture exports are direct-capacity-bound exactly as production;
        # with this pin the resolver says packed for the capacities used here.
        direct = (False if DIRECT_ROUTES is None else bool(DIRECT_ROUTES(
            variant['capacity'], variant['top_k'], direct_exl3=True)))
        return list(POLICY_TILE(None, hidden_size=5120, intermediate_size=variant['intermediate'],
                                token_count=variant['capacity'], direct_topk_routes=direct))

    # How many ranks share one export per width, from the layout contract.
    expected_rank_count = {'tp2-rank': 2, 'tp3-rank': 3}

    def build_fixture(self, root, role, bits, paired=False, require=(), tiles=()):
        capacities = (1, 80)
        args = argparse.Namespace(
            output=root / 'package', build_dir=root / 'build', role=role,
            bits=list(bits), capacities=','.join(map(str, capacities)), paired_tp4=paired,
            residency=[], require_layout=list(require), tile=list(tiles),
            cxx='unused-cxx', cuda_include=root, cuda_libdir=root,
            cuda_driver=root / 'libcuda.so', runtime=root / 'libcute_dsl_runtime.so')
        args.runtime.write_bytes(b'runtime fixture')
        pinned = ModuleType('_pinned_sparkinfer')
        pinned.REVISION = 'fixture-revision'
        exporter = ModuleType('export_b12x_v41_exl3_aot')
        torch = ModuleType('torch')
        torch.cuda = SimpleNamespace(
            get_device_properties=Mock(return_value=SimpleNamespace(
                major=12, minor=1 if role == 'spark' else 0, multi_processor_count=48)),
            empty_cache=Mock())

        def export(raw, width, experts, capacity, tiers, routing, topk, dtype, **options):
            self.assertEqual(routing, 'auto')
            raw.mkdir(parents=True)
            meta = dict(capacity=capacity, intermediate=width, experts=experts,
                        top_k=topk, output_dtype=dtype, bits=list(tiers), blocks_per_sm=1,
                        sparkinfer_revision=pinned.REVISION)
            if paired:
                meta.update(paired_boundary=options['paired_boundary'],
                            descriptor_rows=4, native_info_version=3)
                self.assertNotIn('tile', options,
                                 'the exporter refuses a tile override on paired builds')
            # The build command may request a tile override and a residency cap.
            self.assertLessEqual(set(options), {'tile', 'blocks_per_sm', 'paired_boundary'},
                                 'unexpected option consumed by the fake export')
            # Same derivation as the exporter: direct vs packed routing decides the
            # route bridge, so m1 needs it too whenever routing is packed.
            direct = (False if DIRECT_ROUTES is None else bool(DIRECT_ROUTES(
                capacity, topk, direct_exl3=len(tiers) == 2)))
            meta['direct'] = direct
            meta['requires_route_preparation'] = not direct
            # The real exporter always records the tile it resolved, paired builds
            # included, and the pinned policy differs per width *and* per capacity,
            # so the fixture asks the planner itself rather than restating numbers.
            if 'tile' in options:
                meta['tile'] = list(options['tile'])
            elif POLICY_TILE is None:
                # A positive placeholder that is clearly not a policy value: the
                # ungated tests only need "a tile was recorded and survives
                # verification", and verify() rejects zeros outright.
                meta['tile'] = list(NEUTRAL_POLICY_TILE)
            else:
                meta['tile'] = list(POLICY_TILE(None, hidden_size=5120,
                    intermediate_size=width, token_count=capacity,
                    direct_topk_routes=direct))
            if meta['requires_route_preparation']:
                routes = raw / 'routes'
                routes.mkdir()
                route_manifest = routes / 'v41_exl3_routes.json'
                route_manifest.write_text('{}')
                meta['route_preparation'] = dict(manifest='routes/v41_exl3_routes.json',
                                                  sha256=package.digest(route_manifest))
            (raw / 'v41_exl3.json').write_text(json.dumps(meta))
            (raw / 'trellis_lut.bin').write_bytes(b'lut fixture')
            return meta

        exporter.export = Mock(side_effect=export)

        def link(command, *, check):
            self.assertTrue(check)
            self.assertEqual(command[0], 'unused-cxx')
            Path(command[-1]).write_bytes(b'linked fixture')

        with patch.dict(sys.modules, {'_pinned_sparkinfer': pinned,
                                     'export_b12x_v41_exl3_aot': exporter, 'torch': torch}), \
                patch.object(package.subprocess, 'run', side_effect=link):
            package.build(args)
        manifest = package.verify(args.output, pinned.REVISION, args.runtime, role)
        self.assertEqual(torch.cuda.empty_cache.call_count, exporter.export.call_count)
        return args.output, manifest, exporter.export.call_args_list, capacities

    def assert_layouts(self, manifest, capacities, expected, bits):
        variants = {v['directory']: v for v in manifest['variants']}
        self.assertEqual(set(variants),
                         {f'{layout}/m{capacity}' for layout in expected for capacity in capacities})
        for directory, variant in variants.items():
            width, experts, topk, dtype = expected[directory.split('/')[0]]
            self.assertEqual((variant['intermediate'], variant['experts'], variant['top_k'],
                              variant['output_dtype']), (width, experts, topk, dtype))
            self.assertEqual(variant['bits'], list(bits))

    def test_spark_disjoint_families_include_tp2_tp3_and_preserve_tp4(self):
        """One Spark package serves TP4, exact TP2 and exact TP3 ownership.

        TP3 is the 6+6+6 split of the 18 whole 128-wide Trellis blocks at I=2304, so
        all three ranks share one width-768 export per capacity - like TP2, and
        unlike the padded 640/512 TP4 pair-split.
        """
        expected = {f'tp4-rank{rank}': (640 if rank < 2 else 512, 384, 6, 'bf16')
                    for rank in range(4)}
        expected.update({f'tp2-rank{rank}': (1152, 384, 6, 'bf16') for rank in range(2)})
        expected.update({f'tp3-rank{rank}': (768, 384, 6, 'bf16') for rank in range(3)})
        for bits in ((2, 3), (3, 4)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                output, manifest, calls, capacities = self.build_fixture(Path(temporary), 'spark', bits)
                self.assert_layouts(manifest, capacities, expected, bits)
                self.assertFalse(manifest.get('paired_tp4', False))
                self.assertEqual(len(calls), 4 * len(capacities))
                for group, width in (('tp2-rank', 1152), ('tp3-rank', 768)):
                    for capacity in capacities:
                        shared = [call for call in calls
                                  if call.args[1] == width and call.args[3] == capacity]
                        self.assertEqual(
                            len(shared), 1,
                            f'compile width {width} once per capacity, not per rank')
                        ranks = sorted(layout for layout in expected if layout.startswith(group))
                        self.assertEqual(len(ranks), self.expected_rank_count[group])
                        first = output / f'{ranks[0]}/m{capacity}'
                        for twin_layout in ranks[1:]:
                            twin = output / f'{twin_layout}/m{capacity}'
                            for path in first.rglob('*'):
                                if path.is_file():
                                    self.assertEqual(
                                        path.read_bytes(),
                                        (twin / path.relative_to(first)).read_bytes())

    def test_requested_layouts_are_recorded_and_reverified(self):
        """A declared contract survives into verify(); its absence does not (v9)."""
        require = ['tp3-rank0', 'tp3-rank1', 'tp3-rank2']
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, _, _ = self.build_fixture(Path(temporary), 'spark', (2, 3),
                                                        require=require)
            self.assertEqual(manifest['requested_layouts'], require)
            self.assertEqual(package.verify(output, 'fixture-revision'), manifest)

    def test_requesting_a_layout_the_role_cannot_build_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError) as caught:
                self.build_fixture(Path(temporary), 'coordinator', (2, 3), require=['tp3-rank0'])
        self.assertIn('unknown EXL3 layout requested: tp3-rank0', str(caught.exception))

    def test_verify_rejects_a_package_missing_a_requested_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, _, _ = self.build_fixture(Path(temporary), 'spark', (2, 3),
                                                        require=['tp3-rank0'])
            tampered = dict(manifest, requested_layouts=['tp3-rank0', 'tp4-rank0', 'tp9-rank0'])
            (output / 'manifest.json').write_text(json.dumps(tampered))
            with self.assertRaises(ValueError) as caught:
                package.verify(output)
        self.assertIn('missing requested layout', str(caught.exception))

    def test_a_spark_only_tile_knob_cannot_be_requested_on_the_coordinator(self):
        """No silent knobs: an override for a profile this role never builds fails.

        Accepting it and ignoring it would ship looking effective, so the profile
        set is scoped to the role being built, exactly like --require-layout.
        """
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError) as caught:
                self.build_fixture(Path(temporary), 'coordinator', (2, 3),
                                   tiles=['tp3-width768=1:64,256,64,256'])
        message = str(caught.exception)
        self.assertIn('must name a coordinator profile', message)
        self.assertIn('rtx-tp1', message, 'the error lists what the role can build')

    def test_the_tile_grammar_documented_in_help_is_the_grammar_parsed(self):
        """The help example has to parse; a typo in it is an invalid shipped knob.

        Reads the example straight out of the argparse help text so the two cannot
        drift apart, and keeps the two historical malformed spellings rejected.
        """
        import argparse
        tool = argparse.ArgumentParser()
        tool.add_argument('--tile', action='append', default=[])
        source = Path(package.__file__).read_text()
        examples = [token for token in source.split() if 'tp3-width768=' in token]
        self.assertTrue(examples, 'the package tool documents no tile example')
        for example in examples:
            cleaned = example.strip("'(),;")
            with self.subTest(example=cleaned):
                parsed = package.tile_overrides([cleaned], [1, 16, 80], 'spark', False)
                self.assertEqual(list(parsed), ['tp3-width768'])
        for bad in ('tp3-width768=64,256,64,256', 'tp3-width768=16=64,256,64,256'):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as caught:
                package.tile_overrides([bad], [1, 16, 80], 'spark', False)
            self.assertIn('must name a spark profile as PROFILE=CAPACITIES', str(caught.exception))

    @unittest.skipIf(POLICY_TILE is None, 'requires the pinned b12x planner')
    def test_recorded_tiles_are_the_planner_values_for_their_own_shape(self):
        """Policy equality alone: what the pin resolved per width and capacity.

        Split from the tamper negatives so those always run; only this comparison
        needs the planner importable.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, manifest, _calls, _ = self.build_fixture(root, 'spark', (2, 3))
            for variant in manifest['variants']:
                with self.subTest(directory=variant['directory']):
                    self.assertEqual(variant['tile'], self.expected_policy_tile(variant))
                    self.assertNotIn('tile_requested', variant)
                    self.assertEqual(package.verify(output, 'fixture-revision')['variants'],
                                     manifest['variants'])

    def test_verify_cross_checks_the_recorded_tile_against_the_export(self):
        """A tile in the manifest is a claim about compiled geometry.

        The export's own record is the authority: an edited manifest must not be
        able to rename the geometry a variant was compiled for, and a requested
        override that the compiler resolved differently must fail closed.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            built = self.build_fixture(root, 'spark', (2, 3), tiles=['tp3-width768=80:64,256,64,256'])
            manifest = json.loads((root / 'package/manifest.json').read_text())
            variant = next(v for v in manifest['variants'] if v['directory'] == 'tp3-rank0/m80')
            self.assertEqual(variant['tile'], [64, 256, 64, 256], 'the override is the compiled tile')
            self.assertEqual(variant['tile_requested'], variant['tile'])
            policy = next(v for v in manifest['variants'] if v['directory'] == 'tp3-rank0/m1')
            self.assertIn('tile', policy, 'a build with no override still records a tile')
            self.assertNotIn('tile_requested', policy)

            for key, expected in (('tile', 'variant tile mismatch'),
                                  ('tile_requested', 'override was not applied')):
                with self.subTest(field=key):
                    tampered = json.loads((root / 'package/manifest.json').read_text())
                    target = next(v for v in tampered['variants'] if v['directory'] == 'tp3-rank0/m80')
                    target[key] = [128, 128, 128, 128]
                    with tempfile.TemporaryDirectory() as other:
                        copy = Path(other) / 'package'
                        shutil.copytree(root / 'package', copy)
                        (copy / 'manifest.json').write_text(json.dumps(tampered))
                        with self.assertRaises(ValueError) as caught:
                            package.verify(copy, 'fixture-revision')
                    self.assertIn(expected, str(caught.exception))

    def test_verify_rejects_malformed_tile_fields_even_when_a_planner_is_absent(self):
        """A recorded tile is data, so it has to be shaped like one.

        Runs without the pinned planner: the check protects every package, and v9
        manifests are exempt only because they carry neither key.
        """
        broken = {
            'zeros': [0, 0, 0, 0],
            'negative': [-1, 64, 64, 64],
            'short': [64, 256, 64],
            'long': [64, 256, 64, 256, 64],
            'string': '64,256,64,256',
            'floats': [64.0, 256.0, 64.0, 256.0],
            'booleans': [True, True, True, True],
        }
        for field in ('tile', 'tile_requested'):
            for label, value in broken.items():
                if field == 'tile_requested' and label in ('zeros', 'short'):
                    continue  # covered by the tile column; same validation path
                with self.subTest(field=field, value=label):
                    with tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        self.build_fixture(root, 'spark', (2, 3),
                                           tiles=['tp3-width768=80:64,256,64,256'])
                        manifest_path = root / 'package/manifest.json'
                        manifest = json.loads(manifest_path.read_text())
                        target = next(v for v in manifest['variants']
                                      if v['directory'] == 'tp3-rank0/m80')
                        target[field] = value
                        manifest_path.write_text(json.dumps(manifest))
                        with self.assertRaises(ValueError) as caught:
                            package.verify(root / 'package', 'fixture-revision')
                        self.assertIn(f'{field} must be four positive integers',
                                      str(caught.exception))

    def test_planner_import_failure_cleans_only_b12x(self):
        """A failed planner import must not unload the shared dependency graph.

        Deleting `b12x` from `sys.modules` is not enough to force the failure path
        in a full-suite run, since the pin is importable from elsewhere; a meta-path
        blocker makes it deterministic. Unloading torch alongside a half-imported
        pin is exactly what corrupted later torch imports, so that is asserted too.
        """
        class BlockB12x:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'b12x' or fullname.startswith('b12x.'):
                    raise ImportError(f'blocked for {self.name}')
                return None
            name = 'b12x-blocker'

        cached = {name: module for name, module in list(sys.modules.items())
                  if name == 'b12x' or name.startswith('b12x.')}
        torch_before = sys.modules.get('torch')
        path_snapshot = list(sys.path)
        blocker = BlockB12x()
        sys.meta_path.insert(0, blocker)
        for name in cached:
            del sys.modules[name]
        try:
            with tempfile.TemporaryDirectory() as temporary:
                self.assertIsNone(_load_planner_tile(source=Path(temporary) / 'absent'))
            leaked = {name for name in sys.modules if name == 'b12x' or name.startswith('b12x.')}
            self.assertEqual(leaked - set(cached), set(), 'half-imported b12x modules leaked')
            self.assertIs(sys.modules.get('torch'), torch_before,
                          'torch and other shared deps must survive a failed pin import')
            self.assertEqual(sys.path, path_snapshot, 'sys.path must be restored verbatim')
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(cached)

    def test_a_v9_package_without_tile_records_still_verifies(self):
        """Backward compatibility: the tile check is conditional on the key existing."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_fixture(root, 'spark', (3, 4))
            manifest_path = root / 'package/manifest.json'
            manifest = json.loads(manifest_path.read_text())
            for variant in manifest['variants']:
                variant.pop('tile', None)
                variant.pop('tile_requested', None)
            manifest_path.write_text(json.dumps(manifest))
            verified = package.verify(root / 'package', 'fixture-revision')
            self.assertEqual(len(verified['variants']), len(manifest['variants']))

    def test_verify_catches_a_contract_that_outruns_the_package(self):
        """A declared capacity with no variant for a required rank is an error.

        Deriving the capacity set from the variants alone would let a build that
        silently lost one capacity for every rank verify cleanly, so the recorded
        capacities drive the cross-product check.
        """
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, _, capacities = self.build_fixture(
                Path(temporary), 'spark', (2, 3), require=['tp3-rank0'])
            self.assertEqual(manifest['requested_capacities'], list(capacities))
            inflated = dict(manifest, requested_capacities=[*capacities, 999])
            (output / 'manifest.json').write_text(json.dumps(inflated))
            with self.assertRaises(ValueError) as caught:
                package.verify(output)
        self.assertIn('tp3-rank0/m999', str(caught.exception))

    def test_tile_override_is_capacity_scoped_and_directory_keyed(self):
        """An A/B retargets one capacity without flattening the policy at others.

        The m16 geometry is where the pinned policy already differs, so scoping is
        what keeps an override honest; the override also must not reuse the
        policy-keyed export directory.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, manifest, calls, _ = self.build_fixture(
                root, 'spark', (2, 3), tiles=['tp3-width768=80:64,256,64,256'])
            overridden = [call for call in calls if call.args[1] == 768 and 'tile' in call.kwargs]
            self.assertEqual([call.args[3] for call in overridden], [80])
            self.assertEqual(overridden[0].kwargs['tile'], (64, 256, 64, 256))
            by_directory = {v['directory']: v for v in manifest['variants']}
            self.assertEqual(by_directory['tp3-rank0/m80']['tile'], [64, 256, 64, 256])
            self.assertEqual(by_directory['tp3-rank0/m80']['tile_requested'], [64, 256, 64, 256])
            # Every variant records the tile the compiler resolved, so an override
            # is distinguished by the *request*, not by a missing key elsewhere.
            self.assertNotIn('tile_requested', by_directory['tp3-rank0/m1'])
            requested = {v['directory'] for v in manifest['variants'] if 'tile_requested' in v}
            self.assertEqual(requested, {f'tp3-rank{rank}/m80' for rank in range(3)})
            # Every capacity keeps its own export directory; the override does not
            # flatten the policy at the others.
            self.assertEqual({v['directory'] for v in manifest['variants']
                              if v['directory'].startswith('tp3')},
                             {f'tp3-rank{rank}/m{cap}' for rank in range(3) for cap in (1, 80)})
            staged = root / 'build'
            self.assertTrue((staged / 'tp3-width768+tile64-256-64-256/m80').is_dir())
            self.assertFalse((staged / 'tp3-width768/m80').exists())
            self.assertTrue((staged / 'tp3-width768/m1').is_dir(), 'policy path unchanged')
            self.assertEqual(package.verify(output, 'fixture-revision'), manifest)

    def test_tile_override_rejects_unscoped_and_paired_requests(self):
        for value, reason in (
                ('tp9-width1=1:64,256,64,256', 'must name a spark profile'),
                ('tp3-width768=99:64,256,64,256', 'not packaged'),
                ('tp3-width768=1:64,256,64', 'four positive tiles'),
                ('tp3-width768:64,256,64,256', 'must name a spark profile')):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError) as caught:
                    self.build_fixture(Path(temporary), 'spark', (2, 3), tiles=[value])
            self.assertIn(reason, str(caught.exception))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError) as caught:
                self.build_fixture(Path(temporary), 'spark', (2, 3), paired=True,
                                   tiles=['tp3-width768=1:64,256,64,256'])
        self.assertIn('not supported for paired', str(caught.exception))

    def test_spark_paired_families_remain_tp4_only(self):
        expected = {f'tp4-rank{rank}': (640, 384, 6, 'bf16') for rank in range(4)}
        for bits in ((2, 3), (3, 4)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                _, manifest, calls, capacities = self.build_fixture(Path(temporary), 'spark', bits, paired=True)
                self.assert_layouts(manifest, capacities, expected, bits)
                self.assertTrue(manifest['paired_tp4'])
                self.assertEqual(len(calls), 2 * len(capacities))
                for variant in manifest['variants']:
                    rank = int(variant['directory'].split('/')[0][-1])
                    self.assertEqual(variant['paired_boundary'], 'last' if rank % 2 == 0 else 'first')

    def test_coordinator_families_unchanged(self):
        expected = {'rtx-tp1': (2304, 384, 6, 'fp32'),
                    'rtx-tp2': (1152, 384, 6, 'fp32'), 'dspark': (2304, 128, 3, 'bf16')}
        for bits in ((2, 3), (3, 4)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                _, manifest, calls, capacities = self.build_fixture(Path(temporary), 'coordinator', bits)
                self.assert_layouts(manifest, capacities, expected, bits)
                self.assertEqual(len(calls), 3 * len(capacities))
                self.assertFalse(manifest.get('paired_tp4', False))


if __name__ == '__main__':
    unittest.main()
