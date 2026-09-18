"""Exercise package contents with a fake compiler; never import CUDA or Torch."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location('exl3_package_profiles',
    Path(__file__).resolve().parents[1] / 'tools/package_v41_exl3_aot.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageProfileTests(unittest.TestCase):
    def build_fixture(self, root, role, bits, paired=False):
        capacities = (1, 80)
        args = argparse.Namespace(
            output=root / 'package', build_dir=root / 'build', role=role,
            bits=list(bits), capacities=','.join(map(str, capacities)), paired_tp4=paired,
            residency=[], cxx='unused-cxx', cuda_include=root, cuda_libdir=root,
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
                        sparkinfer_revision=pinned.REVISION,
                        requires_route_preparation=capacity > 1)
            if paired:
                meta.update(paired_boundary=options['paired_boundary'],
                            descriptor_rows=4, native_info_version=3)
            else:
                self.assertEqual(options, {})
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

    def test_spark_disjoint_families_include_tp2_and_preserve_tp4(self):
        expected = {f'tp4-rank{rank}': (640 if rank < 2 else 512, 384, 6, 'bf16')
                    for rank in range(4)}
        expected.update({f'tp2-rank{rank}': (1152, 384, 6, 'bf16') for rank in range(2)})
        for bits in ((2, 3), (3, 4)):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as temporary:
                output, manifest, calls, capacities = self.build_fixture(Path(temporary), 'spark', bits)
                self.assert_layouts(manifest, capacities, expected, bits)
                self.assertFalse(manifest.get('paired_tp4', False))
                self.assertEqual(len(calls), 3 * len(capacities))
                for capacity in capacities:
                    tp2_calls = [call for call in calls if call.args[1] == 1152 and call.args[3] == capacity]
                    self.assertEqual(len(tp2_calls), 1, 'compile TP2 once per capacity, not per rank')
                    rank0 = output / f'tp2-rank0/m{capacity}'
                    for path in rank0.rglob('*'):
                        if path.is_file():
                            twin = output / f'tp2-rank1/m{capacity}' / path.relative_to(rank0)
                            self.assertEqual(path.read_bytes(), twin.read_bytes())

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
