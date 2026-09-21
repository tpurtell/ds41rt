"""Configure the real EXL3 build rules without compiling or accessing CUDA."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


import importlib.util
from types import ModuleType

RULES = Path(__file__).resolve().parents[2] / 'native/cmake/v41_exl3.cmake'

_spec = importlib.util.spec_from_file_location('exl3_cmake_tile_grammar',
    Path(__file__).resolve().parents[1] / 'tools/package_v41_exl3_aot.py')
package = importlib.util.module_from_spec(_spec)
_pinned = ModuleType('_pinned_sparkinfer')
_pinned.REVISION = 'pinned-for-tests'
import sys
from unittest.mock import patch

with patch.dict(sys.modules, {'_pinned_sparkinfer': _pinned}):
    _spec.loader.exec_module(package)

DISJOINT_SPARK_LAYOUTS = [f'tp4-rank{i}' for i in range(4)] + ['tp2-rank0', 'tp2-rank1'] \
                         + ['tp3-rank0', 'tp3-rank1', 'tp3-rank2']


def documented_tile_examples():
    """Tile examples scraped from the TILES cache documentation, not any other option's.

    A help example is a promise: if it does not configure and parse, the knob ships
    looking usable while being invalid.
    """
    doc = re.search(r'DS41RT_V41_EXL3_TILES[^)]*CACHE STRING "([^"]+)"',
                    RULES.read_text()).group(1)
    # Match the grammar shape itself so the count cannot drift with the wording.
    # Loose token scan: finding the candidates is this test's job, validating them
    # is the parser's and CMake's, which the two tests below then drive.
    return [value.rstrip(',').strip('()')
            for value in re.findall(r'[A-Za-z0-9._+-]+=(?:all|[0-9+]+):[0-9,]+', doc)]


def harness(architecture='121'):
    return f'''
cmake_minimum_required(VERSION 3.24)
project(exl3_options NONE)
set(DS41RT_ENABLE_CUDA ON)
set(DS41RT_CUDA_ARCHITECTURES {architecture} CACHE STRING "test architecture")
set(CUDAToolkit_INCLUDE_DIRS "${{CMAKE_CURRENT_SOURCE_DIR}}")
set(Python3_EXECUTABLE python3)
add_library(CUDA::cudart SHARED IMPORTED)
set_target_properties(CUDA::cudart PROPERTIES IMPORTED_LOCATION /unused/libcudart.so)
add_library(CUDA::cuda_driver SHARED IMPORTED)
set_target_properties(CUDA::cuda_driver PROPERTIES IMPORTED_LOCATION /unused/libcuda.so)
add_custom_target(ds41rt_verify_sparkinfer_source)
include("{RULES}")
'''


@unittest.skipUnless(shutil.which('cmake') and shutil.which('ninja'), 'requires CMake and Ninja')
class Exl3CmakeOptionsTests(unittest.TestCase):
    @staticmethod
    def cmake(source: Path, build: Path, options):
        return subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja', *options],
                              text=True, capture_output=True)

    def configure(self, *options):
        """Configure once into a throwaway tree and return the generated rules."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'CMakeLists.txt').write_text(harness())
            result = self.cmake(root, root / 'build', options)
            rules = root / 'build/build.ninja'
            return result, rules.read_text() if rules.exists() else ''

    def test_explicit_paired_residency_reaches_export_command(self):
        result, rules = self.configure('-DDS41RT_V41_EXL3_PAIRED_TP4=ON',
                                       '-DDS41RT_V41_EXL3_BITS=3;4',
                                       '-DDS41RT_V41_EXL3_RESIDENCY=80=2;16=1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--paired-tp4 --residency 80=2 --residency 16=1', rules)

    def test_default_keeps_disjoint_export(self):
        for architecture in ('120', '121'):
            with self.subTest(architecture=architecture):
                result, rules = self.configure(f'-DDS41RT_CUDA_ARCHITECTURES={architecture}')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('--paired-tp4', rules)
                self.assertNotIn('--residency', rules)

    def test_default_families_declare_all_disjoint_layouts(self):
        for architecture, layouts in (
            ('121', DISJOINT_SPARK_LAYOUTS),
            ('120', ['rtx-tp1', 'rtx-tp2', 'dspark']),
            ('120f', ['rtx-tp1', 'rtx-tp2', 'dspark']),
        ):
            with self.subTest(architecture=architecture):
                result, rules = self.configure(f'-DDS41RT_CUDA_ARCHITECTURES={architecture}')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for family, bits in (('23', '2 3'), ('34', '3 4')):
                    self.assertIn(f'--bits {bits}', rules)
                    for layout in layouts:
                        for capacity in (1, 16, 80, 256, 1024, 4096):
                            for name in ('v41_exl3.json', 'trellis_lut.bin', 'libds41rt_exl3.so'):
                                self.assertIn(f'exl3-k{family}/{layout}/m{capacity}/{name}', rules)
                if architecture != '121':
                    self.assertNotIn('/tp2-rank', rules)
                    self.assertNotIn('/tp3-rank', rules)

    def test_spark_package_requests_exactly_its_declared_layouts(self):
        """The requested-layout contract is one comma-joined argument per family.

        A CMake list would expand into separate arguments and the package tool's
        argparse would reject the command, so this pins the shape that reaches the
        build: every declared Spark rank, including the exact width-768 TP3 group.
        """
        expected = {
            '121': ','.join(DISJOINT_SPARK_LAYOUTS),
            '120': 'rtx-tp1,rtx-tp2,dspark',
        }
        for architecture, layouts in expected.items():
            with self.subTest(architecture=architecture):
                result, rules = self.configure(f'-DDS41RT_CUDA_ARCHITECTURES={architecture}')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                requests = re.findall(r'--require-layout +(\S+)', rules)
                self.assertEqual(requests, [layouts, layouts], 'one request per tier family')
                self.assertNotIn('--require-layout tp4-rank0 tp4-rank1', rules)

    def test_paired_package_requests_only_tp4(self):
        result, rules = self.configure('-DDS41RT_V41_EXL3_PAIRED_TP4=ON',
                                       '-DDS41RT_V41_EXL3_BITS=3;4')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(set(re.findall(r'--require-layout +(\S+)', rules)),
                         {'tp4-rank0,tp4-rank1,tp4-rank2,tp4-rank3'})

    def test_tile_override_reaches_export_and_is_a_build_dependency(self):
        """An A/B override is a dependency, not just a command-line detail.

        Make never restates a custom command's line, so a changed capacity/tile
        selection has to move a file the export depends on.
        """
        result, rules = self.configure('-DDS41RT_V41_EXL3_TILES=tp3-width768=16+80:64,256,64,256')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--tile tp3-width768=16+80:64,256,64,256', rules)
        for family in ('k23', 'k34'):
            stamp = f'v41_exl3_{family}_config.stamp'
            self.assertIn(stamp, rules, f'{family} export must depend on its config stamp')

    def test_reconfiguring_a_tile_rewrites_the_config_stamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, build = Path(temporary), Path(temporary) / 'build'
            (root / 'CMakeLists.txt').write_text(harness())

            def stamps():
                return {path.name: path.read_text() for path in build.rglob('v41_exl3_k*_config.stamp')}

            first = self.cmake(root, build, [])
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            before = stamps()
            self.assertEqual(sorted(before), ['v41_exl3_k23_config.stamp',
                                              'v41_exl3_k34_config.stamp'])
            # Reconfiguring with the same values must not churn the stamp, or every
            # build would pay for a full re-export.
            self.assertEqual(self.cmake(root, build, []).returncode, 0)
            self.assertEqual(stamps(), before)
            changed = self.cmake(root, build,
                                 ['-DDS41RT_V41_EXL3_TILES=tp3-width768=all:128,128,128,128'])
            self.assertEqual(changed.returncode, 0, changed.stdout + changed.stderr)
            after = stamps()
            self.assertNotEqual(after, before, 'a tile change must invalidate the export')
            self.assertIn('tp3-width768=all:128,128,128,128', after['v41_exl3_k23_config.stamp'])
            self.assertTrue(before['v41_exl3_k23_config.stamp'].endswith('tiles=\n'),
                            'a default configuration records no tiles')
            # Capacities and layouts move the stamp for both families too.
            narrowed = self.cmake(root, build, ['-DDS41RT_V41_EXL3_CAPACITIES=1;16;80',
                                                '-DDS41RT_V41_EXL3_TILES=tp3-width768=all:128,128,128,128'])
            self.assertEqual(narrowed.returncode, 0, narrowed.stdout + narrowed.stderr)
            self.assertNotEqual(stamps(), after)

    def test_the_tile_examples_documented_in_cmake_configure(self):
        """Every tile example in the cache documentation must be a valid value.

        A stale example is a shipped knob that fails at configure time, so the
        examples are scraped from the file and driven through the real gate.
        """
        documented = documented_tile_examples()
        self.assertTrue(documented, 'v41_exl3.cmake documents no tile example')
        for example in documented:
            example = example.rstrip(',').strip('()')
            with self.subTest(example=example):
                result, rules = self.configure(f'-DDS41RT_V41_EXL3_TILES={example}')
                self.assertEqual(result.returncode, 0, result.stdout[-500:] + result.stderr[-500:])
                self.assertIn(f'--tile {example}', rules)

    def test_the_documented_tile_examples_parse_in_the_package_tool(self):
        """The CMake examples and the Python parser must agree on the grammar."""
        documented = documented_tile_examples()
        self.assertEqual(len(documented), 2, f'expected both documented examples, got {documented}')
        for example in documented:
            with self.subTest(example=example):
                parsed = package.tile_overrides([example], [1, 16, 80], 'spark', False)
                self.assertEqual(list(parsed), ['tp3-width768'])
                self.assertIn(int(example.split(':')[0].split('=')[1].split('+')[0]
                                 if example.split(':')[0].split('=')[1] != 'all' else 1), [1, 16, 80])

    def test_malformed_tile_override_fails_configuration(self):
        for value in ('tp3-width768=16:64,256,64', 'tp3-width768:64,256,64,256',
                      'tp3 width768 all:64,256,64,256', 'nonsense'):
            with self.subTest(value=value):
                result, _ = self.configure(f'-DDS41RT_V41_EXL3_TILES={value}')
                self.assertNotEqual(result.returncode, 0, value)
                self.assertIn('EXL3 tile override must be PROFILE=CAPACITIES', result.stderr)

    def test_paired_families_remain_tp4_only(self):
        for bits, family in (('2;3', '23'), ('3;4', '34')):
            with self.subTest(family=family):
                result, rules = self.configure('-DDS41RT_V41_EXL3_PAIRED_TP4=ON',
                                               f'-DDS41RT_V41_EXL3_BITS={bits}')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('--paired-tp4', rules)
                self.assertNotIn('/tp2-rank', rules)
                self.assertNotIn('/tp3-rank', rules)
                for rank in range(4):
                    self.assertIn(f'exl3-k{family}/tp4-rank{rank}/m16/libds41rt_exl3.so', rules)

    def test_paired_default_rejects_multiple_families(self):
        result, _ = self.configure('-DDS41RT_V41_EXL3_PAIRED_TP4=ON')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exactly one decoder tier family', result.stderr)

    def test_invalid_layouts_fail_during_configuration(self):
        paired = '-DDS41RT_V41_EXL3_PAIRED_TP4=ON'
        cases = [
            (['-DDS41RT_V41_EXL3_RESIDENCY=80=2'], 'require paired TP4'),
            ([paired, '-DDS41RT_CUDA_ARCHITECTURES=120'], 'requires SM121'),
            ([paired, '-DDS41RT_V41_EXL3_BITS=2;3;4'], 'exactly two decoder tiers'),
            ([paired, '-DDS41RT_V41_EXL3_RESIDENCY=80=3'], 'capacity=1 or capacity=2'),
            ([paired, '-DDS41RT_V41_EXL3_RESIDENCY=17=2'], 'selected, nonduplicate'),
            ([paired, '-DDS41RT_V41_EXL3_RESIDENCY=80=1;80=2'], 'selected, nonduplicate'),
        ]
        for options, error in cases:
            with self.subTest(options=options):
                result, _ = self.configure('-DDS41RT_V41_EXL3_BITS=3;4', *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)


if __name__ == '__main__':
    unittest.main()
