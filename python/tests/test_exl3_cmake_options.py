"""Configure the real EXL3 build rules without compiling or accessing CUDA."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


RULES = Path(__file__).resolve().parents[2] / 'native/cmake/v41_exl3.cmake'


@unittest.skipUnless(shutil.which('cmake') and shutil.which('ninja'), 'requires CMake and Ninja')
class Exl3CmakeOptionsTests(unittest.TestCase):
    def configure(self, *options):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.24)
project(exl3_options NONE)
set(DS41RT_ENABLE_CUDA ON)
set(DS41RT_CUDA_ARCHITECTURES 121 CACHE STRING "test architecture")
set(CUDAToolkit_INCLUDE_DIRS "${{CMAKE_CURRENT_SOURCE_DIR}}")
set(Python3_EXECUTABLE python3)
add_library(CUDA::cudart SHARED IMPORTED)
set_target_properties(CUDA::cudart PROPERTIES IMPORTED_LOCATION /unused/libcudart.so)
add_library(CUDA::cuda_driver SHARED IMPORTED)
set_target_properties(CUDA::cuda_driver PROPERTIES IMPORTED_LOCATION /unused/libcuda.so)
add_custom_target(ds41rt_verify_sparkinfer_source)
include("{RULES}")
''')
            result = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'),
                                     '-G', 'Ninja', *options], text=True, capture_output=True)
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
            ('121', [f'tp4-rank{i}' for i in range(4)] + ['tp2-rank0', 'tp2-rank1']),
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

    def test_paired_families_remain_tp4_only(self):
        for bits, family in (('2;3', '23'), ('3;4', '34')):
            with self.subTest(family=family):
                result, rules = self.configure('-DDS41RT_V41_EXL3_PAIRED_TP4=ON',
                                               f'-DDS41RT_V41_EXL3_BITS={bits}')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('--paired-tp4', rules)
                self.assertNotIn('/tp2-rank', rules)
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
