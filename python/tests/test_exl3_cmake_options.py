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
                result, _ = self.configure(*options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)


if __name__ == '__main__':
    unittest.main()
