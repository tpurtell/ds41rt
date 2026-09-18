"""Compile the generated bridge against CPU-only CUDA stubs; no GPU access."""
import ctypes as ct
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / 'tools/export_b12x_v41_exl3_aot.py'


def exporter():
    spec = importlib.util.spec_from_file_location('exl3_export_grid_test', TOOL)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'_pinned_sparkinfer': ModuleType('_pinned_sparkinfer')}):
        spec.loader.exec_module(module)
    return module


@unittest.skipUnless(shutil.which('c++'), 'requires a host C++ compiler')
class Exl3BridgeGridTests(unittest.TestCase):
    def test_actual_generated_bridge_clamps_grid_without_changing_scalar_table(self):
        for blocks in (1, 2):
            with self.subTest(blocks=blocks), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                common = '''#pragma once
#include <cstdint>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaLibrary_t = void*;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorMemoryAllocation=2, cudaErrorInvalidDevice=10;
struct cudaDeviceProp { int major, minor, multiProcessorCount; };
static int major=12, minor=0, sms=188, current=0, last_grid=-1;
static int cudaGetDevice(int* out) { *out=current; return 0; }
static int cudaGetDeviceProperties(cudaDeviceProp* out,int) { *out={major,minor,sms}; return 0; }
static int cudaLibraryUnload(void*) { return 0; }
extern "C" void set_device(int ma,int mi,int sm,int id) { major=ma;minor=mi;sms=sm;current=id; }
extern "C" int observed_grid() { return last_grid; }
'''
                (root / 'stub.h').write_text(common)
                objects = []
                for role in ('core', 'sum'):
                    label = 'v41_exl3_' + role
                    module_type = 'ds41rt_' + label + '_Kernel_Module_t'
                    (root / (label + '.h')).write_text(f'''#include "stub.h"
struct {module_type} {{ void* module=nullptr; }};
static void _mlir_ds41rt_{label}_cuda_init(void**) {{}}
static void _mlir_ds41rt_{label}_cuda_load_to_device(void**) {{}}
static int wrapper_{role}({module_type}*,void*,int32_t active_m,int32_t grid_x,cudaStream_t) {{
    last_grid=grid_x; return 0;
}}
''')
                    objects.append(dict(label=label, wrapper=f'wrapper_{role}', parameters=[
                        f'{module_type} *module', 'void *input', 'int32_t active_m',
                        'int32_t grid_x', 'cudaStream_t stream']))
                manifest = dict(compute=[12, 0], sms=188, blocks_per_sm=blocks,
                                capacity=16, hidden=5120, intermediate=2304, experts=384,
                                top_k=6, bits=[2, 3], output_dtype='fp32', objects=objects)
                exporter().write_bridge(root, manifest)
                result = subprocess.run(['c++', '-shared', '-fPIC', '-std=c++17',
                                         str(root / 'v41_exl3_bridge.cc'), '-o', str(root / 'bridge.so')],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                lib = ct.CDLL(str(root / 'bridge.so'))
                lib.set_device.argtypes = [ct.c_int] * 4
                lib.ds41rt_exl3_create.argtypes = [ct.POINTER(ct.c_void_p)]
                lib.ds41rt_exl3_destroy.argtypes = [ct.c_void_p]
                lib.ds41rt_exl3_core.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p),
                                                ct.POINTER(ct.c_int32), ct.c_void_p]
                for sms in (170, 188, 200):
                    lib.set_device(12, 0, sms, 0)
                    context = ct.c_void_p()
                    self.assertEqual(lib.ds41rt_exl3_create(ct.byref(context)), 0)
                    pointers = (ct.c_void_p * 1)(1)
                    for requested in (1, 170 * blocks, 188 * blocks, 200 * blocks):
                        scalars = (ct.c_int32 * 2)(16, requested)
                        self.assertEqual(lib.ds41rt_exl3_core(context, pointers, scalars, None), 0)
                        self.assertEqual(lib.observed_grid(), min(requested, min(sms, 188) * blocks))
                        self.assertEqual(list(scalars), [16, requested])
                    for rows, grid in ((0, 1), (17, 1), (1, 0), (1, -1)):
                        self.assertNotEqual(lib.ds41rt_exl3_core(context, pointers,
                                             (ct.c_int32 * 2)(rows, grid), None), 0)
                    lib.set_device(12, 0, sms, 1)
                    self.assertNotEqual(lib.ds41rt_exl3_core(context, pointers,
                                         (ct.c_int32 * 2)(1, 1), None), 0)
                    lib.set_device(12, 0, sms, 0)
                    lib.ds41rt_exl3_destroy(context)
                for major, minor, sms in ((12, 1, 188), (11, 0, 188), (12, 0, 0)):
                    lib.set_device(major, minor, sms, 0)
                    context = ct.c_void_p()
                    self.assertNotEqual(lib.ds41rt_exl3_create(ct.byref(context)), 0)
                    self.assertFalse(context.value)

    def test_invalid_export_grid_rejected(self):
        for sms, blocks in ((0, 1), (188, 0), (188, 3), (2**31, 1)):
            with self.subTest(sms=sms, blocks=blocks):
                with self.assertRaisesRegex(ValueError, 'cooperative grid capacity'):
                    exporter().write_bridge(Path('/unused'), dict(sms=sms, blocks_per_sm=blocks))


if __name__ == '__main__':
    unittest.main()
