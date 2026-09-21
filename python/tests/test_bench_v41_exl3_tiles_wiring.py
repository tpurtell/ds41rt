"""End-to-end CPU wiring of `bench_v41_exl3_tiles.main` behind a fake compiler.

The real tool is SM121-only; a Spark session costs minutes of weight loading
before a wiring bug could surface. This fixture runs the actual `main()` flow
(slice contract -> checkpoint identity -> weight load -> tier family -> real
pinned planner -> variant race -> invariants -> oracle gates -> timing ->
JSON record) against numpy-backed fake tensors, a fake safetensors reader,
and fake b12x compile/bind/run entry points. The tensors carry the REAL
publication geometries (from the staged manifest: w1/w3 trellis
(320,144,16*bits) int16, w2 trellis (144,320,16*bits) int16, scale vectors
float16, mcg int32) plus block/position MARKER content, so the fake prepare
asserts the harness sliced exactly the rank window under test. The tile
resolver, the direct-topk decision and the route-pack capacity are the REAL
pinned pure functions. Nothing here measures kernel numeric behavior (that is
what the GPU run is for).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'python' / 'tools'
TARGET = 'wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1'
EXPERTS, HIDDEN, TOTAL = 384, 5120, 2304
PROJECTIONS = ('w1', 'w3', 'w2')
FAKE_LAYERS = range(41)  # the staged publication's target-layer population


def _load_pinned_pure():
    """Real pure planner/capacity functions from the pinned tree; no CUDA."""
    path_snapshot, modules_before = list(sys.path), set(sys.modules)
    try:
        sys.path.insert(0, str(ROOT / 'third_party' / 'sparkinfer'))
        from b12x.moe.fused_moe._impl import (
            _projection_mixed_direct_topk_routes, _projection_mixed_tile_config)
        from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
        return {'tile': _projection_mixed_tile_config,
                'direct': _projection_mixed_direct_topk_routes,
                'capacity': route_pack_capacity}
    except Exception:
        for name in set(sys.modules) - modules_before:
            if name == 'b12x' or name.startswith('b12x.'):
                sys.modules.pop(name, None)
        return None
    finally:
        sys.path[:] = path_snapshot


PURE = _load_pinned_pure()


def load_harness():
    pinned = types.ModuleType('_pinned_sparkinfer')
    pinned.REVISION = 'pinned-for-tests'
    pinned.VERSION = '0.0.0'
    pinned.LOCK_DATA = {'source_tree_sha256': 'tree', 'revision': 'pinned-for-tests'}
    spec = importlib.util.spec_from_file_location(
        'bench_v41_exl3_tiles_wired', TOOLS / 'bench_v41_exl3_tiles.py')
    with patch.dict(sys.modules, {'_pinned_sparkinfer': pinned}):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


harness = load_harness()


class Tensor:
    """numpy-backed stand-in for exactly the tensor surface this tool touches."""

    def __init__(self, npv):
        self.np = np.asarray(npv)

    @property
    def shape(self):
        return self.np.shape

    def clone(self):
        return Tensor(self.np.copy())

    def contiguous(self):
        return self

    def to(self, what):
        return self

    def float(self):
        return self

    def abs(self):
        return Tensor(np.abs(self.np))

    def max(self):
        return Tensor(self.np.max())

    def norm(self):
        return Tensor(np.linalg.norm(self.np.astype(np.float64)))

    def flatten(self):
        return Tensor(self.np.reshape(-1))

    def topk(self, k):
        order = np.argsort(-np.abs(self.np).reshape(-1))[:k]
        return types.SimpleNamespace(indices=Tensor(order))

    def all(self):
        return Tensor(bool(self.np.all()))

    def item(self):
        return self.np.item()

    def tolist(self):
        return self.np.tolist()

    def numel(self):
        return int(self.np.size)

    def element_size(self):
        return int(self.np.itemsize)

    def data_ptr(self):
        return int(self.np.ctypes.data)

    def mul_(self, factor):
        self.np = (self.np * factor).astype(self.np.dtype)
        return self

    def copy_(self, src):
        self.np[...] = src.np
        return self

    def roll(self, shifts, dims):
        return Tensor(np.roll(self.np, shifts, axis=dims))

    def clamp_min(self, value):
        return Tensor(np.maximum(self.np, value))

    def reshape(self, *shape):
        return Tensor(self.np.reshape(*shape))

    def mean(self, axis=None, keepdims=False):
        return Tensor(self.np.mean(axis=axis, keepdims=keepdims))

    def sum(self, axis=None, keepdims=False):
        return Tensor(self.np.sum(axis=axis, keepdims=keepdims))

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            key = key.np
        if isinstance(key, tuple):
            key = tuple(k.np if isinstance(k, Tensor) else k for k in key)
        return Tensor(self.np[key])

    def _unwrap(self, other):
        return other.np if isinstance(other, Tensor) else other

    def __sub__(self, other):
        return Tensor(self.np - self._unwrap(other))

    def __mul__(self, other):
        return Tensor(self.np * self._unwrap(other))

    def __truediv__(self, other):
        with np.errstate(invalid='ignore', divide='ignore'):
            return Tensor(self.np / self._unwrap(other))

    def __mod__(self, other):
        return Tensor(self.np % self._unwrap(other))

    def __add__(self, other):
        return Tensor(self.np + self._unwrap(other))

    def __gt__(self, other):
        return Tensor(self.np > self._unwrap(other))

    def __bool__(self):
        return bool(np.all(self.np))

    def __float__(self):
        return float(np.asarray(self.np).reshape(-1)[0])

    def __len__(self):
        return len(self.np)


def _mod(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class FakeTorch:
    """Namespace shaped like the parts of `torch` this tool touches, plus a
    fake mixed-Trellis executor whose outputs are a deterministic,
    mutation-sensitive function of the routed inputs."""

    def __init__(self):
        self.bfloat16 = 'bfloat16'
        self.int32 = 'int32'
        self.int16 = 'int16'
        self.float32 = 'float32'
        self.__version__ = 'fake-torch'
        self.version = types.SimpleNamespace(cuda='fake')
        self.device = lambda *a, **kw: 'cuda:0'
        self.compiles = []
        self.replays = 0
        props = types.SimpleNamespace(
            name='Fake GB10', major=12, minor=1, multi_processor_count=48,
            shared_memory_per_block_optin=233472, uuid='GPU-fake-uuid',
            total_memory=96 << 30)
        torch = self

        class CUDAGraph:
            def __init__(self):
                self.fns = []

            def replay(self):
                torch.replays += 1
                for fn in self.fns:
                    fn()

        @contextmanager
        def graph(captured):
            torch.cuda._capturing = captured
            try:
                yield nullcontext()
            finally:
                torch.cuda._capturing = None

        class Event:
            def __init__(self, enable_timing=False):
                pass

            def record(self):
                pass

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 0.5  # ms; the tool scales to 5us/sample

        self.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index=0: props,
            device=lambda *a: 'cuda:0',
            synchronize=lambda: None,
            memory_stats=lambda device=None: {'allocation.all.allocated': 1234},
            CUDAGraph=CUDAGraph,
            graph=graph,
            Event=Event,
            _capturing=None)

    def manual_seed(self, seed):
        pass

    def _shape(self, shape):
        return tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)

    def randn(self, *shape, device=None, dtype=None):
        shape = self._shape(shape[0] if len(shape) == 1 and isinstance(shape[0], tuple)
                            else shape)
        return Tensor(np.random.default_rng(7).standard_normal(shape).astype(np.float32))

    def zeros(self, shape, dtype=None, device=None):
        return Tensor(np.zeros(self._shape(shape), dtype=np.float32))

    def ones(self, shape, dtype=None, device=None):
        return Tensor(np.ones(self._shape(shape), dtype=np.float32))

    def empty(self, shape, dtype=None, device=None):
        return Tensor(np.zeros(self._shape(shape), dtype=np.float32))

    def randperm(self, n, device=None):
        return Tensor(np.random.default_rng(3).permutation(n).astype(np.int64))

    def arange(self, n, device=None):
        return Tensor(np.arange(n, dtype=np.int64))

    def softmax(self, t, dim=1):
        exp = np.exp(t.np - t.np.max(axis=dim, keepdims=True))
        return Tensor(exp / exp.sum(axis=dim, keepdims=True))

    def stack(self, items):
        return Tensor(np.stack([t.np for t in items]))

    def cat(self, items, dim=0):
        return Tensor(np.concatenate([t.np for t in items], axis=dim))

    def equal(self, a, b):
        return bool(np.array_equal(a.np, b.np))

    def isfinite(self, t):
        return Tensor(np.isfinite(t.np))

    def make_b12x_modules(self, pure, expect):
        """Fake b12x namespace; `expect` describes the rank window under test
        so the fake prepare/compile assert the harness sliced the right one."""
        torch = self
        start, width, bits_for = expect['start'], expect['width'], expect['bits_for']

        @dataclasses.dataclass(frozen=True)
        class ProjectionTrellisTierWeights:
            bits: int
            w13: object
            w2: object
            gate_experts: tuple
            up_experts: tuple
            down_experts: tuple

        @dataclasses.dataclass(frozen=True)
        class PreparedTier:
            bits: int
            w13: object
            w2: object
            w2_global_scale: object

        @dataclasses.dataclass(frozen=True)
        class Prepared:
            tiers: tuple
            global_to_combined: object
            descriptor_map: object
            rotations: object
            gate_counts: tuple
            up_counts: tuple

        def members_of(bits, projection):
            return tuple(e for e in range(EXPERTS) if bits_for(e, projection) == bits)

        def prepare_projection_native_trellis_weights(native_tiers, **kw):
            assert len(native_tiers) == 2, 'the disjoint TP3 package is two-tier'
            blocks = width // 16
            for tier in native_tiers:
                bits = tier.bits
                gate, up, down = (members_of(bits, p) for p in PROJECTIONS)
                # Membership tuples must be exactly the experts of that width.
                assert tier.gate_experts == gate, (bits, 'gate membership')
                assert tier.up_experts == up, (bits, 'up membership')
                assert tier.down_experts == down, (bits, 'down membership')
                # Real planes: (members, hidden/16, width/16, 16*bits) int16,
                # sliced to THIS rank's block window, not the padded whole.
                assert tier.w13.np.shape == (len(gate) + len(up), HIDDEN // 16,
                                             blocks, 16 * bits), tier.w13.np.shape
                assert tier.w2.np.shape == (len(down), blocks, HIDDEN // 16,
                                            16 * bits), tier.w2.np.shape
                if tier.w13.numel():
                    assert tier.w13.np.dtype == np.int16
                    assert tier.w2.np.dtype == np.int16
                # Marker content proves the SLICE START: the axis holding
                # block ids must read exactly [start//16, start//16 + blocks).
                want = np.arange(start // 16, start // 16 + blocks)
                if len(gate) + len(up):
                    assert np.array_equal(tier.w13.np[0, 0, :, 0], want)
                    assert np.array_equal(tier.w13.np[-1, -1, :, -1], want)
                if len(down):
                    assert np.array_equal(tier.w2.np[0, :, 0, 0], want)
                    assert np.array_equal(tier.w2.np[-1, :, -1, -1], want)
            # Position-marker scale vectors prove per-rank slicing too:
            # intermediate_rotations is cat(w1.svh, w3.svh, w2.suh) slices.
            want_scale = np.arange(start, start + width).astype(np.float16)
            for name in ('gate_suh', 'up_suh', 'down_svh'):
                assert kw[name].np.shape == (EXPERTS, HIDDEN), name
                assert kw[name].np.dtype == np.float16
            ir = kw['intermediate_rotations']
            assert ir.np.shape == (EXPERTS, 3 * width), ir.np.shape
            assert ir.np.dtype == np.float16
            for third in (ir.np[:, :width], ir.np[:, width:2 * width],
                          ir.np[:, 2 * width:]):
                assert np.array_equal(third[0], want_scale)
                assert np.array_equal(third[-1], want_scale)
            assert kw['num_experts'] == EXPERTS and kw['hidden_size'] == HIDDEN
            assert kw['intermediate_size'] == width
            return Prepared(
                tiers=tuple(PreparedTier(t.bits, t.w13, t.w2, torch.ones(1))
                            for t in native_tiers),
                global_to_combined=torch.zeros(EXPERTS),
                descriptor_map=torch.zeros(EXPERTS * 3),
                rotations=torch.zeros(1),
                gate_counts=tuple(len(t.gate_experts) for t in native_tiers),
                up_counts=tuple(len(t.up_experts) for t in native_tiers))

        def compile_mixed_trellis(**kw):
            torch.compiles.append(kw)
            assert kw['intermediate_size'] == expect['width']
            if kw['intermediate_size'] == 768:
                # Hard contract: the disjoint TP3 race never forces residency
                # and never carries paired-boundary semantics.
                assert kw['force_blocks_per_sm'] is None, kw['force_blocks_per_sm']
                assert 'paired_boundary' not in kw
            # The route geometry must equal the production pure-function
            # decision: the real direct-topk helper and the real route-pack
            # capacity with the export tool's ceil semantics, on the block
            # the harness passes explicitly.  Every compile option the export
            # tool passes for the shipped package must be explicit here too.
            assert kw['direct_topk_routes'] == pure['direct'](
                kw['size_m'], 6, direct_exl3=True), 'direct-route decision drifted'
            assert kw['moe_block_size'] == 8, 'package block geometry drifted'
            assert kw['trellis_codebook'] == 'mcg'
            assert kw['rotation_input_dtype'] == 'bf16'
            assert kw['full_rotation_output_dtype'] == 'bf16'
            assert kw['swiglu_limit'] == 10.0
            assert kw['route_ids_dtype'] == torch.int32
            block = kw['moe_block_size']
            slots = (kw['size_m'] * 6 if kw['direct_topk_routes']
                     else pure['capacity'](kw['size_m'] * 6, block, EXPERTS, topk=6)[1])
            blocks = (slots if kw['direct_topk_routes']
                      else (slots + block - 1) // block)
            assert kw['max_m_blocks'] == blocks, (kw['max_m_blocks'], blocks)
            tile = tuple(kw['force_tile_config'])
            return types.SimpleNamespace(
                fc1_tile_k=tile[0], fc1_tile_n=tile[1], fc2_tile_k=tile[2],
                fc2_tile_n=tile[3], blocks_per_sm=1, tier0_bits=kw['tier0_bits'],
                tier1_bits=kw['tier1_bits'], sms=kw['sms'],
                shared_memory_bytes=40960, size_m=kw['size_m'])

        @dataclasses.dataclass
        class Buffers:
            rotation_gate: object
            rotation_up: object
            fc1: object
            activated: object
            fc2: object
            output: object
            packed_route_indices: object
            block_expert_ids: object
            packed_route_count: object
            expert_offsets: object
            expert_counts: object
            fc1_scratch: object
            fc2_scratch: object
            workspace: object

        def make_mixed_trellis_buffers(launch, device=None, sms=None):
            values = {field.name: torch.zeros(launch.size_m * 64)
                      for field in dataclasses.fields(Buffers)}
            values["output"] = torch.zeros((launch.size_m, HIDDEN))
            # Mirror production aliasing: fc2 IS the rotation_gate buffer.
            values["fc2"] = values["rotation_gate"]
            return Buffers(**values)

        def bind_mixed_trellis(*args, **kw):
            return types.SimpleNamespace(launch=args[-1], device='cuda:0')

        def run_bound_mixed_trellis(x, weights, ids, binding, buffers):
            def body():
                rows = int(x.shape[0])
                view = buffers.output.np[:rows]
                value = ((x.np[:rows].mean(axis=1) + ids.np[:rows].sum(axis=1) * 1e-6
                          + weights.np[:rows].sum(axis=1)) * 0.5)[:, None]
                np.copyto(view, np.broadcast_to(value, view.shape))
            capturing = torch.cuda._capturing
            if capturing is not None:
                capturing.fns.append(body)
            body()
            return Tensor(buffers.output.np[:int(x.shape[0])])

        trellis = _mod('b12x.moe.fused_moe.trellis',
                       ProjectionTrellisTierWeights=ProjectionTrellisTierWeights,
                       prepare_projection_native_trellis_weights=(
                           prepare_projection_native_trellis_weights))
        host = _mod('b12x.moe._shared.kernels.w4a16.host',
                    route_pack_capacity=pure['capacity'])
        impl = _mod('b12x.moe.fused_moe._impl',
                    _projection_mixed_tile_config=pure['tile'],
                    _projection_mixed_direct_topk_routes=pure['direct'])
        mixed = _mod('b12x.moe._shared.kernels.w4a16.mixed_trellis',
                     compile_mixed_trellis=compile_mixed_trellis,
                     make_mixed_trellis_buffers=make_mixed_trellis_buffers,
                     bind_mixed_trellis=bind_mixed_trellis,
                     run_bound_mixed_trellis=run_bound_mixed_trellis)
        chain = {}
        for dotted in ('b12x', 'b12x.moe', 'b12x.moe.fused_moe', 'b12x.moe._shared',
                       'b12x.moe._shared.kernels', 'b12x.moe._shared.kernels.w4a16'):
            chain[dotted] = _mod(dotted)
        chain['b12x.moe.fused_moe'].trellis = trellis
        chain['b12x.moe.fused_moe']._impl = impl
        chain['b12x.moe._shared.kernels.w4a16'].host = host
        chain['b12x.moe._shared.kernels.w4a16'].mixed_trellis = mixed
        chain.update({'b12x.moe.fused_moe.trellis': trellis,
                      'b12x.moe.fused_moe._impl': impl,
                      'b12x.moe._shared.kernels.w4a16.host': host,
                      'b12x.moe._shared.kernels.w4a16.mixed_trellis': mixed})
        return chain


def default_bits(expert, projection):
    return 3 if expert % 2 else 4


def fake_snapshot(root: Path, bits_for):
    """Snapshot dir + weight index (layer 30) + staged quantize_config with a
    checkpoint-wide tensor_storage over every fake layer, all consistent with
    `bits_for`."""
    weight_map = {}
    for expert in range(EXPERTS):
        for projection in PROJECTIONS:
            for suffix in ('trellis', 'suh', 'svh', 'mcg'):
                weight_map[f'layers.30.ffn.experts.{expert}.{projection}.{suffix}'] = \
                    'model-00001-of-00001.safetensors'
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
    (root / 'model-00001-of-00001.safetensors').write_bytes(b'fake shard')
    storage = {f'layers.{layer}.ffn.experts.{expert}.{projection}':
               {'bits_per_weight': bits_for(expert, projection), 'quant_format': 'exl3'}
               for layer in FAKE_LAYERS for expert in range(EXPERTS)
               for projection in PROJECTIONS}
    (root / 'quantize_config.json').write_text(json.dumps(
        {'bits': 3, 'checkpoint_format': 'exl3', 'codebook': 'mcg',
         'tensor_storage': storage}))
    return root


def fake_safe_open(bits_for=None, magic=None, layer_bits=None):
    bits_for = bits_for or default_bits

    def width_for(expert, projection):
        # `layer_bits` lets the SAMPLED layer deviate from the checkpoint-wide
        # manifest table; that mismatch must fail closed.
        return (layer_bits or bits_for)(expert, projection)

    class Reader:
        def __init__(self, path, framework=None, device=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_tensor(self, name):
            parts = name.split('.')
            expert, projection, suffix = int(parts[4]), parts[5], parts[6]
            bits = width_for(expert, projection)
            if suffix == 'mcg':
                # published int32 magic word (stored via uint32 bitcast)
                return Tensor(np.frombuffer(
                    np.uint32(0xCBAC1FED if magic is None else magic).tobytes(),
                    dtype=np.int32))
            if suffix == 'trellis':
                blocks = TOTAL // 16
                marker = np.arange(blocks, dtype=np.int16)
                if projection == 'w2':
                    # published (144, 320, 16*bits) int16, block id on axis 0
                    return Tensor(np.broadcast_to(
                        marker[:, None, None],
                        (blocks, HIDDEN // 16, 16 * bits)).copy())
                # published (320, 144, 16*bits) int16, block id on axis 1
                return Tensor(np.broadcast_to(
                    marker[None, :, None],
                    (HIDDEN // 16, blocks, 16 * bits)).copy())
            if suffix == 'suh':
                if projection == 'w2':
                    # published (2304,) float16 with position markers
                    return Tensor(np.arange(TOTAL).astype(np.float16))
                return Tensor(np.ones(HIDDEN, dtype=np.float16))
            if projection == 'w2':
                return Tensor(np.ones(HIDDEN, dtype=np.float16))
            # published (2304,) float16 with position markers
            return Tensor(np.arange(TOTAL).astype(np.float16))

    return Reader


class WiredRunCase(unittest.TestCase):
    def run_main(self, start=768, width=768, capacity=16, extra_argv=(),
                 bits_for=None, layer_bits=None, magic=None, expect_error=None):
        if PURE is None:
            self.skipTest('pinned b12x pure planner functions are unavailable')
        bits_for = bits_for or default_bits
        fake = FakeTorch()
        modules = fake.make_b12x_modules(
            PURE, {'start': start, 'width': width, 'bits_for': bits_for})
        safetensors = _mod('safetensors',
                           safe_open=fake_safe_open(bits_for, magic, layer_bits))
        pinned = types.ModuleType('_pinned_sparkinfer')
        pinned.REVISION = 'pinned-for-tests'
        pinned.VERSION = '0.0.0'
        pinned.LOCK_DATA = {'source_tree_sha256': 'tree'}
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp) / 'snap'
            snapshot.mkdir()
            fake_snapshot(snapshot, bits_for)
            out = Path(temp) / 'record.json'
            argv = ['--snapshot', str(snapshot), '--intermediate', str(width),
                    '--slice-start', str(start), '--capacity', str(capacity),
                    '--output', str(out)] + list(extra_argv)
            with patch.dict(sys.modules, {**modules, 'torch': fake,
                                          'safetensors': safetensors,
                                          '_pinned_sparkinfer': pinned}, clear=False):
                if expect_error is not None:
                    with self.assertRaises(expect_error):
                        harness.main(argv)
                    return None, fake
                harness.main(argv)
            return json.loads(out.read_text()), fake


class Tp3WiringTests(WiredRunCase):
    def test_tp3_m16_full_flow_contract(self):
        record, fake = self.run_main()
        self.assertEqual(record['layout']['mode'], 'tp3-disjoint-width768')
        self.assertEqual(record['layout']['rank'], 1)
        self.assertEqual(record['layout']['slice_start'], 768)
        self.assertEqual(record['layout']['planner_tile'], [128, 128, 128, 128])
        self.assertEqual(record['layout']['routing'], 'packed')
        self.assertEqual(record['layout']['direct_topk_routes'], False)
        slots = PURE['capacity'](16 * 6, 8, EXPERTS, topk=6)[1]
        self.assertEqual(record['layout']['route_slots'], slots)
        self.assertEqual(record['layout']['max_m_blocks'], (slots + 7) // 8)
        self.assertEqual(record['tier_family']['tiers'], [3, 4])
        self.assertEqual(record['tier_family']['resolution'],
                         'derived-from-checkpoint-trellis-widths-matching-global')
        self.assertEqual(record['tier_family']['checkpoint_global_family'], [3, 4])
        self.assertEqual(record['tier_family']['sampled_layer_family'], [3, 4])
        self.assertEqual(record['tier_family']['checkpoint_global']['projection_count'],
                         len(list(FAKE_LAYERS)) * EXPERTS * 3)
        self.assertEqual(record['tier_family']['checkpoint_global']['histogram'],
                         {'3': 192 * 3 * 41, '4': 192 * 3 * 41})
        self.assertEqual(record['non_policy'], True)
        self.assertEqual(record['promotes_default'], False)
        # Real planner at m16 resolves (128,128,128,128); the identical
        # candidate is recorded as the planner duplicate and never recompiled.
        statuses = {v['name']: v['status'] for v in record['variants']}
        self.assertEqual(statuses['baseline'], 'compiled')
        self.assertEqual(statuses['k64-n256'], 'compiled')
        self.assertEqual(statuses['k64-n128'], 'compiled')
        self.assertEqual(statuses['k128-n128'], 'duplicate-of-production-planner')
        compiled = {v['name']: v for v in record['variants'] if v['status'] == 'compiled'}
        self.assertEqual(compiled['baseline']['tile'], [128, 128, 128, 128])
        self.assertEqual(compiled['k64-n256']['tile'], [64, 256, 64, 256])
        self.assertEqual(compiled['k64-n128']['tile'], [64, 128, 64, 128])
        self.assertTrue(all(v['blocks_per_sm_forced'] is None for v in compiled.values()))
        self.assertTrue(all(v['blocks_per_sm'] == 1 for v in compiled.values()))
        self.assertTrue(all(v['tier_bits'] == [3, 4] for v in compiled.values()))
        # E2E: the priced workspace must merge the production fc2/rotation_gate
        # alias mirrored by the fake buffers into unique spans, not sum fields.
        # 14 buffer fields, 13 of them cap*64-sized — but fc2 IS rotation_gate,
        # so only 12 distinct small spans plus the (cap, HIDDEN) output.
        per_buf = 16 * 64 * 4
        out_buf = 16 * HIDDEN * 4
        self.assertEqual({v['workspace_bytes'] for v in compiled.values()},
                         {12 * per_buf + out_buf})
        # A fieldwise sum (the bug the merging prevents) would be one span more.
        self.assertTrue(all(v['workspace_bytes'] < 13 * per_buf + out_buf
                            for v in compiled.values()))
        self.assertTrue(all(k['intermediate_size'] == 768 and
                            k['force_blocks_per_sm'] is None and
                            k['direct_topk_routes'] is False
                            for k in fake.compiles))
        self.assertTrue(record['passed'])
        self.assertEqual(len(record['results']), 4)  # (1,6) (8,16) (8,30) (16,30)
        for result in record['results']:
            self.assertIsNotNone(result['timings'])
            self.assertIsNone(result['same_tile_residency_bitwise'])
            self.assertEqual(result['timing_allocations'], 0)
            self.assertEqual(result['timing_iterations'], 100)
            self.assertEqual(len(result['timing_order']), 6)
            self.assertIn('clocks_before', result)
            self.assertIn('clocks_after', result)
            self.assertEqual([entry['replay_allocations'] for entry in result['invariants']],
                             [0, 0, 0])
            self.assertTrue(all(entry['replay_deterministic'] and
                                entry['workspace_addresses_stable']
                                for entry in result['invariants']))
            self.assertTrue(all(check['passed'] for check in result['checks']))
            for entry in result['row_consistency']:
                self.assertTrue(entry['passed'], entry)
            self.assertEqual(len(result['timings']), 3)
            for timing in result['timings']:
                self.assertEqual(len(timing['samples_us']), 6)

    def test_every_rank_window_slices_its_own_sixth(self):
        for start in (0, 768, 1536):
            with self.subTest(start=start):
                record, _ = self.run_main(start=start)
                self.assertEqual(record['layout']['rank'], start // 768)
                self.assertEqual(record['layout']['slice_start'], start)
                # The per-rank dimension/marker asserts live inside the fake
                # prepare/compile: a run that reached 'passed' sliced exactly
                # [start, start+768) of the 2304-wide intermediate everywhere.
                self.assertTrue(record['passed'])

    def test_tp3_m80_baseline_is_k64n256_and_the_m16_geometry_is_raced(self):
        record, _ = self.run_main(capacity=80)
        statuses = {v['name']: v['status'] for v in record['variants']}
        self.assertEqual(statuses['k64-n256'], 'duplicate-of-production-planner')
        self.assertEqual(statuses['k128-n128'], 'compiled')
        self.assertEqual(record['layout']['planner_tile'], [64, 256, 64, 256])
        slots = PURE['capacity'](80 * 6, 8, EXPERTS, topk=6)[1]
        self.assertEqual(record['layout']['max_m_blocks'], (slots + 7) // 8)
        raced = {v['name']: v['tile'] for v in record['variants']
                 if v['status'] == 'compiled'}
        self.assertEqual(raced['baseline'], [64, 256, 64, 256])
        self.assertEqual(raced['k64-n128'], [64, 128, 64, 128])
        self.assertEqual(raced['k128-n128'], [128, 128, 128, 128])
        self.assertTrue(record['passed'])
        self.assertEqual(len(record['results']), 6)  # + (24,30) (64,30)

    def test_declared_tiers_are_confirmed_and_reach_the_compiler(self):
        record, fake = self.run_main(extra_argv=['--tiers', '3,4'])
        self.assertEqual(record['tier_family']['resolution'],
                         'declared-and-confirmed-matching-checkpoint-global')
        self.assertEqual(record['tier_family']['declared'], [3, 4])
        self.assertTrue(all(k['tier0_bits'] == 3 and k['tier1_bits'] == 4
                            for k in fake.compiles))

    def test_uniform_k4_checkpoint_forms_45_and_a_wrong_declaration_fails_closed(self):
        record, fake = self.run_main(bits_for=lambda e, p: 4)
        self.assertEqual(record['tier_family']['derived_family'], [4, 5])
        self.assertEqual(record['tier_family']['tiers'], [4, 5])
        self.assertEqual(record['tier_family']['checkpoint_global_family'], [4, 5])
        self.assertTrue(all(k['tier0_bits'] == 4 and k['tier1_bits'] == 5
                            for k in fake.compiles))
        self.assertTrue(record['passed'])
        self.run_main(extra_argv=['--tiers', '3,4'], bits_for=lambda e, p: 4,
                      expect_error=ValueError)

    def test_sampled_layer_cannot_legitimise_a_different_global_family(self):
        # Checkpoint-wide mixed [3,4], but every tensor of the SAMPLED layer
        # reads width 4 -> the sampled family is [4,5]; the run must refuse
        # before compiling anything.
        _, fake = self.run_main(bits_for=default_bits,
                                layer_bits=lambda e, p: 4, expect_error=ValueError)
        self.assertEqual(fake.compiles, [])

    def test_three_width_family_fails_closed_before_compilation(self):
        def bits(e, p):
            if e == 5 and p == 'w1':
                return 2
            return 3 if p == 'w2' else 4
        _, fake = self.run_main(bits_for=bits, expect_error=ValueError)
        self.assertEqual(fake.compiles, [])

    def test_mcg_magic_violation_fails_closed_before_compilation(self):
        _, fake = self.run_main(magic=0xDEADBEEF, expect_error=ValueError)
        self.assertEqual(fake.compiles, [])

    def test_unconfirmable_checkpoint_fails_closed(self):
        self.run_main(extra_argv=['--checkpoint', TARGET], expect_error=ValueError)

    def test_programmatic_argv_is_the_recorded_command(self):
        record, _ = self.run_main(extra_argv=['--tiers', '3,4'])
        command = record['command']
        self.assertEqual(command[0], '--snapshot')
        self.assertIn('--slice-start', command)
        self.assertIn('--tiers', command)
        # The passed argv, not the pytest process argv: pytest never sees
        # '--output' for this invocation.
        self.assertIn('--output', command)


if __name__ == '__main__':
    unittest.main()
