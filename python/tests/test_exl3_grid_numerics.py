"""Opt-in exact-AOT grid qualification; no export, JIT, or service startup.

On an idle 188-SM SM120 host, point DS41RT_EXL3_GRID_AOT at one or more
path-separated shipping variant directories (rtx-tp1, rtx-tp2, dspark; k23/k34).
Run: python3 -m unittest discover -s python/tests -p test_exl3_grid_numerics.py -v
Synthetic packed weights exercise six experts across both tiers. This compares
only grid invariance, not checkpoint accuracy or actual RTX 5090 hardware.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('DS41RT_EXL3_GRID_AOT'), 'opt-in exact-AOT GPU qualification')
class Exl3GridNumericsTests(unittest.TestCase):
    def test_export_grid_vs_170_and_changed_input_graph(self):
        import torch

        props = torch.cuda.get_device_properties(0)
        self.assertEqual((props.major, props.minor, props.multi_processor_count), (12, 0, 188))
        torch.cuda.set_device(0)
        for directory in os.environ['DS41RT_EXL3_GRID_AOT'].split(os.pathsep):
            with self.subTest(aot=directory):
                self.qualify(Path(directory), torch)

    def qualify(self, root, torch):
        meta_path = root / 'v41_exl3.json'
        meta = json.loads(meta_path.read_text())
        self.assertEqual(meta['compute'], [12, 0])
        self.assertEqual(meta['sms'], 188)
        self.assertIn(meta['bits'], ([2, 3], [3, 4]))
        self.assertIsNone(meta.get('paired_boundary'))
        self.assertIn(meta['intermediate'], (1152, 2304))
        self.assertIn(meta['experts'], (128, 384))
        torch.manual_seed(4105170)
        hidden, width, experts = meta['hidden'], meta['intermediate'], meta['experts']
        capacity, topk = meta['capacity'], meta['top_k']
        # Use every exported allocation and alias verbatim, including the
        # workspace whose counters remain at offsets 752/753 on either grid.
        buffers = {}
        for name, spec in meta['buffers'].items():
            if name == spec['allocation']:
                dtype = getattr(torch, spec['dtype'].removeprefix('torch.'))
                tensor = torch.empty(spec['shape'], dtype=dtype, device='cuda')
                self.assertEqual(tensor.numel() * tensor.element_size(), spec['bytes'])
                if spec['zero_on_create']:
                    tensor.zero_()
                buffers[name] = tensor
        for name, spec in meta['buffers'].items():
            owner = buffers[spec['allocation']]
            self.assertLessEqual(spec['bytes'], owner.numel() * owner.element_size())
            buffers[name] = owner
        original_addresses = {name: value.data_ptr() for name, value in buffers.items()}
        x = (torch.randn(capacity, hidden, device='cuda') * .01).to(torch.bfloat16)
        original_x = x.clone()
        ids = torch.arange(capacity * topk, device='cuda', dtype=torch.int32).reshape(capacity, topk) % 6
        weights = torch.softmax(torch.randn(capacity, topk, device='cuda'), dim=1)
        # Three physical experts per tier, with a full exported descriptor
        # namespace. No routed expert addresses any unpopulated storage.
        descriptor = torch.full((3, 2 * experts), -1, device='cuda', dtype=torch.int32)
        descriptor[:, :6] = torch.tensor([(e % 2 << 9) | (e // 2) for e in range(6)],
                                        device='cuda', dtype=torch.int32)
        expert_map = torch.full((experts,), -1, device='cuda', dtype=torch.int32)
        expert_map[:6] = torch.arange(6, device='cuda', dtype=torch.int32)
        rotations = torch.ones(2 * experts, 3 * width, device='cuda', dtype=torch.float16)
        gate = torch.ones(2 * experts, hidden, device='cuda', dtype=torch.float16)
        up = gate.clone()
        down = torch.full_like(gate, .01)
        lut_raw = (root / 'trellis_lut.bin').read_bytes()
        self.assertEqual(hashlib.sha256(lut_raw).hexdigest(), meta['trellis_lut']['sha256'])
        lut = torch.frombuffer(bytearray(lut_raw), dtype=torch.float16).to('cuda')
        pointers = dict(buffers)
        pointers.update(rotation_input_ptr=x, raw_topk_ids=ids, topk_weights_ptr=weights,
                        descriptor_map_ptr=descriptor, global_to_combined_ptr=expert_map,
                        intermediate_rotations_ptr=rotations, gate_suh_ptr=gate, up_suh_ptr=up,
                        svh_ptr=down, trellis_lut_ptr=lut, fc2_ptr=buffers['fc2'],
                        output_ptr=buffers['output'], route_expert_ids_ptr=ids,
                        expert_map_ptr=expert_map)
        scalars = dict(route_num_experts=experts, weight_num_experts=2 * experts)
        for tier, bits in enumerate(meta['bits']):
            plane_words = hidden * width * bits // 32
            pointers[f't{tier}_w13_ptr'] = torch.randint(-2**31, 2**31-1, (6 * plane_words,),
                                                       device='cuda', dtype=torch.int32)
            pointers[f't{tier}_w2_ptr'] = torch.randint(-2**31, 2**31-1, (3 * plane_words,),
                                                      device='cuda', dtype=torch.int32)
            for key in ('w13_scales', 'w2_scales'):
                pointers[f't{tier}_{key}_ptr'] = torch.zeros(16, device='cuda', dtype=torch.uint8)
            for key in ('w13_global', 'w2_global'):
                pointers[f't{tier}_{key}_ptr'] = torch.ones(experts, device='cuda', dtype=torch.float32)
            scalars.update({f'tier{tier}_num_experts': experts, f'tier{tier}_fc2_experts': 3,
                            f'tier{tier}_gate_experts': 3, f'tier{tier}_up_experts': 3})
        lib = ct.CDLL(str(root / 'libds41rt_exl3.so'))
        lib.ds41rt_exl3_create.argtypes = [ct.POINTER(ct.c_void_p)]
        lib.ds41rt_exl3_destroy.argtypes = [ct.c_void_p]
        for role in ('core', 'sum'):
            fn = getattr(lib, 'ds41rt_exl3_' + role)
            fn.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_int32), ct.c_void_p]
            fn.restype = ct.c_int
        context = ct.c_void_p()
        self.assertEqual(lib.ds41rt_exl3_create(ct.byref(context)), 0)
        route_lib, route_context, graph = None, ct.c_void_p(), None
        try:
            if meta['requires_route_preparation']:
                route_path = root / meta['route_preparation']['manifest']
                self.assertEqual(hashlib.sha256(route_path.read_bytes()).hexdigest(),
                                 meta['route_preparation']['sha256'])
                route_lib = ct.CDLL(str(route_path.parent / 'libv41_exl3_routes.so'))
                route_lib.ds41rt_exl3_routes_create.argtypes = [ct.POINTER(ct.c_void_p)]
                route_lib.ds41rt_exl3_routes_destroy.argtypes = [ct.c_void_p]
                route_lib.ds41rt_exl3_routes_launch.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p),
                    ct.POINTER(ct.c_uint64), ct.c_int32, ct.c_void_p]
                route_tensors = [ids, expert_map] + [buffers[name] for name in
                    ('packed_route_indices', 'block_expert_ids', 'packed_route_count', 'expert_offsets', 'expert_counts')]
                route_p = (ct.c_void_p * 7)(*[t.data_ptr() for t in route_tensors])
                route_bytes = (ct.c_uint64 * 7)(*[t.numel() * t.element_size() for t in route_tensors])
                self.assertEqual(route_lib.ds41rt_exl3_routes_create(ct.byref(route_context)), 0)

            def poison():
                # A reduced grid must produce every live value itself, not reuse
                # results from the baseline that share these exact allocations.
                # Do not poison workspace: barrier count/sense persists between
                # launches and is initialized only once, as in the daemon.
                for name in ('rotation_gate', 'rotation_up', 'fc1', 'activated',
                             'fc2', 'output', 'fc1_scratch', 'fc2_scratch'):
                    buffers[name].fill_(float('nan'))
                for name in ('packed_route_indices', 'block_expert_ids',
                             'packed_route_count', 'expert_offsets', 'expert_counts'):
                    buffers[name].fill_(-777)

            def native(rows, sms):
                scalars.update(active_m=rows, grid_x=sms * meta['blocks_per_sm'])
                stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
                if route_lib is not None:
                    # Match the daemon's live ID view, not its larger allocation.
                    route_bytes[0] = rows * topk * ids.element_size()
                    self.assertEqual(route_lib.ds41rt_exl3_routes_launch(
                        route_context, route_p, route_bytes, rows, stream), 0)
                for entry in meta['objects']:
                    p = (ct.c_void_p * len(entry['pointer_slots']))(*[pointers[n].data_ptr() for n in entry['pointer_slots']])
                    s = (ct.c_int32 * len(entry['scalar_slots']))(*[scalars[n] for n in entry['scalar_slots']])
                    role = entry['label'].rsplit('_', 1)[1]
                    self.assertEqual(getattr(lib, 'ds41rt_exl3_' + role)(context, p, s, stream), 0)

            for rows in sorted({1, min(3, capacity), max(1, capacity - 1), capacity}):
                poison()
                native(rows, 188)
                expected = buffers['output'][:rows].clone()
                poison()
                native(rows, 170)
                torch.cuda.synchronize()
                self.assertTrue(bool(torch.isfinite(expected).all()))
                self.assertTrue(bool(expected.abs().sum() > 0))
                self.assertTrue(torch.equal(expected, buffers['output'][:rows]))
            rows = min(3, capacity)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                native(rows, 170)
            for factor in (0., -.75, 1.):
                x.copy_(original_x * factor)
                ids.copy_(ids.roll(1, dims=1))
                poison()
                native(rows, 188)
                expected = buffers['output'][:rows].clone()
                poison()
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(expected, buffers['output'][:rows]))
                self.assertEqual(bool(expected.abs().sum() > 0), factor != 0.)
            self.assertEqual(original_addresses, {name: value.data_ptr() for name, value in buffers.items()})
            print(json.dumps(dict(aot=str(root), passed=True, export_sms=188, simulated_sms=170,
                bits=meta['bits'], intermediate=width, capacity=capacity, top_k=topk,
                manifest_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256((root / 'libds41rt_exl3.so').read_bytes()).hexdigest(),
                bitwise_equal=True, changed_input_graph=True, scope='synthetic exact-object grid invariance only')))
        finally:
            torch.cuda.synchronize()
            if graph is not None:
                del graph
            if route_context.value:
                self.assertEqual(route_lib.ds41rt_exl3_routes_destroy(route_context), 0)
            lib.ds41rt_exl3_destroy(context)


if __name__ == '__main__':
    unittest.main()
