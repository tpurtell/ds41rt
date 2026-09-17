#!/usr/bin/env python3
"""Compare full checkpoint FP8 projections with output-channel halves on two GPUs."""
import argparse
import ctypes as C
import json
import struct
from pathlib import Path

import torch

P, I = C.c_void_p, C.c_int32


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ('abi', 'capacity', 'k', 'n')] + [
        (n, C.c_uint64) for n in ('scratch', 'values', 'row_scales', 'mma_scales', 'weight_scales')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lib = C.CDLL(str(args.native_lib))

    def bind(name, types):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, I
        def checked(*values):
            status = fn(*values)
            if status:
                raise RuntimeError(f'{name}: CUDA status {status}')
        return checked

    info = bind('ds41rt_v41_fp8_matrix_info', [I, I, I, P])
    init = bind('ds41rt_v41_fp8_matrix_initialize', [I, I, I, P])
    pack = bind('ds41rt_v41_fp8_matrix_pack_scales', [P, P, I, I, P])
    initialize = bind('ds41rt_v41_fp8_initialize_scratch', [P, P, C.c_uint64, P, P])
    launch = bind('ds41rt_v41_fp8_launch', [P, P, P, P, P, C.c_uint64, P, P, I, P])
    index = json.loads((args.snapshot / 'model.safetensors.index.json').read_text())['weight_map']

    def weight(name):
        with (args.snapshot / index[name]).open('rb') as source:
            size = struct.unpack('<Q', source.read(8))[0]
            entry = json.loads(source.read(size))[name]
            begin, end = entry['data_offsets']
            source.seek(8 + size + begin)
            return torch.frombuffer(bytearray(source.read(end - begin)), dtype=torch.uint8).reshape(entry['shape'])

    results = []
    for layer in (2, 20):
        for name, k, n in (('wq_b', 1280, 32768), ('wo_b', 8192, 5120)):
            raw = weight(f'layers.{layer}.attn.{name}.weight')
            scales = weight(f'layers.{layer}.attn.{name}.scale')
            assert raw.shape == (n, k) and scales.shape == (n // 32, k // 32)
            for capacity, rows in ((1, 1), (16, 6), (80, 64), (256, 129)):
                owners = []
                for gpu, start, width in ((0, 0, n), (0, 0, n // 2), (1, n // 2, n // 2)):
                    with torch.cuda.device(gpu):
                        stream = torch.cuda.Stream()
                        shape, kernel = Info(), P()
                        info(capacity, k, width, C.byref(shape))
                        init(capacity, k, width, C.byref(kernel))
                        w = raw[start:start + width].to(f'cuda:{gpu}')
                        ws = scales[start // 32:(start + width) // 32].to(f'cuda:{gpu}')
                        packed = torch.empty(shape.weight_scales, dtype=torch.uint8, device=gpu)
                        scratch = torch.empty(shape.scratch, dtype=torch.uint8, device=gpu)
                        alpha = torch.empty(1, dtype=torch.float32, device=gpu)
                        x = torch.empty((rows, k), dtype=torch.bfloat16, device=gpu)
                        y = torch.empty((rows, width), dtype=torch.bfloat16, device=gpu)
                        pack(ws.data_ptr(), packed.data_ptr(), k, width, stream.cuda_stream)
                        initialize(kernel, scratch.data_ptr(), scratch.numel(), alpha.data_ptr(), stream.cuda_stream)
                        stream.synchronize()
                        def run(kernel=kernel, x=x, w=w, packed=packed, scratch=scratch, alpha=alpha, y=y, stream=stream):
                            launch(kernel, x.data_ptr(), w.data_ptr(), packed.data_ptr(), scratch.data_ptr(),
                                   scratch.numel(), alpha.data_ptr(), y.data_ptr(), rows, stream.cuda_stream)
                        x.zero_(); torch.cuda.synchronize(gpu)
                        run(); stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            run()
                        owners.append((gpu, x, y, stream, graph, run, ws))
                for seed in (41, 97):
                    generator = torch.Generator().manual_seed(seed)
                    values = torch.randn((rows, k), generator=generator, dtype=torch.float32).to(torch.bfloat16)
                    outputs = []
                    for gpu, x, y, stream, graph, run, ws in owners:
                        with torch.cuda.device(gpu), torch.cuda.stream(stream):
                            x.copy_(values)
                            graph.replay()
                            stream.synchronize()
                            outputs.append(y.float().cpu())
                    actual = torch.cat(outputs[1:], dim=1)
                    expected = outputs[0]
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
                    row = dict(layer=layer, projection=name, rows=rows, capacity=capacity, seed=seed,
                               max_abs_error=(actual - expected).abs().max().item(), exact=torch.equal(actual, expected))
                    results.append(row)
                    print(row, flush=True)
                    args.output.write_text(json.dumps(dict(samples=results, passed=False), indent=2))
                del owners, outputs, actual, expected
    args.output.write_text(json.dumps(dict(samples=results, passed=True), indent=2))


if __name__ == '__main__':
    main()
