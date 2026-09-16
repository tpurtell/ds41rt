#!/usr/bin/env python3
"""Check packed PLE CUDA decoding against an independent CPU bit-level oracle."""
import argparse
import ctypes
import json
from pathlib import Path

import torch


def reference(weights, scales, global_scale):
    packed = weights.cpu().to(torch.int32)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    raw = scales.cpu().to(torch.int32)
    exponent, mantissa = (raw >> 3) & 15, raw & 7
    scale = torch.where(exponent == 0, mantissa.float() * 2**-9,
                        (1 + mantissa.float() / 8) * torch.pow(2., exponent - 7))
    scale = torch.where((raw & 128) != 0, -scale, scale)
    return ((lut[codes] * scale.repeat_interleave(16, dim=1)) * global_scale).bfloat16()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    parser.add_argument("--fixture-dir", type=Path)
    args = parser.parse_args()
    lib = ctypes.CDLL(args.native_lib)
    fn = lib.ds41rt_cuda_engram_nvfp4_dequant_bf16_async
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    results = []
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream().cuda_stream
            graph_stream = torch.cuda.Stream(device=device)
            for rows in (1, 24, 384, 4096):
                weights = (torch.arange(rows * 128) % 256).byte().reshape(rows, 128).cuda()
                # All finite positive E4M3 encodings, including zero/subnormals.
                scales = (torch.arange(rows * 16) % 127).byte().reshape(rows, 16).cuda()
                storage = torch.full((rows * 256 + 32,), 123., dtype=torch.bfloat16, device="cuda")
                out = storage[16:-16].view(rows, 256)
                for global_scale in (.000123, .125, 1., 13.75):
                    def launch():
                        assert fn(weights.data_ptr(), scales.data_ptr(), global_scale,
                                  out.data_ptr(), rows, stream) == 0
                    launch()
                    torch.cuda.synchronize()
                    expected = reference(weights, scales, ctypes.c_float(global_scale).value)
                    actual_bits, expected_bits = out.cpu().view(torch.int16), expected.view(torch.int16)
                    mismatch = actual_bits != expected_bits
                    assert not mismatch.any(), (device, rows, global_scale, mismatch.sum().item(), actual_bits[mismatch][:16].tolist(), expected_bits[mismatch][:16].tolist())
                    assert (storage[:16] == 123).all() and (storage[-16:] == 123).all()
                    # Capture, then mutate inputs before replay to catch stale capture data.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_stream):
                        capture_stream = torch.cuda.current_stream().cuda_stream
                        assert fn(weights.data_ptr(), scales.data_ptr(), global_scale,
                                  out.data_ptr(), rows, capture_stream) == 0
                    weights.bitwise_xor_(255)
                    scales.copy_(scales.roll(1, dims=1))
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = reference(weights, scales, ctypes.c_float(global_scale).value)
                    actual_bits, expected_bits = out.cpu().view(torch.int16), expected.view(torch.int16)
                    mismatch = actual_bits != expected_bits
                    assert not mismatch.any(), (device, rows, global_scale, mismatch.sum().item(), actual_bits[mismatch][:16].tolist(), expected_bits[mismatch][:16].tolist())
                for invalid in (0., -1., float("nan"), float("inf")):
                    assert fn(weights.data_ptr(), scales.data_ptr(), invalid,
                              out.data_ptr(), rows, stream) != 0
                assert fn(None, scales.data_ptr(), 1., out.data_ptr(), rows, stream) != 0
                assert fn(weights.data_ptr(), scales.data_ptr(), 1., out.data_ptr(), 0, stream) != 0
                results.append(dict(device=device, rows=rows, global_scales=4,
                                    bitwise_equal=True, graph_changed_inputs=True,
                                    output_guards=True, invalid_arguments_rejected=True))
            if args.fixture_dir:
                for fixture in json.loads((args.fixture_dir / "manifest.json").read_text()):
                    layer, rows = fixture["layer"], fixture["hash_rows"]
                    assert fixture["encoding"] == "Nvfp4"
                    def read(suffix, width):
                        data = bytearray((args.fixture_dir / f"layer{layer}-{suffix}.bin").read_bytes())
                        return torch.frombuffer(data, dtype=torch.uint8).reshape(rows, width).cuda()
                    weights, scales = read("weights", 128), read("scales", 16)
                    out = torch.empty((rows, 256), dtype=torch.bfloat16, device="cuda")
                    assert fn(weights.data_ptr(), scales.data_ptr(), fixture["global_scale"],
                              out.data_ptr(), rows, stream) == 0
                    torch.cuda.synchronize()
                    expected = reference(weights, scales, fixture["global_scale"])
                    assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16))
                    (args.fixture_dir / f"layer{layer}-expected.bin").write_bytes(
                        expected.view(torch.uint8).numpy().tobytes())
                    results.append(dict(device=device, checkpoint_layer=layer, rows=rows,
                                        bitwise_equal=True))
    assert results, "No CUDA devices were tested"
    print(json.dumps(dict(cases=results), indent=2))


if __name__ == "__main__":
    main()
