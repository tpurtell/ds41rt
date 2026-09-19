#!/usr/bin/env python3
"""Byte-exact GPU gate for NVFP4 load-time padding; pass native library path."""
import ctypes as C
import sys
import torch


class Buffer(C.Structure):
    _fields_ = [("ptr", C.c_void_p), ("bytes", C.c_size_t),
                ("device_id", C.c_int), ("flags", C.c_uint64)]


def buffer(tensor):
    return Buffer(tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor.device.index, 0)


def swizzle(plain):
    rows, cols = plain.shape
    row = torch.arange(rows, device=plain.device)[:, None]
    col = torch.arange(cols, device=plain.device)[None, :]
    offset = (row // 128) * (cols * 128) + (col // 4) * 512
    offset = offset + (row % 32) * 16 + ((row // 32) % 4) * 4 + col % 4
    out = torch.empty_like(plain).flatten()
    out[offset.flatten()] = plain.flatten()
    return out


def main():
    library = C.CDLL(sys.argv[1])
    fn = library.ds41rt_cuda_nvfp4_pad_expert_async
    fn.argtypes = [C.POINTER(Buffer), C.POINTER(Buffer), C.c_size_t, C.c_size_t, C.c_void_p]
    fn.restype = C.c_int
    torch.manual_seed(72)
    stream = torch.cuda.current_stream().cuda_stream
    for source_n, kernel_n in [(64, 128), (576, 640), (1152, 1152)]:
        src = [torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
               for shape in [(2*source_n, 2560), (2*source_n, 320),
                             (5120, source_n//2), (5120, source_n//16)]]
        original = [t.clone() for t in src]
        expected = []
        for index, tensor in enumerate(src):
            if index < 2:
                padded = torch.zeros(2*kernel_n, tensor.shape[1], dtype=torch.uint8, device="cuda")
                padded[:source_n] = tensor[:source_n]
                padded[kernel_n:kernel_n+source_n] = tensor[source_n:]
            else:
                padded = torch.zeros(5120, kernel_n//(2 if index == 2 else 16), dtype=torch.uint8, device="cuda")
                padded[:, :tensor.shape[1]] = tensor
            expected.append(swizzle(padded) if index % 2 else padded.flatten())
        dst = [torch.full((t.numel()+64,), 0xA5, device="cuda", dtype=torch.uint8) for t in expected]
        sources = (Buffer*4)(*(buffer(t) for t in src))
        destinations = (Buffer*4)(*(buffer(t[:-64]) for t in dst))
        assert fn(sources, destinations, source_n, kernel_n, stream) == 0
        torch.cuda.synchronize()
        for actual, want, source, before in zip(dst, expected, src, original):
            assert torch.equal(actual[:-64], want)
            assert bool((actual[-64:] == 0xA5).all()), "destination overrun"
            assert torch.equal(source, before), "source modified"
        # Validate every plane before launching any write.
        for index in range(4):
            for arr in (sources, destinations):
                saved = arr[index].bytes
                arr[index].bytes = saved - 1
                assert fn(sources, destinations, source_n, kernel_n, stream) != 0
                arr[index].bytes = saved
        for invalid in [(0, kernel_n), (source_n+1, kernel_n), (source_n, 8193)]:
            assert fn(sources, destinations, *invalid, stream) != 0
        print(f"PASS: {source_n}->{kernel_n}, four planes exact; bounds and immutability", flush=True)


if __name__ == "__main__":
    main()
