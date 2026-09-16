"""Two-GPU shard/full projection check; run with Python containing CUDA PyTorch.

Usage: python native/tests/v41_vocabulary_shard_selftest.py /path/to/native.so
Uses synthetic BF16 weights and independent per-rank streams/workspaces.
"""
import ctypes as C
import sys

import torch


def main():
    lib = C.CDLL(sys.argv[1])
    ptr = C.c_void_p
    create = lib.ds41rt_v41_vocabulary_shard_create
    create.argtypes = [ptr, C.c_uint64, C.c_int32, C.POINTER(ptr)]
    full_create = lib.ds41rt_v41_vocabulary_head_create
    full_create.argtypes = [ptr, C.c_uint64, C.POINTER(ptr)]
    launch = lib.ds41rt_v41_vocabulary_head_launch
    launch.argtypes = [ptr, ptr, ptr, ptr, C.c_int32, ptr]
    destroy = lib.ds41rt_v41_markov_destroy
    destroy.argtypes = [ptr]
    merge = lib.ds41rt_v41_vocabulary_merge_greedy
    merge.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, C.c_int32, C.c_int32, ptr]
    assert torch.cuda.device_count() >= 2, 'two CUDA GPUs required'
    torch.manual_seed(41)
    vocab, width, split = 129280, 5120, 64640
    weight = torch.empty((vocab, width), device='cuda:0', dtype=torch.bfloat16).normal_(std=0.02)
    inputs = torch.empty((80, width), device='cuda:0', dtype=torch.bfloat16).normal_(std=0.2)
    weights = [weight[:split], weight[split:].to('cuda:1'), weight]
    xs = [inputs, inputs.to('cuda:1'), inputs]
    sizes = [split, vocab - split, vocab]
    devices = [0, 1, 0]
    streams, workspaces, outputs, handles = [], [], [], []
    for device, size in zip(devices, sizes):
        with torch.cuda.device(device):
            streams.append(torch.cuda.Stream())
            workspaces.append(torch.empty(4 * 1024 * 1024, device=device, dtype=torch.uint8))
            outputs.append(torch.empty((80, size), device=device, dtype=torch.float32))
            handle = ptr()
            if size == vocab:
                status = full_create(workspaces[-1].data_ptr(), workspaces[-1].numel(), C.byref(handle))
            else:
                status = create(workspaces[-1].data_ptr(), workspaces[-1].numel(), size, C.byref(handle))
            assert status == 0 and handle.value, status
            handles.append(handle)
    for device in (0, 1):
        torch.cuda.synchronize(device)

    def run(rank, rows):
        with torch.cuda.device(devices[rank]):
            status = launch(handles[rank], xs[rank].data_ptr(), weights[rank].data_ptr(),
                            outputs[rank].data_ptr(), rows, streams[rank].cuda_stream)
            assert status == 0, status

    def gpu_merge(ids0, scores0, ids1, scores1, capture=False):
        with torch.cuda.device(0):
            ids0, ids1 = [x.to(device='cuda:0', dtype=torch.int32) for x in (ids0, ids1)]
            scores0, scores1 = [x.to(device='cuda:0', dtype=torch.float32) for x in (scores0, scores1)]
            out_ids = torch.empty_like(ids0)
            out_scores = torch.empty_like(scores0)
            for device in (0, 1):
                torch.cuda.synchronize(device)
            args = [x.data_ptr() for x in (ids0, scores0, ids1, scores1, out_ids, out_scores)]
            args += [ids0.numel(), split, streams[0].cuda_stream]
            assert merge(*args) == 0
            streams[0].synchronize()
            if capture:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=streams[0]):
                    assert merge(*args) == 0
                graph.replay()
                streams[0].synchronize()
                graph.reset()
            # Outputs may not alias an input or each other.
            bad = args.copy()
            bad[4] = ids0.data_ptr()
            assert merge(*bad) != 0
            return out_ids.cpu(), out_scores.cpu()

    def verify(rows):
        for stream in streams:
            stream.synchronize()
        full = outputs[2][:rows].cpu()
        parts = [o[:rows].cpu() for o in outputs[:2]]
        combined = torch.cat(parts, dim=1)
        torch.testing.assert_close(combined, full, atol=2e-5, rtol=2e-5)
        scores = torch.stack([p.max(dim=1).values for p in parts], dim=1)
        ids = torch.stack([parts[0].argmax(dim=1), parts[1].argmax(dim=1) + split], dim=1)
        winners = ids.gather(1, scores.argmax(dim=1)[:, None]).squeeze(1)
        assert torch.equal(winners, full.argmax(dim=1))
        local = [o[:rows].max(dim=1) for o in outputs[:2]]
        gpu_ids, gpu_scores = gpu_merge(local[0].indices, local[0].values,
                                       local[1].indices, local[1].values, capture=rows == 3)
        assert torch.equal(gpu_ids.long(), winners)
        assert torch.equal(gpu_scores, full.max(dim=1).values)
        # Independent FP32 reference on columns spanning the shard boundary.
        columns = [0, split - 1, split, vocab - 1]
        reference = xs[0][:rows].float().cpu() @ weight[columns].float().cpu().T
        torch.testing.assert_close(full[:, columns], reference, atol=2e-5, rtol=2e-5)
        print(f'PASS rows={rows} shard/full max_error={(combined-full).abs().max().item():.8g} global greedy exact', flush=True)

    try:
        for rows in (1, 3, 16, 80):
            for rank in range(3):
                run(rank, rows)
            verify(rows)
        graphs = []
        for rank in range(3):
            with torch.cuda.device(devices[rank]):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=streams[rank]):
                    run(rank, 3)
                graphs.append(graph)
        for _ in range(3):
            for rank, graph in enumerate(graphs):
                with torch.cuda.device(devices[rank]):
                    graph.replay()
            verify(3)
        for graph in graphs:
            graph.reset()
        with torch.cuda.device(0):
            for count in (0, -1, vocab + 1):
                bad = ptr(1)
                assert create(workspaces[0].data_ptr(), workspaces[0].numel(), count, C.byref(bad)) != 0
                assert not bad.value
            # Head handles support up to 128 rows (including K7 verification).
            assert launch(handles[0], xs[0].data_ptr(), weights[0].data_ptr(), outputs[0].data_ptr(), 129, streams[0].cuda_stream) != 0
            assert launch(handles[1], xs[0].data_ptr(), weights[0].data_ptr(), outputs[0].data_ptr(), 1, streams[0].cuda_stream) != 0
        edge_ids, edge_scores = gpu_merge(
            torch.tensor([0, split - 1, 1, -1, 0, 0, 0]),
            torch.tensor([1., 2., -5., 1., 1., float('nan'), 1.]),
            torch.tensor([0, 0, split - 1, 0, vocab - split, 0, 0]),
            torch.tensor([2., 2., -4., 2., 2., 2., float('inf')]), capture=True)
        assert edge_ids.tolist() == [split, split - 1, vocab - 1, -1, -1, -1, -1]
        assert torch.isnan(edge_scores[3:]).all()
        print('PASS GPU merge ties, boundary offsets, invalid candidates, and graph replay', flush=True)
        print('PASS graph replay, invalid shard/row bounds, and wrong-device rejection', flush=True)
    finally:
        for rank, handle in enumerate(handles):
            with torch.cuda.device(devices[rank]):
                streams[rank].synchronize()
                assert destroy(handle) == 0


if __name__ == '__main__':
    main()
