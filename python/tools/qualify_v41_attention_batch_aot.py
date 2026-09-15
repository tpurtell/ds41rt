"""Mixed-request native batch ABI checks with an independent closed-form oracle."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import torch
from compare_v41_upstream_attention import View


@torch.inference_mode()
def run(libs, requests, per_request):
    rows = requests*per_request
    q = torch.zeros(rows, 64, 512, dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(64, device="cuda")
    outputs = [torch.empty_like(q), torch.empty_like(q)]
    scratch = torch.empty(rows, 10, 64, 514, device="cuda")
    metadata, bounds, views, owners, expected_settings = [], [], [], [], []
    all_pages = []
    for request in range(requests):
        factor = (.5, 1., 2., .25)[request % 4]
        window_value = (1., .5, 2.)[request % 3]
        values = [torch.zeros(128, 512, dtype=torch.uint8, device="cuda"),
                  torch.full((per_request, 512), window_value, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8),
                  torch.full((512, 256), 0x22, dtype=torch.uint8, device="cuda"),
                  torch.full((4, 256), 0x55, dtype=torch.uint8, device="cuda")]
        source_scales = torch.full((512, 32), factor, device="cuda")
        source_scales[256:].mul_(2)
        scales = [torch.full((128, 16), 127, dtype=torch.uint8, device="cuda"),
                  torch.full((per_request, 16), 127, dtype=torch.uint8, device="cuda"),
                  source_scales.to(torch.float8_e4m3fn).view(torch.uint8),
                  torch.full((4, 32), factor, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)]
        end = torch.tensor([128], dtype=torch.uint64, device="cuda")
        source_end = torch.tensor([512], dtype=torch.uint64, device="cuda")
        pages = torch.tensor([1, 0], dtype=torch.uint32, device="cuda")
        all_pages.append(pages)
        view = View((C.c_void_p*4)(*[x.data_ptr() for x in values]),
                    (C.c_void_p*4)(*[x.data_ptr() for x in scales]), end.data_ptr(),
                    pages.data_ptr(), source_end.data_ptr(), per_request, 512, 4, 2, 2)
        owners.extend(values+scales+[end, source_end, pages])
        for local in range(per_request):
            views.append(view)
            metadata.append([128, 0, per_request, 128+local, 0, 514, 512, 2, 0, 2])
            bounds.append(128 if request % 2 else 0)
            expected_settings.append((factor, window_value, local+1, local+1 if request % 2 else 128))
    host_views = (View*rows)(*views)
    desc = torch.tensor([list((C.c_uint64*15).from_buffer_copy(v)) for v in views], dtype=torch.uint64, device="cuda")
    meta = torch.tensor(metadata, dtype=torch.uint64, device="cuda")
    begin = torch.tensor(bounds, dtype=torch.uint64, device="cuda")
    selected = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
    selected[:, :5] = torch.tensor([0, 255, 256, 512, 513], device="cuda")
    calls = []
    for i, lib in enumerate(libs):
        validate = lib.ds41rt_v41_sparse_attention_batch_validate
        validate.argtypes = [C.c_void_p]*5 + [C.c_int32] + [C.c_void_p]*4 + [C.c_uint64, C.c_int32, C.c_int32]
        status = validate(q.data_ptr(), sink.data_ptr(), meta.data_ptr(), selected.data_ptr(), outputs[i].data_ptr(),
            rows, C.cast(host_views, C.c_void_p), desc.data_ptr(), begin.data_ptr(), scratch.data_ptr(), scratch.numel()*4, 10, 2)
        assert status == 0, status
        fn = lib.ds41rt_v41_sparse_attention_batch if i == 0 else lib.ds41rt_v41_sparse_attention_batch_aot
        fn.argtypes = [C.c_void_p]*5 + [C.c_int32] + [C.c_void_p]*4 + [C.c_int32, C.c_int32]

        def launch(fn=fn, output=outputs[i]):
            status = fn(q.data_ptr(), sink.data_ptr(), meta.data_ptr(), selected.data_ptr(), output.data_ptr(),
                rows, desc.data_ptr(), torch.cuda.current_stream().cuda_stream, begin.data_ptr(), scratch.data_ptr(), 10, 2)
            assert status == 0, status
        calls.append(launch)

    def check(recycled=False, invalid=False, first_masked=False):
        coefficient = (10 if recycled else 11) - ((1 if recycled else 2) if first_masked else 0)
        expected = torch.tensor([(n*w+coefficient*f)/(windows+6-int(first_masked))
                                 for f,w,n,windows in expected_settings], device="cuda")
        if invalid:
            expected[:per_request] = 0
        expected = expected[:, None, None].expand_as(q).bfloat16()
        torch.testing.assert_close(outputs[0], expected, atol=.004, rtol=.004)
        torch.testing.assert_close(outputs[1], expected, atol=.008, rtol=.008)
        if invalid:
            assert torch.equal(outputs[1][:per_request], torch.zeros_like(outputs[1][:per_request]))

    for launch in calls:
        launch()
    check()
    graphs = []
    for launch in calls:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                launch()
        graphs.append(graph)
    for page in all_pages:
        page.copy_(torch.tensor([0, 1], dtype=torch.uint32, device="cuda"))
    meta[:per_request, 0] = 127
    for graph in graphs:
        graph.replay()
    check(recycled=True, invalid=True)
    # Restore valid workload before timing; no new capture or allocation.
    meta[:per_request, 0] = 128
    for graph in graphs:
        graph.replay()
    check(recycled=True)
    # A missing first key must not discard a tile containing later live keys.
    selected[:, 0] = -1
    for graph in graphs:
        graph.replay()
    check(recycled=True, first_masked=True)
    selected[:, 0] = 0
    samples = [[], []]
    for repeat in range(5):
        for i in ([0, 1] if repeat % 2 == 0 else [1, 0]):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graphs[i].replay()
            stop.record()
            stop.synchronize()
            samples[i].append(start.elapsed_time(stop)*1000/400)
    return dict(requests=requests, rows_per_request=per_request, rows=rows,
                median_us=[statistics.median(x) for x in samples], samples_us=samples,
                checks="closed-form FP4/FP8 values, private proposals, mixed bounds, recycled pages, malformed request, missing first key and graph replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    libs = [C.CDLL(args.baseline), C.CDLL(args.candidate)]
    for lib in libs:
        assert lib.ds41rt_v41_sparse_attention_initialize() == 0
    results = []
    provenance = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        main_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        gpu_state=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,power.limit,clocks.mem", "--format=csv"], text=True),
        launches_per_graph=20, replays_per_sample=20, repetitions=5)
    for requests, per_request in ((2, 1), (2, 4), (16, 1), (16, 4)):
        result = run(libs, requests, per_request)
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(dict(scope="synthetic mixed-request batch attention only",
            provenance=provenance,
            library_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.baseline, args.candidate)],
            results=results), indent=2)+"\n")


if __name__ == "__main__":
    main()
