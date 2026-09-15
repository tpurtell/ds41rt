"""Kernel-only comparison; identical cache bytes, layout preparation untimed.

Run with PYTHONPATH pointing at the candidate b12x checkout. This is not an
engine benchmark: upstream does not yet consume the native four-source ABI.
"""
import argparse
import ctypes as C
import json
import hashlib
import statistics
import subprocess
from pathlib import Path

import torch
import b12x
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention._shared.mla.compressed_reference import (
    pack_deepseek_v41_cache_reference as pack,
    unpack_deepseek_v41_cache_reference as unpack,
)
from b12x.preparation import PreparationSession, PreparedCall


class View(C.Structure):
    _fields_ = [("values", C.c_void_p * 4), ("scales", C.c_void_p * 4),
                ("window_end", C.c_void_p), ("pages", C.c_void_p),
                ("source_end", C.c_void_p), ("window_capacity", C.c_uint64),
                ("source_capacity", C.c_uint64), ("private_capacity", C.c_uint64),
                ("page_stride", C.c_uint32), ("compressed", C.c_uint32)]


@torch.inference_mode()
def compare(lib, rows, candidate=None):
    torch.manual_seed(4100 + rows)
    q = (torch.randn(rows, 64, 512, device="cuda") * .2).bfloat16()
    sink = torch.randn(64, device="cuda")
    swa = pack(torch.randn(128 + rows, 512, device="cuda").bfloat16(),
               page_size=64, cache_kind="swa")
    indexed = pack(torch.randn(512, 512, device="cuda").bfloat16(),
                   page_size=256, cache_kind="indexed")
    swar, ixr = swa.view(-1, 528), indexed.view(-1, 288)
    values = [swar[:128, :512].contiguous(), swar[128:128+rows, :512].contiguous(),
              ixr[:, :256].contiguous(), torch.zeros(1, 256, dtype=torch.uint8, device="cuda")]
    scales = [swar[:128, 512:].contiguous(), swar[128:128+rows, 512:].contiguous(),
              ixr[:, 256:].contiguous(), torch.zeros(1, 32, dtype=torch.uint8, device="cuda")]
    end = torch.tensor([128], dtype=torch.uint64, device="cuda")
    source_end = torch.tensor([512], dtype=torch.uint64, device="cuda")
    pages = torch.tensor([0, 1], dtype=torch.uint32, device="cuda")
    metadata = torch.tensor([[128, 0, rows, 128+r, 0, 512, 512, 0, 0, 1]
                             for r in range(rows)], dtype=torch.uint64, device="cuda")
    selected = torch.arange(512, dtype=torch.int32, device="cuda").repeat(rows, 1)
    swa_indices = (torch.arange(128, dtype=torch.int32, device="cuda")[None]
                   + torch.arange(rows, dtype=torch.int32, device="cuda")[:, None] + 1)
    lens = torch.full((rows,), 128, dtype=torch.int32, device="cuda")
    ilens = torch.full((rows,), 512, dtype=torch.int32, device="cuda")
    table = pages.view(torch.int32).repeat(rows, 1)
    outputs = [torch.empty_like(q), torch.empty_like(q)]
    scratch = torch.empty(rows, 10, 64, 514, device="cuda")
    view = View((C.c_void_p*4)(*[v.data_ptr() for v in values]),
                (C.c_void_p*4)(*[s.data_ptr() for s in scales]), end.data_ptr(),
                pages.data_ptr(), source_end.data_ptr(), rows, 512, 1, 2, 2)
    fn = lib.ds41rt_v41_sparse_attention_split
    fn.argtypes = [C.c_void_p]*5 + [C.c_int32, C.c_int32, C.POINTER(View),
                                  C.c_void_p, C.c_void_p, C.c_uint64, C.c_int32]

    def native():
        rc = fn(q.data_ptr(), sink.data_ptr(), metadata.data_ptr(), selected.data_ptr(),
                outputs[0].data_ptr(), rows, 128, C.byref(view),
                torch.cuda.current_stream().cuda_stream, scratch.data_ptr(),
                scratch.numel()*4, 10)
        assert rc == 0, rc

    with PreparationSession(device=q.device, autotune=False, compile_workers=2) as session:
        caps = mla.Caps(device=q.device, num_q_heads=64, max_q_rows=rows,
                        max_width=640, swa_width=128, indexed_width=512,
                        max_page_table_width=2, swa_page_size=64, indexed_page_size=256,
                        cache_format="deepseek_v41", mode="decode", max_chunks_per_row=4,
                        use_cuda_graph=True)
        invocation = mla.invocation_from_tensors(q=q, swa_k_cache=swa,
            indexed_k_cache=indexed, attn_sink=sink, out=outputs[1])
        plan = mla.plan(caps, invocation=invocation)
        bind_args = dict(q=q, swa_indices=swa_indices, swa_lengths=lens,
                         indexed_indices=selected, indexed_lengths=ilens, indexed_page_table=table)
        run_args = dict(swa_k_cache=swa, indexed_k_cache=indexed, attn_sink=sink,
                        sm_scale=512**-.5, out=outputs[1])

        def prepare(state):
            (spec,) = state.scratch_specs()
            storage = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
            binding = state.bind_for_preparation(scratch=storage, **bind_args)
            return PreparedCall(run=lambda: state.run(binding, **run_args),
                                output=outputs[1], owners=(storage, binding))

        session.prepare((plan.request(name="comparison", prepare_call=prepare),))
        (spec,) = plan.scratch_specs()
        storage = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
        binding = mla.bind(plan, scratch=storage, **bind_args)

        def upstream():
            return mla.run(binding=binding, **run_args)

        native()
        upstream()
        if candidate is not None:
            descriptor_words = list((C.c_uint64 * 15).from_buffer_copy(view))
            descriptors = torch.tensor([descriptor_words]*rows, dtype=torch.uint64, device="cuda")
            bounds = torch.zeros(rows, dtype=torch.uint64, device="cuda")
            partials = torch.empty(rows, 64, 10, 512, dtype=torch.bfloat16, device="cuda")
            lses = torch.empty(rows, 64, 10, dtype=torch.float32, device="cuda")
            upstream_output = outputs[1].clone()
            candidate_fn = candidate.ds41rt_attention_probe
            candidate_fn.argtypes = [C.c_void_p]*9 + [C.c_int32, C.c_void_p]

            def upstream():
                rc = candidate_fn(q.data_ptr(), descriptors.data_ptr(), metadata.data_ptr(),
                    selected.data_ptr(), bounds.data_ptr(), sink.data_ptr(), partials.data_ptr(),
                    lses.data_ptr(), outputs[1].data_ptr(), rows, torch.cuda.current_stream().cuda_stream)
                assert rc == 0, rc

            upstream()
            torch.testing.assert_close(outputs[1], upstream_output, atol=.008, rtol=.008)
        torch.cuda.synchronize()
        keys = torch.cat((unpack(swa, page_size=64, cache_kind="swa")[swa_indices.long()],
                          unpack(indexed, page_size=256, cache_kind="indexed").expand(rows, -1, -1)), dim=1)
        logits = torch.einsum("rhd,rkd->rhk", q.float(), keys) * 512**-.5
        probs = torch.cat((logits, sink[None, :, None].expand(rows, -1, -1)), dim=-1).softmax(-1)
        reference = torch.einsum("rhk,rkd->rhd", probs[:, :, :-1], keys)
        errors = []
        for output in outputs:
            assert torch.isfinite(output).all()
            delta = output.float() - reference
            errors.append(dict(max_abs=delta.abs().max().item(), rms=delta.square().mean().sqrt().item()))
        torch.testing.assert_close(outputs[0].float(), reference, atol=.003, rtol=.02)
        # Report upstream numerical differences; real-model qualification is separate.
        session.freeze()
        graphs = []
        for launch in (native, upstream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
                    launch()
            graphs.append(graph)
        if candidate is not None:
            # Exercise actual C ABI graph replay with changed query and invalid
            # request metadata, without resolving or loading any new kernel.
            original_q = q.clone()
            q.mul_(.7)
            for graph in graphs:
                graph.replay()
            changed_logits = torch.einsum("rhd,rkd->rhk", q.float(), keys) * 512**-.5
            changed_probs = torch.cat((changed_logits, sink[None, :, None].expand(rows, -1, -1)), dim=-1).softmax(-1)
            changed_reference = torch.einsum("rhk,rkd->rhd", changed_probs[:, :, :-1], keys)
            torch.testing.assert_close(outputs[1].float(), changed_reference, atol=.008, rtol=.008)
            metadata[0, 0] = 127
            for graph in graphs:
                graph.replay()
            for output in outputs:
                assert torch.equal(output[0], torch.zeros_like(output[0]))
            metadata[0, 0] = 128
            q.copy_(original_q)
        samples = [[], []]
        for repeat in range(5):
            for i in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                for _ in range(5):
                    graphs[i].replay()
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graphs[i].replay()
                stop.record()
                stop.synchronize()
                samples[i].append(start.elapsed_time(stop)*1000/(50*20))
        return dict(rows=rows, native_parts=10, mode="decode", errors=errors,
                    median_us=[statistics.median(x) for x in samples], samples_us=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", required=True)
    parser.add_argument("--candidate-library", help="Qualify/timing the native AOT producer+merge instead of the interleaved kernel")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 2, 7, 16, 32])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(args.native_library)
    assert lib.ds41rt_v41_sparse_attention_initialize() == 0
    candidate = C.CDLL(args.candidate_library) if args.candidate_library else None
    if candidate is not None:
        manifest_path = Path(args.candidate_library).parent / "v41_attention.json"
        manifest = json.loads(manifest_path.read_text())
        for name, expected_hash in manifest["artifacts"].items():
            assert hashlib.sha256((manifest_path.parent/name).read_bytes()).hexdigest() == expected_hash, name
        candidate.ds41rt_attention_probe_initialize()
    source = Path(b12x.__file__).resolve().parent.parent
    provenance = dict(
        candidate_revision=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        native_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        native_library_sha256=hashlib.sha256(Path(args.native_library).read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        gpu_state=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,power.limit,clocks.mem", "--format=csv"], text=True),
        launches_per_graph=20, replays_per_sample=50, repetitions=5,
        candidate_library_sha256=(hashlib.sha256(Path(args.candidate_library).read_bytes()).hexdigest()
                                  if args.candidate_library else None),
        candidate_manifest_sha256=(hashlib.sha256(manifest_path.read_bytes()).hexdigest() if candidate else None),
    )
    if candidate:
        assert manifest["sparkinfer_revision"] == provenance["candidate_revision"]
    results = []
    for rows in args.rows:
        result = compare(lib, rows, candidate)
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(dict(scope=("native direct producer+merge; no repacking; baseline then candidate" if candidate else "kernel-only; repacking excluded; native then upstream"),
                                              provenance=provenance, results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
