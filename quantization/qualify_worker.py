#!/usr/bin/env python3
"""Compare authenticated Spark K3/K4 searches with RTX on real V4.1 weights.

Uses a synthetic diagonal raw Hessian. This qualifies execution equivalence,
not corpus calibration or final model quality. Reports exact packed equality
and worker checkpoint reuse; any mismatch fails the qualification.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.utils.exl3_projection_checkpoint import build_projection_request
from gptqmodel.utils.exl3_remote import (
    EXL3RemoteClient, RemoteEndpoint, EXL3_HESSIAN_CAPTURE_CONTRACT,
    EXL3_HESSIAN_NUMERICAL_CONTRACT, EXL3_HESSIAN_SYMMETRY_CONTRACT,
    validate_exl3_hessian_metrics,
)
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3


@torch.inference_mode()
def run(snapshot, identity, url, token, device):
    torch.backends.cuda.matmul.allow_tf32 = False
    endpoint = RemoteEndpoint(identity["name"], url, identity["preflight_sha256"], identity["image_digest"])
    client = EXL3RemoteClient(endpoints=[endpoint], token=token, coordinator_slots=[],
                              timeout_seconds=600, max_attempts=1)
    client.qualify(endpoint)
    source = V41Source(snapshot)
    for namespace in ("layers", "mtp"):
        for projection in ("w1", "w3", "w2"):
            name = f"{namespace}.0.ffn.experts.0.{projection}"
            weight = source.decoded(name + ".weight", "cpu").T.contiguous().float()
            hessian = torch.eye(weight.shape[0], dtype=torch.float32) * 512
            for bits in (3, 4):
                started = time.monotonic()
                contract = dict(bits=bits, codebook="mcg", apply_out_scales=None, sigma_reg=.025,
                    seed=787, hessian_capture=EXL3_HESSIAN_CAPTURE_CONTRACT,
                    hessian_numerical=EXL3_HESSIAN_NUMERICAL_CONTRACT,
                    hessian_symmetry=EXL3_HESSIAN_SYMMETRY_CONTRACT,
                    execution=client.execution_contract(endpoint))
                request = build_projection_request(module_full_name=name, layer_index=0,
                    input_weight=weight, hessian=hessian, sample_count=1024,
                    quantizer_contract=contract,
                    family_join={"diagnostic": "ds41rt-real-weight-worker-qualification-v1"}, route_evidence=None)
                packed, result, transport = client.quantize(endpoint=endpoint, request_manifest=request,
                                                           input_weight=weight, hessian=hessian)
                validate_exl3_hessian_metrics(result["quantizer_metrics"], sample_count=1024, sigma_reg=.025)
                args = dict(K=bits, devices=[torch.device(device)], apply_out_scales=None,
                            sigma_reg=.025, seed=787, mcg=True)
                _, _, local = quantize_exl3(weight.to(device),
                    dict(H=hessian.to(device, copy=True), count=1024, finalized=False), args, return_weight_q=False)
                if args.get("q_fallback") is not False:
                    raise AssertionError("local reference used fallback")
                differences = [key for key in packed if key not in local
                               or not torch.equal(packed[key].cpu(), local[key].cpu())]
                if set(packed) != set(local) or differences:
                    raise AssertionError(f"packed RTX/Spark mismatch for {name} K{bits}: {differences}")
                repeated, _, replay = client.quantize(endpoint=endpoint, request_manifest=request,
                                                       input_weight=weight, hessian=hessian)
                if not replay["worker_checkpoint_hit"] or any(
                        not torch.equal(packed[key], repeated[key]) for key in packed):
                    raise AssertionError("worker checkpoint reuse differs")
                print(json.dumps(dict(event="projection_passed", name=name, bits=bits,
                    seconds=time.monotonic() - started, packed_exact=True, checkpoint_hit=True,
                    initial_transport=transport, remote_metrics=result["quantizer_metrics"],
                    local_metrics=args["error_metrics"])), flush=True)
    print(json.dumps(dict(event="qualification_passed", identity=identity, cases=12)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--identity-log", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    identity = json.loads([line for line in args.identity_log.read_text().splitlines()
                           if line.startswith("{")][-1])
    run(args.snapshot, identity, args.url, args.token_file.read_bytes().strip(), args.device)
