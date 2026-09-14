#!/usr/bin/env python3
"""Exercise all six real search devices and durable assignment reload.

Uses real source weights with synthetic raw Hessians, never production capture.
A diagnostic barrier holds ten acquired staging leases before starting search,
ensuring both RTX devices and both staging positions on every Spark are covered.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

import torch
import gptqmodel
from gptqmodel.utils.exl3_remote import EXL3RemoteClient, RemoteEndpoint, CoordinatorSlot
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from distributed_search import DistributedSearch


class DiagnosticClient(EXL3RemoteClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.barrier = threading.Barrier(10, timeout=120)

    def acquire_slot(self, key):
        lease = super().acquire_slot(key)
        try:
            self.barrier.wait()
        except BaseException:
            lease.release()
            raise
        return lease


def coordinator_identity(image_digest):
    root = Path(gptqmodel.__file__).resolve().parent
    digest = hashlib.sha256()
    for parent in (root, root.parent / "gptqmodel_ext", Path(__file__).resolve().parent):
        for path in sorted(parent.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cu", ".cpp", ".h", ".cuh"}:
                digest.update(str(path.relative_to(parent)).encode() + b"\0")
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    report = dict(scope="development-container-distributed-search-diagnostic", image_digest=image_digest,
                  source_sha256=digest.hexdigest(), python=sys.version,
                  versions={name: importlib.metadata.version(name) for name in ("torch", "triton", "safetensors")},
                  gpus=[dict(device=f"cuda:{i}", uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(2)])
    fingerprint = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    slots = [CoordinatorSlot(gpu["device"], gpu["uuid"], fingerprint, image_digest) for gpu in report["gpus"]]
    return report, slots


def run(snapshot, reports, token_file, image_digest):
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.device_count() != 2:
        raise ValueError("diagnostic requires exactly two RTX GPUs")
    endpoints = []
    for ordinal, host in enumerate(("ostrich", "dodo", "emu", "kiwi"), 1):
        passed = [json.loads(line) for line in (reports / f"{host}-worker-qualification.log").read_text().splitlines()
                  if line.startswith('{"event": "qualification_passed"')]
        if len(passed) != 1 or passed[0].get("cases") != 12:
            raise ValueError("worker lacks its complete real-weight numerical qualification")
        identity = passed[0]["identity"]
        endpoints.append(RemoteEndpoint(host, f"http://172.22.2.{ordinal}:17841", identity["preflight_sha256"], identity["image_digest"]))
    local, slots = coordinator_identity(image_digest)
    identity = dict(diagnostic="ds41rt-six-device-search-v1", coordinator=local)
    source = V41Source(snapshot)
    jobs = [(f"{('layers' if i < 5 else 'mtp')}.0.ffn.experts.{i}.w{(1, 3, 2)[i % 3]}", 3 + i % 2) for i in range(10)]
    def hessian(name):
        width = 2304 if name.endswith("w2") else 5120
        return dict(H=torch.eye(width, dtype=torch.float32) * 512, count=1024, finalized=False)
    token = token_file.read_bytes().strip()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ds41rt-six-device-search-") as directory:
        first = None
        for wave in range(2):
            client = DiagnosticClient(endpoints=endpoints, coordinator_slots=slots, token=token,
                timeout_seconds=600, max_attempts=1, assignment_store_path=Path(directory) / "assignments.json")
            for endpoint in endpoints:
                client.qualify(endpoint)
            search = DistributedSearch(source, client, identity)
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(search, name, hessian(name), bits) for name, bits in jobs]
                results = []
                for (name, bits), future in zip(jobs, futures):
                    packed, metrics = future.result()
                    results.append(({key: value.cpu() for key, value in packed.items()}, metrics))
                    print(json.dumps(dict(event="distributed_case_complete", wave=wave, name=name, bits=bits,
                                          execution=metrics["execution"], assignment=metrics["assignment"])), flush=True)
            executions = [metrics["execution"] for _, metrics in results]
            if ({item.get("device") for item in executions if item["kind"] == "coordinator"} != {"cuda:0", "cuda:1"}
                    or {item.get("name") for item in executions if item["kind"] == "remote_worker"} != {e.name for e in endpoints}):
                raise AssertionError("not all six execution devices were exercised")
            if first is None:
                first = results
            else:
                for (old, old_metrics), (new, new_metrics) in zip(first, results):
                    if old_metrics["execution"] != new_metrics["execution"] or old_metrics["assignment"] != new_metrics["assignment"]:
                        raise AssertionError("durable assignment changed after client reconstruction")
                    for key in old:
                        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
        for (name, bits), (packed, _) in zip(jobs, first):
            args = dict(K=bits, devices=[torch.device("cuda:0")], apply_out_scales=None, sigma_reg=.025, seed=787, mcg=True)
            weight = source.decoded(name + ".weight", "cuda:0").T.contiguous().float()
            captured = hessian(name)
            captured["H"] = captured["H"].to("cuda:0")
            _, _, reference = quantize_exl3(weight, captured, args, return_weight_q=False)
            if args.get("q_fallback") is not False:
                raise AssertionError("serial reference used fallback")
            for key in packed:
                torch.testing.assert_close(packed[key], reference[key].cpu(), rtol=0, atol=0)
        print(json.dumps(dict(event="distributed_qualification_passed", cases=10, waves=2, coordinator=local,
                              serial_reference_exact=True, durable_assignments_exact=True,
                              seconds=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    run(**vars(parser.parse_args()))
