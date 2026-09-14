#!/usr/bin/env python3
"""Read-only production-Hessian throughput probe; no replay or run mutation.

Compare fixed windows, continuous six-device dispatch, and two RTX devices.
Each variant has fresh remote request identity to prevent checkpoint cache hits.
All packed results must exactly match already committed production candidates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import queue
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import uuid

import torch
from gptqmodel.utils.exl3_remote import EXL3RemoteClient, RemoteEndpoint
from gptqmodel.utils.v41_checkpoint import _load_state
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from candidate_queue import CandidateQueue
from distributed_search import DistributedSearch
from qualify_distributed import coordinator_identity


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    db = sqlite3.connect(f"file:{args.run_root / 'run.sqlite'}?mode=ro", uri=True)
    identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
    def get(key, verify=False):
        row = db.execute("SELECT record FROM artifacts WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    candidates = db.execute("SELECT key FROM artifacts WHERE key LIKE 'blocks/base/000/gate_up/expert-%/w%-k3' ORDER BY key LIMIT ?", (args.jobs,)).fetchall()
    if len(candidates) != args.jobs:
        raise ValueError("insufficient committed reference candidates")
    jobs = []
    for (key,) in candidates:
        prefix, expert, projection = key.rsplit('/', 2)
        parents = get(key)['parents']
        if len(parents) != 1:
            raise ValueError("expected one Hessian parent")
        jobs.append((prefix, int(expert.removeprefix('expert-')), projection.split('-')[0], 3, next(iter(parents)), 'layers.0'))
    endpoints = []
    for ordinal, host in enumerate(('ostrich', 'dodo', 'emu', 'kiwi'), 1):
        reports = [json.loads(line) for line in (args.reports / f'{host}-worker-qualification.log').read_text().splitlines() if line.startswith('{"event": "qualification_passed"')]
        if len(reports) != 1 or reports[0]['cases'] != 12:
            raise ValueError('missing worker qualification')
        worker = reports[0]['identity']
        endpoints.append(RemoteEndpoint(host, f'http://172.22.2.{ordinal}:17841', worker['preflight_sha256'], worker['image_digest']))
    local, slots = coordinator_identity(args.image_digest)
    source = V41Source(args.snapshot)
    token = args.token_file.read_bytes().strip()
    # Exclude extension compilation from measured search; serialize cold load.
    prewarm_exllamav3_extension()
    for variant in args.variants:
        with tempfile.TemporaryDirectory(prefix='ds41rt-throughput-') as temporary:
            client = EXL3RemoteClient(endpoints=endpoints, coordinator_slots=slots, token=token,
                timeout_seconds=600, max_attempts=1, assignment_store_path=Path(temporary) / 'assignments')
            for endpoint in endpoints:
                client.qualify(endpoint)
            search = DistributedSearch(source, client, dict(diagnostic='throughput-v1', nonce=uuid.uuid4().hex, coordinator=local))
            if variant == 'rtx-only':
                available = queue.Queue()
                for device in ('cuda:0', 'cuda:1'):
                    available.put(device)
                @torch.inference_mode()
                def search(name, hessian, bits):
                    device = available.get()
                    try:
                        weight = source.decoded(name + '.weight', device).T.contiguous().float()
                        options = dict(K=bits, devices=[torch.device(device)], apply_out_scales=None, sigma_reg=.025, seed=787, mcg=True)
                        _, _, packed = quantize_exl3(weight, {**hessian, 'H': hessian['H'].to(device, copy=True)}, options, return_weight_q=False)
                        if options.get('q_fallback') is not False:
                            raise AssertionError('fallback')
                        return packed, {**options['error_metrics'], 'execution': {'device': device}}
                    finally:
                        available.put(device)
                search.max_workers = 2
            counts = Counter()
            def publish(key, kind, result, parents):
                record = get(key)
                reference = _load_state(args.run_root / record['path'], expected_sha256=record['sha256'], expected_provenance={**identity, 'artifact': key}, kind='projection')
                if set(result['packed']) != set(reference['packed']):
                    raise AssertionError('packed buffer inventory differs')
                for name, tensor in result['packed'].items():
                    torch.testing.assert_close(tensor.cpu(), reference['packed'][name].cpu(), rtol=0, atol=0)
                execution = result['quantizer_metrics']['execution']
                counts[execution.get('device', execution.get('name'))] += 1
            driver = SimpleNamespace(search=search, identity=identity,
                journal=SimpleNamespace(root=args.run_root, get=get),
                _load=lambda *a: None, _publish=publish, progress=lambda event: None)
            started = time.monotonic()
            print(json.dumps(dict(event='throughput_started', variant=variant, jobs=len(jobs))), flush=True)
            if variant == 'fixed':
                # Reproduce old 16-job subset / ten-job window barriers.
                for subset in range(0, len(jobs), 16):
                    for start in range(subset, min(subset + 16, len(jobs)), 10):
                        with CandidateQueue(driver) as pending:
                            for job in jobs[start:min(start + 10, subset + 16, len(jobs))]:
                                pending.submit(job)
            else:
                with CandidateQueue(driver) as pending:
                    for job in jobs:
                        pending.submit(job)
            print(json.dumps(dict(event='throughput_passed', variant=variant, jobs=len(jobs),
                seconds=time.monotonic() - started, devices=dict(counts), packed_exact=True)), flush=True)
    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'snapshot', 'reports', 'token-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image-digest', required=True)
    parser.add_argument('--jobs', type=int, default=64)
    parser.add_argument('--variants', nargs='+', choices=('fixed', 'continuous', 'rtx-only'), default=['fixed', 'continuous', 'rtx-only'])
    run(parser.parse_args())
