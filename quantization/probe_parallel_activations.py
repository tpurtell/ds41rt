#!/usr/bin/env python3
"""Two-RTX native activation publication parity, without quantizer searches."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import sqlite3

import torch
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.utils.v41_checkpoint import save_frontier, load_frontier, load_routed_batch, _load_state
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch
from gptqmodel.utils.v41_mixed_replay import install_projection

from block_driver import BlockDriver
from run_store import RunStore
from wavefront import Wavefront
from native_kernels import SerializedHostKernels


@torch.inference_mode()
def run(snapshot, mixed_fixture=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(snapshot / "inference"))
    import kernel
    kernel = SerializedHostKernels(kernel)
    source = V41Source(snapshot)
    blocks = [(source.load_decoded_block(0, f"cuda:{index}", native_kernels=kernel, namespace="mtp"),
               torch.device(f"cuda:{index}")) for index in range(2)]
    adapter = source.load_dspark_input("cuda:0", native_kernels=kernel)
    torch.manual_seed(82)
    states = []
    for ordinal in range(4):
        features = {index: torch.randn(1, 131, 5120, dtype=torch.bfloat16) * .1
                    for index in adapter.target_layer_ids}
        ids = torch.randint(adapter.embed.weight.shape[0], (1, 131))
        states.append(adapter.prepare_joint(features, ids, positions=torch.tensor([1, 7, 127, 129])))
    del adapter
    with tempfile.TemporaryDirectory(prefix="ds41rt-parallel-probe-") as directory:
        journal = RunStore(directory, {"diagnostic": "two-rtx-native-activation-v1"})
        try:
            driver = BlockDriver(source, journal, {"diagnostic": "two-rtx-native-activation-v1"}, device="cuda:0")
            wavefront = Wavefront(driver)
            keys, provenance = [], {}
            for ordinal, state in enumerate(states):
                key = f"inputs/{ordinal}"
                path = Path(directory) / f"input-{ordinal}.safetensors"
                provenance[key] = {"input": ordinal}
                save_frontier(state, path, provenance=provenance[key])
                journal.record_file(key, "replay", path)
                keys.append(key)
            driver._publish("inventory", "inventory", {"keys": tuple(keys)}, tuple(keys))
            for routed in (True, False):
                started = time.monotonic()
                outputs, output_provenance = wavefront._activate(blocks, "probe", 0, keys, provenance,
                                                                 routed=routed, parent="inventory")
                elapsed = time.monotonic() - started
                for ordinal, key in enumerate(outputs):
                    record = journal.get(key)
                    if routed:
                        actual = load_routed_batch(journal.root / record["path"], expected_sha256=record["sha256"],
                                                   expected_provenance=output_provenance[key])
                        expected = V41RoutedBatch.from_replay(blocks[0][0], states[ordinal], "cuda:0")
                        for field in ("hidden", "logits", "weights", "indices"):
                            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
                    else:
                        actual = load_frontier(journal.root / record["path"], expected_sha256=record["sha256"],
                                               expected_provenance=output_provenance[key])
                        expected = states[ordinal].advance(blocks[0][0], "cuda:0")
                        torch.testing.assert_close(actual.hidden, expected.hidden, rtol=0, atol=0)
                        torch.testing.assert_close(actual.pre_mix, expected.pre_mix, rtol=0, atol=0)
                print(json.dumps(dict(event="parallel_activation_exact", routed=routed, batches=len(keys), seconds=elapsed)), flush=True)
            if mixed_fixture is not None:
                # Read the old synthetic-search fixture without opening a mutable
                # RunStore or pretending its Hessians came from these new inputs.
                fixture_db = sqlite3.connect(f"file:{mixed_fixture.resolve()}/run.sqlite?mode=ro", uri=True)
                try:
                    fixture_identity = json.loads(fixture_db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
                    def fixture_load(key, kind):
                        record = json.loads(fixture_db.execute("SELECT record FROM artifacts WHERE key=?", (key,)).fetchone()[0])
                        if record["kind"] != kind:
                            raise ValueError("mixed fixture artifact kind mismatch")
                        return _load_state(mixed_fixture / record["path"], expected_sha256=record["sha256"],
                            expected_provenance={**fixture_identity, "artifact": key}, kind=kind)
                    def fixture_phase(block, namespace, layer, routed_keys, *, routed_provenance):
                        if namespace != "mtp" or layer != 0:
                            raise ValueError("fixture is only the previously searched mtp0 diagnostic")
                        for phase in ("gate_up", "down"):
                            phase_key = f"blocks/mtp/000/{phase}/complete"
                            marker = fixture_load(phase_key, "phase")
                            for expert, projection, candidate_key in marker["selected"]:
                                candidate = fixture_load(candidate_key, "projection")
                                driver._publish(candidate_key, "projection", {**candidate,
                                    "diagnostic_fixture_identity": fixture_identity}, ())
                                install_projection(block, expert, projection, candidate["packed"], device="cuda:0")
                            driver._publish(phase_key, "phase", marker, tuple(key for _, _, key in marker["selected"]))
                        return phase_key
                    driver.run = fixture_phase
                    complete = wavefront.process(blocks[0][0], "mtp", 0, keys,
                        input_provenance=provenance, replica=blocks[1])
                    for ordinal, key in enumerate(complete["output_keys"]):
                        actual = wavefront._input(key, complete["output_provenance"][key], 1)
                        expected = states[ordinal].advance(blocks[0][0], "cuda:0")
                        torch.testing.assert_close(actual.hidden, expected.hidden, rtol=0, atol=0)
                        torch.testing.assert_close(actual.pre_mix, expected.pre_mix, rtol=0, atol=0)
                    self_check = wavefront.process(blocks[0][0], "mtp", 0, keys,
                        input_provenance=provenance, replica=blocks[1])
                    if self_check != complete:
                        raise AssertionError("completed mixed wavefront reload changed")
                    print(json.dumps(dict(event="mixed_replica_exact", batches=len(keys),
                                          selected_projections=384, completed_reload=True)), flush=True)
                finally:
                    fixture_db.close()
            print(json.dumps(dict(event="parallel_activation_probe_passed", mixed_fixture=mixed_fixture is not None)), flush=True)
        finally:
            journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mixed-fixture", type=Path, help="read-only old synthetic mtp0 phase fixture")
    args = parser.parse_args()
    run(args.snapshot, args.mixed_fixture)
