#!/usr/bin/env python3
"""Two-RTX native activation publication parity, without quantizer searches."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import torch
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.utils.v41_checkpoint import save_frontier, load_frontier, load_routed_batch
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch

from block_driver import BlockDriver
from run_store import RunStore
from wavefront import Wavefront
from native_kernels import SerializedHostKernels


@torch.inference_mode()
def run(snapshot):
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
            print(json.dumps(dict(event="parallel_activation_probe_passed",
                                   note="native weights, not selected mixed-weight replication")), flush=True)
        finally:
            journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot)
