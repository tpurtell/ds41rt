#!/usr/bin/env python3
"""Native real-token main-to-dSpark smoke replay, not a quantization launcher.

Loads one source block at a time and round-trips every boundary through the
durable frontier format. Reports finite outputs and memory, not model quality.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import torch
from tokenicer import Tokenicer
from transformers import DeepseekV41Config
from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41NgramHashState

from gptqmodel.utils.v41_source import V41Source
from gptqmodel.utils.v41_checkpoint import save_frontier, load_frontier


@torch.inference_mode()
def run(snapshot, device, batch_size):
    config = DeepseekV41Config.from_pretrained(snapshot, local_files_only=True)
    tokenizer = Tokenicer.load(str(snapshot), model_config=config, local_files_only=True,
                               trust_remote_code=False).tokenizer
    text = "Explain how a memory-mapped file lets the operating system reclaim pages without losing data."
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"].repeat(batch_size, 1)
    print(json.dumps(dict(event="inputs", text=text, ids=ids.tolist())), flush=True)
    hashes = DeepseekV41NgramHashState(config.get_text_config())
    hashes.bind_tokenizer(tokenizer)
    source = V41Source(snapshot)
    sys.path.insert(0, str(snapshot / "inference"))
    import kernel
    inputs = source.load_main_input(hashes, device)
    started = time.monotonic()
    try:
        state = inputs.prepare(ids)
    finally:
        inputs.close()
    embedding = inputs.embed
    del inputs
    identity = dict(diagnostic="native-main-draft-smoke", snapshot=snapshot.name,
                    token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix="ds41rt-replay-") as directory:
        path = Path(directory) / "frontier.safetensors"
        def advance(namespace, layer, state):
            tick = time.monotonic()
            block = source.load_decoded_block(layer, device, native_kernels=kernel, namespace=namespace)
            result = state.advance(block, device)
            if not torch.isfinite(result.hidden).all() or not torch.isfinite(result.pre_mix).all():
                raise AssertionError("nonfinite source replay state")
            checksum = save_frontier(result, path, provenance={**identity, "namespace": namespace})
            result = load_frontier(path, expected_sha256=checksum,
                                   expected_provenance={**identity, "namespace": namespace})
            del block
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps(dict(event="boundary", namespace=namespace, layer=layer,
                seconds=time.monotonic() - tick, elapsed=time.monotonic() - started,
                gpu_allocated_bytes=torch.cuda.memory_allocated(device),
                gpu_peak_bytes=torch.cuda.max_memory_allocated(device),
                host_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                target_layers=list(result.target_features), checkpoint_sha256=checksum)), flush=True)
            return result
        for layer in range(config.get_text_config().num_hidden_layers):
            state = advance("layers", layer, state)
        adapter = source.load_dspark_input(device, native_kernels=kernel, embedding=embedding)
        draft = adapter.prepare(state.target_features, ids, position=ids.shape[1] - 2)
        del state, adapter, embedding
        for layer in range(config.get_text_config().num_nextn_predict_layers):
            draft = advance("mtp", layer, draft)
    print(json.dumps(dict(event="passed", elapsed=time.monotonic() - started)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    run(args.snapshot, args.device, args.batch_size)
