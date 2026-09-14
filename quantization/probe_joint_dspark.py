#!/usr/bin/env python3
"""Real-weight joint dSpark replay diagnostic; no production candidates."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.utils.v41_replay import V41ReplayBatch


@torch.inference_mode()
def run(snapshot, device):
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(snapshot / "inference"))
    import kernel
    source = V41Source(snapshot)
    adapter = source.load_dspark_input(device, native_kernels=kernel)
    torch.manual_seed(51)
    positions = torch.tensor([1, 7, 127, 128, 256])
    features = {index: torch.randn(1, 258, 5120, dtype=torch.bfloat16) * .1
                for index in adapter.target_layer_ids}
    ids = torch.randint(adapter.embed.weight.shape[0], (1, 258))
    joint = adapter.prepare_joint(features, ids, positions=positions)
    singles = [V41ReplayBatch(0, joint.hidden[i:i+1].clone(), joint.pre_mix[i:i+1].clone(), {},
               {"main_x": joint.kwargs["main_x"][:, :int(position)+1].clone()}, {})
               for i, position in enumerate(positions)]
    del adapter, features
    for stage in range(3):
        block = source.load_decoded_block(stage, device, native_kernels=kernel, namespace="mtp")
        output = joint.advance(block, device)
        repeated = joint.advance(block, device)
        if not torch.equal(output.hidden, repeated.hidden) or not torch.isfinite(output.hidden).all():
            raise AssertionError("joint replay is not finite and repeatable")
        singles = [state.advance(block, device) for state in singles]
        separate = torch.cat([state.hidden for state in singles])
        print(json.dumps(dict(stage=stage, repeatable=True,
            single_anchor_max_abs=(output.hidden.float()-separate.float()).abs().max().item(),
            single_anchor_exact=torch.equal(output.hidden, separate),
            hidden_shape=list(output.hidden.shape), shared_main_shape=list(output.kwargs["main_x"].shape))), flush=True)
        joint = output
        del block, repeated
        gc.collect()
    print(json.dumps(dict(event="joint_smoke_passed", anchors=positions.tolist(),
                          note="single-anchor differences are reported, not a joint reference parity gate")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.snapshot, args.device)
