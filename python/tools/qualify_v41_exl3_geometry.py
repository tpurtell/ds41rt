#!/usr/bin/env python3
"""Exercise B12x mixed K3/K4 geometry before native V4.1 AOT integration.

This is a component correctness probe, not a serving or throughput benchmark.
It reuses B12x's serial-tier comparison, route masking, repeatability and CUDA
graph replay checks with V4.1's hidden width and the selected TP width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import _pinned_sparkinfer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate", type=int, choices=(512, 640, 1152, 2304), required=True)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from tests.moe.test_w4a16_mixed_trellis import test_mixed_two_tier_matches_serial_and_captures

    props = torch.cuda.get_device_properties(0)
    record = {
        "scope": "component serial-tier equivalence and graph replay; not serving qualification",
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "gpu": props.name, "compute": [props.major, props.minor],
        "sm_count": props.multi_processor_count, "torch": torch.__version__,
        "hidden": 5120, "intermediate": args.intermediate,
        "tiers": [3, 4], "direct": args.direct, "swiglu_limit": args.swiglu_limit,
        "started_at": time.time(),
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    try:
        test_mixed_two_tier_matches_serial_and_captures(
            torch.int32, "mcg", (3, 4), args.direct, "bf16",
            geometry=(5120, args.intermediate),
            swiglu_limit=args.swiglu_limit,
        )
        record["passed"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["completed_at"] = time.time()
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
