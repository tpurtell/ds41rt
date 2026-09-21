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
    parser.add_argument("--intermediate", type=int, choices=(512, 640, 768, 1152, 2304), required=True)
    parser.add_argument("--tiers", type=int, nargs=2, default=[3, 4], metavar=("TIER0", "TIER1"),
                        help="Decoder tier pair to qualify; the release packages the "
                             "uniform-K2 (2,3) and staged (3,4) families separately")
    parser.add_argument("--tile", type=int, nargs=4, default=None, metavar=("FC1_K", "FC1_N", "FC2_K", "FC2_N"),
                        help="Compare this tile geometry explicitly")
    parser.add_argument("--production-tile", action="store_true",
                        help="Resolve the tile with the pinned production planner for this "
                             "width/capacity/routing and pass it explicitly")
    parser.add_argument("--capacity", type=int, default=16,
                        help="Token capacity whose production tile --production-tile resolves")
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
        "tiers": list(args.tiers), "direct": args.direct, "swiglu_limit": args.swiglu_limit,
        # The pinned probe has its own test-local default (64,128,64,128 direct /
        # 128,128,128,128 grouped), which is NOT the production planner. Never record
        # an implicit tile as production geometry.
        "tile": None, "tile_source": "pinned test-local default, not the production planner",
        # The pinned probe's own shape: fixed rows/experts, not a capacity run.
        "probe_rows": 2, "probe_experts": 2,
        "started_at": time.time(),
        "passed": False,
    }
    if args.tile and args.production_tile:
        raise SystemExit("--tile and --production-tile are mutually exclusive")
    if args.tile or args.production_tile:
        from b12x.moe.fused_moe._impl import _projection_mixed_direct_topk_routes, \
            _projection_mixed_tile_config
        if args.production_tile:
            # Resolve the planner's route/capacity pair independently of what this
            # probe runs with: the probe is a fixed 2-row, 2-expert component check,
            # so a production tile named for capacity N is an inspected plan key and
            # never a claim that N tokens were executed here.
            planned_route = bool(args.direct or _projection_mixed_direct_topk_routes(
                args.capacity, 6, direct_exl3=len(args.tiers) == 2))
            args.tile = list(_projection_mixed_tile_config(
                None, hidden_size=5120, intermediate_size=args.intermediate,
                token_count=args.capacity, direct_topk_routes=planned_route))
            record["planned_tile_capacity"] = args.capacity
            record["planned_tile_route"] = planned_route
            record["tile_source"] = (
                f"production planner tile for capacity {args.capacity} on "
                f"{'direct' if planned_route else 'packed'} routes; probe ran "
                f"{record['probe_rows']} rows with direct={args.direct}")
        else:
            record["tile_source"] = "explicit offline A/B subject"
        record["tile"] = list(args.tile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    try:
        test_mixed_two_tier_matches_serial_and_captures(
            torch.int32, "mcg", tuple(args.tiers), args.direct, "bf16",
            geometry=(5120, args.intermediate),
            tile_config=tuple(args.tile) if args.tile else None,  # None => test-local default
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
