#!/usr/bin/env python3
"""Choose the physical GPUs for a one- or two-RTX release launch."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


MIB = 1 << 20
GIB = 1 << 30
GROUP_BYTES = 455_680
DEFAULT_POOL_TOKENS = 14 * 1_048_576

# Fixed owners include weights, execution workspaces, snapshots, CUDA contexts,
# and the constant (window/state) part of the C16 cache. These are conservative
# K7 measurements; K5 needs slightly less memory. Per-group storage follows the
# 3/2 compressed-source ownership split on logical GPUs 0/1. The fixed totals
# embed the default 20 native TP2 routed layers, so the format-aware model
# subtracts them and adds the checkpoint's own per-layer bytes.
DUAL_FIXED_WITH_HEADROOM = (93_078_948_736, 96_299_387_520)
DUAL_GROUP_BYTES = (270_336, 180_224)
# Native packed TP2 rank weights per routed layer; staging remains reserved.
DUAL_ROUTED_LAYER_BYTES = 3_609_722_880
DUAL_FIXED_EX_EXPERTS = (
    DUAL_FIXED_WITH_HEADROOM[0] - 20 * DUAL_ROUTED_LAYER_BYTES,
    DUAL_FIXED_WITH_HEADROOM[1] - 20 * DUAL_ROUTED_LAYER_BYTES,
)
# Per-TP2-rank routed-layer bytes by expert checkpoint format. The native
# value is the K7 measurement; EXL3 values come from the v7 K2 dual plan
# (69,122,129,920 B / 40 layers, ~2% slack) and the v5 K3.25 residency
# measurements. Attention, draft, vision and cache owners are format
# independent, so the fixed base is shared.
DUAL_LAYER_BYTES_BY_FORMAT = {
    "native": DUAL_ROUTED_LAYER_BYTES,
    "exl3-k23": 1_760_000_000,
    "exl3-k34": 2_800_000_000,
}


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    pci: str
    total_mib: int
    free_mib: int
    reclaim_mib: int = 0

    @property
    def effective_free_mib(self) -> int:
        return min(self.total_mib, self.free_mib + self.reclaim_mib)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("auto", "1", "2"), required=True)
    parser.add_argument("--primary-uuid", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-context-tokens", type=int, required=True)
    parser.add_argument("--retained-turns", type=int, required=True)
    parser.add_argument("--kv-pool-size", default="")
    parser.add_argument("--memory-reservation", default="")
    parser.add_argument("--compact-spark-tp2", action="store_true",
                        help="single RTX EXL3 two-Spark ceiling is capped by physical VRAM")
    parser.add_argument("--minimum-expert-layers", type=int, choices=range(1, 41), default=20)
    parser.add_argument("--expert-format", choices=tuple(DUAL_LAYER_BYTES_BY_FORMAT), default="native")
    parser.add_argument("--reclaim-pid", action="append", type=int, default=[])
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    return parser.parse_args()


def run(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "").strip()
        raise SelectionError(f"GPU query failed: {' '.join(command)}{': ' + detail if detail else ''}") from error


def inventory(tool: str, reclaim_pids: set[int]) -> list[Gpu]:
    raw = run([tool, "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free", "--format=csv,noheader,nounits"])
    reclaim: dict[str, int] = {}
    if reclaim_pids:
        apps = run([tool, "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
        for line in apps.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3 or not fields[1].isdigit() or not fields[2].isdigit():
                continue
            if int(fields[1]) in reclaim_pids:
                reclaim[fields[0]] = reclaim.get(fields[0], 0) + int(fields[2])
    result = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise SelectionError(f"invalid nvidia-smi GPU row: {line}")
        index, uuid, pci, total, free = fields
        if not (index.isdigit() and total.isdigit() and free.isdigit()):
            raise SelectionError(f"non-numeric nvidia-smi GPU row: {line}")
        result.append(Gpu(int(index), uuid, pci, int(total), int(free), reclaim.get(uuid, 0)))
    if not result:
        raise SelectionError("nvidia-smi reported no GPUs")
    return result


def p2p_read_pairs(tool: str) -> set[tuple[int, int]]:
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", run([tool, "topo", "-p2p", "r"]))
    lines = [line.split() for line in raw.splitlines() if line.strip()]
    header = next((line for line in lines if len(line) >= 2 and all(re.fullmatch(r"GPU\d+", item) for item in line)), None)
    if header is None:
        raise SelectionError("cannot parse nvidia-smi peer-access matrix")
    pairs: set[tuple[int, int]] = set()
    for row in lines:
        if not row or not re.fullmatch(r"GPU\d+", row[0]) or len(row) < len(header) + 1:
            continue
        source = int(row[0][3:])
        for target_name, status in zip(header, row[1:]):
            if status == "OK":
                pairs.add((source, int(target_name[3:])))
    return pairs


def parse_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]{1,6})?)(B|MB|GB|MiB|GiB)?", value)
    if not match:
        raise SelectionError(f"invalid KV pool size: {value}")
    scale = {None: 1, "B": 1, "MB": 10**6, "GB": 10**9, "MiB": MIB, "GiB": GIB}[match.group(2)]
    try:
        result = int(Decimal(match.group(1)) * scale)
    except InvalidOperation as error:
        raise SelectionError(f"invalid KV pool size: {value}") from error
    if result <= 0:
        raise SelectionError(f"memory size rounds to zero bytes: {value}")
    return result


def reservation_bytes(value: str, total_mib: int) -> int:
    if value.endswith("%"):
        try:
            percent = Decimal(value[:-1])
        except InvalidOperation as error:
            raise SelectionError(f"invalid memory reservation: {value}") from error
        result = int(total_mib * MIB * percent / Decimal(100))
    else:
        result = parse_bytes(value)
    if result <= 0 or result > total_mib * MIB:
        raise SelectionError(f"memory reservation exceeds device total or rounds to zero: {value}")
    return result


def desired_groups(args: argparse.Namespace) -> int:
    minimum = 2 * args.concurrency + 2 * args.retained_turns
    if args.kv_pool_size:
        groups = parse_bytes(args.kv_pool_size) // GROUP_BYTES
        if groups < minimum:
            raise SelectionError(
                f"explicit KV pool provides {groups} groups but minimum admission needs {minimum}"
            )
        return groups
    if args.memory_reservation:
        return minimum
    groups = math.ceil(args.max_context_tokens / 512) * args.concurrency
    return min(groups, DEFAULT_POOL_TOKENS // 512) + args.concurrency + 2 * args.retained_turns


def required_mib(role: int, groups: int, minimum_expert_layers: int = 20,
                 expert_format: str = "native") -> int:
    layer_bytes = DUAL_LAYER_BYTES_BY_FORMAT[expert_format]
    required = (DUAL_FIXED_EX_EXPERTS[role] + DUAL_GROUP_BYTES[role] * groups
                + minimum_expert_layers * layer_bytes)
    return math.ceil(required / MIB)


def fits(gpu: Gpu, role: int, groups: int, reservation: str, minimum_expert_layers: int = 20,
         expert_format: str = "native") -> bool:
    required = required_mib(role, groups, minimum_expert_layers, expert_format)
    if gpu.total_mib < required or gpu.effective_free_mib < required:
        return False
    return not reservation or reservation_bytes(reservation, gpu.total_mib) >= required * MIB


def main() -> int:
    args = parse_args()
    try:
        gpus = inventory(args.nvidia_smi, set(args.reclaim_pid))
        primary = next((gpu for gpu in gpus if gpu.uuid == args.primary_uuid), None)
        if primary is None:
            raise SelectionError(f"configured primary GPU is absent: {args.primary_uuid}")
        if args.compact_spark_tp2:
            if args.mode != "1" or not args.memory_reservation:
                raise SelectionError("compact Spark TP2 requires mode 1 and a memory ceiling")
            if parse_bytes(args.memory_reservation) > 32 * GIB:
                raise SelectionError("compact Spark TP2 memory ceiling exceeds 32GiB")
        if args.mode == "1":
            if args.kv_pool_size:
                parse_bytes(args.kv_pool_size)
            if args.memory_reservation:
                # Compact runtime clamps its absolute ceiling to physical total;
                # a nominal 32GiB board can report slightly less usable VRAM.
                reservation = args.memory_reservation
                if args.compact_spark_tp2:
                    reservation = str(min(parse_bytes(reservation), primary.total_mib * MIB))
                reservation_bytes(reservation, primary.total_mib)
            print(json.dumps({
                "count": 1,
                "groups": None,
                "required_mib": None,
                "gpus": [primary.__dict__ | {"effective_free_mib": primary.effective_free_mib}],
                "decision": "forced",
            }))
            return 0
        groups = desired_groups(args)
        requirements = [required_mib(role, groups, args.minimum_expert_layers, args.expert_format)
                        for role in (0, 1)]
        peers: set[tuple[int, int]] = set()
        if len(gpus) > 1:
            try:
                peers = p2p_read_pairs(args.nvidia_smi)
            except SelectionError:
                if args.mode == "2":
                    raise
        candidates = sorted(
            (
                gpu for gpu in gpus
                if gpu.uuid != primary.uuid
                and (primary.index, gpu.index) in peers
                and (gpu.index, primary.index) in peers
                and fits(gpu, 1, groups, args.memory_reservation, args.minimum_expert_layers,
                         args.expert_format)
            ),
            key=lambda gpu: (gpu.effective_free_mib, gpu.total_mib, -gpu.index),
            reverse=True,
        )
        dual = fits(primary, 0, groups, args.memory_reservation, args.minimum_expert_layers,
                    args.expert_format) and bool(candidates)
        if args.mode == "2" and not dual:
            details = ", ".join(
                f"GPU {gpu.index}: {gpu.effective_free_mib}/{gpu.total_mib} MiB effective-free/total"
                for gpu in gpus
            )
            raise SelectionError(
                f"forced two-RTX mode is infeasible; logical GPUs need {requirements[0]}/{requirements[1]} MiB; {details}"
            )
        count = 2 if args.mode == "2" or (args.mode == "auto" and dual) else 1
        selected = [primary] + ([candidates[0]] if count == 2 else [])
        print(json.dumps({
            "count": count,
            "groups": groups if count == 2 else None,
            "required_mib": requirements if count == 2 else None,
            "gpus": [gpu.__dict__ | {"effective_free_mib": gpu.effective_free_mib} for gpu in selected],
            "decision": "forced" if args.mode != "auto" else ("automatic-dual" if count == 2 else "automatic-single"),
        }))
        return 0
    except SelectionError as error:
        print(f"ds41rt release: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
