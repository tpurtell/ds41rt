#!/usr/bin/env python3
"""Diagnostic: does the real Rust scheduler tie seed change the native FFN result?

Two stages:

1. CPU (always runnable): validate the owners/route-word export produced by
   ``rust/crates/ds41rt-transport/examples/v41_tp_ep_tie_seed_owners.rs`` and
   reproduce the production reduction arithmetic exactly on the CPU:
   per-rank ordered FP32 sum over the six route slots with one BF16 rounding
   (``compact_routes``), then the coordinator's ordered FP32 rank-plane add with
   one final BF16 rounding (``reduce_compact``). A known-cancellation fixture
   proves the ordered sum is not a parallel/reassociated sum.

2. Native (GPU lease required, staged): feed the exported per-rank worker masks
   (sentinel 384 / weight 0 already applied by the Rust transport, never
   bit-packed here) into the emitted native ABI per rank, compact each rank's
   FP32 route planes to BF16 with the runtime kernel, collect all four BF16
   planes on the CPU and finish with the ordered rank add.

Metrics are BF16 bit-equality, rel_L2 and cosine only. The FFN hidden output is
not logits, so this cannot claim a greedy/argmax flip.

This is a diagnostic; it changes no default, gate, production file or frozen
harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SENTINEL = 384
TOPK = 6
HIDDEN = 5120
REL_TOL, COS_TOL = 0.01, 0.9999


# --------------------------------------------------------------------------- #
# Production-equivalent CPU reduction
# --------------------------------------------------------------------------- #

def float2bfloat16_rn(x: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 with round-to-nearest-even, returned as float32.

    Non-finite inputs are rejected explicitly: the bit trick used here can turn a
    NaN payload into inf, so the diagnostic fails rather than silently masking a
    non-finite FFN result.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite value rejected before BF16 conversion")
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 and return the uint16 bit pattern (RNE).

    Bit comparison is the auditable equality here: `np.array_equal` on floats
    treats +0.0 and -0.0 as equal, while their BF16 bits differ (0x0000 vs
    0x8000). Non-finite inputs are rejected.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite value rejected before BF16 conversion")
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> np.uint32(16)
    return rounded.astype(np.uint16)


def plane_sha256_bits(bits: np.ndarray) -> str:
    """sha256 over the little-endian uint16 BF16 plane bytes."""
    payload = np.ascontiguousarray(bits, dtype="<u2").tobytes()
    return hashlib.sha256(payload).hexdigest()


def compact_routes_bf16(routes: np.ndarray) -> np.ndarray:
    """Native `compact_routes<6>`: ordered FP32 slot add, one BF16 rounding.

    `routes` is [rows, 6, 5120] float32. Each add is an explicit float32 op so
    the order p0 + p1 + p2 + p3 + p4 + p5 is preserved (never a parallel sum).
    """
    assert routes.dtype == np.float32 and routes.ndim == 3 and routes.shape[1] == TOPK
    value = routes[:, 0, :].astype(np.float32, copy=True)
    for slot in range(1, TOPK):
        value = (value + routes[:, slot, :]).astype(np.float32)
    return float2bfloat16_rn(value)


def reduce_compact_planes_bf16(planes: list[np.ndarray], shared: np.ndarray | None = None) -> np.ndarray:
    """Native `reduce_compact<Ranks>`: ordered FP32 rank add, one BF16 rounding.

    `planes` are BF16-valued float32 [rows,5120] in rank order. Shared is added
    once after the rank sum and before the single final rounding.
    """
    assert planes, "at least one plane"
    value = planes[0].astype(np.float32, copy=True)
    for plane in planes[1:]:
        value = (value + plane.astype(np.float32)).astype(np.float32)
    if shared is not None:
        value = (value + shared.astype(np.float32)).astype(np.float32)
    return float2bfloat16_rn(value)


def _scalar_ordered(values: list[float]) -> float:
    """Independent scalar float32 ordered add (struct-based IEEE single)."""
    import struct

    def f32(v: float) -> float:
        return struct.unpack("<f", struct.pack("<f", v))[0]

    total = f32(values[0])
    for value in values[1:]:
        total = f32(total + f32(value))
    return total


def reducer_selfcheck() -> dict:
    """Known-cancellation fixture for the ordered add + BF16 rounding."""
    import torch

    # Cancellation: a parallel/tree sum of (2^24, 1, -2^24, 1) is not 1.
    plane = np.array([[[16777216.0, 256.0, 1.0, 0.00390625]]], dtype=np.float32)
    p0 = plane[:, 0, :]
    p1 = np.array([[1.0, 1.0, 2.0, 0.0]], dtype=np.float32)
    p2 = np.array([[-16777216.0, 2.0, -2.0, 0.0]], dtype=np.float32)
    p3 = np.array([[1.0, -1.0, -1.0, 0.0]], dtype=np.float32)
    ordered = reduce_compact_planes_bf16([p0, p1, p2, p3])
    scalar = np.array(
        [[_scalar_ordered([p0[0, c], p1[0, c], p2[0, c], p3[0, c]]) for c in range(4)]],
        dtype=np.float32,
    )
    expected = float2bfloat16_rn(scalar)
    assert np.array_equal(ordered, expected), (ordered, expected)
    # 2^24 + 1 - 2^24 + 1 == 1 under ordered FP32 (the +1s are absorbed).
    assert expected[0, 0] == 1.0, expected
    # BF16 rounding must agree with torch for the same values.
    torch_bf16 = torch.from_numpy(scalar).bfloat16().float().numpy()
    assert np.array_equal(expected, torch_bf16), (expected, torch_bf16)
    # Six-slot compaction is ordered and rounds once: (256,1,1,1,1,1) sums to
    # 261, and BF16 (7-bit mantissa, ulp 2 at 256) rounds the tie to even 260.
    routes = np.zeros((1, TOPK, 4), dtype=np.float32)
    routes[0, 0, 0] = 256.0
    for slot in range(1, TOPK):
        routes[0, slot, 0] = 1.0
    compacted = compact_routes_bf16(routes)
    assert compacted[0, 0] == 260.0, compacted
    return {
        "cancellation_ordered_bf16": float(expected[0, 0]),
        "six_slot_compact": float(compacted[0, 0]),
        "torch_bf16_agrees": True,
    }


# --------------------------------------------------------------------------- #
# Owner-export validation (CPU)
# --------------------------------------------------------------------------- #

def load_export_json(export: dict) -> dict:
    assert export.get("schema") == "v41-tie-seed-owner-export-v1", export.get("schema")
    tp, ep = int(export["tp"]), int(export["ep"])
    world = tp * ep
    experts = export["fixed_experts"]
    assert len(experts) == TOPK and len(set(experts)) == TOPK
    for name in ("case_same_id", "case_cross_id"):
        case = export[name]
        assert len(case["owners"]) == 384
        assert len(case["encoded_words"]) == TOPK
        assert len(case["rank_mask"]) == world
        for entry in case["rank_mask"]:
            rank, group, tp_rank = entry["rank"], entry["group"], entry["tp_rank"]
            assert group == rank // tp and tp_rank == rank % tp, (name, entry)
            assert len(entry["ids"]) == TOPK and len(entry["weights"]) == TOPK
            for slot, expert in enumerate(experts):
                owner = case["owners"][expert]
                if group == owner:
                    assert entry["ids"][slot] == expert, (name, rank, slot, entry["ids"][slot])
                    assert abs(entry["weights"][slot] - 1.0 / TOPK) < 1e-6
                else:
                    assert entry["ids"][slot] == SENTINEL, (name, rank, slot, entry["ids"][slot])
                    assert entry["weights"][slot] == 0.0
    same = export["case_same_id"]
    cross = export["case_cross_id"]
    if ep == 1:
        assert same["owners"] == cross["owners"], "EP1 must be seed-insensitive"
        assert same["owners_hash"] == cross["owners_hash"]
    else:
        assert export["owners_differ"] is True, "cross-id pair must actually differ"
        assert same["owners_hash"] != cross["owners_hash"]
    return export


def load_export(path: Path) -> dict:
    """Validate the owner export loaded from disk."""
    return load_export_json(json.loads(path.read_text()))


def analyse_export(export: dict) -> dict:
    same, cross = export["case_same_id"], export["case_cross_id"]
    owner_delta = sum(1 for a, b in zip(same["owners"], cross["owners"]) if a != b)
    rank_delta = 0
    for a, b in zip(same["rank_mask"], cross["rank_mask"]):
        if a["ids"] != b["ids"] or a["weights"] != b["weights"]:
            rank_delta += 1
    return {
        "topology": export["topology"],
        "same_id": export["same_id"],
        "cross_id": export["cross_id"],
        "owner_delta_experts": owner_delta,
        "rank_mask_delta_ranks": rank_delta,
        "ep1_negative": int(export["ep"]) == 1,
    }


def rank_inputs(case: dict, rank: int, tp: int, ep: int) -> tuple[int, int, list[int], list[float]]:
    """Pure mapping from an exported case to one rank's native request inputs.

    Returns `(group, tp_rank, ids, weights)` exactly as the Rust transport built
    them (sentinel 384 / weight 0 already applied); no owner bit-packing here.
    """
    assert 0 <= rank < tp * ep, rank
    entry = case["rank_mask"][rank]
    group, tp_rank = rank // tp, rank % tp
    assert entry["rank"] == rank and entry["group"] == group and entry["tp_rank"] == tp_rank
    ids = [int(value) for value in entry["ids"]]
    weights = [float(value) for value in entry["weights"]]
    assert len(ids) == TOPK and len(weights) == TOPK
    for owner in case["owners"]:
        assert 0 <= owner < ep or owner == 255, owner
    return group, tp_rank, ids, weights


def net_result(rank_planes_bf16: list[np.ndarray], shared: np.ndarray | None = None) -> np.ndarray:
    """Production-equivalent coordinator finish: ordered FP32 rank add, one BF16.

    `rank_planes_bf16` are BF16-valued float32 [rows,5120] planes in physical
    rank order. Shared is documented as excluded/constant-zero for this M1
    diagnostic and defaults to None.
    """
    return reduce_compact_planes_bf16(rank_planes_bf16, shared)


def arena_slot(expert_id: int) -> int:
    """Arena slot for an expert id: identity, never compacted.

    The transport-exported masks keep the original expert ids (`0,383,1,...`),
    so each source expert must be packed into its own arena slot; any base
    compaction would make the native ABI read an empty slot.
    """
    assert 0 <= expert_id < 384, expert_id
    return expert_id


def artifact_slug(tp: int, ep: int) -> str:
    """Filesystem-safe topology slug for raw plane artifacts (`tp2ep2`)."""
    assert tp > 0 and ep > 0
    return f"tp{tp}ep{ep}"


def _topology_config(export: dict) -> tuple[int | None, int]:
    """Map the exported topology to (`_v41_expert_native.library` spark_tp, I)."""
    tp, ep = int(export["tp"]), int(export["ep"])
    if (tp, ep) == (2, 2):
        return 2, 1152
    if (tp, ep) == (4, 1):
        return None, 576
    if (tp, ep) == (3, 2):
        return 3, 768
    raise ValueError(f"unsupported exported topology tp={tp} ep={ep}")


# --------------------------------------------------------------------------- #
# Native stage (GPU lease required; implemented, not executed on CPU)
# --------------------------------------------------------------------------- #

def _build_rank_arena(snapshot, layer, intermediate, tp_rank, gids, lib, torch):
    """Slice the official checkpoint at one TP rank and pack with the native packer.

    Each resident expert is packed into arena slot `expert_id` (identity, no
    compaction/remap), because the transport-exported worker masks carry the
    original expert ids `0,383,1,...`; remapping would feed the native ABI a
    zero arena slot. Mirrors the slice/pack path in
    `qualify_v41_replicated_native.run_checkpoint`. Returns `(arena, source_sha256)`.
    """
    import safetensors

    from _v41_expert_native import L, P, check

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    lo, hi = tp_rank * intermediate, (tp_rank + 1) * intermediate
    files: dict = {}

    def tensor(key: str):
        shard = index[key]
        if shard not in files:
            files[shard] = safetensors.safe_open(snapshot / shard, framework="pt", device="cpu")
        return files[shard].get_tensor(key)

    digests: list[bytes] = []
    sliced: dict[str, tuple] = {}
    for name in ("w1", "w3", "w2"):
        weight_rows, scale_rows = [], []
        for gid in gids:
            weight = tensor(f"layers.{layer}.ffn.experts.{gid}.{name}.weight").view(torch.uint8)
            scale = tensor(f"layers.{layer}.ffn.experts.{gid}.{name}.scale").view(torch.uint8)
            if name == "w2":
                weight = weight[:, lo // 2:lo // 2 + intermediate // 2]
                scale = scale[:, lo // 32:lo // 32 + intermediate // 32]
            else:
                weight = weight[lo:hi, :]
                scale = scale[lo:hi, :]
            digests.append(weight.contiguous().numpy().tobytes())
            digests.append(scale.contiguous().numpy().tobytes())
            weight_rows.append(weight.contiguous().cuda())
            scale_rows.append(scale.contiguous().cuda())
        sliced[name] = (torch.stack(weight_rows), torch.stack(scale_rows))

    sizes = (L * 4)()
    check(lib.ds41rt_v41_expert_packed_sizes(intermediate, sizes))
    per_bytes = [int(sizes[i]) for i in range(4)]
    arena = [torch.zeros(384 * per_bytes[i], dtype=torch.uint8, device="cuda") for i in range(4)]
    stream = torch.cuda.current_stream().cuda_stream
    for k, gid in enumerate(gids):
        slot = arena_slot(gid)
        src = (P * 6)(
            sliced["w1"][0][k].data_ptr(), sliced["w3"][0][k].data_ptr(),
            sliced["w2"][0][k].data_ptr(), sliced["w1"][1][k].data_ptr(),
            sliced["w3"][1][k].data_ptr(), sliced["w2"][1][k].data_ptr(),
        )
        dst = (P * 4)(*[arena[i].data_ptr() + slot * per_bytes[i] for i in range(4)])
        check(lib.ds41rt_v41_pack_expert_async(src, dst, intermediate, stream))
    torch.cuda.synchronize()
    digest = hashlib.sha256(b"".join(digests)).hexdigest()
    return arena, digest


def abi_layout(token_accumulation: bool, abi_version: int) -> tuple[str, str, tuple]:
    """Return `(kind, compact_symbol, output_shape)` for the native output ABI.

    v2 emits FP32 route planes `[rows,6,5120]` compacted by
    `compact_routes_bf16_async`; v3 emits FP32 tokens `[rows,5120]` compacted by
    `compact_tokens_bf16_async`. This mirrors `qualify_v41_replicated_native` and
    assumes nothing about the actual build until `info` is observed.
    """
    if token_accumulation:
        assert abi_version == 3, abi_version
        return ("fp32_tokens", "ds41rt_v41_compact_tokens_bf16_async", (1, HIDDEN))
    assert abi_version == 2, abi_version
    return ("fp32_routes", "ds41rt_v41_compact_routes_bf16_async", (1, TOPK, HIDDEN))


def quantize_wire(x, capacity: int, torch_module):
    """BF16 [capacity,5120] -> contiguous 5280-byte FP8 E4M3 + UE8M0 K32 rows.

    Inlined (same contract as `qualify_v41_replicated_native`) so this diagnostic
    depends only on `_v41_expert_native`, torch and numpy at the lease.
    """
    wire = torch_module.empty((capacity, 5280), device="cuda", dtype=torch_module.uint8)
    qa = wire[:, :5120].view(torch_module.uint32)
    qs = wire[:, 5120:]
    blocks = x.float().reshape(capacity, 5120 // 32, 32)
    exponent = torch_module.ceil(
        torch_module.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    qa.copy_((blocks / torch_module.exp2(exponent)[..., None])
             .to(torch_module.float8_e4m3fn).view(torch_module.uint32)
             .reshape(capacity, 5120 // 4))
    qs.copy_((exponent + 127).to(torch_module.uint8))
    return wire


def _run_rank_plane(lib, spark_tp, arena, case, rank, tp, ep, x, torch) -> tuple[np.ndarray, dict]:
    """Run the native ABI for one rank and return (BF16 [1,5120] plane, ABI info)."""
    from _v41_expert_native import Native, check

    _, _, ids, weights = rank_inputs(case, rank, tp, ep)
    wire = quantize_wire(x, 1, torch)
    native_ids = torch.tensor([ids], dtype=torch.int32, device="cuda")
    native_weights = torch.tensor([weights], dtype=torch.float32, device="cuda")
    native = Native(lib, 1, arena, wire, native_ids, native_weights, spark_tp=spark_tp)
    kind, symbol, shape = abi_layout(
        bool(native.token_accumulation), int(native.info.abi_version))
    rows = TOPK if kind == "fp32_routes" else 1
    # A view over the caller-owned output (no copy); its pointer is stable, but
    # it must not be read until after `run`.
    source = native.output[:rows].reshape(shape)
    # Execute BEFORE reading or reducing any output.
    native.run(1)
    # Same-stream ordering: the finiteness reduction is queued after the launch.
    if not bool(torch.isfinite(source).all()):
        raise FloatingPointError(f"native {kind} output is non-finite at rank {rank}")
    # Bound compact entry point with ctypes argtypes [P,P,U,P]; `.data_ptr()` is
    # an int converted by ctypes, never a truncated raw pointer.
    compact = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device="cuda")
    check(getattr(lib, symbol)(
        source.data_ptr(), compact.data_ptr(), 1,
        torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    result = compact.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError(f"native compacted plane is non-finite at rank {rank}")
    info = {
        "rank": rank,
        "abi_version": int(native.info.abi_version),
        "token_accumulation": bool(native.token_accumulation),
        "output_kind": kind,
    }
    return result, info


def run_native(args, export: dict) -> dict:
    """Feed the exported Rust worker masks into the native ABI per rank.

    Real execution (needs a CUDA device and the frozen native library): per rank
    the native AOT kernel runs on the exact top-6 M1 IDs/weights the Rust
    transport produced, its FP32 route planes are compacted to BF16 with the
    runtime kernel, all physical-rank BF16 planes are collected on the CPU, and
    the production-ordered FP32 rank add produces the net result. Shared expert
    is excluded (constant zero for this diagnostic).
    """
    import torch

    from _v41_expert_native import library

    if not torch.cuda.is_available():
        raise RuntimeError(
            "--native requires a CUDA device and the frozen native library; "
            "re-run with --native under the GPU lease")
    if args.native_lib is None:
        raise SystemExit("--native requires --native-lib")

    spark_tp, intermediate = _topology_config(export)
    tp, ep = int(export["tp"]), int(export["ep"])
    world = tp * ep
    if int(args.layer) != int(export["layer"]):
        raise SystemExit(
            f"--layer {args.layer} must equal the exported layer {export['layer']}")
    lib = library(str(args.native_lib), spark_tp=spark_tp)
    snapshot = Path(args.snapshot).expanduser()
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"snapshot index missing: {index_path}")
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    gids = [int(value) for value in export["fixed_experts"]]
    generator = torch.Generator(device="cuda").manual_seed(0x51A7_5EED)
    x = torch.randn(1, HIDDEN, generator=generator, device="cuda").mul_(0.5).bfloat16()

    def collect(case: dict, trials: int) -> dict:
        per_rank: dict[int, list[np.ndarray]] = {rank: [] for rank in range(world)}
        abi: dict[int, dict] = {}
        hashes: list[str] = []
        for tp_rank in range(tp):
            arena, digest = _build_rank_arena(
                snapshot, args.layer, intermediate, tp_rank, gids, lib, torch)
            hashes.append(digest)
            for group in range(ep):
                rank = group * tp + tp_rank
                for _ in range(trials):
                    plane, info = _run_rank_plane(
                        lib, spark_tp, arena, case, rank, tp, ep, x, torch)
                    per_rank[rank].append(plane)
                    abi[rank] = info
            del arena
            torch.cuda.empty_cache()
        finals = [net_result([per_rank[rank][trial] for rank in range(world)])
                  for trial in range(trials)]
        # BF16 bit planes: uint16 representation is the auditable equality, since
        # float equality would equate +0.0 and -0.0 (0x0000 vs 0x8000).
        rank_bits = {rank: [bf16_bits(plane) for plane in per_rank[rank]]
                     for rank in range(world)}
        final_bits = [bf16_bits(final) for final in finals]
        return {
            "per_rank": per_rank,
            "finals": finals,
            "rank_bits": rank_bits,
            "final_bits": final_bits,
            "weight_source_sha256": hashes,
            "abi": abi,
        }

    input_bits_sha = plane_sha256_bits(bf16_bits(x.float().cpu().numpy()))
    wire = quantize_wire(x, 1, torch)
    input_wire_sha = hashlib.sha256(wire.cpu().numpy().tobytes()).hexdigest()
    same = collect(export["case_same_id"], 3)
    cross = collect(export["case_cross_id"], 1)
    # The same physical rank/experts must pack identical bytes in both collects.
    assert same["weight_source_sha256"] == cross["weight_source_sha256"], (
        "weight source hash must match between the same-id and cross-id cases")
    # Bit-exact same-id determinism on the uint16 BF16 planes.
    same_trials_exact = all(
        np.array_equal(same["final_bits"][0], trial) for trial in same["final_bits"][1:])
    for rank in range(world):
        bits = same["rank_bits"][rank]
        assert all(np.array_equal(bits[0], value) for value in bits[1:]), (
            f"same request id must be bit-identical per rank {rank}")
    final_bits_same = same["final_bits"][0]
    final_same = same["finals"][0]
    if not np.isfinite(final_same).all():
        raise FloatingPointError("net result is non-finite")
    if not np.count_nonzero(final_same):
        raise AssertionError("net result is all zero; the active mask did not execute")
    # Optional raw artifacts so the reported hashes are auditable.
    artifacts: dict[str, str] = {}
    if args.planes_dir is not None:
        args.planes_dir.mkdir(parents=True, exist_ok=True)

        def write(name: str, bits: np.ndarray) -> str:
            path = args.planes_dir / name
            path.write_bytes(np.ascontiguousarray(bits, dtype="<u2").tobytes())
            return str(path)

        slug = artifact_slug(tp, ep)
        for rank in range(world):
            artifacts[f"same_rank{rank}"] = write(
                f"{slug}_same_rank{rank}.u16", same["rank_bits"][rank][0])
            artifacts[f"cross_rank{rank}"] = write(
                f"{slug}_cross_rank{rank}.u16", cross["rank_bits"][rank][0])
        artifacts["same_final"] = write(f"{slug}_same_final.u16", final_bits_same)
        artifacts["cross_final"] = write(f"{slug}_cross_final.u16", cross["final_bits"][0])
    report = {
        "topology": export["topology"],
        "spark_tp": spark_tp,
        "intermediate": intermediate,
        "snapshot_revision": snapshot.name,
        "snapshot_index_sha256": index_sha,
        "layer": int(args.layer),
        "input_seed": "0x51A75EED",
        "input_activation_sha256_bits": input_bits_sha,
        "input_wire_sha256": input_wire_sha,
        "weight_source_sha256": same["weight_source_sha256"],
        "abi": same["abi"],
        "final_nonzero_elements": int(np.count_nonzero(final_same)),
        "same_id_triplicate_exact_bits": bool(same_trials_exact),
        "same_rank_sha256_bits": {
            str(rank): [plane_sha256_bits(bits) for bits in same["rank_bits"][rank]]
            for rank in range(world)},
        "same_final_sha256_bits": plane_sha256_bits(final_bits_same),
        "cross_id": {
            "final_bit_equal": bool(
                np.array_equal(final_bits_same, cross["final_bits"][0])),
            "final_rel_l2": float(
                np.linalg.norm(cross["finals"][0] - final_same)
                / max(float(np.linalg.norm(final_same)), 1e-30)),
            "final_cosine": _cosine(final_same, cross["finals"][0]),
            "rank_bit_equal": [
                bool(np.array_equal(same["rank_bits"][rank][0], cross["rank_bits"][rank][0]))
                for rank in range(world)
            ],
            "cross_final_sha256_bits": plane_sha256_bits(cross["final_bits"][0]),
            "cross_rank_sha256_bits": {
                str(rank): plane_sha256_bits(cross["rank_bits"][rank][0])
                for rank in range(world)},
        },
        "artifacts": artifacts,
        "note_zero_sign": "bit equality uses BF16 uint16 planes; -0.0 differs from +0.0",
    }
    if ep == 1:
        assert report["cross_id"]["final_bit_equal"], "EP1 negative control must be identical"
    if not same_trials_exact:
        raise AssertionError("same request id must be deterministic across triplicates")
    return report


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owners-json", type=Path, required=True)
    parser.add_argument("--verify-reducer", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--native-lib", type=Path)
    parser.add_argument("--planes-dir", type=Path,
                        help="Write raw little-endian uint16 BF16 planes here")
    parser.add_argument("--snapshot",
                        default="/root/.cache/huggingface/hub/models--deepseek-ai--"
                                "DeepSeek-V4.1-Flash/snapshots/"
                                "dba1be0a40aa45a94ad051997016db3960a90277")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report: dict = {"kind": "v41-tie-seed-replay-diagnostic"}
    report["reducer_selfcheck"] = reducer_selfcheck()
    export = load_export(args.owners_json)
    report["owners_export_sha256"] = hashlib.sha256(args.owners_json.read_bytes()).hexdigest()
    report["analysis"] = analyse_export(export)
    if args.native:
        report["native"] = run_native(args, export)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
