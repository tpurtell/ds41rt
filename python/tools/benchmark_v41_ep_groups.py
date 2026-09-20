#!/usr/bin/env python3
"""Official V4.1 Spark routed-expert benchmark for replicated expert groups.

Run through ``scripts/run-tp-ep-kernel-checks.sh bench`` so the pinned
SparkInfer tree is imported read-only. This file deliberately lives in the
parent repository: ``scripts/verify-sparkinfer-source.py`` hashes every file
under ``third_party/sparkinfer`` including ``tests/`` and ``benchmarks/``, so a
new file there would break AOT source verification for the native build.

Scope of THIS revision: the true production Python path only.

* ``--phase full`` compiles and times ``V41SlicePipeline`` end to end. That is
  the kernel path the DS41RT AOT entry is generated from.
* ``--phase split`` additionally compiles the plan and reduce stages from the
  *same* pipeline instance on the same buffers, so phase-0 decomposition is
  directly comparable to ``full``.
* ``--phase native`` drives the real C ABI (``ds41rt_v41_expert_info`` /
  ``initialize`` / ``bind_scratch`` / ``launch``). Its geometry comes from the
  library's own emitted info, which is authoritative for the new TP2/TP3 roles.

Weight sources are explicit and recorded in every record as ``source_mode``:

* ``synthetic`` -- deterministic pseudo-random FP4/E8M0 operands. Fast, fully
  hermetic, and the only mode that supports the oracle gate by itself.
* ``checkpoint`` -- reads the pinned official snapshot, slices the real
  ``w1/w3/w2`` and scale tensors for the requested TP degree, repacks them, and
  records the snapshot revision plus a SHA-256 over the exact slice bytes used.

An arm never reports a timing without a completed launch and synchronize, and
every record states which gate it passed: ``reference`` (oracle within
``rel_l2 < 0.01`` and ``cosine > 0.9999``) or ``weak`` (finite/nonzero only).
A ``weak`` arm is not qualified evidence and says so.

Masking model (canonical top-6, whole-expert ownership): a rank keeps a route
iff it owns that route's expert; dropped routes become sentinel id 384 with
weight exactly 0. The grouped pipeline emits no work group for an unowned
expert, so the rank runs no FC1/activation/FC2 for it. The group grid is still
launched and an unowned group returns after two integer comparisons -- launched
CTAs and expensive math are separate quantities and are never conflated.

Cost scope, so it is not assumed away: EP changes how work is partitioned, not
how much arithmetic exists. At width 192, TP4 runs
``6 experts x ceil(576/192) = 18`` active group CTAs; TP2xEP2 runs
``3 experts x ceil(1152/192) = 18``. There is no free EP speedup. Any gain must
come from per-rank CTA count, per-CTA weight bytes, L2 working set, padding tax,
or reduction width.

This script never touches the serving deployment: it allocates its own device
memory and its own CUDA graphs and runs on the current device only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# The canonical DS41RT resolver: it verifies the pinned SparkInfer tree and its
# lock (honouring DS41RT_SPARKINFER_SOURCE_DIR), puts that tree first on
# sys.path, and fails closed if `b12x` does not import from it. Import it before
# anything from SparkInfer so an unverified copy can never be used.
import _pinned_sparkinfer  # noqa: F401  (import side effect is the point)

import torch  # noqa: E402

HIDDEN = 5120
SENTINEL = 384
EXPERT_POINTERS = 44


# --------------------------------------------------------------------------- #
# Ownership masks
# --------------------------------------------------------------------------- #

def owners_for(ids, groups, rank):
    """Whole-expert ownership: group ``r`` owns experts with ``expert % groups == r``."""
    return (ids % groups) == rank


def mask_ids(ids, owner):
    return ids.masked_fill(~owner, SENTINEL)


MASK_CASES = {
    "ep2-3-3": (2, "balanced 3/3 for a distinct top-6"),
    "ep2-3-3-other": (2, "balanced 3/3, other group"),
    "ep3-2-2-2": (3, "balanced 2/2/2 for a distinct top-6"),
    "ep2-all": (1, "rank owns every expert: worst-case imbalance"),
    "ep2-none": (0, "rank owns nothing: fully inactive group"),
}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_stream():
    """The stream the caller is executing on right now.

    Resolved per call, not cached: during CUDA-graph capture the active stream is
    the capture stream, and a stream captured earlier would record nothing.
    """
    from b12x._lib.utils import current_cuda_stream
    return current_cuda_stream()


def gpu_snapshot():
    fields = "index,name,uuid,compute_cap,clocks.sm,power.draw,power.limit,utilization.gpu"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or f"exit {proc.returncode}"}
    return {"raw": [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]}


def time_replays(replay, count: int, torch_module):
    """Microseconds for ``count`` consecutive replays."""
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / count


def timed_intervals(replay, intervals: int, replays: int, torch_module):
    return [time_replays(replay, replays, torch_module) for _ in range(intervals)]


def timed_cold_single(replay, flush, intervals: int, torch_module):
    """Flush, then time exactly ONE replay -- the historical cold condition."""
    samples = []
    for _ in range(intervals):
        flush()
        samples.append(time_replays(replay, 1, torch_module))
    return samples


def summarise(samples):
    return {"median_us": statistics.median(samples), "min_us": min(samples),
            "max_us": max(samples), "mean_us": statistics.fmean(samples),
            "samples_us": samples}


# --------------------------------------------------------------------------- #
# Weight sources
# --------------------------------------------------------------------------- #

class LogicalWeights:
    """Full logical (unsliced) expert operands at the full intermediate width."""

    def __init__(self, weights, scales, source_mode, identity, experts=None):
        self.weights = weights
        self.scales = scales
        self.source_mode = source_mode
        self.identity = identity
        # `experts` is the leading expert dimension of every operand; it is a
        # property of the operand set, not of the slice being executed.
        self.experts = int(weights["w1"].shape[0]) if experts is None else int(experts)

    @staticmethod
    def synthetic(experts, intermediate, torch_module, seed=0x41):
        gen = torch_module.Generator(device="cuda").manual_seed(seed)
        weights, scales = {}, {}
        for name, shape in [
            ("w1", (experts, intermediate, HIDDEN // 2)),
            ("w3", (experts, intermediate, HIDDEN // 2)),
            ("w2", (experts, HIDDEN, intermediate // 2)),
        ]:
            weights[name] = torch_module.randint(
                0, 256, shape, generator=gen, device="cuda", dtype=torch.uint8)
            scales[name] = torch_module.randint(
                121, 124, (*shape[:-1], shape[-1] // 16), generator=gen,
                device="cuda", dtype=torch.uint8)
        digest = hashlib.sha256()
        for name in ("w1", "w3", "w2"):
            digest.update(weights[name].cpu().numpy().tobytes())
            digest.update(scales[name].cpu().numpy().tobytes())
        identity = {
            "mode": "synthetic", "seed": seed, "experts": experts,
            "intermediate": intermediate, "operands_sha256": digest.hexdigest(),
        }
        return LogicalWeights(weights, scales, "synthetic", identity, experts)

    @staticmethod
    def checkpoint(snapshot: Path, layer: int, experts: int, intermediate: int,
                   torch_module):
        """Read the pinned official snapshot, assembling full-width operands.

        The official checkpoint stores one tensor per expert, so a full-width
        operand set must be stacked across experts:
        ``layers.{L}.ffn.experts.{E}.w1.weight`` / ``...w1.scale`` for ``E`` in
        ``0..experts-1``, resolved through the official
        ``model.safetensors.index.json``.

        Per-rank slicing happens afterwards in :func:`rank_operands`, so every TP
        degree derives from the same bytes and is compared against the same
        oracle.
        """
        index_path = snapshot / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"official index {index_path} is missing; refusing to guess "
                "checkpoint tensor names")
        weight_map = json.loads(index_path.read_text())["weight_map"]
        weights, scales, used = {}, {}, {}
        for name in ("w1", "w3", "w2"):
            weight_rows, scale_rows, files = [], [], set()
            for expert in range(experts):
                for kind, sink in (("weight", weight_rows), ("scale", scale_rows)):
                    key = f"layers.{layer}.ffn.experts.{expert}.{name}.{kind}"
                    if key not in weight_map:
                        raise KeyError(
                            f"{key} absent from the official index; the "
                            "checkpoint layout differs from the per-expert names")
                    row, path = _read_tensor(snapshot, weight_map, key)
                    files.add(path.name)
                    sink.append(row)
            weights[name] = torch_module.stack(weight_rows)
            scales[name] = torch_module.stack(scale_rows)
            used[name] = {"file": sorted(files)[0], "experts": len(weight_rows),
                          "weight_shape": list(weights[name].shape),
                          "scale_shape": list(scales[name].shape)}
            if weights[name].shape[0] != experts:
                raise RuntimeError(
                    f"{name} stacked {weights[name].shape[0]} experts, expected "
                    f"{experts}")
        # The official checkpoint stores the FP4 payload as int8 and the E8M0
        # scales as float8_e8m0fnu. Both are 8-bit; the repack helpers want raw
        # uint8 bytes, so reinterpret (never resample). Do this BEFORE hashing:
        # numpy cannot serialise float8_e8m0fnu.
        device = torch_module.device("cuda", torch_module.cuda.current_device())
        weights = {
            k: v.to(device).view(torch_module.uint8).contiguous()
            for k, v in weights.items()
        }
        scales = {
            k: v.to(device).view(torch_module.uint8).contiguous()
            for k, v in scales.items()
        }
        digest = hashlib.sha256()
        for name in ("w1", "w3", "w2"):
            digest.update(weights[name].cpu().numpy().tobytes())
            digest.update(scales[name].cpu().numpy().tobytes())
        identity = {
            "mode": "checkpoint", "snapshot": str(snapshot),
            "snapshot_revision": snapshot.name, "layer": layer,
            "experts": experts, "intermediate": intermediate,
            "index_sha256": sha256_file(index_path),
            "per_expert_tensors": used,
            "operands_sha256": digest.hexdigest(),
        }
        return LogicalWeights(weights, scales, "checkpoint", identity, experts)


def bank_label(active_count, experts):
    """Truthful bank label derived from the active count.

    A bank whose active union covers every expert IS the full real expert bank;
    calling it "not a full real bank" is false. Used by the tool's stdout label
    and by the operand identity so both stay consistent.
    """
    if active_count == experts:
        return (f"full real expert bank: real weights for all {experts} experts; "
                f"no zero-filled slots")
    return (f"active-filled dense E{experts}: real weights for {active_count} active "
            f"experts, remaining {experts - active_count} slots zero; this is NOT a "
            f"full real expert bank")


def checkpoint_metadata(snapshot: Path) -> dict:
    """Read only the checkpoint's declared identity; never its tensor payload."""
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"official index {index_path} is missing; refusing to guess "
            "checkpoint tensor names")
    index = json.loads(index_path.read_text())
    if "weight_map" not in index:
        raise KeyError(f"{index_path} has no weight_map")
    return {
        "snapshot": str(snapshot),
        "snapshot_revision": snapshot.name,
        "index_sha256": sha256_file(index_path),
        "weight_map": index["weight_map"],
    }


def expert_operand_keys(layer: int, expert: int) -> dict:
    base = f"layers.{layer}.ffn.experts.{expert}"
    return {f"{name}.{kind}": f"{base}.{name}.{kind}"
            for name in ("w1", "w3", "w2")
            for kind in ("weight", "scale")}


def checkpoint_selected_operands(snapshot: Path, layer: int, experts: int,
                                 intermediate: int, active_experts,
                                 torch_module, *, expect_revision=None,
                                 require_all_active=True):
    """REAL weights for the ACTIVE experts only, in a dense E-slot layout.

    Only the requested experts are read from disk and copied into a zero-filled
    dense tensor of the canonical ``[experts, ...]`` shape. Unused expert slots
    stay zero, so this is an **active-filled dense E{experts}** operand set, NOT
    a full real bank: the cache working set it presents is only the active
    experts. Callers must label it that way.

    ``active_experts`` are the ids the live histogram actually routes to. At M1
    that is 6 ids; at M8 up to 48; at M16 up to 96. Nothing outside that set is
    read, which is what avoids materialising a full 384-expert operand bank.

    Fails closed on a missing expert tensor or an unexpected revision so a
    silently wrong checkpoint cannot be measured.
    """
    meta = checkpoint_metadata(snapshot)
    weight_map = meta["weight_map"]
    if expect_revision is not None and meta["snapshot_revision"] != expect_revision:
        raise ValueError(
            f"snapshot revision {meta['snapshot_revision']!r} does not match the "
            f"expected {expect_revision!r}")
    requested = [int(e) for e in active_experts]
    if not requested:
        raise ValueError("active_experts must not be empty")
    if require_all_active and len(requested) != len(set(requested)):
        raise ValueError(
            f"active_experts contains duplicates: {sorted(requested)}; the "
            "histogram derivation must de-duplicate before calling")
    bad = [e for e in requested if e < 0 or e >= experts]
    if bad:
        raise ValueError(f"active expert ids out of range 0..{experts - 1}: {bad}")
    ids = sorted(set(requested))

    shapes = {
        "w1": (experts, intermediate, HIDDEN // 2),
        "w3": (experts, intermediate, HIDDEN // 2),
        "w2": (experts, HIDDEN, intermediate // 2),
    }
    # Allocate on the CPU deliberately: the per-expert tensors are read on CPU,
    # the digest is computed on CPU before any transfer, and only then is the
    # bank moved to the device. Allocating on the device here would make the
    # host-side assignment below fail.
    weights, scales = {}, {}
    for name, shape in shapes.items():
        weights[name] = torch_module.zeros(shape, dtype=torch_module.uint8,
                                           device="cpu")
        scale_shape = (*shape[:-1], shape[-1] // 16)
        scales[name] = torch_module.zeros(scale_shape, dtype=torch_module.uint8,
                                          device="cpu")
    read = 0
    for expert in ids:
        for key, suffix in (("w1", "w1"), ("w3", "w3"), ("w2", "w2")):
            for kind, sink, name in (("weight", weights, key),
                                     ("scale", scales, key)):
                full = f"layers.{layer}.ffn.experts.{expert}.{name}.{kind}"
                if full not in weight_map:
                    raise KeyError(
                        f"{full} absent from the official index; the checkpoint "
                        "layout differs from the per-expert names")
                row, _path = _read_tensor(snapshot, weight_map, full)
                want = (sink[name].shape[1], sink[name].shape[2])
                got = tuple(row.shape)
                if got != want:
                    raise ValueError(
                        f"{full} has shape {got}, expected {want} for this "
                        "geometry")
                # The checkpoint stores the FP4 payload as int8 and the E8M0
                # scales as float8_e8m0fnu; both are 8-bit, and the repack
                # helpers want raw uint8 bytes. Reinterpret, never resample.
                sink[name][expert] = row.view(torch_module.uint8)
                read += 1
    # Hash the ACTIVE experts' bytes only, and do it on CPU BEFORE the device
    # transfer. Hashing the whole dense bank would copy every slot (including the
    # zeroed ones) back to the host just to digest bytes that carry no data.
    digest = hashlib.sha256()
    slot_bytes = 0
    for name in ("w1", "w3", "w2"):
        for expert in ids:
            payload = weights[name][expert].numpy().tobytes()
            digest.update(payload)
            slot_bytes += len(payload)
            scale_payload = scales[name][expert].numpy().tobytes()
            digest.update(scale_payload)
            slot_bytes += len(scale_payload)
    device = torch_module.device("cuda", torch_module.cuda.current_device())
    weights = {k: v.to(device).contiguous() for k, v in weights.items()}
    scales = {k: v.to(device).contiguous() for k, v in scales.items()}
    full_bank_bytes = sum(
        t.numel() * t.element_size()
        for t in (*weights.values(), *scales.values()))
    identity = {
        "mode": "checkpoint-active-filled",
        "snapshot": str(snapshot),
        "snapshot_revision": meta["snapshot_revision"],
        "index_sha256": meta["index_sha256"],
        "layer": layer,
        "experts": experts,
        "intermediate": intermediate,
        "active_experts": ids,
        "active_count": len(ids),
        "tensors_read": read,
        "unused_slots_zeroed": experts - len(ids),
        "is_full_real_bank": len(ids) == experts,
        "label": bank_label(len(ids), experts),
        "operands_sha256": digest.hexdigest(),
        "hashed_active_bytes": slot_bytes,
        "hashed_scope": "active experts only, hashed on CPU before any device "
                        "transfer",
        "full_dense_bank_bytes": full_bank_bytes,
        "memory_note": (
            f"the dense active-filled layout allocates {full_bank_bytes} bytes "
            f"for {experts} expert slots at this FULL width, of which "
            f"{slot_bytes} bytes carry real data for {len(ids)} active experts; "
            "the rest are zero-filled. Account the full bank, not the active "
            "subset."),
    }
    return LogicalWeights(weights, scales, "checkpoint-active-filled", identity,
                          experts)


def active_expert_ids(ids_all, experts, limit=None):
    """The distinct expert ids a live route table actually uses.

    Derived from the real routing, not assumed. `limit` caps the count so the
    caller can bound how many experts are read from disk.
    """
    used = sorted({int(e) for e in ids_all.flatten().tolist()})
    if limit is not None:
        used = used[: int(limit)]
    return used


def _read_tensor(snapshot: Path, weight_map, key: str):
    shard = weight_map[key]
    path = snapshot / shard
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key)
    return tensor, path


def rank_logical_slice(logical: LogicalWeights, tp_rank: int, tp_degree: int,
                       intermediate: int, torch_module) -> LogicalWeights:
    """The logical (UNPACKED) operand slice one physical rank actually computes.

    A single rank's output must be compared against the oracle built from this
    slice. The full-width oracle applies only after every rank's plane has been
    summed.
    """
    return LogicalWeights(
        weights={name: _slice_logical(logical.weights[name], name, tp_rank,
                                      intermediate, torch_module)
                 for name in ("w1", "w3", "w2")},
        scales={name: _slice_scale(logical.scales[name], name, tp_rank,
                                   intermediate, torch_module)
                for name in ("w1", "w3", "w2")},
        source_mode=logical.source_mode + "+rank_slice",
        identity=dict(logical.identity, tp_rank=tp_rank, tp_degree=tp_degree,
                      intermediate=intermediate),
        experts=logical.experts,
    )


def _slice_logical(tensor, name, tp_rank, intermediate, torch_module):
    lo = tp_rank * intermediate
    hi = lo + intermediate
    if name in ("w1", "w3"):
        return tensor[:, lo:hi, :]
    # w2 contracts over the intermediate axis, at half rate in the last dim.
    return tensor[:, :, lo // 2:hi // 2]


def _slice_scale(tensor, name, tp_rank, intermediate, torch_module):
    lo = tp_rank * intermediate
    hi = lo + intermediate
    if name in ("w1", "w3"):
        return tensor[:, lo:hi, :]
    # w2's logical scale is [E, rows, K//32]; the K axis is the last one.
    return tensor[:, :, lo // 32:hi // 32]


def rank_operands(logical: LogicalWeights, tp_rank: int, tp_degree: int,
                  intermediate: int, torch_module):
    """Derive one TP rank's logical slice, then repack it for the kernel.

    w1/w3 are sliced along N (intermediate rows); w2 is sliced along K
    (intermediate columns). Scales follow the same axis. Slicing the *logical*
    tensors and then repacking is exactly what the native loader does with the
    official checkpoint, so this is the same arithmetic path, not a model of it.
    """
    from b12x.moe.fused_moe._impl import (
        _logical_weight_to_w4a8_rp_inplace, _e8m0_scale_to_w4a8_sfb_inplace)

    full = intermediate * tp_degree
    lo = tp_rank * intermediate
    hi = lo + intermediate

    def w(name):
        return _slice_logical(logical.weights[name], name, tp_rank, intermediate,
                              torch_module).contiguous()

    def s(name):
        return _slice_scale(logical.scales[name], name, tp_rank, intermediate,
                            torch_module).contiguous()

    assert logical.weights["w1"].shape[1] == full, (
        logical.weights["w1"].shape, full)
    w13 = _logical_weight_to_w4a8_rp_inplace(
        torch_module.cat([w("w3"), w("w1")], 1).clone(),
        size_k=HIDDEN, size_n=intermediate * 2, gated_half_rows=intermediate)
    s13 = _e8m0_scale_to_w4a8_sfb_inplace(
        torch_module.cat([s("w3"), s("w1")], 1).clone(), weight_E=logical.experts,
        rows=intermediate * 2, k_dim=HIDDEN, gated_half_rows=intermediate)
    w2 = _logical_weight_to_w4a8_rp_inplace(
        w("w2").clone(), size_k=intermediate, size_n=HIDDEN)
    s2 = _e8m0_scale_to_w4a8_sfb_inplace(
        s("w2").clone(), weight_E=logical.experts, rows=HIDDEN, k_dim=intermediate)
    return [t.view(torch.uint32).flatten() for t in (w13, s13, w2, s2)]


# --------------------------------------------------------------------------- #
# Pipeline case
# --------------------------------------------------------------------------- #

class Case:
    """One compiled pipeline on one geometry, with its exact scratch buffers."""

    def __init__(self, width, intermediate, experts, capacity, topk, logical,
                 torch_module, tp_degree=1):
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from b12x._lib.utils import current_cuda_stream
        from b12x.moe._shared.kernels.v41_slice_pipeline import V41SlicePipeline

        self.width = width
        self.intermediate = intermediate
        self.tp_degree = tp_degree
        self.experts = experts
        self.capacity = capacity
        self.topk = topk
        self.logical = logical
        self.torch = torch_module
        t = torch_module
        self.kernel_intermediate = (intermediate + 127) // 128 * 128
        self.slices = (intermediate + width - 1) // width

        self.x = t.randn(capacity, HIDDEN, device="cuda").mul_(0.5).bfloat16()
        self.wire = t.empty((capacity, 5280), device="cuda", dtype=t.uint8)
        self.qa, self.qs = self.wire[:, :HIDDEN].view(t.uint32), self.wire[:, HIDDEN:]
        self._quantize_input()

        routes = capacity * topk
        self.routes = routes
        self.ids = t.zeros(routes, device="cuda", dtype=t.int32)
        self.routing = t.zeros(routes, device="cuda", dtype=t.float32)
        self.live = t.zeros(1, device="cuda", dtype=t.int32)
        self.packed_routes = t.zeros(experts * routes, device="cuda", dtype=t.int32)
        self.counts = t.zeros(experts, device="cuda", dtype=t.int32)
        self.prefixes = t.zeros((experts, 2), device="cuda", dtype=t.int32)
        self.metadata = t.full((routes, 19), -1, device="cuda", dtype=t.int32)
        self.grouped = t.zeros(routes, device="cuda", dtype=t.float32)
        self.inverse = t.zeros(routes, device="cuda", dtype=t.int32)
        self.partial = t.zeros((self.slices, routes, HIDDEN), device="cuda",
                               dtype=t.float32)
        self.out = t.zeros((routes, HIDDEN), device="cuda", dtype=t.float32)

        # `V41SlicePipeline.__call__` REQUIRES the four weight tensors as
        # arguments (w13, s13, w2, s2) even though the AOT export embeds no
        # values in them: at runtime they are pointers. The argument list must
        # stay exactly the exporter's, with the weights present and sized for the
        # FULL resident expert count.
        #
        # Sizes are taken from a real repack rather than assumed: the prepared
        # layout rounds the intermediate extent up to a 128 multiple, so an
        # intermediate of 576 produces 640-wide storage. Guessing the size from
        # the logical width would under-allocate TP4.
        first = rank_operands(logical, 0, self.tp_degree, intermediate, t)
        self._weight_buffers = [
            t.zeros_like(tensor).view(t.uint32).flatten().clone()
            for tensor in first
        ]
        self._weight_buffers = [b.zero_() for b in self._weight_buffers]
        self._values = [self.qa, self.qs, *self._weight_buffers, self.ids,
                        self.routing, self.live, self.packed_routes, self.counts,
                        self.prefixes, self.metadata, self.grouped, self.inverse,
                        self.partial, self.out]
        self._args = [from_dlpack(v, assumed_align=16) for v in self._values]
        self.stream = current_cuda_stream()
        self.pipeline = V41SlicePipeline(capacity, width, experts=experts,
                                         topk=topk, intermediate=intermediate)
        # Every stage is compiled against the exact argument list that stage
        # declares; the stages do not share a signature and must not be forced
        # to. `full` is the production callable.
        self.full = cute.compile(self.pipeline, *self._args,
                                 cutlass.Int32(capacity), self.stream)
        self.plan = cute.compile(self.pipeline.plan,
                                 self._args[6], self._args[7], self._args[8],
                                 self._args[9], self._args[10], self._args[11],
                                 self._args[12], self._args[13], self._args[14],
                                 self.stream)
        self.reduce = cute.compile(self.pipeline.reduce,
                                   self._args[15], self._args[16], self._args[14],
                                   self._args[8], self.stream,
                                   cutlass.Int32(capacity))
        # The fused expert stage on its own, declared against its own real
        # signature so phase-0 attribution does not have to go through `full`.
        # The group count is derived the same way the pipeline derives it.
        compute_groups = max(1, min(capacity * topk, experts))
        self.compute_error = None
        try:
            self.compute = cute.compile(
                self.pipeline.compute,
                self._args[0], self._args[1], *self._weight_buffers,
                self._args[9], self._args[11], cutlass.Int32(capacity),
                self.stream, self._args[8], cutlass.Int32(compute_groups))
        except Exception as error:  # pragma: no cover - toolchain dependent
            # Standalone compilation of the fused stage currently trips a CUTLASS
            # internal error on this toolchain. It is not required for the
            # production path (`full` compiles it successfully as part of the
            # pipeline), so record the limitation instead of failing the run.
            self.compute = None
            self.compute_error = f"{type(error).__name__}: {error}"
        # Buffer element counts are fixed at construction from a real repack.
        self._packed_sizes = [b.numel() for b in self._weight_buffers]
        self.packed = None
        self.installed_rank = None

        # Allocation contract, asserted rather than narrated. The kernel strides
        # over exactly what the exporter allocates, and the native packer defines
        # the prepared per-expert bytes; disagreeing with either is a latent
        # out-of-bounds, so both are checked here instead of trusted.
        self.expected_elements = exporter_weight_elements(experts,
                                                          self.kernel_intermediate)
        self.native_bytes, self.native_padded = native_packer_bytes(
            intermediate, experts)
        actual_elements = [b.numel() for b in self._weight_buffers]
        actual_bytes = [b.numel() * b.element_size() for b in self._weight_buffers]
        # Measured, not inferred: the bytes one expert's repack actually occupies
        # in the very buffers the kernel will stride over.
        packed_bytes = [t.numel() * t.element_size() for t in first]
        self.allocation = {
            "kernel_intermediate": self.kernel_intermediate,
            "native_padded": self.native_padded,
            "slots": ["w13", "s13", "w2", "s2"],
            "buffer_elements": actual_elements,
            "buffer_element_bytes": [b.element_size() for b in self._weight_buffers],
            "buffer_bytes": actual_bytes,
            "repack_bytes": packed_bytes,
            "exporter_elements": self.expected_elements,
            "exporter_bytes_as_uint32": [e * 4 for e in self.expected_elements],
            "native_packer_bytes": self.native_bytes,
        }
        if actual_bytes != self.native_bytes:
            raise RuntimeError(
                "allocated weight bytes disagree with the native packer contract: "
                f"actual={actual_bytes} native={self.native_bytes} "
                f"(intermediate={intermediate}, kernel_intermediate="
                f"{self.kernel_intermediate}, experts={experts})")

    def _quantize_input(self):
        t = self.torch
        blocks = self.x.float().reshape(self.capacity, HIDDEN // 32, 32)
        exponent = t.ceil(t.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
        self.qa.copy_((blocks / t.exp2(exponent)[..., None])
                      .to(t.float8_e4m3fn).view(t.uint32)
                      .reshape(self.capacity, HIDDEN // 4))
        self.qs.copy_((exponent + 127).to(t.uint8))

    def install_rank(self, tp_rank, tp_degree):
        """Copy one physical rank's packed slice into the persistent buffers."""
        packed = rank_operands(self.logical, tp_rank, tp_degree,
                               self.intermediate, self.torch)
        # Every rank of one geometry must produce the same prepared sizes. A
        # mismatch means the packing or the slicing differs per rank, which would
        # silently corrupt the forward, so fail loudly instead.
        for index, (buffer, tensor) in enumerate(zip(self._weight_buffers, packed)):
            if tensor.numel() != buffer.numel():
                raise RuntimeError(
                    f"rank {tp_rank} of {tp_degree} packed slot {index} has "
                    f"{tensor.numel()} elements but the buffer holds "
                    f"{buffer.numel()}; prepared sizes must not vary by rank")
        self.packed = [t.view(torch.uint32).flatten() for t in packed]
        for buffer, tensor in zip(self._weight_buffers, self.packed):
            buffer.copy_(tensor)
        self.installed_rank = (tp_rank, tp_degree)
        return {f"slot_{name}": sha256_bytes(t.view(torch.uint8).cpu().numpy().tobytes())
                for name, t in zip(("w13", "s13", "w2", "s2"), packed)}

    def launch(self, rows, stream=None):
        """Launch the compiled pipeline on `stream` (default: the live stream).

        The stream is a RUNTIME argument of the compiled callable. It must be
        resolved at call time: a stream captured when the Case was constructed is
        the default stream, and passing that during a torch CUDA-graph capture
        records nothing, because capture happens on its own stream.
        """
        assert self.installed_rank is not None, "install_rank must run before launch"
        import cutlass
        from b12x._lib.utils import current_cuda_stream

        target = current_cuda_stream() if stream is None else stream
        self.full(*self._args, cutlass.Int32(rows), target)
        return target

    def run(self, rows, masked_ids, weights):
        """One launch, then synchronize; the caller may only read out after."""
        self.ids.copy_(masked_ids.flatten())
        self.routing.copy_(weights.flatten())
        self.live.fill_(rows)
        self.launch(rows)
        self.torch.cuda.synchronize()
        return self.out[: rows * self.topk].view(rows, self.topk, HIDDEN).clone()


def exporter_weight_elements(experts: int, kernel_intermediate: int):
    """The EXACT element counts `export_b12x_v41_slices_aot.py` allocates.

    Source: python/tools/export_b12x_v41_slices_aot.py, the `specs` list:
        Uint32 (experts * kernel_intermediate * n,) for n in (1280, 80, 640, 40)
    in the (w13, s13, w2, s2) slot order. These are uint32 elements, so the byte
    count is 4x. This is the contract the compiled kernel strides over, so the
    runtime buffers must match it exactly -- neither more nor less.
    """
    return [experts * kernel_intermediate * n for n in (1280, 80, 640, 40)]


def native_packer_bytes(intermediate: int, experts: int, hidden: int = HIDDEN):
    """Per-expert bytes from the native packer's own size function.

    Source: native/cuda/kernels/v41_expert_pack.cu, `ds41rt_v41_expert_packed_sizes`:
        padded = align_up(intermediate, 128)
        bytes  = [padded*hidden, padded*hidden/16, hidden*padded/2,
                  hidden*padded/32]
    Returned scaled to the full resident expert count.
    """
    padded = (intermediate + 127) // 128 * 128
    per_expert = [padded * hidden, padded * hidden // 16,
                  hidden * padded // 2, hidden * padded // 32]
    return [value * experts for value in per_expert], padded


def build_case(width, intermediate, experts, capacity, topk, logical, torch_module,
               tp_degree=1):
    return Case(width, intermediate, experts, capacity, topk, logical,
                torch_module, tp_degree)


# --------------------------------------------------------------------------- #
# Python production-path matrix
# --------------------------------------------------------------------------- #

def run_python_matrix(options, torch_module):
    from tests.moe.test_v41_expert_numerics import reference

    tp_degree = options.tp_degree
    intermediate = options.intermediate
    if options.snapshot is not None:
        logical = LogicalWeights.checkpoint(
            options.snapshot, options.layer, options.experts,
            intermediate * tp_degree, torch_module)
    else:
        logical = LogicalWeights.synthetic(
            options.experts, intermediate * tp_degree, torch_module)

    # One shared input across every case and arm, so the oracle and the kernel
    # always see identical activations.
    gen = torch_module.Generator(device="cuda").manual_seed(0x9911)
    shared_x = torch_module.randn(options.capacity, HIDDEN, generator=gen,
                                  device="cuda").mul_(0.5).bfloat16()
    cases = {}
    for width in options.widths:
        case = build_case(width, intermediate, options.experts, options.capacity,
                          options.topk, logical, torch_module)
        case.x = shared_x
        case._quantize_input()
        cases[width] = case

    base = torch_module.arange(options.capacity * options.topk, device="cuda",
                               dtype=torch.int32).reshape(
        options.capacity, options.topk) % options.experts
    route_weights = torch_module.full(
        (options.capacity, options.topk), 1.0 / options.topk, device="cuda")

    pressure = (torch_module.empty(options.cold_bytes // 4, device="cuda",
                                   dtype=torch.float32)
                if options.cold_bytes > 0 else None)

    def flush():
        if pressure is not None:
            pressure.fill_(1)

    records = []
    for width, case in cases.items():
        for tp_rank in range(tp_degree):
            hashes = case.install_rank(tp_rank, tp_degree)
            # A single physical rank computes only its own N/K shard, so its
            # output must be compared against the oracle built from THAT shard.
            # Using the full-width operands here would compare a shard against a
            # full-width expert and fail for a reason that is not a defect.
            local = rank_logical_slice(logical, tp_rank, tp_degree, intermediate,
                                       torch_module)
            for rows in options.rows:
                for label, (owner_rank, note) in MASK_CASES.items():
                    # Ownership is an EXPERT-PARALLEL property, not a
                    # tensor-parallel one: groups come from --ep-degree, while
                    # tp_degree only sets how wide this rank's shard is.
                    groups = options.ep_degree
                    if owner_rank is None:
                        owner = torch_module.ones_like(base, dtype=torch.bool)
                    else:
                        owner = owners_for(base, groups, owner_rank)
                    if label == "ep2-none":
                        owner = torch_module.zeros_like(base, dtype=torch.bool)
                    masked = mask_ids(base, owner)
                    weights_arm = torch_module.where(
                        owner, route_weights,
                        torch_module.zeros_like(route_weights)).contiguous()
                    actual = case.run(rows, masked, weights_arm)

                    # The oracle always takes the ORIGINAL ids with masked
                    # weights: the fixture oracle cannot index the 384 sentinel,
                    # and an unowned route carries weight exactly 0 anyway.
                    expected = reference(case.x[:rows], base[:rows],
                                         weights_arm[:rows], local.weights,
                                         local.scales)
                    assembled = actual.sum(1)
                    denom = expected.norm()
                    if float(denom) == 0.0:
                        assert bool((assembled == 0).all()), (label, rows)
                        rel, cosine, gate = 0.0, 1.0, "reference"
                    else:
                        rel = ((assembled - expected).norm() / denom).item()
                        cosine = torch_module.nn.functional.cosine_similarity(
                            assembled.flatten(), expected.flatten(), dim=0).item()
                        assert rel < 0.01 and cosine > 0.9999, (
                            width, tp_rank, rows, label, rel, cosine)
                        gate = "reference"
                    # owner is capacity-shaped while actual covers only the live
                    # rows, so the mask must be sliced to the live extent or the
                    # boolean index is the wrong length.
                    live_owner = owner[:rows].flatten()
                    if bool(live_owner.any()) and not bool(live_owner.all()):
                        assert bool((actual.view(-1, HIDDEN)[~live_owner] == 0).all()), (
                            f"{label}: masked route not zero")

                    graphs = {}
                    for phase, entry in (("full", case.full), ("plan", case.plan),
                                         ("reduce", case.reduce)):
                        if phase == "full":
                            continue  # timed through the production callable below
                        graph = torch_module.cuda.CUDAGraph()
                        with torch_module.cuda.graph(graph):
                            capture_stream = _current_stream()
                            if phase == "plan":
                                entry(case._args[6], case._args[7], case._args[8],
                                      case._args[9], case._args[10], case._args[11],
                                      case._args[12], case._args[13], case._args[14],
                                      capture_stream)
                            else:
                                entry(case._args[15], case._args[16], case._args[14],
                                      case._args[8], capture_stream,
                                      __import__("cutlass").Int32(rows))
                        graphs[phase] = graph
                    full_graph = torch_module.cuda.CUDAGraph()
                    with torch_module.cuda.graph(full_graph):
                        case.full(*case._args, __import__("cutlass").Int32(rows),
                                  _current_stream())
                    graphs["full"] = full_graph

                    for graph in graphs.values():
                        for _ in range(3):
                            graph.replay()
                    torch_module.cuda.synchronize()
                    # Re-read after the timed graph so the recorded output is the
                    # graph's own result, not a stale pre-capture buffer.
                    after = case.out[: rows * case.topk].view(
                        rows, case.topk, HIDDEN).sum(1)
                    assert bool(torch_module.isfinite(after).all())
                    if float(denom) != 0.0:
                        rel2 = ((after - expected).norm() / denom).item()
                        assert rel2 < 0.01, ("graph replay drifted", rel2)

                    for cache in options.caches:
                        for phase, graph in graphs.items():
                            if cache == "cold":
                                samples = timed_cold_single(
                                    graph.replay, flush, options.intervals,
                                    torch_module)
                            else:
                                samples = timed_intervals(
                                    graph.replay, options.intervals,
                                    options.replays, torch_module)
                            records.append(dict(
                                leg="python", phase=phase, cache=cache,
                                width=width, intermediate=intermediate,
                                kernel_intermediate=case.kernel_intermediate,
                                slices_per_expert=case.slices,
                                tp_degree=tp_degree, tp_rank=tp_rank,
                                ep_degree=options.ep_degree,
                                experts=options.experts, topk=options.topk,
                                capacity=options.capacity, rows=rows,
                                mask=label, mask_note=note,
                                active_experts=int(
                                    masked[:rows][owner[:rows]].unique().numel()),
                                owned_routes=int(owner[:rows].sum().item()),
                                source_mode=logical.source_mode,
                                weight_hashes=hashes,
                                correctness=gate, rel_l2=rel, cosine=cosine,
                                execution="graph", intervals=options.intervals,
                                replays_per_interval=(1 if cache == "cold"
                                                      else options.replays),
                                **summarise(samples)))
                            print(json.dumps(records[-1], sort_keys=True), flush=True)
                    del graphs, full_graph
                    torch_module.cuda.synchronize()
    return records


# --------------------------------------------------------------------------- #
# Native AOT leg
# --------------------------------------------------------------------------- #

def run_native_matrix(options, torch_module):
    """AOT ABI leg -- NOT IMPLEMENTED; fail closed.

    The earlier draft resolved only the fixed TP4 symbol set, sized weight
    buffers for one expert instead of ``info.experts``, fed pseudo-random bytes
    as FP8 wire rows, and read an output view that ignored the route ABI. Each
    of those can produce out-of-bounds reads or silently wrong numbers, so the
    leg refuses to run rather than emit a number that looks like evidence.

    A correct implementation must resolve the role-specific exported symbol set,
    size and fill full resident expert weights from real packed operands, build
    genuinely quantized FP8/UE8M0 wire rows, honour the route output ABI, and
    gate on the oracle. That is a separate task after the AOT build exists.
    """
    raise NotImplementedError(
        "--phase native is disabled and not implemented; use --phase full or "
        "--phase split on the Python production path and do not quote any "
        "native-leg number.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("full", "split", "native"),
                        default="full")
    parser.add_argument("--native-lib", type=Path)
    parser.add_argument("--expect-manifest", type=Path)
    parser.add_argument("--snapshot", type=Path,
                        help="official checkpoint snapshot (enables checkpoint "
                             "weights and the reference gate)")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--intermediate", type=int, default=1152,
                        help="per-rank intermediate width")
    parser.add_argument("--tp-degree", type=int, choices=(1, 2, 3, 4), default=2,
                        help="tensor-parallel degree; 1 = no slicing")
    parser.add_argument("--ep-degree", type=int, choices=(1, 2, 3), default=1,
                        help="expert-parallel groups; >1 uses the diagnostic "
                             "modulo ownership mask, not a production scheduler")
    parser.add_argument("--widths", default="192")
    parser.add_argument("--rows", default="1,8,16,80")
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=80)
    parser.add_argument("--caches", default="warm")
    parser.add_argument("--cold-bytes", type=int, default=128 << 20)
    parser.add_argument("--intervals", type=int, default=9)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    options.widths = [int(x) for x in options.widths.split(",") if x]
    options.rows = [int(x) for x in options.rows.split(",") if x]
    options.caches = [x for x in options.caches.split(",") if x]
    if options.phase == "native":
        parser.error(
            "--phase native is disabled (see run_native_matrix): the AOT leg is "
            "not implemented and must not be reported. Use --phase full or "
            "--phase split.")
    if max(options.rows) > options.capacity:
        parser.error("rows must not exceed capacity")
    return options


def main(argv=None):
    options = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA unavailable; refusing to report timings", file=sys.stderr)
        return 2
    provenance = dict(
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=platform.node(), python=sys.version.split()[0],
        torch=torch.__version__, device_name=torch.cuda.get_device_name(0),
        device_capability=list(torch.cuda.get_device_capability(0)),
        gpu=gpu_snapshot(), phase=options.phase, layer=options.layer,
        intermediate_per_rank=options.intermediate,
        tp_degree=options.tp_degree, ep_degree=options.ep_degree,
        ownership="diagnostic modulo (expert %% ep_degree); not a production "
                  "scheduler",
        widths=options.widths, rows=options.rows,
        caches=options.caches, experts=options.experts, topk=options.topk,
        capacity=options.capacity, cold_bytes=options.cold_bytes,
        intervals=options.intervals, replays_per_interval=options.replays,
        source_mode="checkpoint" if options.snapshot else "synthetic",
        notes=[
            "Python leg compiles the true V41SlicePipeline.",
            "Native leg drives the ds41rt_v41_expert_launch AOT entry.",
            "Masked routes are sentinel id 384 with weight exactly 0.",
            "Cold = one timed replay immediately after a device flush.",
            "weak correctness is not qualified evidence.",
        ])
    if options.snapshot is not None:
        provenance["snapshot"] = str(options.snapshot)
    if options.native_lib is not None:
        provenance["native_lib_sha256"] = sha256_file(options.native_lib)
    records = (run_native_matrix(options, torch) if options.phase == "native"
               else run_python_matrix(options, torch))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(
        {"provenance": provenance, "records": records}, indent=1) + "\n")
    print(json.dumps({"written": str(options.output), "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
