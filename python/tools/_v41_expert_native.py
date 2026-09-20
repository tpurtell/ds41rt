"""ctypes bindings for official TP4 and local dSpark expert qualification."""

import ctypes as C
import torch

P = C.c_void_p
I = C.c_int32
U = C.c_uint32
L = C.c_uint64


class Info(C.Structure):
    _fields_ = (
        [
            (n, U)
            for n in [
                "abi_version",
                "role",
                "experts",
                "hidden_size",
                "logical_intermediate",
                "kernel_intermediate",
                "topk",
                "capacity_rows",
            ]
        ]
        + [("scratch_bytes", L)]
        + [
            (n, I)
            for n in [
                "max_rows",
                "rows_padded",
                "max_tasks",
                "max_phys_tiles",
                "max_active_clusters",
            ]
        ]
        + [("input_dtype", U)]
    )


class Launch(C.Structure):
    _fields_ = (
        [("tensors", P * 44)]
        + [
            (n, I)
            for n in [
                "num_tokens",
                "max_rows",
                "scatter_rows",
                "rows_padded",
                "max_tasks",
                "max_phys_tiles",
                "max_active_clusters",
            ]
        ]
        + [("stream", P)]
    )


assert C.sizeof(Info) == 64 and C.sizeof(Launch) == 392


SPARK_TP_PREFIX = {2: "ds41rt_v41_spark_tp2_expert_", 3: "ds41rt_v41_spark_tp3_expert_"}
# Only the five per-family launch-ABI entry points are namespaced. The packer
# size query and the packer itself are canonical symbols shared by every family.
FAMILY_SYMBOLS = frozenset({
    "ds41rt_v41_expert_info",
    "ds41rt_v41_expert_initialize",
    "ds41rt_v41_expert_bind_scratch",
    "ds41rt_v41_expert_initialize_scratch_async",
    "ds41rt_v41_expert_launch",
})


def namespaced_symbol(name, prefix):
    """Family symbol for the launch-ABI entry points; canonical for everything else."""
    if prefix != "ds41rt_v41_expert_" and name in FAMILY_SYMBOLS:
        return name.replace("ds41rt_v41_expert_", prefix, 1)
    return name


def expert_symbol_prefix(*, local=False, tp2=False, spark_tp=None):
    """Native symbol prefix for one expert family.

    `spark_tp=2|3` selects the replicated-group Spark TP2/TP3 AOT families
    (roles 5/6); `local`/`tp2` keep the historical RTX families unchanged, and
    the default is the canonical TP4 family. At most one family may be selected.
    """
    assert sum(bool(x) for x in (local, tp2, spark_tp is not None)) <= 1
    if spark_tp is not None:
        assert spark_tp in SPARK_TP_PREFIX, spark_tp
        return SPARK_TP_PREFIX[spark_tp]
    if tp2:
        return "ds41rt_v41_tp2_expert_"
    if local:
        return "ds41rt_v41_local_expert_"
    return "ds41rt_v41_expert_"


def library(path, *, local=False, tp2=False, spark_tp=None):
    prefix = expert_symbol_prefix(local=local, tp2=tp2, spark_tp=spark_tp)
    lib = C.CDLL(path)
    for name, args in {
        "ds41rt_v41_expert_info": [I, C.POINTER(Info)],
        "ds41rt_v41_expert_initialize": [I, C.POINTER(P)],
        "ds41rt_v41_expert_bind_scratch": [P, P, L, C.POINTER(P)],
        "ds41rt_v41_expert_initialize_scratch_async": [P, P, L, P],
        "ds41rt_v41_expert_launch": [P, C.POINTER(Launch)],
        "ds41rt_v41_expert_packed_sizes": [U, C.POINTER(L)],
        "ds41rt_v41_pack_expert_async": [C.POINTER(P), C.POINTER(P), U, P],
        "ds41rt_v41_compact_routes_bf16_async": [P, P, U, P],
        "ds41rt_v41_compact_tokens_bf16_async": [P, P, U, P],
    }.items():
        selected = namespaced_symbol(name, prefix)
        fn = getattr(lib, selected)
        if selected != name:
            setattr(lib, name, fn)
        fn.argtypes = args
        fn.restype = I
    if prefix != "ds41rt_v41_expert_":
        lib.ds41rt_v41_expert_output_kind = getattr(lib, prefix + "output_kind")
    return lib


def check(code):
    assert code == 0, code


class Native:
    def __init__(self, lib, capacity, weights, wire, ids, routing, *, coordinator=False, full_backbone=False, tp2=False, spark_tp=None, storage=None):
        assert spark_tp in (None, 2, 3)
        assert sum((coordinator, full_backbone, tp2, spark_tp is not None)) <= 1
        self.lib = lib
        self.info = info = Info()
        self.handle = P()
        check(lib.ds41rt_v41_expert_info(capacity, C.byref(info)))
        expected = ((5, 384, 5120, 1152, 1152, 6, capacity, 7) if spark_tp == 2 else
                    (6, 384, 5120, 768, 768, 6, capacity, 7) if spark_tp == 3 else
                    (3, 384, 5120, 1152, 1152, 6, capacity, 7) if tp2 else
                    (0, 128, 5120, 2304, 2304, 3, capacity, 1)
                    if coordinator else (2, 384, 5120, 2304, 2304, 6, capacity, 7)
                    if full_backbone else (1, 384, 5120, 576, 640, 6, capacity, 7))
        assert (
            info.abi_version,
            info.role,
            info.experts,
            info.hidden_size,
            info.logical_intermediate,
            info.kernel_intermediate,
            info.topk,
            info.capacity_rows,
            info.input_dtype,
        ) == (info.abi_version, *expected)
        assert info.abi_version in (2, 3)
        self.token_accumulation = info.abi_version == 3
        if self.token_accumulation:
            query = lib.ds41rt_v41_expert_output_kind
            query.argtypes = [I, C.POINTER(U)]
            query.restype = I
            kind = U(99)
            check(query(capacity, C.byref(kind)))
            assert kind.value == 1
        check(lib.ds41rt_v41_expert_initialize(capacity, C.byref(self.handle)))
        if storage is not None:
            assert storage.dtype == torch.uint8 and storage.is_cuda and storage.is_contiguous()
            assert storage.numel() >= info.scratch_bytes
        self.storage = (torch.empty(info.scratch_bytes, device="cuda", dtype=torch.uint8)
                        if storage is None else storage)
        self.args = Launch()
        slots = self.args.tensors
        check(
            lib.ds41rt_v41_expert_bind_scratch(
                self.handle, self.storage.data_ptr(), self.storage.numel(), slots
            )
        )
        w13, s13, w2, s2 = [x.data_ptr() for x in weights]
        for slot, pointer in [
            (22, w13),
            (23, s13),
            (24, w2),
            (25, s2),
            (26, s13),
            (27, s2),
            (28, slots[34]),
            (29, slots[34]),
            (30, w13),
            (31, s13),
            (32, w2),
            (33, s2),
            (38, slots[37]),
            (39, slots[37]),
            (0, wire.data_ptr()),
            (1, ids.data_ptr()),
            (2, routing.data_ptr()),
        ]:
            slots[slot] = pointer
        check(
            lib.ds41rt_v41_expert_initialize_scratch_async(
                self.handle,
                self.storage.data_ptr(),
                self.storage.numel(),
                torch.cuda.current_stream().cuda_stream,
            )
        )
        for name in [
            "max_rows",
            "rows_padded",
            "max_tasks",
            "max_phys_tiles",
            "max_active_clusters",
        ]:
            setattr(self.args, name, getattr(info, name))
        offset = slots[41] - self.storage.data_ptr()
        output_rows = capacity if self.token_accumulation else capacity * info.topk
        self.output = (
            self.storage[offset : offset + output_rows * 5120 * 4]
            .view(torch.float32)
            .reshape(output_rows, 5120)
        )

    def run(self, rows):
        self.args.num_tokens = rows
        self.args.scatter_rows = rows * self.info.topk
        self.args.stream = torch.cuda.current_stream().cuda_stream
        check(self.lib.ds41rt_v41_expert_launch(self.handle, C.byref(self.args)))
