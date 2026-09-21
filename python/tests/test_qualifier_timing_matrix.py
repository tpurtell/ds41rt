"""CPU regression tests for the replicated-native timing matrix harness.

These exercise the real planning and identity helpers the device timing path
depends on, so a matrix cannot silently measure the wrong shape, report an
unverified slice width, or attribute a number to a library/manifest that does
not describe the arm. No CUDA is required: every test runs with
`CUDA_VISIBLE_DEVICES=""` and executes the production helper code.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
QUALIFIER = TOOLS / "qualify_v41_replicated_native.py"


def _load_qualifier():
    pytest.importorskip("torch")
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location("ds41rt_qualifier_timing", QUALIFIER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"qualifier import unavailable: {error}")
    return module


def test_timing_rows_are_validated_sorted_and_deduplicated():
    q = _load_qualifier()
    assert q.parse_timing_rows("64,1,8,16,8") == [1, 8, 16, 64]
    assert q.parse_timing_rows("1") == [1]
    for bad in ("", ",", "0", "-1", "1,two"):
        with pytest.raises(SystemExit):
            q.parse_timing_rows(bad)


def test_rows_must_be_covered_by_a_capacity_and_are_never_clamped():
    q = _load_qualifier()
    grid = q.assert_rows_covered([1, 8, 16, 64], [1, 16, 80])
    assert grid == [(1, 1), (16, 1), (16, 8), (16, 16),
                    (80, 1), (80, 8), (80, 16), (80, 64)]
    # 256/4096 are not covered by a 1/80 capacity list: fail instead of clamping.
    for rows, capacities in (([1, 256], [1, 80]), ([4096], [1, 80])):
        with pytest.raises(SystemExit):
            q.assert_rows_covered(rows, capacities)
    # With the precompiled variants present the same rows are accepted.
    assert (256, 256) in q.assert_rows_covered([256], [1, 80, 256])


def test_width_parser_distinguishes_auto_from_an_explicit_scalar():
    q = _load_qualifier()
    assert q.parse_width("auto") is None
    assert q.parse_width("") is None
    for value in ("64", "128", "192"):
        assert q.parse_width(value) == int(value)
    with pytest.raises(SystemExit):
        q.parse_width("96")


def test_manifest_identity_must_match_the_measured_arm():
    q = _load_qualifier()
    manifest = dict(spark_tp_degree=6,
                    geometry=dict(intermediate=384, experts=q.ARENA_EXPERTS))
    assert q.verify_manifest_identity(manifest, 384, 6) is manifest
    wrong_degree = dict(manifest, spark_tp_degree=3)
    wrong_intermediate = dict(manifest, geometry=dict(intermediate=768, experts=q.ARENA_EXPERTS))
    wrong_experts = dict(manifest, geometry=dict(intermediate=384, experts=128))
    for bad in (wrong_degree, wrong_intermediate, wrong_experts):
        with pytest.raises(SystemExit):
            q.verify_manifest_identity(bad, 384, 6)


def test_tensor_hash_is_content_addressed_and_includes_dtype():
    q = _load_qualifier()
    torch = pytest.importorskip("torch")
    a = torch.arange(8, dtype=torch.int32)
    b = torch.arange(8, dtype=torch.int32)
    c = torch.arange(8, dtype=torch.int32)
    c[0] = 99
    d = torch.arange(8, dtype=torch.int64)
    assert q._tensor_sha256(a) == q._tensor_sha256(b)
    assert q._tensor_sha256(a) != q._tensor_sha256(c)
    assert q._tensor_sha256(a) != q._tensor_sha256(d)


def test_tensor_hash_accepts_bfloat16_without_a_numpy_dtype():
    """The timed activation is bf16; hashing it must not need a bf16 numpy dtype."""
    q = _load_qualifier()
    torch = pytest.importorskip("torch")
    a = torch.arange(16, dtype=torch.bfloat16)
    b = torch.arange(16, dtype=torch.bfloat16)
    c = torch.arange(16, dtype=torch.bfloat16)
    c[-1] = torch.tensor(7.0, dtype=torch.bfloat16)
    assert q._tensor_sha256(a) == q._tensor_sha256(b)
    assert q._tensor_sha256(a) != q._tensor_sha256(c)
    assert isinstance(q._tensor_sha256(a), str) and len(q._tensor_sha256(a)) == 64


def test_routing_hash_combines_ids_and_weights_without_dtype_mixing():
    q = _load_qualifier()
    torch = pytest.importorskip("torch")
    ids = torch.tensor([[377, 378, 379, 380, 381, 382]], dtype=torch.int32)
    weights = torch.full((1, 6), 1.0 / 6)
    same = q._routing_sha256(ids, weights)
    assert same == q._routing_sha256(ids.clone(), weights.clone())
    other_ids = ids.clone()
    other_ids[0, 0] = 383
    assert q._routing_sha256(other_ids, weights) != same
    other_weights = weights.clone()
    other_weights[0, 0] = 0.5
    assert q._routing_sha256(ids, other_weights) != same


def test_seeded_input_repeats_for_the_same_shape_and_seed():
    """The harness seeds one activation per arm; equal shape+seed must be identical."""
    q = _load_qualifier()
    torch = pytest.importorskip("torch")

    def draw(seed, capacity):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(capacity, 5120, generator=generator).mul_(0.5).bfloat16()

    first = draw(0x51, 8)
    assert q._tensor_sha256(first) == q._tensor_sha256(draw(0x51, 8))
    assert q._tensor_sha256(draw(0x52, 8)) != q._tensor_sha256(first)
    # A different capacity is a different shape and therefore a different hash.
    assert q._tensor_sha256(draw(0x51, 16)) != q._tensor_sha256(first)


def test_file_hash_matches_hashlib(tmp_path):
    q = _load_qualifier()
    path = tmp_path / "libds41rt_native.so"
    path.write_bytes(b"ds41rt" * 1000)
    assert q._sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_forward_region_selects_the_compiled_shape():
    q = _load_qualifier()
    torch = pytest.importorskip("torch")

    class FakeInfo:
        topk = 6

    class FakeNative:
        pass

    token = FakeNative()
    token.output = torch.arange(6 * 5120, dtype=torch.float32).reshape(6, 5120)
    token.info = FakeInfo()
    token.token_accumulation = True
    assert tuple(q._native_forward(None, token, 3).shape) == (3, 5120)

    routes = FakeNative()
    routes.output = torch.arange(6 * 6 * 5120, dtype=torch.float32).reshape(6 * 6, 5120)
    routes.info = FakeInfo()
    routes.token_accumulation = False
    assert tuple(q._native_forward(None, routes, 4).shape) == (4, 5120)


class _OwnScopeNames(ast.NodeVisitor):
    """Collect Load/Store names in a function's OWN scope.

    Nested function/lambda/class bodies are separate scopes: their names are
    bound in this scope but their bodies are not scanned, so a nested parameter
    is never reported as an outer free variable.
    """

    def __init__(self):
        self.loaded = set()
        self.bound = set()

    def visit_Name(self, node):
        (self.loaded if isinstance(node.ctx, ast.Load) else self.bound).add(node.id)

    def visit_Import(self, node):
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.bound.add(node.name)

    def visit_Lambda(self, node):
        return None

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.bound.update(node.names)

    def visit_Nonlocal(self, node):
        self.bound.update(node.names)


def test_module_level_functions_define_every_free_name():
    """Executable undefined-name lint: would have caught the `reference` NameError.

    Module-level functions run with module globals plus their own locals; a name
    that is neither (nor a builtin) is a runtime NameError the moment that branch
    executes. Imports inside OTHER functions do not count, which is exactly the
    failure the first GPU preflight hit.
    """
    q = _load_qualifier()
    tree = ast.parse(Path(QUALIFIER).read_text())
    available = set(vars(q)) | set(dir(builtins))
    problems = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        visitor = _OwnScopeNames()
        for statement in node.body:
            visitor.visit(statement)
        params = {a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
        if node.args.vararg:
            params.add(node.args.vararg.arg)
        if node.args.kwarg:
            params.add(node.args.kwarg.arg)
        unresolved = sorted(visitor.loaded - params - visitor.bound - available)
        if unresolved:
            problems.append((node.name, unresolved))
    assert not problems, problems


class _FakeTensor:
    def __init__(self, numel=6 * 5120, itemsize=2):
        self._numel = numel
        self._itemsize = itemsize

    def __getitem__(self, _):
        return self

    def copy_(self, _):
        return self

    def fill_(self, _):
        return self

    def mul_(self, _):
        return self

    def bfloat16(self):
        return self

    def data_ptr(self):
        return 0

    def numel(self):
        return self._numel

    def element_size(self):
        return self._itemsize

    def all(self):
        return self

    def __bool__(self):
        return True


class _FakeGraph:
    def replay(self):
        return None

    def reset(self):
        return None


class _FakeCuda:
    def CUDAGraph(self):
        return _FakeGraph()

    def graph(self, _):
        return _NullContext()

    def synchronize(self):
        return None

    def memory_allocated(self):
        return 0

    def current_stream(self):
        return SimpleNamespace(cuda_stream=0)

    def empty_cache(self):
        return None


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeGenerator:
    def manual_seed(self, _):
        return self


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()
        # Dtype sentinels: the harness only forwards them to mocked calls.
        self.int32 = "int32"
        self.bfloat16 = "bfloat16"
        self.float32 = "float32"

    def Generator(self, device=None):
        return _FakeGenerator()

    def randn(self, *_, **__):
        return _FakeTensor()

    def empty(self, *_, **__):
        return _FakeTensor()

    def isfinite(self, tensor):
        return tensor


def test_run_timing_branch_executes_with_mocks_and_emits_the_record_schema(tmp_path):
    """Execute the timing branch (no CUDA) and assert the JSON record contract.

    Every device-facing helper is replaced by a mock, so this is a real
    execution of `_run_timing`'s control flow and record construction, not an
    AST assertion. It fails if a name is unresolved, an option is misread, or a
    record key disappears.
    """
    q = _load_qualifier()
    manifest_path = tmp_path / "v41_experts.json"
    manifest_path.write_text(json.dumps(dict(
        role="spark_tp6", spark_tp_degree=6,
        geometry=dict(experts=q.ARENA_EXPERTS, intermediate=384),
        variants=[dict(capacity_rows=1, width=64), dict(capacity_rows=80, width=192)])))

    q.report_compiled_widths = lambda *a, **k: {1: 64, 80: 192}
    q.assert_native_role_geometry = lambda *a, **k: None
    q.check = lambda code: None
    q._sha256_file = lambda path: "0" * 64
    q._tensor_sha256 = lambda tensor: "a" * 64
    q._routing_sha256 = lambda ids, routing: "b" * 64
    q._quantize_wire = lambda *a, **k: _FakeTensor()
    q._fill_routing = lambda *a, **k: (_FakeTensor(), _FakeTensor())
    q._route_counts = lambda ids, rw: (18, 18)
    q._time_graph = lambda torch_module, graph, intervals, launches: (
        [1.0] * intervals, [2.0] * intervals)
    q._oracle_compact_mask_checks = lambda *a, **k: (
        dict(rel_l2=0.001, cosine=0.999999, compact_bf16_rel_l2=0.002,
             compact_bf16_cosine=0.9999, mask_zero=True, all_inactive_zero=True),
        _FakeTensor(), _FakeTensor(), _FakeGraph())

    class FakeInfo:
        role = 7
        abi_version = 2
        logical_intermediate = 384
        kernel_intermediate = 384
        topk = 6

    q.Info = lambda: FakeInfo()

    class FakeNative:
        def __init__(self):
            self.info = FakeInfo()
            self.token_accumulation = False
            self.output = _FakeTensor()
            self.storage = _FakeTensor(numel=1024)

        def run(self, rows):
            return None

    q.Native = lambda *a, **k: FakeNative()
    # `C.byref` is only a pointer hand-off to the mocked info call.
    q.C = SimpleNamespace(byref=lambda obj: obj)
    lib = SimpleNamespace(
        ds41rt_v41_expert_info=lambda capacity, out: 0,
        ds41rt_v41_compact_routes_bf16_async=lambda *a, **k: 0,
        ds41rt_v41_compact_tokens_bf16_async=lambda *a, **k: 0)

    options = SimpleNamespace(
        manifest=manifest_path, native_lib=tmp_path / "lib.so", width=None,
        spark_tp=6, tp4_legacy=False, capacities="1,80", timing_rows="1,8",
        active_counts="6", cold_bytes=1024, timing_seed=0x51, repeats=2,
        warm_replays=2, single_replays=2, cold_replays=2)

    records = q._run_timing(options, _FakeTorch(), lib, arena=None,
                            intermediate=384, role=7, gids=[0, 1, 2],
                            base=q.ARENA_EXPERTS - 3, weights={}, scales={},
                            reference=lambda *a, **k: None)

    assert [(r["capacity"], r["rows"], r["active"]) for r in records] == [
        (1, 1, 6), (80, 1, 6), (80, 8, 6)]
    for record in records:
        for key in ("kind", "role", "spark_tp", "tp4_legacy", "capacity", "rows",
                    "active", "abi_version", "valid_route_count",
                    "weighted_route_count", "intermediate", "kernel_intermediate",
                    "width", "timing_seed", "input_sha256", "route_sha256",
                    "lib_sha256", "manifest_sha256", "manifest_role",
                    "manifest_spark_tp_degree", "scratch_bytes",
                    "compact_output_bytes", "rel_l2", "cosine",
                    "kernel_only", "kernel_plus_compact"):
            assert key in record, (key, sorted(record))
        assert record["kind"] == "native_timing"
        assert record["lib_sha256"] == "0" * 64
        assert record["input_sha256"] == "a" * 64
        assert record["route_sha256"] == "b" * 64
        assert record["valid_route_count"] == 18
        assert record["weighted_route_count"] == 18
        assert record["width"] == {1: 64, 80: 192}[record["capacity"]]
        for variant in ("kernel_only", "kernel_plus_compact"):
            for key in ("warm_single_device_us", "warm_amortized_device_us",
                        "warm_single_host_us", "warm_amortized_host_us",
                        "cold_device_us", "warm_single_samples_us",
                        "warm_amortized_samples_us", "cold_samples_us"):
                assert key in record[variant], (variant, key)
            assert len(record[variant]["warm_single_samples_us"]) == 2
            assert len(record[variant]["warm_amortized_samples_us"]) == 2


def test_fill_routing_skips_inactive_slots_with_the_unassigned_sentinel():
    """active<6 must not dispatch: slots >= active carry SENTINEL with weight 0."""
    q = _load_qualifier()
    torch = pytest.importorskip("torch")
    gids = [0, 1, 2, 3, 4, 5, 383]
    base = q.ARENA_EXPERTS - len(gids)
    for active, rows in ((3, 2), (6, 2)):
        ids = torch.full((rows, 6), 7, dtype=torch.int32)
        rw = torch.full((rows, 6), 3.0)
        oracle_ids, routing = q._fill_routing(
            torch, ids, rw, rows, base, gids, active, device="cpu")
        # Oracle-space ids stay valid indices; only the weight decides participation.
        assert int(oracle_ids.min()) >= 0 and int(oracle_ids.max()) < len(gids)
        assigned = ids[:, :active]
        assert bool(((assigned >= base) & (assigned < base + len(gids))).all())
        assert bool(torch.allclose(
            rw[:, :active], torch.full((rows, active), 1.0 / active)))
        if active < 6:
            assert bool((ids[:, active:] == q.SENTINEL).all())
            assert bool((rw[:, active:] == 0).all())
        else:
            assert int((ids == q.SENTINEL).sum().item()) == 0
        valid, weighted = q._route_counts(ids[:rows], rw[:rows])
        assert valid == rows * active and weighted == rows * active


def test_active_six_is_unchanged_and_route_counts_expose_the_ambiguity():
    q = _load_qualifier()
    torch = pytest.importorskip("torch")
    ids = torch.full((1, 6), q.SENTINEL, dtype=torch.int32)
    rw = torch.zeros((1, 6))
    q._fill_routing(torch, ids, rw, 1, q.ARENA_EXPERTS - 7,
                    [0, 1, 2, 3, 4, 5, 383], 6, device="cpu")
    assert bool((ids != q.SENTINEL).all())
    assert q._route_counts(ids, rw) == (6, 6)
    # An assigned id with a zero weight is the exact ambiguity the record exposes.
    ids2 = torch.tensor([[377, 378, q.SENTINEL]], dtype=torch.int32)
    rw2 = torch.tensor([[0.5, 0.0, 0.0]])
    assert q._route_counts(ids2, rw2) == (2, 1)
