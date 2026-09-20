# Native Spark TP×EP topology, ownership wire and transport

Status: **implemented in `rust/crates/ds41rt-transport/`, CPU tests only**. No GPU,
native/FFI, container, service, WIP build or workload action was performed for
this document. Every runtime/live claim is marked pending and must be produced by
the owning kernel/daemon agent.

Owner: transport agent (`rust/crates/ds41rt-transport/**`, this file). The
architecture record `docs/tp-ep-architecture-audit.md` is **pending** and is the
authority for rank placement, per-rank budgets and kernel geometry; this document
covers only the transport contract it consumes.

## 1. Scope

Replicated expert groups (`EP`) keep a complete copy of all 384 routed experts in
every group and tensor-parallel shard each group internally. A route is assigned
to exactly one group; every physical rank of that group receives it. This is
**not** classic disjoint `ep_moe` expert sharding and must not be confused with
it.

This change adds the topology, the ownership wire encoding and the generic
N-rank transport to `ds41rt-transport` only:

- `V41SparkTopology`: validated `TP×EP` mapping, physical rank count and
  topology-bound executor identities.
- Native owner route words in the existing 12-byte route entry plus the
  `1 << 18` request flag.
- Ownership-aware request parse/validate/unpack on the worker side.
- Generic 2/3/4/6-rank constructors for `V41Tp4ChunkReceiver`, `V41Tp4Planes`,
  `V41Tp4Roce`, `V41Tp4Tcp` and the internal `LocalTp4Client`.
- Generic protocol flag-validator rules for the new flag.

All pre-existing public constructors and behaviour are preserved; the legacy
TP4/TP2 paths are byte-compatible and the new contract is opt-in.

## 2. Supported topologies

`physical_rank = group * TP + tp_rank`, therefore `group = physical_rank / TP`
and `tp_rank = physical_rank % TP`. Group-major ordering makes every group a
contiguous peer range. Unsupported pairs are rejected with an explicit error;
no dummy or placeholder rank is created.

| Layout | `TP` | `EP` | World | Executor ids | Status |
| --- | ---: | ---: | ---: | --- | --- |
| `TP2EP1` | 2 | 1 | 2 | `5, 6` | native, new |
| `TP3EP1` | 3 | 1 | 3 | `7, 8, 9` | native, new |
| `TP4EP1` | 4 | 1 | 4 | `1, 2, 3, 4` | legacy production, preserved |
| `TP2EP2` | 2 | 2 | 4 | `17..20` | native, primary new target |
| `TP3EP2` | 3 | 2 | 6 | `11..16` | native, hardware-unqualified |
| `TP2EP3` | 2 | 3 | 6 | `21..26` | native, hardware-unqualified |

`V41SparkTopology::new(tp, ep)` accepts only these six pairs. The executor-id
namespaces are pairwise disjoint, including the two six-rank layouts, so a
receiver built for one topology rejects a response whose `executor_id` belongs to
another. This binds topology and rank, not checkpoint or deployment identity.

## 3. Wire contract

### 3.1 Request flag

`V41_NATIVE_GROUP_REQUEST_FLAG = 1 << 18` selects the
ownership contract. Bit 18 was verified unused for protocol flags repository
wide. The flag is:

- legal only together with `EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16` and the
  optional debug-checksum flag;
- mutually exclusive with the paired EXL3 flag `1 << 17`;
- **never carried in a response**. Worker response headers and response chunks
  strip it exactly as they already strip the paired flag, and the generic
  response validator rejects it if present.

### 3.2 Owner route word

The 12-byte route entry is unchanged. The route word is the existing
`expert_id: u32` field, reinterpreted only when the flag is set:

```
bits  0..8   true routed expert id, 0..=383
bits  9..11  one-hot owning-group bitmap, exactly one bit set
             (bit 9 + group for group 0..=2)
bits 12..31  reserved, must be zero
```

- `V41NativeOwnerRouteWord::encode` requires `expert_id < 384` and
  `owner < 3`, and writes the one-hot bit `9 + owner`.
- `V41NativeOwnerRouteWord::decode(word, group_count)` additionally requires
  `1 <= group_count <= 3`, zero reserved bits, `expert_id < 384`, **exactly one**
  owner bit set and that bit's group index `< group_count`. Empty or multi-bit
  owner fields are rejected, so a route can never be owned by two groups.
- `V41NativeOwnershipBatch` requires every occurrence of an expert in a batch to
  carry the same owner and rejects conflicting ownership.

Route order, the exact FP32 gate weights, the canonical six unique experts per
token and the hidden payload are unchanged; only the owner bits are added, so
the frame keeps its canonical size.

### 3.3 Request/response identity

Response matching continues to use `request_id`, `placement_version`,
`layer_id`, geometry and `executor_id`. `placement_version` is **not** rewritten:
it remains the existing request identity. Topology binding is carried by the
disjoint executor-id namespace (and by the request flag on the request side).
The same encoded request is sent to every physical rank.

Because the owner bits only need `EP` and the compact contract, two topologies
with the same `EP` (for example `TP2EP1` vs `TP3EP1`) produce parse-identical
requests. They are separated at the transport boundary: the receiver is built
with the topology's canonical executor ids, so the other topology's workers
cannot satisfy response coverage. Deployment configuration and the launch
handshake are responsible for giving every worker the matching topology; a
mismatched worker may compute, but its response is rejected before it can reach
the assembler. `TP3EP2` vs `TP2EP3` (both world 6) are separated the same way,
by `11..16` vs `21..26`.

## 4. Worker unpack semantics

`V41BackboneRequest::parse_native_group(frame, max_rows, topology)` validates the
flag, the topology group count, reserved bits, per-row canonical shape and
batch-wide ownership consistency.

`copy_native_group_routes_into(ids, weights, group)` fills caller-owned kernel
descriptor arrays for the worker's local group:

| Route | `ids[i]` | `weights[i]` |
| --- | --- | --- |
| owned by `group` | true expert `0..383` | original exact FP32 gate weight |
| owned by another group | `384` sentinel | `0.0` |

The existing native kernel maps an out-of-range route id to inverse `-1` and
contributes zero, so every physical rank returns a compact BF16 `[M, 5120]`
partial of the same shape and **no CPU computation fallback exists**. Empty
groups (a group that owns no route in a batch) produce sentinel-only descriptors
and a zero partial; they still participate in coverage.

The coordinator sums the physical rank planes in ordered FP32, adds the shared
expert once and rounds once to final BF16. This transport adds **no**
Spark-to-Spark reduction; every physical rank plane returns directly to the
coordinator.

## 5. Public API

```rust
use ds41rt_transport::v41_expert::{
    V41SparkTopology, V41NativeOwnerRouteWord, V41NativeOwnershipBatch,
    V41_NATIVE_GROUP_REQUEST_FLAG, V41_NATIVE_UNASSIGNED_EXPERT_ID,
    V41_NATIVE_INACTIVE_OWNER, V41_ROUTED_EXPERTS, V41_MAX_NATIVE_GROUPS,
    V41BackboneRequest, V41Tp4ChunkReceiver, V41Tp4Planes, V41Tp4Roce, V41Tp4Tcp,
};
```

Topology:

```rust
let topology = V41SparkTopology::new(2, 2)?;         // TP2 × EP2
topology.tp(); topology.ep(); topology.group_count();
topology.world_size();                                // TP * EP
topology.group(global_rank)?;                         // global_rank / TP
topology.tp_rank(global_rank)?;                       // global_rank % TP
topology.global_rank(group, tp_rank)?;
topology.executor_id(global_rank)?;                   // canonical, topology-bound
topology.executor_ids();                              // Vec<u64>, rank order
topology.rank_of_executor(executor_id);               // Option<usize>
```

Request construction (coordinator; `owners` is caller-provided, e.g. from
`ds41rt_core::replicated_expert_schedule`, `owners[expert_id]` is `0..EP` or
`INACTIVE_REPLICATED_EXPERT_GROUP` = 255):

```rust
request.with_native_group_owners(&owners, topology)?;
```

Worker:

```rust
let native = V41BackboneRequest::parse_native_group(frame, capacity, topology)?;
native.is_native_group();
native.native_topology();
native.copy_native_group_routes_into(&mut ids, &mut weights, local_group)?;
// local_group = topology.group(global_rank)?
let response = native.response(topology.executor_id(global_rank)?, &partials)?;      // full plane
let response = native.response_chunk(id, start_row, &partials, &mut idx, budget)?;   // chunked
```

Transport:

```rust
let mut transport = V41Tp4Roce::new_topology(topology, &peers, capacity, config)?;
transport.topology();      // Some(topology)
transport.world_size();    // TP * EP
transport.execute(&request, sink).await?;   // or dispatch().await?.receive(sink).await
transport.reset_connections();
```

Generic non-ownership constructors for validated 2/3/4/6-rank worlds:
`V41Tp4Roce::new_ranks`, `V41Tp4Tcp::new_ranks`,
`V41Tp4ChunkReceiver::new_ranks`, `V41Tp4Planes::new_ranks`; the
topology-bound `V41Tp4Tcp::new_topology` mirrors `V41Tp4Roce::new_topology`.
Legacy `V41Tp4Roce::new`/`new_tp2`,
`V41Tp4ChunkReceiver::new`/`new_tp2`, `V41Tp4Planes::new` and
`v41_spark_executor_id(2|4, rank)` are unchanged.

Flag and topology must agree, and dispatch fails closed before any I/O:
`new_topology` transports accept only requests carrying the native group flag,
while `new`/`new_ranks` transports reject flagged requests. There is no silent
canonical fallback for an EP1 topology-bound transport; legacy EP1 canonical
traffic uses `new`/`new_tp2`/`new_ranks`.

`V41Tp4ChunkReceiver::received_rows()` keeps its legacy `[u32; 4]` shape;
`received_rows_slice()` returns one counter per physical rank and
`world_size()` reports the expected rank count.

## 6. Daemon integration sequence

1. Resolve the topology once at startup and validate `TP * EP == SPARK_COUNT`
   in config/launcher (owned by the daemon/config agents).
2. Coordinator: build the canonical `ExpertProtocolV2Request`, apply
   `with_native_group_owners(&assignment, topology)`, construct
   `V41Tp4Roce::new_topology(topology, &peers, capacity, config)`, then
   `execute`/`dispatch` exactly as today. All ranks receive the same request.
3. Worker: parse with `parse_native_group(frame, capacity, local_topology)`,
   unwrap with `copy_native_group_routes_into(ids, weights, local_group)`, run the
   existing kernel, respond with `topology.executor_id(global_rank)`.
4. Cancellation/failure: unchanged. Dropping a pending wave resets every
   connection (`reset_connections`), including 3- and 6-rank transports.

`owners` must be unique per expert per batch. The scheduler guarantees an active
expert has exactly one owner; transport rejects inactive or out-of-range owners
for routed experts and conflicting owners across the batch.
`with_native_group_owners` runs the full canonical shape validation (six unique
experts per row, canonical route spans, finite non-negative weights, expert range)
**before** rewriting any route word, so a rejected assignment leaves the request
byte-identical and it can be corrected and retried; the successful path performs
no heap allocation beyond what the caller already owns.

## 7. Tests

CPU-only, no GPU, no network peers required at construction time:

- topology mapping for all six layouts, group-major round trip, disjoint
  namespaces across layouts, rejection of every unsupported pair;
- every legal owner word round trip plus reserved-bit, expert-range and
  `owner >= EP` rejection;
- encode/unwrap exactly-once across every supported topology, including an
  inactive group, with sentinel id `384` and zero weight for unowned routes and
  a scratch-reuse re-run;
- inactive/missing/out-of-range owners, conflicting ownership, duplicate
  experts, negative weights, reserved bits and flag mismatch rejection, with a
  regression proving a rejected assignment leaves the request byte-identical and
  retryable;
- topology/flag fail-closed dispatch on both RoCE and TCP (legacy transport
  rejects a flagged request; topology-bound transport rejects an unflagged one)
  before any socket I/O;
- canonical and paired EXL3 consumers rejecting the new flag; the generic flag
  validator bounding the flag and the response validator rejecting it;
- 3- and 6-rank receiver/plane coverage: out-of-order arrival, duplicates,
  truncation, stale `request_id`, foreign-topology executor ids, wrong world
  size, row overlap and bad final-chunk markers;
- generic `V41Tp4Roce`/`V41Tp4Tcp` `new_ranks`/`new_topology` construction bounds
  and topology binding, plus a live-loopback six-rank TP2EP3 TCP round trip that
  covers every rank/group.

Run isolated (never share the default target directory with WIP builds):

```bash
CARGO_TARGET_DIR=runs/tp-ep-transport/target \
  cargo test --manifest-path rust/Cargo.toml -p ds41rt-transport
```

### 7.1 Repair note

`rust/crates/ds41rt-transport/src/v41_expert/tcp_tests.rs` carried a committed
syntax error from an earlier `timing` refactor
(`fn config() -> TcpTransportConfig { timing: false, ...}` duplicated the struct
literal fragment), which prevented the whole test target from compiling. It is
inside this module's exclusive ownership and was fixed minimally so the legacy
TCP tests compile and run unchanged.

## 8. Evidence and open items

- The kernel-side claim that an out-of-range route id yields inverse `-1` and a
  zero contribution is the kernel agent's qualification, not established here.
- Live 2-RTX + 4-Spark TP2×EP2 behaviour, reductions, coordinator assembly and
  performance are PENDING; this document contains no measured number.
- Six-rank (`TP3EP2`, `TP2EP3`) hardware is not connected and remains
  hardware-unqualified; transport support is contractual only.
- `docs/tp-ep-architecture-audit.md` remains the authority for rank placement and
  per-rank budgets; if it disagrees with §2, the audit wins and this contract
  must be revised.
