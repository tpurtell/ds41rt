# Six-node execution readiness (1RTX6 TP3EP2 / 2RTX6 TP2EP3 / 2RTX6 TP3EP2)

Status: **CPU readiness package. Templates and dry-run tests only; no execution.**
Arm identity is **NOT ready** until the actual captures in section 5 exist. The
per-rail RDMA binding review is complete (section 4): the software mechanism is
source-verified, and the only **PROVISIONAL** item left is the physical A↔B
reachability for moa `.12` plus forwarding the device map on both roles. No
GPU/remote/build/default change was made, the frozen runner is untouched, and no
network fix is applied.

## 1. Frozen identity and the G3 reference

| artifact | sha256 |
| --- | --- |
| `scripts/qualify-ds41-tp-ep-e2e.py` (frozen four-Spark runner) | `e197146aa6c98de0c8c731a698de69cbb0f3dadf3edb6fe20e3690e82a6008c4` |
| `scripts/fixtures/tp-ep-e2e-corpus.jsonl` (v3) | `823f203ca1ce054b352a7e97f8acf6102116d5fa89b7226f167b26914778b750` |
| `scripts/fixtures/tp-ep-e2e-corpus-v4-six.jsonl` (v4 six) | `99034171dba1fe8e6c64e41f9cbf07ab960d6c77932079f44a70c789cf836d85` |
| `scripts/qualify-ds41-tp-ep-e2e-six.py` | `551773a98fdf6a819d96e17611b047dc9481165b7ac894e4dc69bca8c58c948a` |

G3 reference (four-Spark TP2EP2, matched N20/KV 5,037,542,400): corpus
`823f203c…b750`, runner `e197146a…08c4`, metadata `2898eb23…`; 372/372 rows
passed the objective, 0 runtime/SSE errors, 0 `cudaGraphInstantiate` failures,
3 failures all `greedy output changed across repeats` (code, topic, mixed).
That artifact is **not accepted** by the gate (repeat inconsistency is tier A).

## 2. Runtime facts (authoritative)

- **One image, one library, all arms.** The arm64 expert image is
  `sha256:dab2b979509231568ff375228fee9814d68cf5fdeb27edc77955cd35c1680234`
  with the frozen library `d453812c5c53e916fe8ea0b54d72a51c19b0df8e4db8ce9b7c1ea5b967b1f513`
  (16,418,544 B); it carries roles 1/5/6, so there is **no separate TP2 vs TP3
  image**. moa is already CDI-running; rhea's image import is in progress after
  the f6 release (`runs/tp-ep-preflight/six-node-prep.md`).
- **Isolated candidate launch, not release defaults.** The G1/G2/G3 candidate
  ran from the staged diag-v2 bytes with coordinator `--listen 127.0.0.1:18000`
  and experts `--listen 0.0.0.0:29441`
  (`runs/tp-ep-preflight/logs/g1-manual-argv-tp4v1-20260920T075941Z.txt`). Do
  **not** blindly use `./run.sh` release defaults: that path can pull release
  artifacts and start default services on port 8000. Reuse the candidate stage
  (`candidate-diag-v2-stage.sh` / `wip-process.sh run candidate-coordinator-18000`
  and `candidate-expert-29441`, no builds), and point the six runner at
  `--base-url http://127.0.0.1:18000`.
- **Global Spark budget.** The global knob is the **minimum over all six hosts:
  109119320064 bytes** (the original four-host value). rhea/moa allow about
  67 MB more (`109186371584`), but that new-host value must **not** be applied to
  all six; per-host overrides are supported if the memory review approves them.
  The candidate argv already carries `--device-budget-bytes 109119320064`.

## 3. Arms, staged files, and exact launch

| arm | RTX | topology | RTX local / remote | KV | draft | qualification |
| --- | --- | --- | --- | --- | --- | --- |
| `1rtx6-tp3ep2` | 1 | TP3×EP2 | 0 / 40 | auto (single) | 5 | quality + memory |
| `2rtx6-tp2ep3` | 2 | TP2×EP3 | 20 / 20 | 5,037,542,400 | 7 | paired speed+quality |
| `2rtx6-tp3ep2` | 2 | TP3×EP2 | 20 / 20 | 5,037,542,400 | 7 | paired speed+quality |

Staged files (copy into `runs/tp-ep-six/`, then fill the REQUIRED tokens):

- `scripts/fixtures/tp-ep-six/site-1rtx6-tp3ep2.config`
- `scripts/fixtures/tp-ep-six/site-2rtx6-tp2ep3.config`
- `scripts/fixtures/tp-ep-six/site-2rtx6-tp3ep2.config`
- `scripts/fixtures/tp-ep-six/startup-1rtx6-tp3ep2.template.json`
- `scripts/fixtures/tp-ep-six/startup-2rtx6-tp2ep3.template.json`
- `scripts/fixtures/tp-ep-six/startup-2rtx6-tp3ep2.template.json`

Filling every `REQUIRED_*` token is a **manual** gate: the arm gate cannot
detect placeholder strings because the argv and metadata agree on them. Only the
standalone arm fails numerically until the memory evidence is real, so 915 must
treat the `REQUIRED_*` scan as mandatory for all three arms. Do not call an arm
identity ready before its actual capture is recorded.

## 4. Per-rail RDMA binding — review resolved (software); one physical fact pending

Ranks 0–3 are `ostrich`, `dodo`, `emu`, `kiwi`; rank 4 is `rhea`, rank 5 is
`moa`. Both rhea/moa report two active 200 Gb/s HDR HCAs
(`runs/tp-ep-preflight/six-node/rdma-172.22.2.5.txt`, `.6`) with IPv4 RoCE v2 at
capture **gid index 5** (rhea A `10.55.0.5` / B `10.55.0.11`; moa A `10.55.0.6`
/ B `10.55.0.12`). Ethernet IP MTU: Fabric A and B `mtu=9000` UP on both hosts
(`six-node-prep.md`); the older `six-node-readiness.md` 1500 reading is
historical. This is link/Ethernet-MTU state only, **not** RDMA validation, and it
is a different protocol domain from Verbs `active_mtu`: `active_mtu` is an IB
enum (`IBV_MTU_1024/2048/4096`, 4096 normally the maximum and the usual reported
value), never "9000". Do not require 9000 in `ibv_devinfo`.

**How a rail is selected (source-verified).**

- **No GID-index knob.** `select_rc_gid` (`native/src/ds41rt_native.cc:299-346`)
  returns the first RoCE v2 + IPv4-mapped GID on the chosen device. Each rhea/moa
  rail device exposes exactly one such GID (index 5), so once the *device* is
  right the GID is deterministic; `modify_rc_qp_to_rtr` uses
  `path_mtu = port_attr.active_mtu` (`:370`), i.e. the IB enum value (max 4096
  normally) — the two endpoints must report the same `active_mtu`, and Ethernet
  IP MTU 9000 is a separate, lower-layer fact.
- **Device selection is the supported lever.**
  `DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=local-ip=rdma_device,...`
  (`verbs.rs:48,51-92`) selects the HCA from the control connection's local IP
  (accept `verbs.rs:3303`, connect `:4180`); port via
  `DS41RT_VERBS_APP_IB_PORT_NUM` (default 1, `:4782`). Build it with
  `discover_rdma_device_map()` (`scripts/phase0-spark-tcp-bench.sh:1147-1168`).
- `scripts/run-tp-ep-native-candidate.sh:356-362` forwards that map (plus IB
  port/lanes) into **both** roles; the **release** `run.sh` forwards only
  `CUDA_VISIBLE_DEVICES`, `DS41RT_RELEASE_CONFIG_SHA256`, `RUST_LOG`
  (`run.sh:383,425`), so a release-path six-rank run has **no** per-host device
  binding today. Use the candidate launcher, or extend `run.sh` (not done here).

**Rail facts (16:03 captures, authoritative; the 15:29 `rhea-extra.txt` B reading
is stale).**

| Host | A | B | route to `.22` |
| --- | --- | --- | --- |
| raptor | `10.55.0.22` (`enp1s0np0`) | none | local (A-only host) |
| rhea | `.5` (`enp1s0f0np0`) | `.11` (`enP2p1s0f0np0`) | A, src `.5` |
| moa | `.6` (`enp1s0f0np0`) | `.12` (`enP2p1s0f0np0`) | B, src `.12` |

**Why the staged `.5` (rhea) + `.12` (moa) mix is intentional.** `run.sh:301-303`
makes `SPARK_n_LANE_A` the coordinator's dial target, so each lane is the rail
that host's own route to `.22` uses; the worker's inbound and outbound then stay
on one rail:

| lane | SYN | reply to `.22` | network requirement |
| --- | --- | --- | --- |
| rhea `.5` | raptor A → rhea A | rhea A → raptor A | fabric A only |
| moa `.12` | raptor A → moa B | moa B → raptor A | **A↔B connectivity** |
| moa `.6` (alt) | raptor A → moa A | moa B → raptor A | B→A only (asymmetric) |

So `.12` is the symmetric choice for moa given its route, and **the only
unresolved item is whether raptor's fabric-A frame can reach moa's fabric-B port
(and the reply back)**. If ops confirms A↔B, `.12` stands with no config change.
If A↔B is isolated, the no-new-NIC fix is a host policy on moa forcing
`10.55.0.22` (or `10.55.0.0/24`) out `enp1s0f0np0`, paired with lane `.6`; a
raptor fabric-B path is the alternative. Neither is authorized here.

Per-host device map (every host, both roles; raptor HCA name to be captured):
`raptor:10.55.0.22=<hca>`, `ostrich:.1=rocep1s0f0,.7=roceP2p1s0f0`,
`dodo:.2=rocep1s0f0,.8=roceP2p1s0f0`, `emu:.3=rocep1s0f0,.9=roceP2p1s0f0`,
`kiwi:.4=rocep1s0f0,.10=roceP2p1s0f0`, `rhea:.5=rocep1s0f0,.11=roceP2p1s0f0`,
`moa:.6=rocep1s0f0,.12=roceP2p1s0f0`.

**Preflight commands (read-only, before any timing).**

```bash
ip -br -4 addr; rdma link show
ibv_devinfo -v | grep -E 'hca_id|GID\[|active_mtu|link_layer'
ip route get 10.55.0.22            # rhea -> A/.5, moa -> B/.12 expected
ip route get <peer_lane_addr>      # raptor -> enp1s0np0 for every peer
ip -4 -o addr show | awk '{print $2, $4}'   # build/verify the ip=device map
ping -M do -s 8972 <moa .12> <moa .6> <rhea .5>   # Ethernet IP MTU 9000 + A/B reachability
# `active_mtu` printed above is the IB enum (max 4096); IP MTU 9000 is `ip -br link`.
# native endpoints report identity, e.g. RDMA RC endpoint created ... gid_index=5
```

**Required match facts (owner: ops/915).**

1. A↔B topology: single L2, routed, or isolated (decides `.12` vs the moa
   policy-route + `.6` path).
2. raptor HCA name(s) and that `.22` maps to the fabric-A device.
3. rhea `ip route get 10.55.0.22` → A, moa → B, matching the lane device.
4. Exactly one IPv4-mapped RoCE v2 GID per rail device at index 5 on all six
   hosts (only rhea/moa captures exist today).
5. NIC/interface IP MTU 9000 on the chosen Ethernet rail (checked with
   `ip -br link`; the 8972-byte DF ping proves it end to end), **and** Verbs
   `active_mtu` (IB enum) **matching on both endpoints** of every QP. Do not
   require 9000 in `ibv_devinfo`: the usual `active_mtu` is `4096` (enum max).
6. `DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP` forwarded on **both** roles (the
   release launcher does not do this today); `--device=/dev/infiniband` present
   (it is).

Owner note: fa33 owns the actual DF-ping/HCA/GID capture (raptor `.22` → moa
`.12`/`.6`, rhea `.5`), read-only, no network changes.

**Not proven:** no probe/ping/MTU/RC handshake was run here; A↔B reachability,
raptor's HCA identity and end-to-end six-rank RoCE remain unverified. Do not call
an arm network-ready before items 1–6 are recorded.

## 5. Unresolved inputs (REQUIRED, owned by 915)

1. Confirm the actual per-host captures for all six ranks: `MemTotal`,
   `MemAvailable`, MTU, GID/source, and an end-to-end RC probe.
2. rhea image import is done; complete the rhea CDI start and static checks
   (moa is already running). One image/library covers roles 1/5/6, so no
   separate TP2/TP3 image is needed.
3. d129 delivered the fa33 network values: DF 8972 bidirectional PASS on the new
   A/B rails, Ethernet MTU 9000 with verbs `active_mtu 4096`, and the device map
   `raptor 10.55.0.22=mlx5_0`, `rhea .5=rocep1s0f0 / .11=roceP2p1s0f0`,
   `moa .6=rocep1s0f0 / .12=roceP2p1s0f0` (see §4). The map union is passed on
   the command line; IP reachability is still not RC proof, so the launcher
   captures the selected device/GID per rank at startup and fails closed if a
   rank logs none. Use MGMT ssh (`--spark-ssh 4=172.22.2.5 --spark-ssh 5=172.22.2.6
   --spark-ssh-bind <local-mgmt-ip>`) only if the fabric control path is flaky.
4. Actual accounted resident/workspace/ring per host, coordinator
   `device_free_bytes`, and the independent log review flag.
5. Power caps (400 W assumed), release identity + all-manifest hashes, and the
   decision on the G3 three repeat failures.

## 6. Exact invocations, 372 scope, and dry-run alignment

Launch through the six-rank gate `scripts/run-tp-ep-six-candidate.sh`, which
enforces exactly six ranks, an approved `TP2EP3`/`TP3EP2` topology and
`SPARK_DEVICE_BUDGET_BYTES <= 109119320064` on the staged fixture config, then
delegates to the existing `scripts/run-tp-ep-native-candidate.sh`. The gate
exits `2` before the launcher runs, so a rejected fixture sends no remote
command; the ceiling is a six-rank qualification input, not a runtime default,
and the four-Spark path is untouched. The launcher never builds, never
creates/renames/removes a container, never calls `run.sh`/`docker run`, and
never stops anything; `plan` changes nothing and `start` is L3-gated. It starts
the coordinator, waits for the published placement plan, starts all six experts,
waits for every rank's readiness line, writes the coordinator-side placement ACK
(dual), starts the coordinator (single), waits for health/native model, sends
**one bounded first request** to initialize the lazy per-connection RDMA
endpoints, and only then captures the selected GID/device per rank (fail-closed
for six ranks if a rank emits none). `plan` also lists the honest
`prestage-prerequisite` commands: `start` stages nothing and only verifies the
destination, so the built library must already be installed on every host. `--dspark-draft-limit` is explicit (7 dual / 5 single), the device
map union is forwarded to both roles, and the pinned SparkInfer source is
exported for both roles. `--placement-directory` is a coordinator-only hidden
flag (the daemon's `expertd-native` has no such flag); worker placement enters
through `--first-layer` from the published plan.

```bash
# 1. Copy the staged config/template to runs/tp-ep-six/ and set the image tags;
#    keep SPARK_DEVICE_BUDGET_BYTES=109119320064 (minimum over all six hosts).
# 2. Plan (dry run, changes nothing): exactly 1 coordinator + 6 expert commands,
#    world 6, ranks 0..5, coordinator 127.0.0.1:18000, experts 0.0.0.0:29441.
scripts/run-tp-ep-six-candidate.sh plan \
  --config scripts/fixtures/tp-ep-six/site-2rtx6-tp3ep2.config --rtx-gpus 2 \
  --host-artifact-root runs/tp-ep-six

# 3. Start (915 only, after the L3 grant). The union map comes from the fa33 file
#    (all local IPs; the native parser selects this host's entry) and is forwarded
#    to both roles. The 1RTX arm uses --rtx-gpus 1 --first-layer 0.
DS41RT_TPEP_L3_GRANT=1 \
DS41RT_SPARKINFER_SOURCE_DIR=/workspace/ds41rt/third_party/sparkinfer \
PYTHONPATH=/workspace/ds41rt/third_party/sparkinfer \
scripts/run-tp-ep-six-candidate.sh start \
  --config scripts/fixtures/tp-ep-six/site-2rtx6-tp3ep2.config --rtx-gpus 2 --first-layer 20 \
  --host-device-map '10.55.0.22=mlx5_0,10.55.0.5=rocep1s0f0,10.55.0.11=roceP2p1s0f0,10.55.0.6=rocep1s0f0,10.55.0.12=roceP2p1s0f0' \
  --tcp-timing 1 \
  --coordinator-cuda-visible-devices <GPU-UUID,GPU-UUID> --run-id <id> \
  --host-artifact-root runs/tp-ep-six --qualified \
  --role-manifest <spark-role-manifest.json> \
  --expect-coordinator-lib-sha256 <hex> --expect-spark-lib-sha256 <hex> \
  --expect-coordinator-daemon-sha256 <hex> --expect-spark-daemon-sha256 <hex>
# MGMT control fallback for the new hosts, only if the fabric ssh path flaps:
#   ... --spark-ssh 4=172.22.2.5 --spark-ssh 5=172.22.2.6 --spark-ssh-bind <local-mgmt-ip>

# 4. status/stop only ever touch the named candidate processes.
scripts/run-tp-ep-six-candidate.sh status --config scripts/fixtures/tp-ep-six/site-2rtx6-tp3ep2.config
scripts/run-tp-ep-six-candidate.sh stop   --config scripts/fixtures/tp-ep-six/site-2rtx6-tp3ep2.config
```

`--tcp-timing` is an explicit **startup diagnostic** forwarded to both roles as
`DS41RT_PROTOCOL_V2_TCP_TIMING`; there is no production default and it is not
enabled silently. The transport emits the client/server GID lines only under
that flag (and only after a connection), so a six-rank run without it fails
closed at the GID capture rather than claiming RC proof from IP reachability.
`COORDINATOR_GPU_UUID` is intentionally absent from the staged configs: the
launcher takes the ordered physical GPU UUIDs on the command line
(`--coordinator-cuda-visible-devices`), never from a hidden lookup.

Run the arms (match `--base-url` to the isolated candidate port):

```bash
python scripts/qualify-ds41-tp-ep-e2e-six.py validate
python scripts/qualify-ds41-tp-ep-e2e-six.py selftest

python scripts/qualify-ds41-tp-ep-e2e-six.py run --arm 1rtx6-tp3ep2 \
  --startup-metadata runs/tp-ep-six/startup-1rtx6-tp3ep2.json \
  --base-url http://127.0.0.1:18000 --output runs/tp-ep-six/1rtx6-tp3ep2.json
python scripts/qualify-ds41-tp-ep-e2e-six.py run --arm 2rtx6-tp2ep3 \
  --startup-metadata runs/tp-ep-six/startup-2rtx6-tp2ep3.json \
  --base-url http://127.0.0.1:18000 --output runs/tp-ep-six/2rtx6-tp2ep3.json
python scripts/qualify-ds41-tp-ep-e2e-six.py run --arm 2rtx6-tp3ep2 \
  --startup-metadata runs/tp-ep-six/startup-2rtx6-tp3ep2.json \
  --base-url http://127.0.0.1:18000 --output runs/tp-ep-six/2rtx6-tp3ep2.json

python scripts/qualify-ds41-tp-ep-e2e-six.py compare \
  --control runs/tp-ep-six/2rtx6-tp2ep3.json \
  --candidate runs/tp-ep-six/2rtx6-tp3ep2.json \
  --output runs/tp-ep-six/compare-2rtx6-paired.json
python scripts/qualify-ds41-tp-ep-e2e-six.py compare \
  --candidate runs/tp-ep-six/1rtx6-tp3ep2.json \
  --output runs/tp-ep-six/compare-1rtx6-standalone.json   # no --control
```

Each arm must produce **372 rows**: 4 frozen cases × `(1+2+4+8+16) × 3` repeats.
`compare` rejects a smaller matrix (canonical-scope gate) even if the arm's own
matrix field agrees with it, and rejects a run missing a case. The runner
records the v4 corpus hash and the frozen runner hash in every output.

## 7. Memory gates (no all-pass assumption)

The Spark host reserve is **20 GiB (21,474,836,480) on every config**, a
different domain from the coordinator headroom.

- Global budget: use `109119320064` (minimum over all six hosts). rhea/moa may
  allow more, but the global knob must not carry the new-host value; per-host
  overrides need memory-review approval.
- Per host: `device_budget_bytes <= mem_total_bytes - 20 GiB`,
  `20 GiB <= mem_available_bytes <= mem_total_bytes`, and
  `accounted_resident_bytes + accounted_workspace_bytes + accounted_ring_bytes
  <= device_budget_bytes`, with a real `source_log`.
- Resident floor per rank (exact TP bytes/layer × remote layers):
  - TP3 all-40 (1RTX6): `2,406,481,920 × 40 = 96,259,276,800`
  - TP3 20-layer (2RTX6 TP3EP2): `48,129,638,400`
  - TP2 20-layer (2RTX6 TP2EP3): `72,194,457,600`
- With the global 109,119,320,064 budget, the TP3 all-40 resident leaves
  ~12.86 GB for workspace + ring per host. **No host may be assumed to pass**;
  measure all six. A 19 GiB reserve fails; a zero or under-sized resident fails.
- Coordinator headroom: **2 GiB applies only to the 1RTX single-RTX arm**
  (`device_free_bytes` is RTX device free, never host free, recorded
  separately). The 2RTX planner headroom is **800 MiB**, not 2 GiB, and is not a
  Spark-domain requirement.
- The standalone gate qualifies only after an independent review sets
  `logs_independently_reviewed: true`; otherwise it reports
  `standalone_memory_schema_ok_review_required`. Quality without memory evidence
  is `standalone_quality_pass_memory_review_required`. Paired arms carry no
  memory qualification claim.

## 8. Captures, cold restart, evidence, acceptance

- Captures: candidate argv + container/process ids, the six host MTU/GID/source
  and probe logs, per-rank memory numbers, `nvidia-smi` power/free series.
- Cold restart: stop the candidate fleet, relaunch from the same staged bytes,
  re-verify identity and run at least the readiness probe plus one bounded case
  before the full matrix. Evidence dir: `runs/tp-ep-six/<arm>/`.
- Acceptance: paired `text_exact` (or `text_divergent_review_qualified` with
  per-output review evidence) over the canonical 372-row scope with matched
  identity/controls and identical family manifests; any greedy-repeat
  inconsistency is a tier A failure (G3 had three). Standalone must be
  `standalone_memory_qualified`. `1rtx6` vs any `2rtx6` arm stays a declared
  forbidden comparison.
- A failing arm may still report **diagnostic** TPS, but every such number must
  be labelled **not accepted**; it is never a qualification claim.

## 9. Handoff checklist for 915

1. Copy the staged configs and templates to `runs/tp-ep-six/`; fill every
   `REQUIRED_*` token and the image tags.
2. Capture the six host facts and the live RC probe; resolve the PROVISIONAL moa
   rail with d129 before timing.
3. Launch the isolated candidate (18000/29441), capture argv, cold-restart once,
   then run the three `run` invocations and two `compare` invocations.
4. Confirm 372 rows per arm and hand the three arm JSONs plus two compare JSONs
   to the parent; label any diagnostic-only numbers as not accepted.
