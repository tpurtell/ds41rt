# DS41RT v9 release execution checklist (CPU-only pre-stage)

**STATUS: PARTIALLY EXECUTED — build, distribution and functional validation are
done and recorded; publication and the experimental canonical qualifier remain.**
Boxes are checked only where a recorded artifact exists, never from intention.

Companion documents: `docs/release-v9-notes.md` (final release notes) and
`docs/release-v9-checklist.md` (qualification gates). This file is the
**execution/runbook** layer: exact paths and known blockers.

## Actual v9 state (2026-09-21)

- Clean build rc0, engine revision `5d0d209509bf1f26731bc588dfe8dc72d1373ad0`;
  bare label, version `v9`, roles `tp2;tp3;tp6`; coordinator local image
  `sha256:786d1d67…` (amd64), Spark local image `sha256:9e8c8248…` (arm64).
- Docker-wide local wipe done on all seven hosts; RDMA distribution to the four
  build Sparks plus `rhea`/`moa`; dist `sha256sum -c` rc=0 over 569 files.
- Functional validation PASS on all four production geometries
  (`RELEASE-GATE-CHECKLIST-ACTUAL.md`): api-smoke and constrained rc=0 everywhere;
  content 30/30 (TP6 1x), 10/10 elsewhere. `drop-page-cache` on `kiwi`/`rhea` is a
  recorded operational prerequisite. TP4 1 RTX dispatcher placement is unknown.
- **Publication done:** `./push-containers.sh v9` published `v9` and `latest` (same
  digest per role) and anonymous fresh-`DOCKER_CONFIG` pulls returned rc=0 with
  matching digests (coordinator index `sha256:786d1d67…`, spark-expert manifest
  `sha256:f0c67407…`; `v9-build/PUBLICATION-EVIDENCE.txt`). `ds41rt.config` already
  points at `:v9`.
- **Remaining:** the GitHub release (separate step) and, separately, the
  experimental canonical six-rank qualifier, which is out of scope and not run. No
  final-image performance campaign was made; the report numbers are validated
  warm-candidate measurements.

## 0. Lease, authority and standing constraints

- [ ] Parent explicitly transfers the hardware/lease to the release executor;
      benchmark owner confirms its exclusive campaign is stopped.
- [ ] No implementation churn during the benchmark campaign — only this new
      scoped checklist file and the frozen drafts may change.
- [ ] Roles to build: `DS41RT_RELEASE_SPARK_TP_ROLES=tp2;tp3;tp6`, with the
      implicit default **TP4** path (never labeled).
- [ ] Publishing to GHCR/GitHub only after every final gate below passes.

## 1. Source freeze and clean worktree

- [ ] Snapshot the user's worktree before anything:
      `git diff --binary > <archive>/user-worktree.patch`,
      `git status --porcelain -uall > <archive>/user-status.txt`,
      `git show HEAD:TO_DELETE_SCAFFOLDING.md > <archive>/deleted/...` (same for
      `TO_SHIP_V1.md`), and copy the modified `ds41rt.config` revision.
- [ ] **Preserve the user's deletions UNCHANGED in the working tree.** Do not
      restore them, do not delete anything else, and do not commit unrelated
      changes to satisfy a clean tree. The isolated release worktree is built
      from the **intended tracked selection** only; the user's deletions and
      config edits stay in the main working tree, archived as evidence.
- [ ] Commit/adopt explicitly only the intended TP6 files, including the
      currently untracked required sources:
      `native/src/v41_spark_tp6_experts.cc`,
      `native/tests/v41_expert_pack_tp6_selftest.cc`,
      `python/tests/test_v41_spark_tp6_contract.py`,
      `examples/configs/tp6ep1-native.config`,
      `scripts/fixtures/tp-ep-six/site-2rtx6-tp6ep1.config`,
      `scripts/tests/test_stop_host_selection.py`.
      A worktree contains **only tracked files**, so an uncommitted `.cc` breaks
      the build.
- [ ] Create a **fresh** isolated worktree at the release commit on root NVMe
      under `~/.cache/ds41rt/builds/<name>` (never `/mnt/scratch`).
      **Do not touch** the baseline-owned worktree
      `~/.cache/ds41rt/builds/v9-baseline-src`.
- [ ] In the worktree: `git submodule update --init --checkout` for
      sparkinfer/xgrammar (and xgrammar's dlpack), then
      `verify-sparkinfer-source.py` and `verify-xgrammar-source.py` against the
      locks.
- [ ] Confirm `git status --porcelain` is empty in the build tree and note the
      exact commit; the image `org.opencontainers.image.revision` must be that
      bare commit (no `-dirty-`).
- [ ] Ancestry recorded with `git merge-base --is-ancestor main dev` and
      `git rev-list --left-right --count main...dev` (at review time:
      fast-forward, `0 6`). Never infer from ahead/behind display.

## 2. Archives and evidence before the wipe (outside Docker)

- [ ] Evidence root created outside the repo and outside tmpfs, e.g.
      `<archive>/v9-<UTC>/`, with subdirs `git/ inventory/ images/ build/
      validation/ performance/`.
- [ ] `git/`: rev-parse of HEAD/dev/main/release branches, `status -uall`,
      `submodule status`, tags, commit manifest.
- [ ] `inventory/`: local and per-Spark `docker ps -a`, `images --digests`,
      `volume ls` (the authoritative named-volume list), `system df`,
      per-image `inspect` labels.
- [ ] `images/`: optional `docker image save` rollback archives of the v8
      coordinator and arm64 Spark images (coordinator ≈ 10.45 GB uncompressed);
      record sizes and confirm free space first.
- [ ] Named-volume archives via read-only container mount
      (`docker run --rm -v <vol>:/v:ro -v <archive>:/b alpine tar -C /v -czf ...`)
      or an explicit written decision that they are disposable caches, plus a
      content manifest.
- [ ] Copy the existing baseline evidence into the archive (it lives outside
      the repo today): `~/.cache/ds41rt-v9-baseline/clean-{1,2}/`,
      `evidence-{1,2}/`, `config/`, `provenance/`, `worker-alloc/`,
      `manifest-baseline-v8.json`, `report-baseline-v8.md`, `HANDOFF.md`.
- [ ] Record the baseline provenance explicitly: the metadata `release: v9`
      correctly means "the v9 campaign's baseline", and the measured images are
      the v8 published pair
      `ghcr.io/tpurtell/ds41rt-{coordinator,spark-expert}:v8` @ `5a56f0e`. This
      is **correct metadata, not a manifest defect**; cite it as "v8 baseline
      measured for the v9 campaign" — never as v9 results.

## 3. Docker-wide local wipe (literal, all projects)

- [ ] Re-verify nothing owned by another lead started since the inventory.
- [ ] All local containers removed; other projects' containers are in scope per
      the user's instruction.
- [ ] `docker image prune -a -f` with **no filter exclusions** (deletes the NGC
      base; accepted).
- [ ] `docker volume prune -f` **and** explicit `docker volume rm` for every
      inventoried named volume (prune is anonymous-only by default and misses
      volumes still referenced by stopped containers); assert
      `docker volume ls` empty.
- [ ] `docker builder prune -af` (a system prune alone leaves the builder
      cache).
- [ ] `docker system df` shows ~0 images/containers/cache; `docker info`
      healthy.
- [ ] Remote cleanup scoped to DS41RT artifacts on the Sparks (do not wipe
      other fleets there).
- [ ] Fresh `DS41RT_RELEASE_REMOTE_BUILD_DIR` so seed staging is new.

## 4. Base re-pull gate (not a credential blocker)

- [ ] Pinned base recorded:
      `nvcr.io/nvidia/pytorch@sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`;
      its anonymous manifest availability is already verified. The missing
      `nvcr.io` credential is **not a blocker by itself** and no credential is
      to be added unless a real access failure occurs.
- [ ] Before the wipe, the lease owner runs the **actual anonymous pull
      availability test** on the target docker daemon (not credential-file
      inspection): pull the pinned base anonymously on raptor and on the Spark
      seed and confirm it lands. This is the gate a lease owner executes later;
      the CPU reviewer does not execute it.
- [ ] If — and only if — that anonymous pull fails, escalate rather than
      silently adding credentials.
- [ ] Re-pull the pinned base on raptor and the Spark seed, then re-tag
      `nvcr.io/nvidia/pytorch:26.05-py3` (or pass `--build-arg BASE_IMAGE=<pinned>`).

## 5. Build-recipe coverage audit (maps to actual recent failures)

Pin the recipe revisions as part of the evidence: the clean build must record
the exporter/CMake/Dockerfile blob hashes, and the resulting binaries must be
compared against the current warm candidate — **a warm candidate is not final
proof**. Current local artifacts are stale v8-era (`dist/` built 2026-09-17;
coordinator native `60fbb788…`, expert native `af21f3dd…`) and will be replaced.

- [ ] **Coordinator full features, RDMA ON for experts.** Confirmed in the
      recipe: `build-release-artifacts.sh` sets `-DDS41RT_ENABLE_RDMA=ON`
      unconditionally; the coordinator native library links `libibverbs.so.1`
      and the release image installs `rdma-core`/`ibverbs-providers`/
      `libibverbs1`. Verify after rebuild that `libibverbs.so.1` is NEEDED and
      resolvable at runtime in both roles.
- [ ] **Expert role RDMA + NCCL.** Expert build sets `nccl=ON`; the arm64
      expert native library links `libibverbs.so.1` and `libnccl.so.2`. Verify
      both resolve inside the Spark image (NCCL is provided by the NGC base).
- [ ] **ARM daemon packaged.** `build.sh` compiles the daemon on the Spark seed
      and the export copies `rust/target/release/ds41rt` from the seed. Verify
      the Spark image's `/opt/ds41rt/bin/ds41rt` is `ELF aarch64` (current
      stale artifact is), that it is the same revision as the coordinator
      label, and that the build host is genuinely aarch64.
- [ ] **Correct architecture / Python runtime for the libs.** Both daemons link
      `libpython3.12.so.1.0` (verified in the current stale artifacts: x86-64
      coordinator, aarch64 expert). Confirm the NGC 26.05 base still ships
      Python 3.12 in both architectures; a base roll would break every daemon
      start with a missing `libpython3.12.so.1.0`.
- [ ] **`libcute_dsl_runtime.so` present.** The release Dockerfile verifies each
      EXL3 family against `python3 -m cutlass.cute.export.aot_config --libdir`
      and the entrypoint prepends it to `LD_LIBRARY_PATH`. Verify the runtime
      exists in both images after rebuild (locally the file is supplied by
      `nvidia-cutlass-dsl==4.6.2` under `nvidia_cutlass_dsl/cu13/lib/`).
- [ ] **HF host mount correct.** `run.sh` mounts `$hf_home` read-only at
      `/root/.cache/huggingface`; verify `HF_HOME`/`hf_home` resolution, that
      the snapshot directory and symlink layout exist on every Spark, and that
      no container bundles weights. Engram tables come from host storage.
- [ ] **Host network and devices.** `run.sh` uses `--network host` and
      `--device=/dev/infiniband` with `--ipc host --ulimit memlock=-1:-1`; the
      Spark launch adds `--gpus all`. Verify verbs devices, GIDs and the
      single-rail (LANE_A) peer list on every rank before declaring readiness.
- [ ] **TP6 role artifacts present in the image.** Verify
      `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6` and that the built-role
      manifest (`V41_EXPERT_TP_AOT.json`) is derived from the actual export.
- [ ] Record the new library hashes after the clean build; do not reuse the
      stale `dist/` hashes as v9 identity.

## 6. Release image validation and headline reruns

- [ ] `./run.sh --dry-run` for the default (TP4EP1) and the TP6EP1 configs.
- [ ] Image identity on every host: bare release revision, `v9` version,
      matching SparkInfer revision, matching role label, optional source
      manifest. `run.sh` validates the SparkInfer revision but **not** the
      engine revision against `git rev-parse HEAD` — add an explicit manual
      check or the assertion itself.
- [ ] Smoke: `/health`, `/v1/models`, one completion, constrained/structured
      output, streaming, long-prefill, cancellation/restart.
      Note: the constrained-smoke short-budget thinking-default mismatch is
      owned by the benchmark owner and is fixed by explicitly disabling
      thinking; **preserve the recorded failure**, do not silently drop it.
- [ ] Rerun the headline content for **1 RTX and 2 RTX**, official TP4 default
      profile (`profile=None`, adaptive, draft widths 5/7 — no automatic
      built-in profile, so no built-in-profile-vs-TP6 confound), plus the TP6EP1
      arm.
- [ ] Single-RTX placement recorded as the diagnostic **5 local / 35 remote**
      geometry. The **40-loaded / 35-dispatched** worker allocation is recorded
      **legacy behavior, not a baseline defect to fix**. Keep the two
      comparisons separate:
      1. **TP6 frame (the saving):** TP6 loading **35** versus a *hypothetical
         TP6 loading 40* — both use the TP6 per-rank shard weight
         (35 x 1,203,240,960 B vs 40 x 1,203,240,960 B), a per-rank weight
         saving of **6.016 GB**. This is **not** a comparison against the TP4
         baseline, whose per-rank shard weight is different.
      2. **TP4 baseline frame (separate):** the published TP4 baseline loads 40
         and dispatches 35 at `first_layer=0`, leaving 5 x 2,005,401,600 B =
         **10.027 GB/rank** loaded and never dispatched; historical, recorded
         for context only.
      Record whether the TP6 35-layer load fits the budget used; do not change
      scope to require a baseline allocation fix.
- [ ] Baseline comparison: use only the **clean** v8 baseline
      (`clean-{1,2}/decode.json`; code 133.71/156.03, weighted 92.30/112.59,
      counting 160.53/218.33; 30/30 cells, 15 assessed passes per arm). The two
      contaminated arms (17:19-17:37 UTC) are quarantined and must not be
      cited. Apply the recorded thresholds honestly: the 5% noise band is **not
      universal** (2x weighted repeat spread was 7.1%; per-content-type spreads
      13-26%), so 5-10% is unresolved, not noise.
- [ ] Every timed sample, warmup, concurrency, KV encoding, sampling and
      acceptance recorded; three samples per cell for headline numbers.
- [ ] Archive `dist/SHA256SUMS`, per-host image labels, and the sanitized build
      argv/options log (never a raw `bash -x` trace with secrets).

## 7. Publication path and its blockers (identified locally; no credential-file or secret inspection)

Local tooling check (no action taken):

- `docker`, `gh`, `jq`, `curl`, `sha256sum`, `rsync` all **present**.
- GitHub CLI is authenticated as `tpurtell` with the `repo` and
  `write:packages` scopes needed for GHCR push (scope check only; no secrets
  were read).
- Git pushes over SSH with `SSH_AUTH_SOCK` set; verify agent reachability to
  `github.com` at publish time without printing any secret. Do not inspect
  credential files going forward.
- **Blocking gaps to resolve before publication:**
  1. the actual **anonymous base-pull availability test** (section 4) must run
     on the target daemon before the wipe; a missing `nvcr.io` credential is
     not itself a blocker and no credential is to be added unless that pull
     genuinely fails;
  2. GHCR push is done from raptor (coordinator) and from `SPARK_0_HOST`
     (ostrich) — confirm `SPARK_0_HOST` is in the built-and-distributed set;
     `rhea`/`moa` are not in the default build/distribution list and must pull
     or receive the image;
  3. `push-containers.sh` also moves `:latest`, which has no separate rollback
     boundary — confirm the release owner accepts that;
  4. package visibility owner action is **fallback only** (packages are already
     public); schedule it only if the anonymous v9 pull fails.
- [ ] Anonymous verification after publication: token from
      `https://ghcr.io/token?scope=repository:tpurtell/<repo>:pull&service=ghcr.io`
      then `GET /v2/tpurtell/<repo>/manifests/v9` returns HTTP 200 with
      `docker-content-digest` equal to the recorded digest (accept OCI index for
      the coordinator and Docker schema-2 for the arm64 Spark image).

## 8. README, report links and final integration

- [ ] README: image references, headline labels and pull lines to `:v9`; link
      the v9 report set.
- [ ] `docs/release-v9-notes.md`: replace every `<PENDING>` with recorded
      evidence; state the official-default status; keep the estimate-vs-measured
      labels honest.
- [ ] Publish the reports with raw-record digests and per-configuration
      qualification; link `report-baseline-v8.md` and the v9 report from the
      notes.
- [ ] Remove/refresh the README delta and stale references:
      `docker/README.md` "v6 images" sentence; dangling
      `TO_DELETE_SCAFFOLDING.md` references (`DEVELOPER.md`, the tracked loader
      example, four `docs/ds41-*.md` files); `docs/HANDOFF_NEXT_SESSION.md`
      links the deleted `TO_Ship_V1.md`.
- [ ] Move this checklist's sibling review artifact
      `runs/release-v9-prep-review.md` out of the repo into the archive.
- [ ] `ds41rt.config` updated to `:v9` and committed with the notes; `main`
      fast-forwarded to the qualified `dev` head and `release/v9` created only
      after publication and hardware validation.

## 9. Owner handoff summary (concise)

| Item | Path / value | Owner |
|---|---|---|
| Draft notes (frozen) | `docs/release-v9-notes.md` | reviewer -> executor |
| Draft qualification gates | `docs/release-v9-checklist.md` | executor |
| Execution runbook (this file) | `docs/release-v9-execution-checklist.md` | executor |
| Review artifact (move to archive) | `runs/release-v9-prep-review.md` | executor |
| Clean v8 baseline numbers | `~/.cache/ds41rt-v9-baseline/clean-{1,2}/decode.json` | benchmark owner 50b |
| Baseline report/manifest | `.../report-baseline-v8.md`, `.../manifest-baseline-v8.json` | benchmark owner 50b |
| Quarantined arms (never cite) | `.../raw/CONTAMINATED/` | benchmark owner 50b |
| Constrained-smoke thinking fix | owned by benchmark owner; preserve failure | benchmark owner 50b |
| `build.sh --dry-run` role-plan test | recommendation to kernel specialist | kernel |
| Anonymous base-pull test (pre-wipe) | pinned `sha256:222d8b18…`; run on raptor + seed | lease owner |
| Spark image distribution to rhea/moa | outside default host list | release executor |
