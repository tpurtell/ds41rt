# DS41RT v9 release checklist

**STATUS: PARTIALLY EXECUTED — the clean build and the functional validation are
done and recorded; the final-image performance campaign, registry publication and
the experimental canonical qualifier remain.** A checked box means "verified with
recorded evidence", not merely "attempted", and no box is checked from memory.
Evidence archive: `~/.cache/ds41rt-v9-archive/v9-20260921T003834Z/`.

## Actual v9 state (2026-09-21)

Verified with recorded evidence:

- Clean build rc0 from the frozen commit, engine revision
  `5d0d209509bf1f26731bc588dfe8dc72d1373ad0`, label bare (no `-dirty-`), version
  `v9`, `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6`.
- Docker-wide local wipe completed on all seven hosts; images distributed over
  RDMA to the four build Sparks plus `rhea`/`moa`; dist `sha256sum -c` rc=0 over
  569 files.
- Functional validation PASS on all four production geometries
  (`RELEASE-GATE-CHECKLIST-ACTUAL.md`): api-smoke and constrained rc=0 everywhere;
  content 30/30 (TP6 1x), 10/10 (TP6 2x, TP4 1x, TP4 2x). The strict-schema 400
  rejection is genuinely exercised, not forced.
- `drop-page-cache` on `kiwi`/`rhea` recorded as an operational prerequisite.
- TP4 1 RTX coordinator dispatch placement is **unknown** (worker LOAD contract is
  `first_layer=0 layers=40`; do not claim all-40-remote).

Publication done: `./push-containers.sh v9` published `v9` and `latest` (same
digest per role) and anonymous fresh-`DOCKER_CONFIG` pulls returned rc=0 with
matching digests. Coordinator index `sha256:786d1d67…` (linux/amd64 child
`sha256:649e5c84…`); spark-expert manifest `sha256:f0c67407…`
(`v9-build/PUBLICATION-EVIDENCE.txt`). The GitHub release is a separate step and
remains pending.

Not done: no final-image performance campaign; reference equivalence not
performed; canonical six-rank qualifier not run and out of scope. The performance
numbers in the v9 reports are validated warm-candidate measurements, not
final-image results.

Boxes below stay unchecked unless the recorded evidence above covers them.

Scope: v9 adds the pure unreplicated TP6EP1 Spark topology, keeps default
TP4EP1 and the approved hybrid topologies, ships the `tp2;tp3;tp6` Spark role
set with implicit default TP4, and widens `stop.sh` cleanup to every configured
Spark host. The official default quant is not re-campaigned unless recorded
below.

## 0. Ownership, freeze and evidence

- [ ] Release executor named; hardware owner (benchmark lead) has released the
      five/six hosts for the rebuild window.
- [ ] Working tree frozen for the build; no other agent writes during the build.
- [ ] `docs/release-v9-notes.md` metrics filled from the actual v9 campaign
      (no v8 numbers carried over, no estimates).
- [ ] Raw evidence archive created outside the repository and outside tmpfs,
      with `SHA256SUMS` over the whole archive and the digest recorded in the
      notes/README.

## 1. Git and source identity

- [ ] Ancestry verified with `git merge-base --is-ancestor main dev` (and
      `git rev-list --left-right --count main...dev` = `0 <n>`), never inferred
      from ahead/behind display. At review time this held (fast-forward).
- [ ] All TP6 changes committed, **including** the untracked required files:
      `native/src/v41_spark_tp6_experts.cc`,
      `native/tests/v41_expert_pack_tp6_selftest.cc`,
      `python/tests/test_v41_spark_tp6_contract.py`,
      `examples/configs/tp6ep1-native.config`,
      `scripts/fixtures/tp-ep-six/site-2rtx6-tp6ep1.config`,
      `scripts/tests/test_stop_host_selection.py`.
- [ ] User-owned local state preserved explicitly: patch/archive of the dirty
      tree (`git diff --binary`, `git status --porcelain -uall`, and the
      `HEAD` copies of `TO_DELETE_SCAFFOLDING.md`/`TO_SHIP_V1.md`) stored in the
      archive. **The clean-tree requirement must not be met by auto-committing
      user changes.** Record which user changes, if any, were intentionally
      scoped into the release.
- [ ] Isolated worktree strategy (confirmed): create a **fresh path** on root
      NVMe under `~/.cache/ds41rt/builds/` (never `/mnt/scratch`); preserve the
      user's deletions **separately** (patch/archive as above) rather than
      re-adding or committing them; explicitly commit only the intended TP6
      files; initialize submodules inside the worktree; and **do not touch the
      baseline-owned worktree** at `~/.cache/ds41rt/builds/v9-baseline-src`.
      A worktree is clean by construction, but untracked files do not exist in
      it — confirm the TP6 files are committed (item above), not merely present
      in the main checkout.
- [ ] TP6 role-plan coverage note: no current test executes `build.sh
      --dry-run` to assert the `DS41RT_RELEASE_SPARK_TP_ROLES` allowlist or the
      `SPARK_TP=6 -> tp6` mapping. Raised as a kernel-specialist recommendation;
      not added here.
- [ ] `git status --porcelain` empty in the build tree; `verify-sparkinfer-source.py`
      and `verify-xgrammar-source.py` pass against the pins.
- [ ] `build.sh` dirty detection reviewed: the clean build must label
      `org.opencontainers.image.revision` with the bare release commit.

## 2. Docker cleanup (literal, all projects) and recovery

- [ ] Inventory captured **before** deletion: `docker ps -a`, `images
      --digests`, `volume ls`, `system df`, per-image labels, local and remote.
- [ ] Optional rollback archives written outside the prune scope:
      `docker image save` for the v8 coordinator and the arm64 Spark image, with
      sizes recorded and `/` capacity confirmed.
- [ ] Named volumes archived read-only (`docker run -v <vol>:/v:ro ... tar`)
      **or** explicitly accepted as disposable build/JIT caches, plus a content
      manifest. Note: `docker volume prune` is anonymous-only by default and
      can miss named volumes still referenced by stopped containers.
- [ ] All containers removed locally and on every configured Spark host.
- [ ] `docker image prune -a -f` with **no filter exclusions** (this deletes
      the NGC base, which is accepted).
- [ ] Inventoried named volumes removed **explicitly by name**; `docker volume
      ls` verified empty.
- [ ] `docker builder prune -af` run locally and on the Spark seed (a system
      prune alone leaves the builder cache).
- [ ] `docker system df` shows ~0 images/containers/cache; `docker info` healthy.
- [ ] Pinned base re-pulled on raptor and the seed and verified by digest, or
      the build passes `--build-arg BASE_IMAGE=<pinned-ref>`.
- [ ] Remote cleanup scoped to DS41RT artifacts on shared Sparks; no other
      fleet's images, containers or caches were touched.
- [ ] Fresh `DS41RT_RELEASE_REMOTE_BUILD_DIR` used so seed staging is new.

## 3. Clean build

- [ ] `./build.sh --dry-run` passes for the default and the six-rank configs.
- [ ] Release build run with the nominated role set
      `DS41RT_RELEASE_SPARK_TP_ROLES=tp2;tp3;tp6` (plus implicit default TP4),
      from the frozen commit, with no concurrent tree edits.
- [ ] Build log captured **sanitized** (no raw `bash -x` with secrets); the
      invoked argv/options recorded.
- [ ] Both images built on the pinned base; Spark image distributed to every
      host that will serve, including `rhea`/`moa` (outside the default
      build/distribution list).
- [ ] Image labels verified on every host: clean
      `org.opencontainers.image.revision` (no `-dirty-`), version `v9`,
      matching `io.ds41rt.sparkinfer.revision`, matching optional
      `io.ds41rt.source-manifest.sha256`, and
      `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6`.
- [ ] `dist/` regenerated and `dist/SHA256SUMS` verified; exporter
      source/options and generated manifests archived (see §5).
- [ ] `run.sh` reviewed against the release: it validates the SparkInfer
      revision but **not** the engine revision against `git rev-parse HEAD`;
      either add that assertion or record an explicit manual check that the
      image revision equals the release commit.

## 4. Functional validation on rebuilt images

Default TP4EP1 deployment:

- [ ] `./run.sh --dry-run` then `./run.sh` on the host set.
- [ ] `/health`, `/v1/models` and one completion (`scripts/api-smoke.sh`).
- [ ] Constrained/structured output (`scripts/api-constrained-smoke.sh`).
- [ ] Streaming smoke and long-prefill smoke.
- [ ] Restart/cancellation/recovery behavior.
- [ ] Known-failure accounting unchanged and recorded (do not silently fix or
      hide pre-existing model failures).

TP6EP1 deployment:

- [ ] Six-rank `run.sh --config examples/configs/tp6ep1-native.config
      --dry-run` passes and the role requirement is satisfied by the image
      label.
- [ ] Six-rank launch reaches readiness with the recorded budget/KV config.
- [ ] Pure-TP6 correctness: routed-expert output across the six disjoint 384
      slices is qualified (component and end-to-end).
- [ ] Cleanup verified with `stop.sh` across all six configured hosts, and a
      four-rank configuration still cleans the fifth/sixth hosts.

Hybrid topologies (as campaigned): `<PENDING which of TP2EP2/TP3EP2/TP2EP3 are
in scope for v9>`.

CPU/source gates:

- [ ] `PYTHONPATH=python/reference .venv/bin/python -m pytest -q scripts/tests`.
- [ ] `python -m pytest -q python/tests` (includes the TP6 contract test).
- [ ] `cargo test --manifest-path rust/Cargo.toml --workspace`.
- [ ] `bash -n build.sh run.sh stop.sh scripts/release-common.sh`;
      `git diff --check`.
- [ ] Native TP6 selftest (`ds41rt_v41_expert_pack_tp6_selftest`) passes or
      records its documented skip.

## 5. Build reproducibility evidence (must be archived)

- [ ] Exporter source revision and blob hashes for
      `python/tools/export_b12x_v41_*_aot.py`, the native helpers and
      `scripts/build-release-artifacts.sh` / `native/cmake/v41_spark_tp_experts.cmake`.
- [ ] Exporter options actually used: `DS41RT_RELEASE_SPARK_TP_ROLES`,
      `DS41RT_RELEASE_EXL3_PAIRED_TP4`, `--role` values, capacity set
      `1/16/80/256/1024/4096`, slice-width maps, CUDA arch 120/121.
- [ ] Generated manifests archived: `exl3/manifest.json`, `V41_EXPERT_AOT.json`,
      `V41_EXPERT_TP_AOT.json`, `V41_FP8_AOT.json`,
      `SPARKINFER_PROVENANCE.json`, `SHA256SUMS`, role export JSON.
- [ ] Binary hashes (`ds41rt`, `libds41rt_native.so`) and image config digests.
- [ ] Launch record: `docker inspect` of the running coordinator and Spark
      containers including `DS41RT_RELEASE_CONFIG_SHA256`.

## 6. Performance and regression

- [ ] v9 campaign run **on the final rebuilt images**, after the benchmark
      lead's published-v8 baselines remain archived and untouched.
- [ ] Every timed sample, warmup, concurrency, KV encoding, sampling and
      acceptance recorded; three samples per cell for headline numbers.
- [ ] Regression check against the v8/parent baseline recorded with its
      conditions; no cross-campaign number is presented as a single run.
- [ ] Contaminated baseline arms (the two first-baseline arms quarantined
      17:19-17:37 UTC) are **excluded**; only clean reruns feed any comparison,
      and each baseline identity (`0cebc06` launcher, `5a56` published v8
      runtime) is quoted.
- [ ] Published-v8 single-RTX placement recorded as the actual diagnostic
      **5 local / 35 remote** geometry, not an all-40-remote claim.
- [ ] Official default quant status stated explicitly (re-campaigned or
      historical, as appropriate), noting the default TP4 profile is `None`
      (adaptive, widths 5/7) so no automatic built-in profile confounds the
      TP6 comparison; the all-40-remote TP6 run is labeled a standalone extra,
      not the primary official-match configuration.

## 7. Publication and promotion

- [x] `./push-containers.sh v9` succeeds; `:v9` and `:latest` pushed for both
      roles (2026-09-21, exit 0).
- [x] Recorded digests match the pushed manifests: coordinator index
      `sha256:786d1d67…`, spark-expert manifest `sha256:f0c67407…`.
- [x] Anonymous public-pull verification passes for both v9 packages (fresh
      `DOCKER_CONFIG`, no host credentials, rc=0 with matching digests on raptor
      and ostrich).
- [x] Package visibility is already public for the existing packages; no owner
      visibility action was needed.
- [x] `ds41rt.config` updated to `:v9` and committed.
- [ ] GitHub release published (separate step, pending).
- [ ] `main` fast-forwarded to the qualified `dev` head (`git merge --ff-only`)
      and `release/v9` created at the qualified commit, only after publication
      and hardware validation.
- [ ] `latest` consumers considered: the tag now points at v9 and has no
      separate rollback boundary (pre-push rollback reference is recorded in the
      publication evidence).

## 8. Documentation

- [ ] README image references, headline labels and pull lines moved to v9 with
      the v9 notes link.
- [ ] `docker/README.md` stale "v6 images" sentence corrected.
- [ ] Dangling `TO_DELETE_SCAFFOLDING.md` references resolved
      (`DEVELOPER.md`, the tracked loader example, and the four
      `docs/ds41-*.md` files) since the user deleted that file.
- [ ] `docs/HANDOFF_NEXT_SESSION.md` updated or marked historical (it links the
      deleted `TO_SHIP_V1.md`).
- [ ] Review artifact `runs/release-v9-prep-review.md` moved out of the repo
      into the evidence archive (it is untracked repo pollution).
- [ ] Final notes contain no `<PENDING>` markers.
