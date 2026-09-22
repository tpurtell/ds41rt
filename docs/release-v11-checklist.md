# DS41RT v11 release checklist

**STATUS: PUBLISHED.** The v11 pair is built from image source
`fb5115466a8c70c063280e25957577f284e903e3`, measured, re-qualified, verified and
published; the runtime default is promoted to `:v11`. Boxes below are checked only
from their named evidence file. It mirrors the published [v10 checklist](release-v10-checklist.md),
whose checked boxes and `runs/v10-release/` evidence remain the reference for
what a completed run looks like. Scope and status of the release itself live in
[release-v11-notes.md](release-v11-notes.md).

Read [OPERATIONS.md](OPERATIONS.md) before starting: development and documentation
land on `dev`; `main` advances only after the final pair is qualified and
published; `release/vX` is created at the qualified commit.

## 0. Preconditions

- [ ] Release executor and hardware owner named; the build window is closed to
      other writers. Build run summary lives at
      `runs/v11-release/build/RUN-SUMMARY.md`.
- [ ] Source freeze: the release source commit is approved by the coordinator
      **after** the runtime tests and the independent review of the sampler
      change pass. Record the exact commit (a bare 40-hex revision, never
      `-dirty-`) in [release-v11-notes.md](release-v11-notes.md) and in the build
      evidence before any image build starts.
- [ ] The runtime change is complete and reviewed: the native target path
      accepts non-greedy sampling (`rust/crates/ds41rt-api/src/native_v41.rs`),
      and the greedy-only tests and qualifier are updated with it
      (`rust/crates/ds41rt-api/src/tests/upstream_native_v41.rs`,
      `scripts/qualify-ds41-native-api.py`). Owning agents: engine lead and test
      agent; **not** the release executor.
- [ ] `ds41rt.build-v11.config` exists and `./build.sh --config
      ds41rt.build-v11.config --dry-run` reports release tag `v11`; the default
      `./build.sh --dry-run` still reports `v10` because `ds41rt.config` is not
      promoted yet (re-verify both and record the output).
- [ ] Evidence directory created under `runs/v11-release/{build,publication}`
      (repo-ignored) with its own `SHA256SUMS` written by
      `scripts/release/write-evidence-sums.sh`.
- [ ] `git status --porcelain` empty in the publishing tree at the frozen
      commit; `main`/`dev` ancestry recorded (`main` is an ancestor of `dev`; no
      force, no rebase).

### Commit and identity separation (record every SHA explicitly)

The git tag, the runtime engine source, the image build source and the
host/documentation commits may each be a **different** commit; this repository's
convention already separates them. The release record must state which commit
produced which artifact and must **never claim the checkout at a tag is
byte-identical to the image build source**. Fill this table in
[release-v11-notes.md](release-v11-notes.md) rather than inferring equality:

| Role | Commit / value | Meaning |
| --- | --- | --- |
| Engine (sampling runtime) | `3f8ea80d8e728cf04832dd520c22a9bbf71f8827` | the reviewed runtime change |
| Image build source | `9d9b3e0ce8bd85f6b8eced515d0ab00be015f2d0` | exports `org.opencontainers.image.revision` |
| Host tools (qualifier/validator) | `6f58206f70934c40515598e697b0fcd9813a548d` | committed before the live host run |
| Release docs / tag commit | the `v11` tag commit | `main`/`release/v11` point here |
| Image identity | local ids `e0e5d631…`/`f1233987…` + registry manifest digests below | the measured artifacts |

- [ ] Every row above is filled with a real SHA/value; the tag annotation names
      the immutable image build source, and the notes state explicitly that the
      tag commit's checkout is not claimed to be byte-identical to the image
      build source.
- [ ] The host tools/qualifier commit is a separate committed SHA from the
      image build source and from the documentation commit.

## 1. Build

Run from a standalone clone of the frozen commit (a linked `git worktree` cannot
host the coordinator leg: its relative submodule gitdir does not resolve under
the `/source` bind mount - see `runs/v10-release/build/RUN-SUMMARY.md` §1).
Launch it as a **managed background job**, not a detached `nohup`, so logs
survive an interrupted session:

```bash
git clone --no-local --no-checkout /home/tj/Developer/ds41rt \
  /home/tj/.cache/ds41rt/builds/v11-release-clone
git -C /home/tj/.cache/ds41rt/builds/v11-release-clone checkout --detach <FROZEN>
git -C /home/tj/.cache/ds41rt/builds/v11-release-clone submodule update --init \
  --checkout -- third_party/sparkinfer third_party/xgrammar
git -C /home/tj/.cache/ds41rt/builds/v11-release-clone/third_party/xgrammar \
  submodule update --init --checkout -- 3rdparty/dlpack

./scripts/release/run-release-build.sh --config ds41rt.build-v11.config \
  --source /home/tj/.cache/ds41rt/builds/v11-release-clone \
  --evidence runs/v11-release/build --label v11
```

- [ ] Build exited 0; `runs/v11-release/build/v11-build.log`, `v11-build.rc` and
      `v11-RUN-SUMMARY.md` written. Expected wall time: about 26-30 minutes
      including RDMA distribution to the four Sparks (v10 measured 26m24s).
- [ ] Build root on root NVMe (`DS41RT_RELEASE_BUILD_ROOT`, default
      `~/.cache/ds41rt/builds/v11-build-root`); **never** `/mnt/scratch`.
      `scripts/assert-build-filesystem.py` passed locally and on the seed Spark.
- [ ] Serialized: no other build, benchmark or container run overlapped the
      build window; the coordinator GPU was held alone.
- [ ] Both images exist under the names `ds41rt.build-v11.config` names:
      `ghcr.io/tpurtell/ds41rt-coordinator:v11` on `raptor` and
      `ghcr.io/tpurtell/ds41rt-spark-expert:v11` on `SPARK_0_HOST`.
- [ ] Spark expert roles are the universal `tp2;tp3;tp6` set on all four workers.
- [ ] Launch identity recorded for the handoff: the frozen build config path and
      its SHA-256 fingerprint (this is what `run.sh` records in-container as
      `DS41RT_RELEASE_CONFIG_SHA256`), both image ids, and the pair revision.
      At preparation `ds41rt.build-v11.config` has fingerprint
      `d7d2ceebfd0eff02fc56101acb09342d9c5c69c4f914b040baa9476c54c206fe`;
      recompute it at the frozen commit rather than copying this value.

## 2. Artifact verification (read-only)

```bash
./scripts/release/verify-release-artifacts.sh --config ds41rt.build-v11.config \
  --evidence runs/v11-release/build \
  --dist /home/tj/.cache/ds41rt/builds/v11-release-clone/dist
```

- [ ] `runs/v11-release/build/10-verify-summary.txt` reports **ALL CHECKS PASS**.
- [ ] Labels asserted per role: bare `org.opencontainers.image.revision`,
      `org.opencontainers.image.version=v11`, `io.ds41rt.sparkinfer.revision`,
      `io.ds41rt.cuda_arch` 120/121, `io.ds41rt.role` coordinator/expert,
      `io.ds41rt.v41.spark_tp_roles` empty/tp2;tp3;tp6, and **no**
      `io.ds41rt.source-manifest.sha256` (clean-commit provenance).
- [ ] `dist/SHA256SUMS` verifies (`sha256sum -c`, v10 had 28 entries).
- [ ] EXL3 k23/k34 packages verify for both roles; `V41_EXPERT_TP_AOT.json`
      `native_library_sha256` equals the shipped `libds41rt_native.so`, and the
      Spark manifest carries `spark_tp_roles=[tp2,tp3,tp6]`.
- [ ] SparkInfer/XGrammar provenance and shipped checksum lists verify.
- [ ] The same Spark image id on every worker that carries the image.

## 3. Build-to-benchmark handoff

The release executor owns the build and the image identity; the benchmark agent
owns the hardware run and the campaign configuration. Hand over exactly:

- the two built images and their ids/labels (from §2), plus the pair revision;
- the frozen source commit and the frozen `ds41rt.build-v11.config` path and
  SHA-256;
- the exclusive hardware window (no build, bootstrap or engine job running) and
  a record of the fleet state at handoff;
- the launch requirement: the runtime default still names `:v10`, so the
  candidate **must** be launched with `--config ds41rt.build-v11.config`.
  `run.sh --config FILE` loads the alternate complete configuration, and
  `run.sh` validates image, source, dependency, model, host and device identity
  before starting. Launch-time overrides the benchmark agent may use (each one
  overrides the config for that launch only): `--concurrency`,
  `--max-context-tokens`, `--max-output-tokens`, `--prefix-cache-entries`,
  `--kv-pool-size`, `--host-cache-bytes`, `--memory-reservation`,
  `--prefill-batch-tokens`, `--rtx-gpus`, `--rtx-expert-layers`,
  `--dspark`/`--no-dspark`, `--dspark-draft-limit`.

- [ ] Handoff recorded (in `runs/v11-release/RUN-SUMMARY.md` or the campaign's
      own record) with the values above; no launch happens before it.
- [ ] The benchmark campaign is not started by the release executor.

## 4. Qualification and performance campaign

Scope for v11 (agreed with the coordinator):

- headline **nine-category** decode table plus counting, as in the v10-era
  benchmark scripts;
- **five strict profiles x 3 repeats**;
- **no** retained-turn and **no** concurrency-matrix campaign required for v11.

Readiness and budget reported by the benchmark agent: its CPU-only scope check
passes (31 tests), and the campaign is estimated at roughly 20-30 minutes after
the build for fixed inputs (around 60-90 minutes if differing input lengths need
separate validation). Size the window with the hardware owner and treat it as
exclusive, like the build.

- [ ] Benchmark arms run on the v11 image pair from the frozen source, with the
      exact launch configuration, config SHA-256 and image identity recorded.
- [ ] Every headline number comes from a raw file named in the benchmark
      package; nothing is transcribed by hand.
- [ ] Sampler behaviour covered: greedy and non-greedy (temperature > 0) both
      served by the native path; the test agent's qualification records the
      bit-level sampling agreement.
- [ ] Results recorded under `runs/v11-release/` and rendered into
      `docs/release-v11-performance.md` (+ raw JSON), with source/model
      revisions, input shapes, sampling, acceptance, warmup, every timed sample,
      memory and graph setup.
- [ ] README headline/version and pull-pair updates are the **benchmark agent's**
      change, landed before promotion; the release executor does not edit
      `README.md`.

## 5. Pre-push baseline (rollback reference)

`./push-containers.sh` publishes the version tag **and** `latest` for both
images, and `latest` has no separate rollback boundary. Record the digests
`latest` resolves to **before** pushing, from the registry's own response:

```bash
EVIDENCE=runs/v11-release/publication
scripts/release-digests.sh capture --config ds41rt.config --tag latest \
  --evidence "$EVIDENCE/pre-push-latest.env"
```

- [ ] Pre-push `latest` digests recorded (coordinator index and Spark manifest);
      they must equal the published v10 pair in
      [release-v10-notes.md](release-v10-notes.md).
- [ ] v11 confirmed absent from both repositories (anonymous `:v11` returns 404)
      so the push creates the tag rather than moving it; record the 404s.
- [ ] Existing packages verified still public: an anonymous `docker buildx
      imagetools inspect` for `:v10`/`:latest` succeeds with a throwaway
      `DOCKER_CONFIG` (verified for this preparation). Re-verify rather than
      assume the owner's visibility setting; only if a package has gone private
      is an owner action needed.

## 6. Publication ordering

Order matters and is not interchangeable:

1. **Freeze** the publishing checkout at the qualified commit (§0).
2. **Record** the pre-push baseline (§5).
3. **Push** v11 and `latest`:

   ```bash
   ./push-containers.sh --config ds41rt.build-v11.config v11
   ```

4. **Capture** the new digests from the registry:

   ```bash
   scripts/release-digests.sh capture --config ds41rt.build-v11.config \
     --evidence "$EVIDENCE/v11-digests.env"
   ```

5. **Verify** by anonymous fresh pull on a matching-architecture host
   (coordinator on `raptor` amd64; Spark image on an arm64 worker such as
   `kiwi`):

   ```bash
   scripts/release-digests.sh verify --config ds41rt.build-v11.config \
     --evidence "$EVIDENCE/v11-digests.env"
   ```

6. **Archive** the evidence (`scripts/release/write-evidence-sums.sh --evidence
   "$EVIDENCE"`) and record the values in
   [release-v11-notes.md](release-v11-notes.md).
7. **Promote** the runtime default in its own change (§7).

Checkboxes:

- [ ] Push exited 0 and published exactly `v11` and `latest` for both roles; no
      other project, tag or repository was touched; the role guard ran and did
      not refuse.
- [ ] The command printed the same revision for both images and the universal
      role set `tp2;tp3;tp6`.
- [ ] Push log archived and sanitized (no credential values).
- [ ] v11 and `latest` resolve to the same digest per role.
- [ ] Anonymous fresh-pull verification passed for both roles, each on its own
      architecture; no credential file was created in the throwaway
      `DOCKER_CONFIG`.
- [x] `v11-digests.env`, `pre-push-latest-recheck.env`, `post-push-latest.env`,
      `push-v11.log` and `v11-absence-recheck.txt` are under
      `runs/v11-release/publication/`; the archive carries a `SHA256SUMS` file.
- [ ] The publication summary in [release-v11-notes.md](release-v11-notes.md)
      carries the real registry digests and the anonymous-verification result,
      never a placeholder.
- [ ] **Pre-publication gate:** `grep -c PENDING docs/release-v11-notes.md`
      reports only the registry and publication fields, and every other fact
      (source revision, image ids, build verification, benchmark results) is
      filled from real evidence with the status at RELEASE-READY. Do not push
      while the non-registry facts are still placeholders.
- [x] **Post-publication gate:** the registry digests and the publication record
      are filled from the registry's own responses,
      `grep -c PENDING docs/release-v11-notes.md` reports **0**, and the status
      line reads PUBLISHED.

## 7. Rollback

There is no server-side "undo" for a moved tag. Decide the rollback before
pushing and keep the §5 baseline.

- [ ] **Revert `latest`** by re-pointing it at the recorded pre-push digest from
      a host with GHCR write credentials: pull by digest, tag `latest`, push;
      the same for the Spark repository on `SPARK_0_HOST`.
- [ ] **Revert a bad `v11`** by publishing a corrected image under a new tag
      (`v11.1`) and promoting that; never overwrite or delete the published
      `v11` manifest, and never re-tag `latest` to a partially-corrected pair.
- [ ] **Partial failure** (one role pushed, the other not): treat the release as
      unpublished; roll `latest` back to the §5 baseline for both roles, then
      rebuild and restart from §6. Never leave `latest` split across the pair.
- [ ] Rollback rehearsal recorded: exact commands, their output and the digest
      each repository ended at.

## 8. Promotion

The promotion change is separate from publication and moves the runtime default
to the published pair:

- [ ] `ds41rt.config` coordinator/Spark inference lines switched to `:v11` and
      committed (its only diff is the release pair).
- [ ] `examples/configs/*` and their README pair lines moved to `:v11` by their
      owning agent; `ds41rt.build-v11.config` becomes identical to
      `ds41rt.config` and its header comment is updated to say so.
- [ ] `./run.sh --dry-run` and the release gate checks pass on the promoted
      default before the documentation switch is announced.
- [ ] `docs/release-v11-notes.md` status moves from DRAFT to PUBLISHED with the
      real digests; `README.md` headline/pull lines are updated by the benchmark
      agent.
- [ ] Rollback of the promotion is the same commit reverted plus the §6 `latest`
      re-point; record both.

## 9. CPU/source gates

- [ ] `bash -n build.sh run.sh wip.sh push-containers.sh
      scripts/release-common.sh scripts/release-digests.sh
      scripts/release/*.sh` - clean.
- [ ] `git diff --check` - clean.
- [ ] Canonical CPU pytest suite from `python/` with the repository Python
      environment (`scripts/run-with-python-env.sh .venv/bin/python -m pytest -q
      --continue-on-collection-errors reference/tests ../scripts/tests`). For
      v10 this was 971 passed / 22 skipped / 26 failed / 221 subtests, where the
      26 failures reproduced at the pre-publication commit and were not caused by
      the release reports. Any v11 failure must be adjudicated the same way, not
      waved through.
- [ ] Rust tests for the changed crates:
      `cargo test --manifest-path rust/Cargo.toml -p ds41rt-api` (sampling
      contract) plus the other crates the runtime change touches.
- [ ] The promotion and reports changes touch only their intended paths; verify
      with `git show --stat` per commit.

## 10. Owner action after everything else

- [ ] GHCR package visibility is confirmed (v10 packages are public and were
      re-verified anonymously during this preparation). If a package is
      private, the repository owner makes it public **manually**; carry this as
      the final handoff item only when it is actually needed.
