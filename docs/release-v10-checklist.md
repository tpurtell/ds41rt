# DS41RT v10 release checklist

**STATUS: PUBLISHED AND ANONYMOUS-VERIFIED.** The pre-push rollback baseline, the
push, the post-push digest captures and the anonymous fresh pulls are recorded
under `runs/v10-release/publication/`, and every box below is checked only from
that evidence. Registry digests are transcribed into
`docs/release-v10-notes.md` from the registry's own responses, never from a local
image id.

This checklist covers publication mechanics. The runtime promotion is committed
in its own change: `ds41rt.config`, the five older `examples/configs/*` native
files and the `README.md` pull/pair lines now name `:v10`, and the two v10 TP3
profiles were re-added to published-pair equality (see
`docs/release-v10-notes.md`).

`ds41rt.build-v10.config` is retained as the explicit release **build** target.
After promotion it is identical to `ds41rt.config` including the
coordinator/Spark release pair (`:v10`), and `./build.sh --dry-run` reports tag
`v10` with or without `--config`. See `docs/release-v9-checklist.md` for the
equivalent v9 gate detail and `runs/v10-release/build/` for the existing build
evidence (repo-ignored).

## 0. Preconditions

- [x] Release executor and hardware owner named; the build window is closed to
      other writers. Build run summary:
      `runs/v10-release/build/RUN-SUMMARY.md` (coordinator `raptor`, workers
      `ostrich`, `dodo`, `emu`, `kiwi`).
- [x] v10 build evidence is complete and archived: clean-build log, engine
      revision label, `dist/SHA256SUMS` verified (`sha256sum -c` clean over 28
      entries), role label `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6`
      (`runs/v10-release/build/10-label-assertions.txt`,
      `10-dist-artifact-verification.txt`).
- [x] Both local images exist under the names `ds41rt.build-v10.config` names:
      `ghcr.io/tpurtell/ds41rt-coordinator:v10` on the coordinator host and
      `ghcr.io/tpurtell/ds41rt-spark-expert:v10` on `SPARK_0_HOST`.
- [x] `./build.sh --config ds41rt.build-v10.config --dry-run` passes and reports
      release tag `v10`; the default `./build.sh --dry-run` now also reports
      `v10` because `ds41rt.config` names the promoted pair.
- [x] Evidence directory created, with its own `SHA256SUMS`; every file written
      below lives under `runs/v10-release/publication/` (repo-ignored) and is
      covered by that file.
- [x] `git status --porcelain` empty in the publishing tree; `main`/`dev`
      ancestry recorded (`main` is an ancestor of `dev`; no force, no rebase).

## 1. Pre-push baseline (rollback reference)

`./push-containers.sh` publishes the version tag **and** `latest` for both
images, and `latest` has no separate rollback boundary. Record the digests
`latest` resolves to **before** pushing, from the registry's own response:

```bash
EVIDENCE=runs/v10-release/publication   # repo-ignored evidence tree
scripts/release-digests.sh capture --config ds41rt.build-v10.config --tag latest \
  --evidence "$EVIDENCE/pre-push-latest.env"
```

- [x] Pre-push `latest` digests recorded: coordinator index
      `sha256:786d1d6704e4cdaaf12ae59f5324bb1a43ce2238c9ec79ff7884fd3c83e8eb7f`,
      Spark manifest
      `sha256:f0c67407adb4228200c1fcb97e3f7210501db120d1e1ee11697f66cfb1ca4ea9`
      (`runs/v10-release/publication/pre-push-latest.env`; values come from the
      capture, not from a local image id or a config digest).
- [x] Pre-push `latest` is confirmed to be the v9 predecessor the rollback in §4
      assumes: both digests equal the published v9 pair recorded in
      `docs/release-v9-notes.md`.
- [x] v10 is confirmed absent from the registry (anonymous `:v10` returns 404 in
      both repositories) so the push below creates the tag rather than moving it:
      `runs/v10-release/publication/v10-absence.txt` records the 404 for both
      repositories.

## 2. Publication ordering

Order matters and is not interchangeable. Run the steps in this order only:

1. **Freeze** the publishing checkout at the qualified commit (§0).
2. **Record** the pre-push baseline (§1).
3. **Push** v10 and `latest` (§2.1).
4. **Capture** the new digests from the registry (§2.2).
5. **Verify** by anonymous fresh pull (§2.3).
6. **Archive** the evidence and record it in the release notes (§2.4).
7. **Promote** the runtime default in its own change (§4; landed before
   publication, so the default already names `:v10`).

### 2.1 Push

```bash
# From the frozen publishing checkout. ds41rt.config is not edited first:
# --config selects the v10 BUILD target; the positional tag stays independent.
./push-containers.sh --config ds41rt.build-v10.config v10
```

- [x] Push exited 0 and published exactly `v10` and `latest` for both roles; no
      other project, tag or repository was touched
      (`runs/v10-release/publication/push-v10.log`).
- [x] The command printed the same revision for both images
      (`3dd9a4ac2be9fd17ecf4cb8b7746efdc900d38f0`) and the universal role set
      `tp2;tp3;tp6`; the publisher's role guard ran and did not refuse.
- [x] Push output/log archived (sanitized; no credential values).

### 2.2 Capture the registry digests

```bash
scripts/release-digests.sh capture --config ds41rt.build-v10.config \
  --evidence "$EVIDENCE/v10-digests.env"
```

The helper is read-only: it obtains one anonymous GHCR pull token per repository
and records the registry's `Docker-Content-Digest` for the coordinator (an OCI
image index) and the Spark expert (a single manifest). It never tags, pushes or
edits a configuration.

- [x] `coordinator.digest` recorded:
      `sha256:2236d94317eb393cd78940efb117bcca14ae06e6d1113c6aeb188b0b22424689`.
- [x] `spark.digest` recorded:
      `sha256:98ddf9cd83626d92297169b04561b722d136de27ac8213a1a9f5702ebe40ec30`.
- [x] `v10` and `latest` digests are confirmed equal per role (the push publishes
      both from the same local image); the post-push `latest` capture agrees with
      the `v10` capture for both roles.
- [x] Evidence file copied into the archive and covered by its `SHA256SUMS`.

### 2.3 Anonymous fresh-pull verification

Run on a host that can reach GHCR. The helper pulls with a throwaway empty
`DOCKER_CONFIG`, so no host credential can be consulted, and requires the pull's
reported `Digest` to equal the capture:

```bash
scripts/release-digests.sh verify --config ds41rt.build-v10.config \
  --evidence "$EVIDENCE/v10-digests.env"
# Optionally confirm the rollback boundary too:
scripts/release-digests.sh verify --config ds41rt.config --tag latest \
  --evidence "$EVIDENCE/pre-push-latest.env"
```

- [x] Both roles verified: anonymous pull rc=0 and the pull's digest equals the
      captured coordinator index / Spark manifest digest. The coordinator (amd64
      OCI index) was verified on `raptor`; the Spark image is a single arm64
      manifest, so it was verified on the arm64 worker `kiwi` with the same
      read-only helper, config and evidence file (`anonymous-verify-*.log`).
- [x] No credential file was created in the throwaway `DOCKER_CONFIG` (the
      helper fails the verification if one appears; no failure was reported).
- [x] The validating hosts' own cached images were not wiped; `kiwi` still
      carries the tagged `v10` and `v9` Spark images.

### 2.4 Evidence record

- [x] `v10-digests.env`, `pre-push-latest.env`, `post-push-latest.env`,
      `v10-absence.txt` and the push log are archived with a `SHA256SUMS` file
      under `runs/v10-release/publication/`; the recorded values are transcribed
      into `docs/release-v10-notes.md`.
- [x] The publication summary in `docs/release-v10-notes.md` carries the real
      registry digests and the anonymous-verification result (never a
      placeholder).

## 3. Rollback

There is no server-side "undo" for a moved tag. Decide the rollback before
pushing and keep the §1 baseline.

- [ ] **Revert `latest`** (the consumer-facing risk) by re-pointing it at the
      recorded pre-push digest, from a host with GHCR write credentials:
      `docker pull <repository>@<pre-push-digest>`, `docker tag
      <repository>@<pre-push-digest> <repository>:latest`, `docker push
      <repository>:latest`. Do the same for the Spark repository on
      `SPARK_0_HOST`.
- [ ] **Revert a bad `v10`** by publishing a corrected image under a new tag
      (`v10.1`) and promoting that; do not overwrite or delete the published
      `v10` manifest, and do not re-tag `latest` to a partially-corrected pair.
- [ ] **Partial failure** (one role pushed, the other not): treat the release as
      unpublished; roll `latest` back to the §1 baseline for both roles, then
      rebuild and restart from §2.1. Never leave `latest` split across the pair.
- [ ] Rollback rehearsal recorded: the exact commands run, their output, and the
      digest each repository ended at.

## 4. Promotion (landed; publication follows)

The promotion change landed before publication, so a v10 server cannot be served
from an unretargeted default. The runtime smoke and the promotion rollback record
remain open:

- [x] `ds41rt.config` coordinator/Spark inference lines switched to `:v10` and
      committed (its only diff is the release pair).
- [x] `examples/configs/*` and `README.md` pull/pair lines moved to `:v10`, with
      the v10 notes link; the two v10 TP3 profiles are no longer exempt from
      published-pair equality.
- [ ] Runtime smoke on the promoted default (`./run.sh --dry-run`, then the
      release gate checks) before the documentation switch is announced.
- [ ] Rollback of the promotion is the same commit reverted plus the §3 `latest`
      re-point; record both.

## 5. CPU/source gates for this release

- [x] `bash -n push-containers.sh scripts/release-digests.sh scripts/release-common.sh`
      — clean.
- [x] Canonical CPU suite, run as the justfile does from `python/` with the
      repository Python environment
      (`run-with-python-env.sh .venv/bin/python -m pytest -q
      --continue-on-collection-errors reference/tests ../scripts/tests`):
      **971 passed, 22 skipped, 26 failed, 221 subtests**. The 26 failures are
      reproduced unchanged at the pre-publication commit `c6871f6` and are not
      caused by the release reports: 25 are the SparkInfer `capture` contract
      tests (`b12x.attention.dsa_indexer` has no `SOURCE_LAYOUT_PAGED` /
      `ImportError` in the checked-out `third_party/sparkinfer`) and one is the
      known six-rank readiness mismatch (`assert '10.55.0.6' == '10.55.0.12'`).
      The report renderer suite is green: 65 passed.
- [x] `scripts/release-digests.sh --help` and `./push-containers.sh --help`
      describe `--config`; `push-containers.sh --help` documents the shared
      `DS41RT_RELEASE_SSH_CONFIG`.
- [x] `git diff --check` — clean.
- [x] The promotion change (`9287203`) touches only its intended paths:
      `ds41rt.config`, `ds41rt.build-v10.config` (comment), the five older
      `examples/configs/*-native.config`, `README.md`,
      `docs/release-v10-notes.md`, the adjacent docs and the promotion tests.
      The measured-report change is the separate
      `Publish v10 TP3 benchmark reports` commit.
