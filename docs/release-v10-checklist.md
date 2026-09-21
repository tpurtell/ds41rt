# DS41RT v10 release checklist

**STATUS: PREPARED — NOT PUBLISHED.** This is the publication skeleton for v10.
No v10 image has been pushed and no registry digest below is real. A box is
checked only from recorded evidence; the two checked entries in §4 are the
committed runtime-promotion change.

This checklist covers publication mechanics. The runtime promotion is committed
in its own change: `ds41rt.config`, the five older `examples/configs/*` native
files and the `README.md` pull/pair lines now name `:v10`, and the two v10 TP3
profiles were re-added to published-pair equality (see
`docs/release-v10-notes.md`). The registry still has no `v10`, so every
publication box below is unchecked and no digest is claimed.

`ds41rt.build-v10.config` is retained as the explicit release **build** target.
After promotion it is identical to `ds41rt.config` including the
coordinator/Spark release pair (`:v10`), and `./build.sh --dry-run` reports tag
`v10` with or without `--config`. See `docs/release-v9-checklist.md` for the
equivalent v9 gate detail and `runs/v10-release/build/` for the existing build
evidence (repo-ignored).

## 0. Preconditions

- [ ] Release executor and hardware owner named; the build window is closed to
      other writers.
- [ ] v10 build evidence is complete and archived outside the repository:
      clean-build log, engine revision label, `dist/SHA256SUMS` verified, role
      label `io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6`.
- [ ] Both local images exist under the names `ds41rt.build-v10.config` names:
      `ghcr.io/tpurtell/ds41rt-coordinator:v10` on the coordinator host and
      `ghcr.io/tpurtell/ds41rt-spark-expert:v10` on `SPARK_0_HOST`.
- [ ] `./build.sh --config ds41rt.build-v10.config --dry-run` passes and reports
      release tag `v10`; the default `./build.sh --dry-run` now also reports
      `v10` because `ds41rt.config` names the promoted pair.
- [ ] Evidence directory created outside the repository, with its own
      `SHA256SUMS`; every file written below goes there.
- [ ] `git status --porcelain` empty in the publishing tree; `main`/`dev`
      ancestry recorded (no force, no rebase).

## 1. Pre-push baseline (rollback reference)

`./push-containers.sh` publishes the version tag **and** `latest` for both
images, and `latest` has no separate rollback boundary. Record the digests
`latest` resolves to **before** pushing, from the registry's own response:

```bash
EVIDENCE=/path/to/v10-evidence        # outside the repository
scripts/release-digests.sh capture --config ds41rt.config --tag latest \
  --evidence "$EVIDENCE/pre-push-latest.env"
```

- [ ] Pre-push `latest` digests recorded: coordinator index `<captured>`,
      Spark manifest `<captured>` (values come from the capture above; do not
      transcribe a local image id or a config digest).
- [ ] Pre-push `latest` is confirmed to be the v9 predecessor the rollback in
      §4 assumes; if it is not, stop and re-derive the rollback plan.
- [ ] v10 is confirmed absent from the registry (anonymous `:v10` returns 404)
      so the push below creates the tag rather than moving it.

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

- [ ] Push exited 0 and published exactly `v10` and `latest` for both roles; no
      other project, tag or repository was touched.
- [ ] The command printed the same revision for both images and the universal
      role set `tp2;tp3;tp6`; a role-less or subset build is refused by the
      publisher and must not be worked around.
- [ ] Push output/log archived (sanitized; no credential values).

### 2.2 Capture the registry digests

```bash
scripts/release-digests.sh capture --config ds41rt.build-v10.config \
  --evidence "$EVIDENCE/v10-digests.env"
```

The helper is read-only: it obtains one anonymous GHCR pull token per repository
and records the registry's `Docker-Content-Digest` for the coordinator (an OCI
image index) and the Spark expert (a single manifest). It never tags, pushes or
edits a configuration.

- [ ] `coordinator.digest` recorded: `<captured>`.
- [ ] `spark.digest` recorded: `<captured>`.
- [ ] `v10` and `latest` digests are confirmed equal per role (the push publishes
      both from the same local image); if they differ, stop and explain why.
- [ ] Evidence file copied into the archive and covered by its `SHA256SUMS`.

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

- [ ] Both roles verified: anonymous pull rc=0 and the pull's digest equals the
      captured coordinator index / Spark manifest digest.
- [ ] No credential file was created in the throwaway `DOCKER_CONFIG` (the
      helper fails the verification if one appears).
- [ ] The validating host's own cached image was not wiped.

### 2.4 Evidence record

- [ ] `v10-digests.env`, `pre-push-latest.env` and the push log are archived with
      a `SHA256SUMS` file; the archive path and its digest replace the pending
      placeholders in `docs/release-v10-notes.md`.
- [ ] `docs/release-v9-checklist.md`-style publication summary written with the
      real digests (never a placeholder; `release-v10-notes.md` currently carries
      explicit `PENDING` placeholders for exactly this reason).

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

## 4. Promotion (landed; publication still pending)

The promotion change landed before publication, so a v10 server cannot be served
from an unretargeted default. The publication steps above (§2.3) and the runtime
smoke remain open:

- [x] `ds41rt.config` coordinator/Spark inference lines switched to `:v10` and
      committed (its only diff is the release pair).
- [x] `examples/configs/*` and `README.md` pull/pair lines moved to `:v10`, with
      the v10 notes link; the two v10 TP3 profiles are no longer exempt from
      published-pair equality.
- [ ] Runtime smoke on the promoted default (`./run.sh --dry-run`, then the
      release gate checks) before the documentation switch is announced.
- [ ] Rollback of the promotion is the same commit reverted plus the §3 `latest`
      re-point; record both.

## 5. CPU/source gates for this preparation

- [ ] `bash -n push-containers.sh scripts/release-digests.sh scripts/release-common.sh`.
- [ ] `.venv/bin/python -m pytest -q scripts/tests` (includes
      `test_push_containers_config.py` and `test_release_digests.py`).
- [ ] `scripts/release-digests.sh --help` and `./push-containers.sh --help`
      describe `--config` and the shared `DS41RT_RELEASE_SSH_CONFIG`.
- [ ] `git diff --check`.
- [ ] The promotion change touches only its intended paths: `ds41rt.config`,
      `ds41rt.build-v10.config` (comment), the five older
      `examples/configs/*-native.config`, `README.md`, `docs/release-v10-notes.md`,
      the adjacent docs and the promotion tests. No measured report, manifest or
      generated campaign document is included.
