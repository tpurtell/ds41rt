# DS41RT v13 release checklist

Status: in progress. Follow [OPERATIONS.md](OPERATIONS.md) for branch and
publication order. The image build source, benchmark images, and release tag
must each be recorded by their own immutable identity.

## Runtime change

- [x] Replace every adaptive dSpark length selector with the online
  bandwidth-balance policy ([design](dspark-bandwidth-policy.md)); delete the
  suffix-removal selector, confidence cutoff, reuse floor, placement cost
  profiles, `DS41RT_ADAPTIVE_COST_*` and the offline fit scripts.
- [x] Core policy tests (estimator recovery, noise/outlier robustness,
  exhaustive-search agreement, warmup/fixed modes, cross-request sharing,
  16-row groups, reliability accounting) and daemon CLI/scheduler/speculative
  tests pass; script tests pass.

## Build and deployment

- [x] Freeze a clean source commit on `dev`, including
  `ds41rt.build-v13.config`, and build from a standalone clone of that commit.
- [x] Build exits successfully and both images carry the same source revision
  and universal Spark roles `tp2;tp3;tp6`.
- [x] Launch that exact pair on one RTX plus four Spark workers with dSpark on;
  the API smoke check passes and `/v1/stats` exports the policy state.
- [ ] Launch the same pair on two RTX plus four Spark workers.

## Evaluation against v12 adaptive

- [ ] Interleaved six-session A/B (v12, v13, v13, v12, v12, v13) on one RTX:
  paired greedy and T0.7/top-p 0.9 batteries, code/topic C1–C16, mixed C1–C16.
- [ ] The same A/B on two RTX.
- [ ] The new policy wins or ties within noise on every family, with no quality
  failures introduced.

## Measurement

- [ ] README decode campaign on the final images for one and two RTX: release
  decode, per-case acceptance, concurrency, mixed traffic, retained-context
  decode with its 2K control, and target-only decode.
- [ ] README and the v13 performance report name source/image identities and
  raw evidence.

## Publication

- [ ] Capture the v12 `latest` registry digests and confirm both `:v13` tags are
  absent before pushing.
- [ ] Publish exactly the v13 coordinator and Spark images and their `latest`
  aliases; capture the registry's returned digests.
- [ ] Verify fresh anonymous pulls on the matching architectures.
- [ ] Promote `ds41rt.config` and examples to v13, verify dry runs, write
  [release-v13-notes.md](release-v13-notes.md), and fast-forward `main` and
  `release/v13` to the qualified release commit. Create and push annotated tag
  `v13`; retain development on `dev`.
- [ ] Publish a non-draft GitHub release with the evidence tarball and hashes.
