# DS41RT v12 release checklist

Status: in progress. Follow [OPERATIONS.md](OPERATIONS.md) for branch and
publication order. The image build source, benchmark images, and release tag
must each be recorded by their own immutable identity.

## Runtime qualification

- [ ] Native selftest passes on the final CUDA source for SM120; SM121 compiles.
- [ ] The 1024-thread GPU sampler's seeded CPU/GPU residual rates are recorded
  per cell from the final binary, with exact mask, tie, greedy, and boundary
  cases passing.
- [ ] The normal and constrained live API qualifier passes on the candidate
  image pair, including seeded replay.
- [ ] Fixed-logit latency is measured on all five requested profiles at one,
  four, and 48 rows, including the top-p survivor path.

## Build and deployment

- [ ] Freeze a clean source commit on `dev`, including
  `ds41rt.build-v12.config`, and build from a standalone clone of that commit.
- [ ] `./build.sh --config ds41rt.build-v12.config --dry-run` selects the v12
  coordinator and Spark pair; the runtime default still selects v11 until
  promotion.
- [ ] Build exits successfully, artifact verification passes, and both images
  carry the same source revision and universal Spark roles `tp2;tp3;tp6`.
- [ ] Launch that exact pair on one RTX plus four Spark workers with dSpark on.

## Measurement

- [ ] Five discarded warmups and three interleaved repeats of the five canonical
  modes complete on nine content types plus the weight-zero counting diagnostic.
- [ ] `validate-sampling-campaign.py` passes on 15 raw reports and the captured
  hardware/image identity; `aggregate-sampling-decode.py` derives the weighted
  and per-content table without manual transcription.
- [ ] README and [release-v12-performance.md](release-v12-performance.md) name
  source/image identities, work and timing policy, repeat spread, exact raw
  evidence hashes, and the measured CPU/GPU mismatch rate.

## Publication

- [ ] Capture the v11 `latest` registry digests before pushing and confirm both
  `:v12` tags are absent.
- [ ] Publish exactly the v12 coordinator and Spark images and their `latest`
  aliases; capture the registry's returned digests.
- [ ] Verify fresh anonymous pulls on the matching architectures and archive
  publication evidence with `SHA256SUMS`.
- [ ] Promote `ds41rt.config` and examples to v12, verify dry runs, update
  [release-v12-notes.md](release-v12-notes.md), and fast-forward `main` and
  `release/v12` to the qualified release commit. Create and push annotated tag
  `v12`; retain development on `dev`.
