# DS41RT v12 release notes

**Status: DRAFT.** This record becomes PUBLISHED only after the v12 image pair
passes live qualification, the five-mode campaign, registry publication, and
fresh-pull verification. The ordered gates are in
[release-v12-checklist.md](release-v12-checklist.md).

V12 moves normal and constrained stochastic target-token selection to the CUDA
sampler in the native serving path. It preserves the existing API controls:
temperature, top-p, top-k, min-p, and deterministic seed plus absolute token
position. Greedy uses its compact device path. Large top-k values beyond the
host's retained-list capacity remain a counted CPU fallback. The GPU sampler
uses a fixed 1024-thread row geometry, three-pass integer radix selection for
top-k, and deterministic integer mass histograms for top-p survivor rows.

The GPU and CPU can select different tokens from the same seeded stochastic
row because their weight accumulation differs. V12 reports that difference by
row class and checks masks, ties, replay, distribution and constrained
decoding. The exact measurements, image identities, and registry digests will
be filled from the final run in
[release-v12-performance.md](release-v12-performance.md).
