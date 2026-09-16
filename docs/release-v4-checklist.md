# DS41RT v4 release checklist

Status: final performance and focused regression review completed; publication
is pending. The published September 14 v3 is unchanged.

- [x] Analyze applicable upstream changes and record native ABI, numerical,
  memory, and performance treatment by component.
- [x] Resolve the SparkInfer merge, preserve native exports, and verify the
  selected source revision and tree hash.
- [x] Compare attention, projections, mHC, index selection, routed/shared
  experts, vocabulary, Engram/loading, and adaptive draft costs; select the
  implementations supported by the recorded comparisons.
- [x] Integrate the tested dependency pin and build flags on dev.
- [x] Build isolated clean candidate images and distribute the matching Spark
  image to all four workers.
- [x] Start target-only and standard dual-RTX dSpark through run.sh on port 8000.
- [x] Pass 32K, 128K, 512K, and 1.04M high-thinking needle retrieval with cold
  and exact reuse; pass vision, divergent/shorter cache branches, retained parent
  isolation, C16 branching, streaming cancellation, and recovery.
- [x] Complete and review exactly three final high-thinking tool-eval runs,
  preserving failures and complete raw evidence.
- [x] Reproduce and correct the post-mixed decode slowdown; verify bounded
  graph reuse with current-input CUDA replay, prefix equivalence, C16 divergent
  branches and cold/exact 1.04M retrieval. Preserve pre-fix measurements.
- [x] Build and launch the corrected revision through standard build.sh/run.sh
  and verify matching images on all four Spark workers.
- [x] Resolve and qualify the single-RTX mixed C16 memory exhaustion without
  reducing the configured KV pool; verify graph retention remains performant.
- [x] Refresh all one/two-RTX performance tables with matched prompts and three
  samples: target/dSpark content types, counting/code/topic concurrency, mixed
  traffic, prefill, and retained-context decode. Add the separate 2K retained
  context measurement. Review any material regressions.
- [x] Verify clean single-RTX launch, final startup times, memory use, cache
  capacity, and standard power/memory-clock settings.
- [x] Replicate all performance tables in README and the linked report; include
  the 400 W limit, standard memory speed, cache bytes and token counts, and
  actual RTX/Spark placement budgets.
- [x] Promote the accepted merged fork to its master branch and verify the
  dependency remains reachable at the locked revision.
- [x] Finalize release version/configuration and build the final source; verify
  the final image identities and standard launch before publication.
  [Final clean-build evidence](sparkinfer-upstream-v4-final-build-20260916.json)
  records revision `3924227`, all five source labels, packaged checksums, and
  the 52.71-second dual-RTX standard launch. Performance qualification and
  remote publication are recorded separately; see the [final report](release-v4-performance.md).
- [ ] Publish coordinator and Spark images, release notes, binaries, evidence,
  and checksums; verify remote digests/assets and release branch/tag identities.
- [ ] Leave the standard service healthy, clean disposable intermediates while
  retaining evidence, and push all final commits.

Routed EXL3/Trellis and parallel attention/projection remain future work, with
analysis recorded in the integration document.
