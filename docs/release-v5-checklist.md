# DS41RT v5 release checklist

Status: qualification and publication complete.

- [x] Add generic 2–5 bpw routed-expert EXL3 loading and execution on RTX and
  DGX Spark; verify mixed 3/4-bit projection shapes and reference numerics.
- [x] Add paired TP4 Spark ownership and calibrated one/two-RTX adaptive cost
  profiles while preserving the native full-checkpoint path.
- [x] Reconstruct the FP4-PLE checkpoint on all four Sparks with 48 hard-linked
  shards and four transferred replacement shards; verify every snapshot.
- [x] Push the qualified SparkInfer dependency to the fork's `master` branch
  and lock its exact source revision and tree hash.
- [x] Build clean coordinator and Spark candidate images and pass all five
  standard `run.sh` deployment smokes, including the FP4-PLE variant.
- [x] Complete matched three-sample full/EXL3 performance qualification across
  one/two RTX cards, including content, concurrency, mixed traffic, retained
  context, prefill, startup, capacity, and memory.
- [x] Complete nine-category adaptive acceptance collection, including
  high-effort reasoning code and grammar-constrained output.
- [x] Complete exactly three fresh high-thinking tool campaigns for EXL3 with
  FP8 PLE and three for EXL3 with FP4 PLE; preserve all failures.
- [x] Pass focused vision and needle checks for the EXL3 deployments.
- [x] Record checkpoint sizes, routed projection tiers, tensor shapes, PLE
  geometry, and hard-link accounting for both quant variants.
- [x] Complete bounded fixed-history exact top-1 agreement against the full
  checkpoint for both EXL3 variants.
- [x] Replicate every v5 performance and analysis table in README and the
  linked report, with the 400 W limit, standard memory speed, and cache byte
  and token capacities stated up front.
- [x] Assemble deterministic coordinator and Spark binary packages containing
  the qualified EXL3 AOT payloads and recursive checksums.
- [x] Publish verified v5/latest coordinator and Spark images, GitHub release
  notes, binary packages, qualification evidence, and release checksums.
- [x] Create `release/v5` at the exact final `v5` tag commit, verify remote
  identities, leave the standard port-8000 service healthy, and clean temporary
  release state.
