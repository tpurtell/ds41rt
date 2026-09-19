# DS41RT v8 release notes

V8 enables the NVFP4 W4A4 optimizations prepared on the development branch and
re-measures that quant on the published images. Only the NVFP4 configurations
were re-campaigned this cycle; the official MXFP4 and EXL3 figures in the README
are the v6 and v7 campaigns respectively and are labelled as such there.

## What changed

The NVFP4 routed-expert path previously used the generic deterministic fused
kernel. This cycle wires the optimizations that were reachable without changing
the checkpoint format:

- **Load-time padding** of the expert payload (`export_b12x_v41_nvfp4_aot.py`,
  `v41_experts/nvfp4.rs`, the FFI launch contract).
- **Adaptive NVFP4 splitting** exposed through the exporter, with the
  corresponding SM121 Spark share-input kernels (SparkInfer revision
  `4b0954148523b5a2e93813f963d483ffd350b9c9`, pushed to the fork before pinning).
- **An NVFP4 speculative cost profile** selected by
  `configure_cost_model(&transport, catalog.nvfp4().is_some())`.

## Measured result

Both layouts, published images, three samples per cell, C1 dSpark decode.

| Measurement | NVFP4 1x v7 | NVFP4 1x v8 | NVFP4 2x v7 | NVFP4 2x v8 |
|---|---:|---:|---:|---:|
| C1 code decode | 86.58 | **112.55** (+30.0%) | 106.38 | **145.96** (+37.2%) |
| Counting decode | 114.06 | **144.66** (+26.8%) | 149.98 | **192.42** (+28.3%) |
| Weighted decode | 66.81 | **81.74** (+22.4%) | 78.57 | **100.78** (+28.3%) |
| Best prefill | 4,141 | **5,237** (+26.5%) | 7,370 | 7,371 |

Prefill gains are shape-dependent. On 1x they are broad (+16 to +22% median at
every suffix length); on 2x they are concentrated in short suffixes (+9.2%
median at +1K, +0.5% at +32K), and because "best prefill" is the maximum cell —
the 0K/+32K cell — the 2x headline barely moves even though most of the matrix
improved. The full 30-cell matrices are in the report.

Both layouts completed the full battery on the published images: retained-context
decode with its separate 2K control, counting/code/topic concurrency scaling
through C16, mixed traffic, target-only decode, startup and post-readiness
memory, and three high-effort tool-call evaluations each (1x 156/157/152, 2x
156/157 over 176).

## Costs

The gain is not free. On the Sparks, per-layer residency rises from
1,911,035,904 to 2,123,372,544 bytes (+11.1%, about 68.8 GB to 76.4 GB over the
36 layers) and layer loading slows from roughly 935 ms to 2,000–2,263 ms. That
is why expert readiness for NVFP4 1x now measures ~171 s end to end against
~47 s at v7, and it reduces the headroom left on a 100 GiB Spark budget.

The NVFP4 2x battery was measured with a **512 MiB global KV pool** (the v7
campaign used the default), which leaves the TP2 pair enough device headroom for
its load-time and request-time allocations. This is part of the measured
configuration, recorded here as the compact EXL3 profile's 2 GiB pool is.

## Container images

- `ghcr.io/tpurtell/ds41rt-coordinator:v8` (`linux/amd64`), digest
  `sha256:08c2d6df9a0a6a365eff2c014172478b40d9f39d06437a1c9244c566181b9e40`
- `ghcr.io/tpurtell/ds41rt-spark-expert:v8` (`linux/arm64`), digest
  `sha256:9907983992916bb8e0f35ab869e12706cfe4613cc6dcd93f4ff53796c2d80bf6`

Both also carry `latest`. Both roles are built from one staged tree at
`5a56f0e`, so they share `org.opencontainers.image.revision`
(`5a56f0eb9d8caea050f84f96332c66786e177607`) and
`io.ds41rt.sparkinfer.revision`
(`4b0954148523b5a2e93813f963d483ffd350b9c9`); `run.sh` requires that equality
across every Spark before it will deploy. `ds41rt.config` points both roles at
`:v8`.

## Reading

The optimizations are real and reproduce on both layouts, with the 2x path
gaining slightly more on decode than 1x. The prefill headline for 2x should be
read together with its matrix, for the suffix reason above. The startup and
expert-residency costs are measured, not estimated, and should be weighed against
the throughput gain when sizing a deployment.

See the [NVFP4 performance report](release-v8-nvfp4-performance.md) for the full
tables, raw-record digests and per-configuration qualification.
