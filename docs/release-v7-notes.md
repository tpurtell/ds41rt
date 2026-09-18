# DS41RT v7

V7 adds native support for two more quantized DeepSeek V4.1 Flash publications
— NVIDIA's ModelOpt NVFP4 (W4A4) and diffbot's EXL3 2.0 bpw — while the
official full checkpoint stays the launcher default and its measurements are
unchanged from v6.

- **NVIDIA `nvidia/DeepSeek-V4.1-Flash-NVFP4` (W4A4).** E2M1 weights with
  E4M3 K16 block scales, activations quantized in the kernel from BF16 rows.
  The loader writes no repacked weight: the checkpoint payload lands in the
  kernel-native `[up; gate]` order with one device-to-device copy per expert,
  and both scale planes are re-laid into 128x4 atoms on device. Runs in the
  same one- and two-card topologies as the official checkpoint, with the
  remainder of the layers on four Sparks.
- **diffbot `DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000`.** Uniform K=2
  trellis. On two cards all forty routed-expert layers are resident and no
  Spark worker is involved. On one card the new **compact** profile holds the
  card to a 32 GiB budget including headroom and serves the rest of the model
  from exactly two Sparks, which is how the checkpoint runs on a 32 GB card.
- **One GPU image for every part in the family.** The AOT artifacts are not
  SM-count specific — the host's SM count only bounds the cooperative grid —
  so the engine accepts a same-capability device with fewer SMs and clamps
  the launch cluster cap to the card. A 5090 needs no rebuild and no separate
  AOT set. (No physical 5090 was available to test; the grid behaviour was
  verified on an RTX PRO 6000 at 188 and 170 SM inventories.)
- Placement now budgets NVFP4 layers at their own measured cost rather than
  the native one, which had over-filled a card at the margin.
- EXL3 routed-expert support, added in v5 and disabled by default, is enabled
  and qualified for this checkpoint.

Container images:

- `ghcr.io/tpurtell/ds41rt-coordinator:v7` (`linux/amd64`), digest
  `sha256:d85608bbe14ce655a3bae4fd61655a0020099f056d357a40b8a3ac5115275c5e`
- `ghcr.io/tpurtell/ds41rt-spark-expert:v7` (`linux/arm64`), digest
  `sha256:aa477ff1c74fe4e815734c1e57d3fad325c5126ddbb324c92fd58f6d02fe856b`

Both also carry `latest`, and each image re-verifies the packaged EXL3 families
against the pinned SparkInfer revision at build time. Both roles are built from
one staged tree at `0107d01`, so they carry the same
`org.opencontainers.image.revision` (`0107d01e3d35d22b1dbc5de70c4e1a32d32d165f`)
and the same `io.ds41rt.sparkinfer.revision`
(`2bcbe122bf34d77fecbaf288df2f395b9c09e79e`); `run.sh` requires that equality
across every Spark before it will deploy.

All reported RTX measurements use a **400 W power limit per card, standard
memory speed, and three samples per cell**. See the
[NVFP4 report](release-v7-nvfp4-performance.md) and the
[EXL3 K2 report](release-v7-exl3-k2-performance.md) for exact results,
configuration detail, and what is still owed; the
[configuration chart](release-v7-configurations.svg) shows each profile's
memory composition.

Against the official checkpoint on the same two-card topology, W4A4 trades
decode throughput for half the routed-expert bytes per rank (weighted 80.1 vs
109.4 tok/s, best prefill 7,432 vs 8,355), and EXL3 2 bpw is the fastest
decode measured here at 145.1 weighted and 337.4 counting tok/s, at a lower
best prefill of 5,702. The single-card compact profile reaches 88.1 weighted
and 2,015 best prefill within its 32 GiB ceiling.
