# Cluster hosts and build filesystem safety

Inventory collected directly over SSH on 2026-09-20. Reachability and hardware
inventory are not serving, checkpoint, or six-rank RDMA qualification.

## Host allocation

- Coordinator: **raptor**, x86-64, two RTX PRO 6000 Blackwell GPUs (SM120).
- Existing Spark order remains **ostrich, dodo, emu, kiwi** (global ranks 0–3).
- **rhea is the fifth Spark (global rank 4); moa is the sixth (global rank 5).**
- User restriction: use rhea/moa **only when a full six-Spark set is needed**.
  Do not use them for independent builds, single-node microbenchmarks, spare
  capacity or replacement of a member of the four-Spark baseline.
- No default serving configuration was changed. Six-Spark hardware is now
  connected, but TP3×EP2 and TP2×EP3 still need full runtime qualification.

For group-major assignment: TP3×EP2 groups are `(ostrich,dodo,emu)` and
`(kiwi,rhea,moa)`; TP2×EP3 groups are `(ostrich,dodo)`, `(emu,kiwi)`,
`(rhea,moa)`. These are fully replicated expert groups, not expert-ID shards.

## New Spark inventory

| Field | rhea | moa |
| --- | --- | --- |
| SSH hostname | `rhea` | `moa` |
| Architecture | aarch64 | aarch64 |
| GPU | NVIDIA GB10 | NVIDIA GB10 |
| Compute capability | 12.1 | 12.1 |
| GPU UUID | `GPU-21b21314-9c5a-41a2-80a4-63b222b2c84a` | `GPU-52635ab1-1d3c-1360-58ea-a25501d3f0aa` |
| GPU PCI address | `0000000F:01:00.0` | `0000000F:01:00.0` |
| Driver | 580.178.04 | 580.178.04 |
| Management (`enP7s7`) | `172.22.2.5/16` | `172.22.2.6/16` |
| Fabric A (`enp1s0f0np0`) | `10.55.0.5/24` | `10.55.0.6/24` |
| Fabric B (`enP2p1s0f0np0`) | `10.55.0.11/24` | `10.55.0.12/24` |
| RAM (`free -b`, total) | 130,661,208,064 bytes | 130,661,208,064 bytes |
| Swap total / used | 17,179,865,088 / 0 bytes | 17,179,865,088 / 0 bytes |
| Root/home filesystem | ext4, `/dev/nvme0n1p2` | ext4, `/dev/nvme0n1p2` |
| Root disk (`df -h`) | 3.7T total, 3.5T available | 3.7T total, 3.5T available |

`rdma link show` reports `rocep1s0f0/1` and `roceP2p1s0f0/1` ACTIVE/LINK_UP
on both hosts. Second ports are disabled. This is link state only: rail routing,
GIDs, bandwidth, and six-rank application transport have not been tested.
The PATH-selected `ibv_devinfo` warns about a missing Linuxbrew verbs config
and reports no devices despite `rdma link` showing active devices. Resolve the
userspace/provider selection before claiming RDMA readiness.

Docker is installed, but `docker ps` and `docker images` returned no entries
at inventory time. No official checkpoint snapshot directory was found at the
standard `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash` path.
Provision matching images, verified source and the pinned official checkpoint
only when preparing a full six-Spark run. No containers, workloads, model
transfers or configuration changes were made on rhea/moa during this inventory.
`nvidia-smi memory.total` reports N/A on these unified-memory GPUs; use actual
RAM and runtime CUDA measurements for admission, not advertised capacity.

## Mandatory build safety

**Never run Cargo or other builds from `/mnt/scratch`.** The user reports an
NTFS driver kernel bug and has locked this drive read-only. Do not remount it,
probe writes, or use a container alias to build on it. This covers source,
Cargo target/cache directories, native build directories and temporary files.

Use a unique `~/.cache/ds41rt/builds/<task>` path on raptor's regular root NVMe.
At inventory time `/` and that cache path are ext4 on `/dev/nvme0n1p2`.
Run `python3 scripts/assert-build-filesystem.py PATH...` before direct builds;
the release/WIP artifact helpers also enforce this before staging. The checker
resolves symlinks and asks `findmnt` for the backing filesystem, so an NTFS bind
mounted at `/scratch` is rejected as well. Unknown/read-only filesystems fail
closed. Do not assume a path name alone identifies its backing filesystem.

### TP/EP maintenance relocation

- Preserved artifacts, source snapshots and exact-service binary backups copied
  from `/mnt/scratch/ds41rt-tpep/` to
  `/home/tj/.cache/ds41rt/builds/tp-ep/`. Old copies left untouched/read-only.
- Fresh daemon Cargo target:
  `/home/tj/.cache/ds41rt/builds/daemon-tp-ep-target`.
  The old `/mnt/scratch/ds41rt-daemon-tp-ep-target` contains filesystem read
  errors (including fingerprint and incremental files); it is **not reused**.
- The old raptor `ds41rt-tpep-dev` container maps `/scratch` to NTFS and must
  remain stopped. The replacement `ds41rt-tpep-nvme-dev` maps `/scratch` to
  `/home/tj/.cache/ds41rt/builds/tp-ep` instead. No shared WIP or serving
  container is recreated or restarted by this relocation.
- Spark-local `/scratch` bindings under `/home/tj/ds41rt-tpep` are separate,
  local filesystems; the raptor ban does not imply they are NTFS. Verify each
  mount before use. Rhea/moa remain restricted as above.

Cargo/native compilation was not run as part of this urgent maintenance.
