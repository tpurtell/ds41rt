//! Generalized official-checkpoint TP shard staging.
//!
//! These tests cover `V41ExpertSelection::BackboneTp { layer, expert, rank,
//! world }`, which stages an explicitly sized shard of one official native
//! FP4/E8M0 backbone routed expert for a TP world of 2, 3 or 4.
//!
//! Two independent layers of evidence:
//!
//! * Catalog-free geometry: `V41ExpertStaging::backbone_tp_geometry` proves
//!   that adjacent shards partition the full expert exactly, that the W2
//!   packed-byte and group-32 scale column offsets stay aligned, and that
//!   invalid world/rank/shape extents are rejected (including overflow).
//! * The real cached official checkpoint (read-only, header + single-expert
//!   reads): for one expert, every generic shard is compared both against an
//!   independent raw-source oracle (applied directly to the safetensors bytes
//!   with test-local slicing arithmetic) and against the pre-existing
//!   `Backbone` (TP4), `BackboneTp2` and `BackboneFull` selectors, so the new
//!   path is not validated only against its own helper.
//!
//! The real-checkpoint test soft-skips when no snapshot is available; point
//! `DS41RT_OFFICIAL_SNAPSHOT` at a local snapshot to force it.
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use ds41rt_loader::{
    read_official_v41_catalog, resolve_snapshot, OfficialV41Catalog, V41ExpertSelection,
    V41ExpertStaging, V41TensorPlacement, OFFICIAL_V41_MODEL_ID,
};

const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 2304;
const W2_ROW_BYTES: usize = INTERMEDIATE / 2; // 1152 packed bytes
const SUFFIXES: [&str; 6] = [
    "w1.weight",
    "w3.weight",
    "w2.weight",
    "w1.scale",
    "w3.scale",
    "w2.scale",
];

/// Geometry algebra: adjacent TP2/TP3/TP4 shards partition the full official
/// expert exactly on both slicing axes.
#[test]
fn explicit_world_2_3_4_shards_partition_the_official_expert_exactly() {
    for world in 2..=4usize {
        let shard_intermediate = INTERMEDIATE / world;
        let weight_column_bytes = shard_intermediate / 2;
        let scale_column_bytes = shard_intermediate / 32;
        assert_eq!(shard_intermediate * world, INTERMEDIATE);
        // Group-32 E8M0 scales and packed FP4 nibbles must both tile the slice.
        assert_eq!(shard_intermediate % 32, 0);
        assert_eq!((shard_intermediate / 2) * 2, shard_intermediate);

        let mut w13_weight = 0usize;
        let mut w13_scale = 0usize;
        let mut w2_weight = 0usize;
        let mut w2_scale = 0usize;
        let mut expected_weight_column = 0usize;
        let mut expected_scale_column = 0usize;
        for rank in 0..world {
            let geometry =
                V41ExpertStaging::backbone_tp_geometry(INTERMEDIATE, HIDDEN, world, rank).unwrap();
            assert_eq!(geometry.world(), world);
            assert_eq!(geometry.rank(), rank);
            assert_eq!(geometry.full_intermediate_size(), INTERMEDIATE);
            assert_eq!(geometry.hidden_size(), HIDDEN);
            assert_eq!(geometry.intermediate_size(), shard_intermediate);
            assert_eq!(geometry.w2_weight_row_bytes(), W2_ROW_BYTES);

            // W1/W3 output-row slices: contiguous rows whose byte extents tile
            // the packed weight and E8M0 scale planes.
            assert_eq!(
                geometry.w13_weight_shard_bytes(),
                shard_intermediate * (HIDDEN / 2)
            );
            assert_eq!(
                geometry.w13_scale_shard_bytes(),
                shard_intermediate * (HIDDEN / 32)
            );
            // W2 input-column slices: adjacent byte/group ranges, no gap or
            // overlap, so rank order reconstructs every source row.
            assert_eq!(geometry.w2_weight_shard_bytes(), HIDDEN * (shard_intermediate / 2));
            assert_eq!(geometry.w2_scale_shard_bytes(), HIDDEN * (shard_intermediate / 32));
            assert_eq!(
                geometry.w2_weight_column_offset_bytes(),
                expected_weight_column
            );
            assert_eq!(
                geometry.w2_scale_column_offset_bytes(),
                expected_scale_column
            );
            // Packed W2 starts on a byte boundary and each scale slice starts
            // on a 32-value group boundary.
            let weight_element_offset = geometry.w2_weight_column_offset_bytes() * 2;
            assert_eq!(weight_element_offset % 32, 0);
            assert_eq!(geometry.w2_scale_column_offset_bytes() * 32 % 32, 0);
            assert!(weight_element_offset + weight_column_bytes * 2 <= INTERMEDIATE);
            assert!(
                geometry.w2_scale_column_offset_bytes() + scale_column_bytes
                    <= INTERMEDIATE / 32
            );

            w13_weight += geometry.w13_weight_shard_bytes();
            w13_scale += geometry.w13_scale_shard_bytes();
            w2_weight += geometry.w2_weight_shard_bytes();
            w2_scale += geometry.w2_scale_shard_bytes();
            expected_weight_column += weight_column_bytes;
            expected_scale_column += scale_column_bytes;
        }
        // Every rank's slices tile the complete source tensor byte-for-byte.
        assert_eq!(w13_weight, INTERMEDIATE * (HIDDEN / 2));
        assert_eq!(w13_scale, INTERMEDIATE * (HIDDEN / 32));
        assert_eq!(w2_weight, HIDDEN * (INTERMEDIATE / 2));
        assert_eq!(w2_scale, HIDDEN * (INTERMEDIATE / 32));
        // The final offsets consume exactly one source row.
        assert_eq!(expected_weight_column, W2_ROW_BYTES);
        assert_eq!(expected_scale_column, INTERMEDIATE / 32);
    }
}

/// A TP3 shard of a 768-wide expert (`I768`) must satisfy the same contract.
#[test]
fn tp3_covers_i768_shards() {
    let geometry = V41ExpertStaging::backbone_tp_geometry(768, HIDDEN, 3, 1).unwrap();
    assert_eq!(geometry.intermediate_size(), 256);
    assert_eq!(geometry.w2_weight_shard_bytes(), HIDDEN * 128);
    assert_eq!(geometry.w2_scale_shard_bytes(), HIDDEN * 8);
    assert_eq!(geometry.w2_weight_column_offset_bytes(), 128);
    assert_eq!(geometry.w2_scale_column_offset_bytes(), 8);
    assert_eq!(geometry.w2_weight_row_bytes(), 384);
}

/// Invalid worlds, ranks, non-tiling shapes and extents that cannot be
/// represented are rejected before any read.
#[test]
fn explicit_tp_geometry_rejects_invalid_world_rank_and_shapes() {
    for world in [0usize, 1, 5, 8, usize::MAX] {
        assert!(
            V41ExpertStaging::backbone_tp_geometry(INTERMEDIATE, HIDDEN, world, 0).is_err(),
            "world {world} must be rejected"
        );
    }
    for (world, rank) in [(2usize, 2usize), (3, 3), (4, 4), (7, 8), (0, 0)] {
        assert!(
            V41ExpertStaging::backbone_tp_geometry(INTERMEDIATE, HIDDEN, world, rank).is_err(),
            "rank {rank} for world {world} must be rejected"
        );
    }
    // 2300 / 2 = 1150 intermediate values, which is not a multiple of 32.
    assert!(V41ExpertStaging::backbone_tp_geometry(2300, HIDDEN, 2, 0).is_err());
    // 2304 is not divisible by 5 (and five ranks are outside the contract).
    assert!(V41ExpertStaging::backbone_tp_geometry(INTERMEDIATE, HIDDEN, 5, 0).is_err());
    // Hidden size must tile the group-32 scale plane.
    assert!(V41ExpertStaging::backbone_tp_geometry(INTERMEDIATE, 5119, 2, 0).is_err());
    assert!(V41ExpertStaging::backbone_tp_geometry(INTERMEDIATE, 0, 2, 0).is_err());
    assert!(V41ExpertStaging::backbone_tp_geometry(0, HIDDEN, 2, 0).is_err());
    // Extent product overflow: 32 * (usize::MAX - 31) does not fit.
    assert!(V41ExpertStaging::backbone_tp_geometry(64, usize::MAX - 31, 2, 0).is_err());
}

fn official_snapshot() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("DS41RT_OFFICIAL_SNAPSHOT") {
        return Some(PathBuf::from(path));
    }
    resolve_snapshot(OFFICIAL_V41_MODEL_ID, None)
        .ok()?
        .snapshot_path
}

fn cached_snapshot(model_id: &str) -> Option<PathBuf> {
    resolve_snapshot(model_id, None).ok()?.snapshot_path
}

/// The generic native selection is scoped to the MXFP4 official checkpoint.
/// Compressed EXL3 and ModelOpt NVFP4 publications must keep their own staging
/// even though the enum variant itself is shared.
#[test]
fn generic_selection_is_rejected_for_exl3_and_nvfp4_publications() {
    const EXL3_MODEL_ID: &str = "diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000";
    const NVFP4_MODEL_ID: &str = "nvidia/DeepSeek-V4.1-Flash-NVFP4";
    let selection = V41ExpertSelection::BackboneTp {
        layer: 0,
        expert: 0,
        rank: 0,
        world: 2,
    };
    let mut checked = 0usize;

    if let Some(snapshot) = cached_snapshot(EXL3_MODEL_ID) {
        if let Ok(catalog) = read_official_v41_catalog(EXL3_MODEL_ID, &snapshot) {
            assert!(catalog.exl3().is_some(), "expected an EXL3 manifest");
            assert!(
                catalog.expert_staging(selection).is_err(),
                "generic native staging must not read a compressed EXL3 expert"
            );
            eprintln!("EXL3 publication rejected generic native TP staging");
            checked += 1;
        }
    }
    if let Some(snapshot) = cached_snapshot(NVFP4_MODEL_ID) {
        if let Ok(catalog) = read_official_v41_catalog(NVFP4_MODEL_ID, &snapshot) {
            assert!(catalog.nvfp4().is_some(), "expected an NVFP4 contract");
            assert!(
                catalog.expert_staging(selection).is_err(),
                "generic native FP4/E8M0 staging must not read a ModelOpt NVFP4 expert"
            );
            assert!(
                catalog.nvfp4_expert_staging(selection).is_err(),
                "NVFP4 staging has no generic TP selection"
            );
            eprintln!("NVFP4 publication rejected generic native TP staging");
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("skipping EXL3/NVFP4 rejection check: no cached compressed publication");
    }
}

struct Staged {
    names: Vec<String>,
    regions: Vec<Vec<u8>>,
    staging_bytes: usize,
    scratch_bytes: usize,
    intermediate: usize,
    world: Option<usize>,
}

fn stage(catalog: &OfficialV41Catalog, selection: V41ExpertSelection) -> Staged {
    let plan = catalog.expert_staging(selection).expect("expert staging plan");
    let mut staging = vec![0u8; plan.staging_bytes()];
    let mut scratch = vec![0u8; plan.minimum_read_scratch_bytes()];
    plan.prefetch().expect("prefetch");
    plan.read_into(&mut staging, &mut scratch).expect("staging read");
    Staged {
        names: plan.tensor_names().to_vec(),
        regions: plan
            .tensor_ranges()
            .iter()
            .map(|range| staging[range.clone()].to_vec())
            .collect(),
        staging_bytes: plan.staging_bytes(),
        scratch_bytes: plan.minimum_read_scratch_bytes(),
        intermediate: plan.intermediate_size(),
        world: plan.tp_world(),
    }
}

fn tensor_axis(catalog: &OfficialV41Catalog, name: &str) -> usize {
    match &catalog.tensor(name).unwrap().placement {
        V41TensorPlacement::BackboneExpertTp4 { axis, .. } => *axis,
        other => panic!("unexpected placement {other:?} for {name}"),
    }
}

fn raw_tensor(catalog: &OfficialV41Catalog, name: &str) -> Vec<u8> {
    let tensor = catalog.tensor(name).unwrap();
    let mut bytes = vec![0u8; usize::try_from(tensor.metadata.byte_length).unwrap()];
    File::open(catalog.snapshot().join(&tensor.shard))
        .unwrap()
        .read_exact_at(&mut bytes, tensor.metadata.byte_offset)
        .unwrap();
    bytes
}

/// Independent oracle: slice the raw safetensors payload with test-local
/// arithmetic, never touching a catalog read helper.
fn raw_shard(catalog: &OfficialV41Catalog, name: &str, world: usize, rank: usize) -> Vec<u8> {
    let raw = raw_tensor(catalog, name);
    assert_eq!(raw.len() % world, 0, "{name} does not divide into {world}");
    let shard_bytes = raw.len() / world;
    if tensor_axis(catalog, name) == 0 {
        return raw[rank * shard_bytes..(rank + 1) * shard_bytes].to_vec();
    }
    let rows = catalog.tensor(name).unwrap().metadata.shape[0];
    let row_bytes = raw.len() / rows;
    let column_bytes = row_bytes / world;
    raw.chunks_exact(row_bytes)
        .flat_map(|row| row[rank * column_bytes..(rank + 1) * column_bytes].iter().copied())
        .collect()
}

/// Reassemble a full tensor from per-rank shard regions: concatenate row
/// slices, or interleave successive column windows row by row.
fn reassemble(regions: &[Vec<u8>], axis: usize, rows: usize) -> Vec<u8> {
    if axis == 0 {
        return regions.concat();
    }
    let mut out = Vec::new();
    for row in 0..rows {
        for region in regions {
            let row_bytes = region.len() / rows;
            out.extend_from_slice(&region[row * row_bytes..(row + 1) * row_bytes]);
        }
    }
    out
}

#[test]
fn generic_world_shards_match_raw_official_source_bytes() {
    let Some(snapshot) = official_snapshot() else {
        eprintln!(
            "skipping official-checkpoint TP staging test: no cached snapshot \
             (set DS41RT_OFFICIAL_SNAPSHOT to force it)"
        );
        return;
    };
    let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, &snapshot)
        .expect("official V4.1 catalog");
    assert!(catalog.exl3().is_none(), "official checkpoint is not EXL3");
    assert!(
        catalog.nvfp4().is_none(),
        "official checkpoint is native FP4/E8M0, not ModelOpt NVFP4"
    );
    let (layer, expert) = (0usize, 0usize);
    let prefix = format!("layers.{layer}.ffn.experts.{expert}");

    // Fixed selectors keep their exact existing extents, scratch bounds and
    // source order; the generic path must reproduce their bytes.
    let full = stage(&catalog, V41ExpertSelection::BackboneFull { layer, expert });
    assert_eq!(full.staging_bytes, 18_800_640);
    assert_eq!(full.intermediate, INTERMEDIATE);
    assert_eq!(full.scratch_bytes, 0);
    assert_eq!(full.world, None);
    let legacy_tp4 = (0..4)
        .map(|rank| stage(&catalog, V41ExpertSelection::Backbone { layer, expert, rank }))
        .collect::<Vec<_>>();
    let legacy_tp2 = (0..2)
        .map(|rank| stage(&catalog, V41ExpertSelection::BackboneTp2 { layer, expert, rank }))
        .collect::<Vec<_>>();
    for staged in &legacy_tp4 {
        assert_eq!(staged.staging_bytes, 4_700_160);
        assert_eq!(staged.intermediate, INTERMEDIATE / 4);
        assert_eq!(staged.scratch_bytes, W2_ROW_BYTES);
    }
    for staged in &legacy_tp2 {
        assert_eq!(staged.staging_bytes, 9_400_320);
        assert_eq!(staged.intermediate, INTERMEDIATE / 2);
        assert_eq!(staged.scratch_bytes, W2_ROW_BYTES);
    }

    for staged in std::iter::once(&full)
        .chain(legacy_tp4.iter())
        .chain(legacy_tp2.iter())
    {
        let expected: Vec<_> = SUFFIXES
            .iter()
            .map(|suffix| format!("{prefix}.{suffix}"))
            .collect();
        assert_eq!(staged.names, expected, "source order must stay W1,W3,W2,S1,S3,S2");
        assert_eq!(staged.regions.len(), 6);
    }

    for world in 2..=4usize {
        let expected_staging = match world {
            2 => 9_400_320,
            3 => 6_266_880,
            _ => 4_700_160,
        };
        let shards = (0..world)
            .map(|rank| {
                stage(
                    &catalog,
                    V41ExpertSelection::BackboneTp {
                        layer,
                        expert,
                        rank,
                        world,
                    },
                )
            })
            .collect::<Vec<_>>();
        for (rank, staged) in shards.iter().enumerate() {
            assert_eq!(staged.world, Some(world));
            assert_eq!(staged.intermediate, INTERMEDIATE / world);
            assert_eq!(staged.names, full.names);
            assert_eq!(staged.staging_bytes, expected_staging);
            // The W2 read scratch bound is one packed source row.
            assert_eq!(staged.scratch_bytes, W2_ROW_BYTES);
            for (slot, name) in staged.names.iter().enumerate() {
                // Independent raw-source oracle.
                assert_eq!(
                    staged.regions[slot],
                    raw_shard(&catalog, name, world, rank),
                    "{name} world {world} rank {rank} differs from source bytes"
                );
                // And the pre-existing fixed selector, where one exists.
                let reference = if world == 4 {
                    Some(&legacy_tp4[rank].regions[slot])
                } else if world == 2 {
                    Some(&legacy_tp2[rank].regions[slot])
                } else {
                    None
                };
                if let Some(reference) = reference {
                    assert_eq!(
                        &staged.regions[slot], reference,
                        "{name} world {world} rank {rank} differs from the fixed selector"
                    );
                }
            }
        }
        // Adjacent shards partition every full source tensor exactly.
        for (slot, name) in full.names.iter().enumerate() {
            let axis = tensor_axis(&catalog, name);
            let regions: Vec<_> = shards.iter().map(|staged| staged.regions[slot].clone()).collect();
            let rebuilt = reassemble(&regions, axis, HIDDEN);
            assert_eq!(
                rebuilt, full.regions[slot],
                "{name} world {world} shards do not reconstruct the expert"
            );
        }
    }

    // Scratch/staging admission is enforced before any read, and a valid plan
    // still reads after prefetch.
    let plan = catalog
        .expert_staging(V41ExpertSelection::BackboneTp {
            layer,
            expert,
            rank: 0,
            world: 3,
        })
        .unwrap();
    let mut staging = vec![0u8; plan.staging_bytes()];
    let mut scratch = vec![0u8; plan.minimum_read_scratch_bytes()];
    assert!(plan
        .read_into(&mut staging[..plan.staging_bytes() - 1], &mut scratch)
        .is_err());
    assert!(plan
        .read_into(&mut staging, &mut scratch[..plan.minimum_read_scratch_bytes() - 1])
        .is_err());
    assert!(staging.iter().all(|&byte| byte == 0));
    plan.prefetch().unwrap();
    plan.read_into(&mut staging, &mut scratch).unwrap();

    // Validation rejects invalid generic selections and keeps the legacy
    // selector bounds unchanged.
    for selection in [
        V41ExpertSelection::BackboneTp { layer, expert, rank: 0, world: 0 },
        V41ExpertSelection::BackboneTp { layer, expert, rank: 0, world: 1 },
        V41ExpertSelection::BackboneTp { layer, expert, rank: 0, world: 5 },
        V41ExpertSelection::BackboneTp { layer, expert, rank: 2, world: 2 },
        V41ExpertSelection::BackboneTp { layer, expert, rank: 3, world: 3 },
        V41ExpertSelection::BackboneTp { layer, expert, rank: 4, world: 4 },
        V41ExpertSelection::BackboneTp { layer: 40, expert, rank: 0, world: 2 },
        V41ExpertSelection::BackboneTp { layer, expert: 384, rank: 0, world: 2 },
        V41ExpertSelection::Backbone { layer, expert, rank: 4 },
        V41ExpertSelection::BackboneTp2 { layer, expert, rank: 2 },
        V41ExpertSelection::BackboneFull { layer: 40, expert },
        V41ExpertSelection::BackboneFull { layer, expert: 384 },
    ] {
        assert!(
            catalog.expert_staging(selection).is_err(),
            "selection {selection:?} must be rejected"
        );
    }
}
