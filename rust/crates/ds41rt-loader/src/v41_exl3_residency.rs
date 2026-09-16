//! Compressed layer residency matching B12x projection-native Trellis storage.
//! No weight dequantization or tier-wide temporary payload is required.
use crate::{OfficialV41Catalog, V41Exl3Manifest};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Serialize)]
pub enum V41Exl3Layer {
    Backbone(usize),
    Dspark(usize),
}

#[derive(Debug, Serialize)]
pub struct V41Exl3ResidentBuffer {
    pub name: String,
    /// Logical bytes; an empty projection still gets a 16-byte dummy allocation.
    pub bytes: usize,
    /// Initial native little-endian int32 words, also used for FP32 unit scales.
    pub initial_words: Vec<i32>,
}

#[derive(Debug, Serialize)]
pub struct V41Exl3Load {
    pub tensor: String,
    pub bytes: usize,
    /// (Buffer index, byte offset). Rotations are read once and copied to each tier.
    pub destinations: Vec<(usize, usize)>,
}

#[derive(Debug, Serialize)]
pub struct V41Exl3Residency {
    pub layer: V41Exl3Layer,
    pub world: usize,
    pub rank: usize,
    pub experts: usize,
    pub intermediate: usize,
    pub tiers: Vec<usize>,
    /// Gate/up/down counts for each tier, passed unchanged to native launches.
    pub projection_counts: Vec<[usize; 3]>,
    pub buffers: Vec<V41Exl3ResidentBuffer>,
    pub loads: Vec<V41Exl3Load>,
}

impl V41Exl3Residency {
    /// Device payload budget, excluding allocator alignment and execution scratch.
    pub fn bytes(&self) -> usize {
        self.buffers.iter().map(|b| b.bytes.max(16)).sum()
    }

    pub fn staging_bytes(&self) -> usize {
        self.loads.iter().map(|l| l.bytes).max().unwrap_or(0)
    }

    pub fn scratch_bytes(&self, catalog: &OfficialV41Catalog) -> Result<usize> {
        self.loads.iter().try_fold(0, |n, load| {
            Ok(n.max(
                catalog
                    .exl3_tensor_slice(&load.tensor, self.world, self.rank)?
                    .scratch_bytes(),
            ))
        })
    }

    /// Read one independent job into caller-owned pinned staging. Compressed
    /// bytes retain their checkpoint layout; destinations can be uploaded on
    /// the caller's stream before reusing this staging slot.
    pub fn read_into(
        &self,
        catalog: &OfficialV41Catalog,
        index: usize,
        staging: &mut [u8],
        scratch: &mut [u8],
    ) -> Result<usize> {
        let load = self
            .loads
            .get(index)
            .context("EXL3 load index out of range")?;
        if let Some(prefix) = load.tensor.strip_suffix(".trellis") {
            let mut mcg = [0u8; 4];
            catalog.read_exl3_tensor_into(
                &format!("{prefix}.mcg"),
                self.world,
                self.rank,
                &mut mcg,
                &mut [],
            )?;
            ensure!(
                u32::from_le_bytes(mcg) == 0xcbac1fed,
                "unsupported EXL3 MCG multiplier for {prefix}"
            );
        }
        let bytes =
            catalog.read_exl3_tensor_into(&load.tensor, self.world, self.rank, staging, scratch)?;
        ensure!(bytes == load.bytes, "EXL3 residency/read size mismatch");
        if !load.tensor.ends_with(".trellis") {
            ensure!(
                staging[..bytes]
                    .chunks_exact(2)
                    .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7c00 != 0x7c00),
                "non-finite EXL3 rotation in {}",
                load.tensor
            );
        }
        Ok(bytes)
    }
}

impl V41Exl3Manifest {
    pub fn residency(
        &self,
        layer: V41Exl3Layer,
        world: usize,
        rank: usize,
    ) -> Result<V41Exl3Residency> {
        let config = self.config.text();
        let (prefix, experts) = match layer {
            V41Exl3Layer::Backbone(layer) => {
                ensure!(
                    layer < config.num_hidden_layers,
                    "EXL3 backbone layer out of range"
                );
                (
                    format!("layers.{layer}.ffn.experts"),
                    config.n_routed_experts,
                )
            }
            V41Exl3Layer::Dspark(stage) => {
                ensure!(
                    stage < config.num_nextn_predict_layers,
                    "EXL3 dSpark stage out of range"
                );
                (
                    format!("mtp.{stage}.ffn.experts"),
                    config.dspark_n_routed_experts,
                )
            }
        };
        let mut projections = Vec::with_capacity(experts);
        let mut bits = BTreeSet::new();
        for expert in 0..experts {
            let mut row = Vec::new();
            for projection in ["w1", "w3", "w2"] {
                let p = self
                    .projections
                    .get(&format!("{prefix}.{expert}.{projection}"))
                    .context("missing EXL3 resident projection")?;
                ensure!(
                    (2..=5).contains(&p.bits),
                    "EXL3 projection tier outside K2..K5"
                );
                bits.insert(p.bits);
                row.push(p);
            }
            projections.push(row);
        }
        // The native mixed ABI needs at least two slots; a uniform checkpoint
        // uses an empty adjacent tier without expanding its compressed payload.
        if bits.len() == 1 {
            let bit = *bits.first().unwrap();
            bits.insert(if bit == 5 { 4 } else { bit + 1 });
        }
        let tiers: Vec<_> = bits.into_iter().collect();
        let partition = projections[0][0].intermediate_partition(world, rank)?;
        let width = partition.end - partition.start;
        let hidden = config.hidden_size;
        let mut counts = vec![[0; 3]; tiers.len()];
        let stride = experts * tiers.len();
        let mut descriptor = vec![-1; 3 * stride];
        let mut locals = vec![[0; 3]; experts];
        for (expert, row) in projections.iter().enumerate() {
            for (projection, p) in row.iter().enumerate() {
                ensure!(
                    p.intermediate_partition(world, rank)? == partition,
                    "inconsistent EXL3 intermediate partition"
                );
                let tier = tiers.binary_search(&p.bits).unwrap();
                let local = counts[tier][projection];
                ensure!(local < 512, "EXL3 descriptor local exceeds nine bits");
                locals[expert][projection] = local;
                descriptor[projection * stride + expert] = ((tier << 9) | local) as i32;
                counts[tier][projection] += 1;
            }
        }
        let mut buffers = Vec::new();
        let mut add = |name: String, bytes: usize, initial_words: Vec<i32>| {
            let index = buffers.len();
            buffers.push(V41Exl3ResidentBuffer {
                name,
                bytes,
                initial_words,
            });
            index
        };
        let mut payload_buffers = Vec::new();
        for (tier, bit) in tiers.iter().enumerate() {
            let one = hidden * width * bit / 8;
            let w13 = add(
                format!("tier{tier}_w13"),
                (counts[tier][0] + counts[tier][1]) * one,
                vec![],
            );
            let w2 = add(format!("tier{tier}_w2"), counts[tier][2] * one, vec![]);
            payload_buffers.push([w13, w2]);
        }
        let gate = add("gate_suh".into(), stride * hidden * 2, vec![]);
        let up = add("up_suh".into(), stride * hidden * 2, vec![]);
        let down = add("down_svh".into(), stride * hidden * 2, vec![]);
        let intermediate = add(
            "intermediate_rotations".into(),
            stride * width * 3 * 2,
            vec![],
        );
        add("descriptor_map".into(), descriptor.len() * 4, descriptor);
        add(
            "global_to_combined".into(),
            experts * 4,
            (0..experts as i32).collect(),
        );
        add("dummy_scales".into(), 4, vec![0]);
        add("unit_scales".into(), experts * 4, vec![0x3f800000; experts]);
        let mut loads = Vec::new();
        for (expert, row) in projections.iter().enumerate() {
            for (projection, p) in row.iter().enumerate() {
                let tier = tiers.binary_search(&p.bits).unwrap();
                let one = hidden * width * p.bits / 8;
                let slot =
                    locals[expert][projection] + if projection == 1 { counts[tier][0] } else { 0 };
                loads.push(V41Exl3Load {
                    tensor: format!("{}.trellis", p.name),
                    bytes: one,
                    destinations: vec![(
                        payload_buffers[tier][usize::from(projection == 2)],
                        slot * one,
                    )],
                });
                for suffix in ["suh", "svh"] {
                    let is_hidden =
                        (projection < 2 && suffix == "suh") || (projection == 2 && suffix == "svh");
                    let (buffer, bytes) = if is_hidden {
                        ([gate, up, down][projection], hidden * 2)
                    } else {
                        (intermediate, width * 2)
                    };
                    let destinations = (0..tiers.len())
                        .map(|copy| {
                            let global = copy * experts + expert;
                            (
                                buffer,
                                if is_hidden {
                                    global * bytes
                                } else {
                                    (global * 3 + projection) * bytes
                                },
                            )
                        })
                        .collect();
                    loads.push(V41Exl3Load {
                        tensor: format!("{}.{suffix}", p.name),
                        bytes,
                        destinations,
                    });
                }
            }
        }
        Ok(V41Exl3Residency {
            layer,
            world,
            rank,
            experts,
            intermediate: width,
            tiers,
            projection_counts: counts,
            buffers,
            loads,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        OfficialV41Config, V41Exl3Projection, V41Exl3ProjectionKind, OFFICIAL_V41_MODEL_ID,
    };
    use std::collections::BTreeMap;

    fn fixture(tiers: &[usize]) -> V41Exl3Manifest {
        let config = OfficialV41Config::from_json(
            OFFICIAL_V41_MODEL_ID,
            include_bytes!("official-v41-config.json"),
        )
        .unwrap();
        let mut projections = BTreeMap::new();
        for expert in 0..384 {
            for (index, (name, kind)) in [
                ("w1", V41Exl3ProjectionKind::Gate),
                ("w3", V41Exl3ProjectionKind::Up),
                ("w2", V41Exl3ProjectionKind::Down),
            ]
            .into_iter()
            .enumerate()
            {
                let name = format!("layers.0.ffn.experts.{expert}.{name}");
                projections.insert(
                    name.clone(),
                    V41Exl3Projection {
                        name,
                        kind,
                        bits: tiers[(expert + index) % tiers.len()],
                        input_features: if index == 2 { 2304 } else { 5120 },
                        output_features: if index == 2 { 5120 } else { 2304 },
                    },
                );
            }
        }
        V41Exl3Manifest {
            config,
            projections,
            ple_quantization: None,
        }
    }

    #[test]
    fn mixed_and_uniform_payload_destinations_cover_buffers_without_overlap() {
        for tiers in [&[2][..], &[3], &[4], &[5], &[3, 4], &[2, 3, 4, 5]] {
            let manifest = fixture(tiers);
            for world in [1, 2, 4] {
                let mut total_width = 0;
                for rank in 0..world {
                    let plan = manifest
                        .residency(V41Exl3Layer::Backbone(0), world, rank)
                        .unwrap();
                    total_width += plan.intermediate;
                    let mut spans = vec![Vec::new(); plan.buffers.len()];
                    for job in &plan.loads {
                        for &(buffer, offset) in &job.destinations {
                            assert_eq!(offset % 16, 0);
                            spans[buffer].push(offset..offset + job.bytes);
                        }
                    }
                    for (buffer, spans) in plan.buffers.iter().zip(&mut spans) {
                        spans.sort_by_key(|span| span.start);
                        let mut cursor = buffer.initial_words.len() * 4;
                        for span in spans {
                            assert_eq!(span.start, cursor, "{}", buffer.name);
                            cursor = span.end;
                        }
                        assert_eq!(cursor, buffer.bytes, "{}", buffer.name);
                    }
                    assert_eq!(plan.loads.len(), 384 * 9);
                    for projection in 0..3 {
                        assert_eq!(
                            plan.projection_counts
                                .iter()
                                .map(|c| c[projection])
                                .sum::<usize>(),
                            384
                        );
                    }
                    let payload: usize = plan
                        .buffers
                        .iter()
                        .filter(|b| b.name.starts_with("tier"))
                        .map(|b| b.bytes)
                        .sum();
                    let expected: usize = manifest
                        .projections
                        .values()
                        .map(|p| 5120 * plan.intermediate * p.bits / 8)
                        .sum();
                    assert_eq!(payload, expected);
                }
                assert_eq!(total_width, 2304);
            }
        }
    }

    #[test]
    fn missing_layer_and_bad_partitions_are_rejected() {
        let manifest = fixture(&[3, 4]);
        assert!(manifest
            .residency(V41Exl3Layer::Backbone(40), 1, 0)
            .is_err());
        assert!(manifest.residency(V41Exl3Layer::Dspark(3), 1, 0).is_err());
        assert!(manifest.residency(V41Exl3Layer::Backbone(0), 3, 0).is_err());
        assert!(manifest.residency(V41Exl3Layer::Backbone(0), 4, 4).is_err());
    }
}
