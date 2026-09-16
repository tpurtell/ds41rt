//! Bounded direct reads of packed EXL3 tensors, retaining complete H128 blocks.
use crate::{OfficialV41Catalog, V41Exl3Projection, V41Exl3ProjectionKind};
use anyhow::{ensure, Context, Result};
use std::{fs::File, os::unix::fs::FileExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41Exl3TensorSlice {
    pub rows: usize,
    pub source_row_bytes: usize,
    pub column_start_bytes: usize,
    pub selected_row_bytes: usize,
}

impl V41Exl3TensorSlice {
    pub fn bytes(self) -> usize {
        self.rows * self.selected_row_bytes
    }
    pub fn scratch_bytes(self) -> usize {
        if self.rows > 1 && self.selected_row_bytes != self.source_row_bytes {
            self.source_row_bytes
        } else {
            0
        }
    }
}

fn tensor_slice(
    p: &V41Exl3Projection,
    suffix: &str,
    world: usize,
    rank: usize,
) -> Result<V41Exl3TensorSlice> {
    let range = p.intermediate_partition(world, rank)?;
    let width = range.end - range.start;
    let down = p.kind == V41Exl3ProjectionKind::Down;
    let contiguous = |full, start, bytes| V41Exl3TensorSlice {
        rows: 1,
        source_row_bytes: full,
        column_start_bytes: start,
        selected_row_bytes: bytes,
    };
    Ok(match suffix {
        "trellis" if !down => {
            let tile_bytes = 16 * p.bits * 2;
            V41Exl3TensorSlice {
                rows: p.input_features / 16,
                source_row_bytes: p.output_features / 16 * tile_bytes,
                column_start_bytes: range.start / 16 * tile_bytes,
                selected_row_bytes: width / 16 * tile_bytes,
            }
        }
        "trellis" => {
            let row_bytes = p.output_features / 16 * 16 * p.bits * 2;
            contiguous(
                p.trellis_bytes(),
                range.start / 16 * row_bytes,
                width / 16 * row_bytes,
            )
        }
        "suh" if down => contiguous(p.input_features * 2, range.start * 2, width * 2),
        "svh" if !down => contiguous(p.output_features * 2, range.start * 2, width * 2),
        "suh" => contiguous(p.input_features * 2, 0, p.input_features * 2),
        "svh" => contiguous(p.output_features * 2, 0, p.output_features * 2),
        "mcg" => contiguous(4, 0, 4),
        _ => anyhow::bail!("unsupported EXL3 tensor suffix {suffix}"),
    })
}

impl OfficialV41Catalog {
    pub fn exl3_tensor_slice(
        &self,
        name: &str,
        world: usize,
        rank: usize,
    ) -> Result<V41Exl3TensorSlice> {
        let manifest = self.exl3().context("checkpoint is not routed EXL3")?;
        let (prefix, suffix) = name.rsplit_once('.').context("missing EXL3 suffix")?;
        let projection = manifest
            .projections
            .get(prefix)
            .context("unknown EXL3 projection")?;
        projection.validate_tensor(&self.tensor(name)?.metadata)?;
        tensor_slice(projection, suffix, world, rank)
    }

    pub fn exl3_tensor_bytes(&self, name: &str, world: usize, rank: usize) -> Result<u64> {
        Ok(u64::try_from(
            self.exl3_tensor_slice(name, world, rank)?.bytes(),
        )?)
    }

    /// Caller owns staging and scratch. This copies compressed tiles and stored
    /// rotations only; it never materializes a dequantized weight matrix.
    pub fn read_exl3_tensor_into(
        &self,
        name: &str,
        world: usize,
        rank: usize,
        dst: &mut [u8],
        scratch: &mut [u8],
    ) -> Result<usize> {
        let slice = self.exl3_tensor_slice(name, world, rank)?;
        ensure!(dst.len() >= slice.bytes(), "EXL3 staging buffer too small");
        ensure!(
            scratch.len() >= slice.scratch_bytes(),
            "EXL3 read scratch too small"
        );
        let tensor = self.tensor(name)?;
        let file = File::open(self.snapshot().join(&tensor.shard))?;
        read_slice(&file, tensor.metadata.byte_offset, slice, dst, scratch)?;
        Ok(slice.bytes())
    }
}

fn read_slice(
    file: &File,
    offset: u64,
    slice: V41Exl3TensorSlice,
    dst: &mut [u8],
    scratch: &mut [u8],
) -> Result<()> {
    if slice.rows == 1 || slice.selected_row_bytes == slice.source_row_bytes {
        file.read_exact_at(
            &mut dst[..slice.bytes()],
            offset
                .checked_add(slice.column_start_bytes as u64)
                .context("EXL3 contiguous offset overflow")?,
        )?;
        return Ok(());
    }
    let batch = scratch.len() / slice.source_row_bytes;
    ensure!(batch > 0, "EXL3 scratch must hold a source row");
    for start in (0..slice.rows).step_by(batch) {
        let rows = batch.min(slice.rows - start);
        let file_offset = (start as u64)
            .checked_mul(slice.source_row_bytes as u64)
            .and_then(|n| offset.checked_add(n))
            .context("EXL3 row offset overflow")?;
        file.read_exact_at(&mut scratch[..rows * slice.source_row_bytes], file_offset)?;
        for row in 0..rows {
            let src = row * slice.source_row_bytes + slice.column_start_bytes;
            let dest = (start + row) * slice.selected_row_bytes;
            dst[dest..dest + slice.selected_row_bytes]
                .copy_from_slice(&scratch[src..src + slice.selected_row_bytes]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires DS41RT_EXL3_SNAPSHOT pointing to a published local snapshot"]
    fn published_projection_reads_match_full_checkpoint_bytes() {
        let path = std::env::var_os("DS41RT_EXL3_SNAPSHOT").expect("DS41RT_EXL3_SNAPSHOT");
        let catalog = crate::read_official_v41_catalog(
            crate::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&path),
        )
        .unwrap();
        for kind in [
            V41Exl3ProjectionKind::Gate,
            V41Exl3ProjectionKind::Up,
            V41Exl3ProjectionKind::Down,
        ] {
            for bits in [3, 4] {
                let p = catalog
                    .exl3()
                    .unwrap()
                    .projections
                    .values()
                    .find(|p| p.kind == kind && p.bits == bits)
                    .unwrap();
                for suffix in ["trellis", "suh", "svh", "mcg"] {
                    let name = format!("{}.{suffix}", p.name);
                    let full = catalog.exl3_tensor_slice(&name, 1, 0).unwrap();
                    let mut expected = vec![0; full.bytes()];
                    catalog
                        .read_exl3_tensor_into(
                            &name,
                            1,
                            0,
                            &mut expected,
                            &mut vec![0; full.scratch_bytes()],
                        )
                        .unwrap();
                    for world in [2, 4] {
                        let mut rebuilt = vec![0; expected.len()];
                        for rank in 0..world {
                            let s = catalog.exl3_tensor_slice(&name, world, rank).unwrap();
                            let mut bytes = vec![0; s.bytes()];
                            catalog
                                .read_exl3_tensor_into(
                                    &name,
                                    world,
                                    rank,
                                    &mut bytes,
                                    &mut vec![0; s.scratch_bytes() * 7],
                                )
                                .unwrap();
                            for row in 0..s.rows {
                                let start = row * s.source_row_bytes + s.column_start_bytes;
                                assert_eq!(
                                    &expected[start..start + s.selected_row_bytes],
                                    &bytes[row * s.selected_row_bytes
                                        ..(row + 1) * s.selected_row_bytes]
                                );
                                rebuilt[start..start + s.selected_row_bytes].copy_from_slice(
                                    &bytes[row * s.selected_row_bytes
                                        ..(row + 1) * s.selected_row_bytes],
                                );
                            }
                        }
                        assert_eq!(rebuilt, expected);
                    }
                }
            }
        }
        println!("K3/K4 gate/up/down packed tensors and rotations reconstruct exactly for TP2/TP4");
    }

    #[test]
    fn packed_tp_slices_reconstruct_full_tensors_past_two_gib() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("packed");
        let file = File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let offset = (1u64 << 31) + 128;
        for kind in [
            V41Exl3ProjectionKind::Gate,
            V41Exl3ProjectionKind::Up,
            V41Exl3ProjectionKind::Down,
        ] {
            let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                (2304, 128)
            } else {
                (128, 2304)
            };
            for bits in 2..=5 {
                let p = V41Exl3Projection {
                    name: "test".into(),
                    kind,
                    bits,
                    input_features: input,
                    output_features: output,
                };
                for suffix in ["trellis", "suh", "svh", "mcg"] {
                    let full = tensor_slice(&p, suffix, 1, 0).unwrap();
                    let payload: Vec<_> = (0..full.bytes())
                        .map(|i| ((i * 17 + i / 251) % 253) as u8)
                        .collect();
                    file.set_len(offset + payload.len() as u64).unwrap();
                    file.write_all_at(&payload, offset).unwrap();
                    for world in [1, 2, 4] {
                        let mut reconstructed = vec![0u8; payload.len()];
                        for rank in 0..world {
                            let s = tensor_slice(&p, suffix, world, rank).unwrap();
                            let mut dst = vec![0xA5; s.bytes()];
                            let mut scratch = vec![0; s.scratch_bytes() * 3];
                            read_slice(&file, offset, s, &mut dst, &mut scratch).unwrap();
                            for row in 0..s.rows {
                                let start = row * s.source_row_bytes + s.column_start_bytes;
                                assert_eq!(
                                    &dst[row * s.selected_row_bytes
                                        ..(row + 1) * s.selected_row_bytes],
                                    &payload[start..start + s.selected_row_bytes]
                                );
                                reconstructed[start..start + s.selected_row_bytes].copy_from_slice(
                                    &dst[row * s.selected_row_bytes
                                        ..(row + 1) * s.selected_row_bytes],
                                );
                            }
                        }
                        assert_eq!(reconstructed, payload);
                    }
                }
            }
        }
    }
}
