//! Bounded gathered native rows; run page-faulting gathers on an I/O worker.
use crate::{EngramEncoding, EngramTable};
use anyhow::{ensure, Context, Result};
use ds41rt_core::{EngramBatch, ENGRAM_ROWS};

pub struct EngramBatchStaging {
    capacity: usize,
    weights: Vec<u8>,
    scales: Vec<u8>,
    text_mask: Vec<u8>,
    order: Vec<(u64, usize)>,
    ready: Option<(usize, usize, EngramEncoding, f32)>,
}
/// Token-major native rows: 24 embeddings of width 256 per token.
/// Image embeddings are zero, scales are one, and the text mask is zero.
pub struct EngramGatherView<'a> {
    pub encoding: EngramEncoding,
    pub global_scale: f32,
    pub weights: &'a [u8],
    pub scales: &'a [u8],
    pub text_mask: &'a [u8],
    pub rows: usize,
    pub layer_index: usize,
}
impl EngramBatchStaging {
    /// Vec payload budget, excluding allocator metadata and request-owned batches.
    pub fn storage_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram staging capacity"
        );
        capacity
            .checked_mul(24 * (256 + 8 + std::mem::size_of::<(u64, usize)>()) + 1)
            .context("engram staging budget overflow")
    }
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram staging capacity"
        );
        Ok(Self {
            capacity,
            weights: vec![0; capacity * 24 * 256],
            scales: vec![127; capacity * 24 * 8],
            text_mask: vec![0; capacity],
            order: Vec::with_capacity(capacity * 24),
            ready: None,
        })
    }
    /// Concatenate request-owned batches without committing their histories.
    /// Read each distinct table row once, in address order, then scatter duplicates
    /// back to their canonical token/head slots using preallocated storage.
    /// Any failure invalidates the view; no eager table load is performed.
    pub fn gather(
        &mut self,
        table: &EngramTable,
        batches: &[&EngramBatch],
        layer_index: usize,
    ) -> Result<EngramGatherView<'_>> {
        self.ready = None;
        ensure!(
            !batches.is_empty() && batches.len() <= 16,
            "engram wave requires 1..16 request batches"
        );
        let expected_rows = *ENGRAM_ROWS
            .get(layer_index)
            .context("invalid engram layer index")?;
        ensure!(
            table.weights().rows() == expected_rows,
            "engram table belongs to another layer"
        );
        let rows = batches.iter().try_fold(0usize, |sum, batch| {
            sum.checked_add(batch.hashes().len())
                .context("engram batch length overflow")
        })?;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "engram wave exceeds staging capacity"
        );
        self.order.clear();
        let mut token = 0;
        for batch in batches {
            for (row, hashes) in batch.hashes().iter().enumerate() {
                let text = batch.is_image(row) == Some(false);
                self.text_mask[token] = u8::from(text);
                if text {
                    for (head, &address) in hashes[layer_index].iter().enumerate() {
                        ensure!(address < expected_rows, "engram hash is outside its table");
                        self.order.push((address, token * 24 + head));
                    }
                }
                token += 1;
            }
        }
        let encoding = table.encoding();
        let weight_bytes = encoding.weight_bytes();
        let scale_bytes = encoding.scale_bytes();
        // FP4's packed weights and scales fit in the original FP8 weight arena.
        // Keep the FP8 allocation/budget unchanged and never expand mapped rows.
        let scale_base = self.capacity * 24 * 128;
        self.weights[..rows * 24 * weight_bytes].fill(0);
        if encoding == EngramEncoding::Fp8 {
            self.scales[..rows * 24 * scale_bytes].fill(127);
        } else {
            self.weights[scale_base..scale_base + rows * 24 * scale_bytes].fill(0x38);
        }
        self.order.sort_unstable();
        let mut previous = None;
        for &(address, slot) in &self.order {
            if let Some((last_address, source)) = previous {
                if last_address == address {
                    self.weights.copy_within(source * weight_bytes..(source + 1) * weight_bytes, slot * weight_bytes);
                    if encoding == EngramEncoding::Fp8 {
                        self.scales.copy_within(source * scale_bytes..(source + 1) * scale_bytes, slot * scale_bytes);
                    } else {
                        self.weights.copy_within(scale_base + source * scale_bytes..scale_base + (source + 1) * scale_bytes,
                            scale_base + slot * scale_bytes);
                    }
                    continue;
                }
            }
            table.weights().gather_into(&[address], &mut self.weights[slot * weight_bytes..(slot + 1) * weight_bytes])?;
            let scales = if encoding == EngramEncoding::Fp8 { &mut self.scales[slot * scale_bytes..(slot + 1) * scale_bytes] }
                else { &mut self.weights[scale_base + slot * scale_bytes..scale_base + (slot + 1) * scale_bytes] };
            table.scales().gather_into(&[address], scales)?;
            previous = Some((address, slot));
        }
        self.ready = Some((rows, layer_index, encoding, table.global_scale()));
        self.view()
    }
    pub fn view(&self) -> Result<EngramGatherView<'_>> {
        let (rows, layer_index, encoding, global_scale) = self.ready.context("engram staging is not complete")?;
        let scale_base = self.capacity * 24 * 128;
        Ok(EngramGatherView {
            encoding, global_scale,
            weights: &self.weights[..rows * 24 * encoding.weight_bytes()],
            scales: if encoding == EngramEncoding::Fp8 { &self.scales[..rows * 24 * 8] }
                else { &self.weights[scale_base..scale_base + rows * 24 * 16] },
            text_mask: &self.text_mask[..rows],
            rows,
            layer_index,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::FileExt;

    #[test]
    fn packed_ple_gather_preserves_rows_duplicates_masks_and_reuse() -> Result<()> {
        let history = ds41rt_core::EngramHistory::new(0)?;
        let batch = history.prepare(0, &[Some(7), None, Some(31)], 3)?;
        let mut staging = EngramBatchStaging::new(8)?;
        let addresses_before = (staging.weights.as_ptr(), staging.scales.as_ptr());
        for encoding in [EngramEncoding::Fp8, EngramEncoding::Nvfp4, EngramEncoding::Fp8] {
            let weights = tempfile::NamedTempFile::new()?;
            let scales = tempfile::NamedTempFile::new()?;
            let wb = encoding.weight_bytes(); let sb = encoding.scale_bytes();
            // Sparse files exercise real 64-bit row offsets without allocating tables.
            weights.as_file().set_len(ENGRAM_ROWS[0] * wb as u64)?;
            scales.as_file().set_len(ENGRAM_ROWS[0] * sb as u64)?;
            let pattern = |address: u64, n: usize| -> Vec<u8> {
                (0..n).map(|col| ((address + col as u64) % 113) as u8).collect()
            };
            for hashes in batch.hashes() {
                for &address in &hashes[0] {
                    weights.as_file().write_all_at(&pattern(address, wb), address * wb as u64)?;
                    scales.as_file().write_all_at(&pattern(address + 3, sb), address * sb as u64)?;
                }
            }
            let map = |file: &tempfile::NamedTempFile, stride| unsafe {
                crate::MappedRows::open(file.path(), 0, ENGRAM_ROWS[0], stride)
            };
            let table = match encoding {
                EngramEncoding::Fp8 => EngramTable::new(map(&weights, wb)?, map(&scales, sb)?)?,
                EngramEncoding::Nvfp4 => EngramTable::new_nvfp4(map(&weights, wb)?, map(&scales, sb)?, 0.125)?,
            };
            for copies in [2,1,2] {
                let batches = vec![&batch; copies];
                let view = staging.gather(&table, &batches, 0)?;
                assert_eq!(view.encoding, encoding);
                assert_eq!(view.global_scale, table.global_scale());
                for token in 0..copies*3 {
                    let source = token%3;
                    assert_eq!(view.text_mask[token], u8::from(source != 1));
                    for head in 0..24 {
                        let slot=token*24+head;
                        let w=&view.weights[slot*wb..(slot+1)*wb];
                        let s=&view.scales[slot*sb..(slot+1)*sb];
                        if source==1 {
                            assert!(w.iter().all(|&v| v==0));
                            assert!(s.iter().all(|&v| v==if encoding==EngramEncoding::Fp8 {127} else {0x38}));
                        } else {
                            let address=batch.hashes()[source][0][head];
                            assert_eq!(w, pattern(address,wb));
                            assert_eq!(s, pattern(address+3,sb));
                        }
                    }
                }
            }
            assert!(staging.gather(&table, &[],0).is_err());
            assert!(staging.view().is_err());
            assert_eq!(addresses_before,(staging.weights.as_ptr(),staging.scales.as_ptr()));
        }
        Ok(())
    }
}
