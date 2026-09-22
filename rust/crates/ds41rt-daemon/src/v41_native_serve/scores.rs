//! Preserve raw scores at retained frontiers so a new grammar can select a
//! different first token without replaying an otherwise exact KV prefix.
//!
//! Chunk 1 of the GPU target-sampler changes where a row's *selection* comes
//! from: greedy and constrained-greedy rows are selected on the device and only
//! their `(id, raw score)` pair crosses the bus, while the rows that still need
//! the CPU sampler download only themselves. `BatchScores` therefore stores the
//! full logit bytes of **some** rows, in batch-row order, instead of requiring
//! every row.
use crate::v41_target_head::SampledTargetRows;
use anyhow::{ensure, Result};
use ds41rt_core::TargetSamplingParams;
use std::sync::Arc;

pub(super) const VOCAB: usize = 129_280;
pub(super) const ROW_BYTES: usize = VOCAB * 4;

/// Materialize one device logit row for the exact full-vocabulary sampler.
fn row_logits(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len() == ROW_BYTES, "target logit row extent differs");
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_ne_bytes(word.try_into().unwrap()))
        .collect())
}

#[derive(Clone)]
pub(super) struct TokenScores {
    bytes: Arc<[u8]>,
    best: u32,
}
impl TokenScores {
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() == ROW_BYTES, "invalid retained logit row");
        let best = argmax(&bytes, None)?;
        Ok(Self { bytes: bytes.into(), best })
    }
    pub fn select(&self, mask: Option<&[u32]>) -> Result<u32> {
        match mask {
            None => Ok(self.best),
            Some(mask) => argmax(&self.bytes, Some(mask)),
        }
    }
    /// Exact full-vocabulary sample at an absolute emitted-token position.
    /// Greedy parameters keep the existing argmax path unchanged.
    pub fn sample(
        &self,
        mask: Option<&[u32]>,
        params: TargetSamplingParams,
        position: u64,
    ) -> Result<u32> {
        if params.is_greedy() {
            return self.select(mask);
        }
        let logits = row_logits(&self.bytes)?;
        params
            .select_token(&logits, mask, position)
            .map(|token| token as u32)
            .map_err(|error| anyhow::anyhow!("{error}"))
    }
}

/// The per-row K1 outputs the device produced, in batch-row order.
///
/// `scores` is the raw maximum logit (valid only for greedy rows); `status` and
/// `status_detail` are the `DS41RT_V41_SAMPLER_*` codes with the `0xFFFFFFFF`
/// "no detail" sentinel already normalized to `0`.
impl SampledTargetRows {
    /// Check the per-row status of the rows whose selection the device produced.
    ///
    /// Two messages are byte-identical to the CPU path (design D1):
    /// `"grammar allows no target token"` for an empty allowed set, and
    /// `"invalid target sampling parameter: temperature"`, which is exactly how
    /// `ds41rt-core`'s `InvalidParameter("temperature")` renders
    /// (`target_sampling.rs`). Two are deliberate supersets, because the device
    /// knows the offending id and the actual width while `scores.rs::argmax`
    /// reports neither: `"non-finite target logit at token {id}"` and
    /// `"invalid grammar mask width: {width}"`. The prefix of each matches the
    /// CPU string, so no caller that matched on the old text breaks.
    pub(crate) fn check_status(&self, rows: &[usize]) -> Result<()> {
        for &row in rows {
            ensure!(row < self.ids.len(), "sampled row is outside the batch");
            match self.status[row] {
                ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_OK => {}
                ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES => {
                    anyhow::bail!("grammar allows no target token");
                }
                ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT => {
                    anyhow::bail!("non-finite target logit at token {}", self.status_detail[row]);
                }
                ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE => {
                    // Exactly the string the CPU sampler renders for
                    // `InvalidParameter("temperature")`.
                    anyhow::bail!("invalid target sampling parameter: temperature");
                }
                ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_MASK_WIDTH => {
                    anyhow::bail!("invalid grammar mask width: {}", self.status_detail[row]);
                }
                code => anyhow::bail!("target sampler failed with status {code}"),
            }
        }
        Ok(())
    }
    /// Attach the full logit rows the caller downloaded, in batch-row order.
    ///
    /// `bytes` must contain exactly one `ROW_BYTES` row per entry of `rows`, and
    /// `rows` must be ascending and free of duplicates, so a row's bytes are its
    /// own slice and every existing per-row accessor keeps working. A row that
    /// is not listed keeps no bytes: reading its logits is an error rather than
    /// a silent zero row.
    pub(crate) fn with_full_logits(mut self, rows: &[usize], bytes: Vec<u8>) -> Result<BatchScores> {
        ensure!(
            bytes.len() == rows.len() * ROW_BYTES,
            "downloaded target logit extent differs"
        );
        ensure!(
            rows.windows(2).all(|pair| pair[0] < pair[1]),
            "sampled rows for download must be strictly ascending"
        );
        let mut packs = vec![usize::MAX; self.ids.len()];
        for (pack, &row) in rows.iter().enumerate() {
            ensure!(row < packs.len(), "sampled row is outside the batch");
            packs[row] = pack;
        }
        Ok(BatchScores { best: std::mem::take(&mut self.ids), bytes, packs })
    }
}

/// The per-row selection batch and the logit bytes of the rows that needed
/// them. `bytes` holds only the rows that were downloaded, packed in the order
/// the caller named them; `packs[row]` maps a batch row to its slice.
pub(super) struct BatchScores {
    pub best: Vec<u32>,
    bytes: Vec<u8>,
    /// For each batch row, its packed position in `bytes`, or `usize::MAX` when
    /// the row has no logits. Storing the mapping (rather than assuming the
    /// packed order equals the batch order) is what lets a sampled batch keep
    /// only the rows it needs.
    packs: Vec<usize>,
}
impl BatchScores {
    pub fn rows(&self) -> usize { self.best.len() }
    /// Whether every row still has its full logits (the legacy full-batch path).
    pub fn has_full_logits(&self) -> bool { self.packs.iter().all(|pack| *pack != usize::MAX) }
    fn row_range(&self, row: usize) -> Result<std::ops::Range<usize>> {
        ensure!(row < self.best.len(), "logit row is outside the batch");
        let pack = self.packs[row];
        ensure!(pack != usize::MAX, "target logit row requires full logits");
        Ok(pack * ROW_BYTES..(pack + 1) * ROW_BYTES)
    }
    pub fn retain_downloaded(&self, row: usize, bytes: &[u8]) -> Result<TokenScores> {
        self.retain_downloaded_with(row, bytes, None)
    }
    /// Retain one downloaded row, cross-checking the device's selection.
    ///
    /// Unmasked rows must reproduce the plain argmax. A **masked** row's
    /// selection is the masked argmax, so the flagged id is validated against
    /// the row's grammar mask instead of re-deriving an unmasked argmax (which
    /// would raise a spurious mismatch for any constrained greedy request).
    pub fn retain_downloaded_with(&self, row: usize, bytes: &[u8],
        mask: Option<&[u32]>) -> Result<TokenScores> {
        ensure!(row < self.best.len() && bytes.len() == ROW_BYTES, "downloaded retained logit extent differs");
        let best = argmax(bytes, mask)?;
        ensure!(best == self.best[row], "GPU and retained CPU greedy selection differ");
        Ok(TokenScores { bytes: Arc::from(bytes), best })
    }
    /// Reuse logits this batch already holds, with the same cross-check.
    fn retain_packed(&self, row: usize, mask: Option<&[u32]>) -> Result<TokenScores> {
        let range = self.row_range(row)?;
        let best = argmax(&self.bytes[range.clone()], mask)?;
        ensure!(best == self.best[row], "GPU and retained CPU greedy selection differs");
        Ok(TokenScores { bytes: Arc::from(&self.bytes[range]), best })
    }
    /// Construction from the device terminal: the ids and (for greedy rows) the
    /// raw maximum logit come from K1. `greedy` marks the rows whose `scores`
    /// entry is a real logit; every other row keeps a zero placeholder because
    /// only the id is meaningful and only its own path reads the score.
    pub fn from_sampled(sampled: &SampledTargetRows, greedy: &[bool]) -> Result<Self> {
        ensure!(
            !sampled.ids.is_empty() && sampled.ids.len() <= 80,
            "invalid sampled target rows"
        );
        ensure!(
            greedy.len() == sampled.rows()
                && sampled.scores.len() == sampled.rows()
                && sampled.status.len() == sampled.rows(),
            "sampled target row extents differ"
        );
        let mut best = Vec::with_capacity(sampled.rows());
        for (row, &id) in sampled.ids.iter().enumerate() {
            ensure!((id as usize) < VOCAB, "invalid sampled token {id} at row {row}");
            if greedy[row] {
                ensure!(
                    sampled.scores[row].is_finite(),
                    "non-finite target logit"
                );
            }
            best.push(id);
        }
        let packs = vec![usize::MAX; best.len()];
        Ok(Self { best, bytes: Vec::new(), packs })
    }
    /// The legacy full-batch construction: every row carries its logits.
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        ensure!(bytes.len() % ROW_BYTES == 0, "invalid target logit batch");
        let best: Vec<u32> = bytes.chunks_exact(ROW_BYTES).map(|row| argmax(row, None)).collect::<Result<_>>()?;
        let packs = (0..best.len()).collect();
        Ok(Self { best, bytes, packs })
    }
    /// Compact construction: the ids are known (device-selected greedy rows or
    /// the compact lane) and no row has logits yet.
    pub fn from_greedy(values: Vec<(u32, f32)>) -> Result<Self> {
        ensure!(!values.is_empty() && values.len() <= 80, "invalid compact target rows");
        ensure!(values.iter().all(|&(id, score)| (id as usize) < VOCAB && score.is_finite()),
            "invalid token or non-finite target logit");
        let best = values.into_iter().map(|v| v.0).collect::<Vec<_>>();
        let packs = vec![usize::MAX; best.len()];
        Ok(Self { best, bytes: Vec::new(), packs })
    }
    /// The completed head buffer stays unchanged until commit returns. Fetch only
    /// a finishing request's frontier; a later grammar still sees all vocabulary scores.
    ///
    /// `mask` is the row's grammar mask when the request is constrained, so the
    /// retained tokens are checked against what the device was asked to select.
    /// A row that was already downloaded is reused in place, without a second
    /// copy or a second transfer.
    pub fn retain_from_device(&self, lib: &ds41rt_ffi::NativeLibrary,
        mut logits: ds41rt_ffi::Ds41rtDeviceBuffer, row: usize, mask: Option<&[u32]>,
    ) -> Result<TokenScores> {
        if self.row_range(row).is_ok() {
            return self.retain_packed(row, mask);
        }
        ensure!(row < self.best.len() && logits.bytes == self.best.len() * ROW_BYTES,
            "compact retained logit extent differs");
        logits.ptr = unsafe { logits.ptr.cast::<u8>().add(row * ROW_BYTES).cast() };
        logits.bytes = ROW_BYTES;
        let mut bytes = vec![0; ROW_BYTES];
        lib.copy_d2h(&mut bytes, logits)?;
        self.retain_downloaded_with(row, &bytes, mask)
    }
    pub fn select(&self, row: usize, mask: Option<&[u32]>) -> Result<u32> {
        ensure!(row < self.best.len(), "selected logit row is outside batch");
        match mask {
            None => Ok(self.best[row]),
            Some(mask) => {
                let range = self.row_range(row)?;
                argmax(&self.bytes[range], Some(mask))
            },
        }
    }
    /// Exact full-vocabulary sample of one row at an absolute emitted-token
    /// position. Requires the full logits of that row; greedy parameters keep
    /// the argmax route.
    pub fn sample(
        &self,
        row: usize,
        mask: Option<&[u32]>,
        params: TargetSamplingParams,
        position: u64,
    ) -> Result<u32> {
        ensure!(row < self.best.len(), "sampled logit row is outside batch");
        if params.is_greedy() {
            return self.select(row, mask);
        }
        let range = self.row_range(row)?;
        let logits = row_logits(&self.bytes[range])?;
        params
            .select_token(&logits, mask, position)
            .map(|token| token as u32)
            .map_err(|error| anyhow::anyhow!("{error}"))
    }
    // Diagnostic only: scores are already on the host for the rows that
    // downloaded them. Callers gate the scan behind the logit trace target so
    // normal serving incurs no extra work.
    pub fn top_two(&self, row: usize) -> Result<[(u32, f32); 2]> {
        let range = self.row_range(row)?;
        let mut top = [(0, f32::NEG_INFINITY); 2];
        for (token, bytes) in self.bytes[range].chunks_exact(4).enumerate() {
            let value = f32::from_ne_bytes(bytes.try_into().unwrap());
            if value > top[0].1 {
                top[1] = top[0];
                top[0] = (token as u32, value);
            } else if value > top[1].1 {
                top[1] = (token as u32, value);
            }
        }
        Ok(top)
    }
    /// Copy only a finishing request's committed frontier, never every decode row.
    /// Reuse a row this batch already holds, trusting its recorded selection.
    ///
    /// No cross-check here: `best[row]` and these bytes were produced by the
    /// same path, and for a constrained row the recorded id is the **masked**
    /// argmax while a bare recomputation would be the unmasked one. The
    /// device-vs-CPU cross-check lives in [`Self::retain_from_device`], where the
    /// id and the bytes come from different producers.
    pub fn retain(&self, row: usize) -> Result<TokenScores> {
        let range = self.row_range(row)?;
        Ok(TokenScores { bytes: Arc::from(&self.bytes[range]), best: self.best[row] })
    }
}

fn argmax(bytes: &[u8], mask: Option<&[u32]>) -> Result<u32> {
    if let Some(mask) = mask {
        ensure!(mask.len() == VOCAB.div_ceil(32), "invalid grammar mask width");
    }
    let mut best = None;
    let mut maximum = f32::NEG_INFINITY;
    for (i, bytes) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_ne_bytes(bytes.try_into().unwrap());
        ensure!(value.is_finite(), "non-finite target logit");
        if mask.is_none_or(|words| words[i / 32] & (1 << (i % 32)) != 0) && value > maximum {
            maximum = value;
            best = Some(i as u32);
        }
    }
    best.ok_or_else(|| anyhow::anyhow!("grammar allows no target token"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(winner: usize) -> Vec<u8> {
        let mut scores = vec![0.0f32; VOCAB];
        scores[winner] = 4.;
        scores[17] = 3.;
        scores.into_iter().flat_map(f32::to_ne_bytes).collect()
    }
    #[test]
    fn compact_scores_reject_invalid_rows_and_require_logits_for_masks() {
        let scores = BatchScores::from_greedy(vec![(17, -2.), (91, 3.)]).unwrap();
        assert_eq!(scores.select(0, None).unwrap(), 17);
        assert!(scores.select(2, None).is_err());
        assert!(scores.select(0, Some(&vec![u32::MAX; VOCAB.div_ceil(32)])).is_err());
        assert!(scores.retain(0).is_err());
        assert!(scores.top_two(0).is_err());
        assert!(!scores.has_full_logits());
        assert!(BatchScores::from_greedy(vec![(0, f32::NAN)]).is_err());
        assert!(BatchScores::from_greedy(vec![(0, f32::INFINITY)]).is_err());
        assert!(BatchScores::from_greedy(vec![(VOCAB as u32, 1.)]).is_err());
        assert!(BatchScores::from_greedy(vec![]).is_err());
    }
    #[test]
    fn new_constraint_reselects_an_exact_cached_frontier() {
        let scores = TokenScores::new(row(91)).unwrap();
        let shared = scores.clone();
        assert!(Arc::ptr_eq(&scores.bytes, &shared.bytes));
        assert_eq!(scores.select(None).unwrap(), 91);
        let mut mask = vec![0; VOCAB.div_ceil(32)];
        mask[0] = 1 << 17;
        assert_eq!(shared.select(Some(&mask)).unwrap(), 17);
        mask[0] = 0;
        assert!(shared.select(Some(&mask)).is_err());
        assert_eq!(scores.select(None).unwrap(), 91);
    }
    #[test]
    fn downloaded_frontier_keeps_full_scores_and_checks_gpu_selection() {
        let compact = BatchScores::from_greedy(vec![(91, 4.)]).unwrap();
        assert!(!compact.has_full_logits());
        let bytes = row(91);
        let retained = compact.retain_downloaded(0, &bytes).unwrap();
        assert!(compact.retain_downloaded(1, &bytes).is_err());
        assert!(compact.retain_downloaded(0, &row(93)).is_err());
        let mut invalid = bytes.clone();
        invalid[..4].copy_from_slice(&f32::NAN.to_ne_bytes());
        assert!(compact.retain_downloaded(0, &invalid).is_err());
        drop(compact); drop(bytes);
        let mut mask = vec![0; VOCAB.div_ceil(32)]; mask[0] = 1 << 17;
        assert_eq!(retained.select(None).unwrap(), 91);
        assert_eq!(retained.select(Some(&mask)).unwrap(), 17);
    }
    #[test]
    fn retained_scores_survive_batch_drop_and_reject_invalid_rows() {
        let batch = BatchScores::new([row(91), row(93)].concat()).unwrap();
        assert_eq!(batch.best, [91, 93]);
        assert!(batch.retain(2).is_err());
        let scores = batch.retain(1).unwrap();
        drop(batch);
        assert_eq!(scores.select(None).unwrap(), 93);
        assert!(TokenScores::new(vec![0; 4]).is_err());
        let mut invalid = row(91);
        invalid[..4].copy_from_slice(&f32::NAN.to_ne_bytes());
        assert!(TokenScores::new(invalid).is_err());
    }
    #[test]
    fn diagnostic_top_two_preserves_greedy_ties_and_negative_scores() {
        let mut values = vec![-10.0f32; VOCAB];
        values[17] = -2.;
        values[91] = -2.;
        let batch = BatchScores::new(values.into_iter().flat_map(f32::to_ne_bytes).collect()).unwrap();
        assert_eq!(batch.best, [17]);
        assert_eq!(batch.top_two(0).unwrap(), [(17, -2.), (91, -2.)]);
        assert!(batch.top_two(1).is_err());
    }
    #[test]
    fn stochastic_sample_is_deterministic_and_masks_first() {
        let batch = BatchScores::new([row(91), row(93)].concat()).unwrap();
        let params =
            ds41rt_core::TargetSamplingParams::new(1.0, 1.0, None, 0.0, 5).unwrap();
        assert!(!params.is_greedy());
        // Same request position and seed replay the same token.
        let first = batch.sample(0, None, params, 0).unwrap();
        assert_eq!(first, batch.sample(0, None, params, 0).unwrap());
        // Greedy parameters keep the argmax route exactly.
        assert_eq!(batch.sample(0, None, ds41rt_core::TargetSamplingParams::greedy(), 0).unwrap(), 91);
        // A grammar mask restricts the draw to the allowed token.
        let mut mask = vec![0u32; VOCAB.div_ceil(32)];
        mask[0] = 1 << 17;
        for position in 0..32 {
            assert_eq!(batch.sample(0, Some(&mask), params, position).unwrap(), 17);
        }
    }
    #[test]
    fn stochastic_sample_requires_full_logits() {
        let compact = BatchScores::from_greedy(vec![(91, 4.)]).unwrap();
        assert!(!compact.has_full_logits());
        let params =
            ds41rt_core::TargetSamplingParams::new(1.0, 1.0, None, 0.0, 5).unwrap();
        assert!(compact.sample(0, None, params, 0).is_err());
        // The greedy route still works without logits.
        assert_eq!(
            compact
                .sample(0, None, ds41rt_core::TargetSamplingParams::greedy(), 0)
                .unwrap(),
            91
        );
    }
    #[test]
    fn token_scores_sample_matches_batch_scores() {
        let bytes = row(91);
        let single = TokenScores::new(bytes.clone()).unwrap();
        let batch = BatchScores::new(bytes).unwrap();
        let params =
            ds41rt_core::TargetSamplingParams::new(0.9, 0.95, Some(8), 0.01, 77).unwrap();
        for position in 0..16 {
            assert_eq!(
                single.sample(None, params, position).unwrap(),
                batch.sample(0, None, params, position).unwrap()
            );
        }
    }

    /// Chunk 1: a sampled batch keeps ids for every row but only the rows the
    /// caller downloaded carry logits. A row without logits must be an error,
    /// never a silent zero row, and the downloaded rows must keep their own
    /// slice.
    #[test]
    fn sampled_rows_keep_ids_and_only_downloaded_logits() {
        let three = || SampledTargetRows {
            ids: vec![91, 93, 95],
            scores: vec![4.0, 3.0, 2.0],
            status: vec![0, 0, 0],
            status_detail: vec![0, 0, 0],
            logits: ds41rt_ffi::Ds41rtDeviceBuffer::default(),
        };
        three().check_status(&[0, 1, 2]).unwrap();
        // All-greedy construction: every row keeps its id and its raw score.
        let greedy = BatchScores::from_sampled(&three(), &[true, true, true]).unwrap();
        assert_eq!(greedy.best, vec![91, 93, 95]);
        assert!(!greedy.has_full_logits());

        // Rows 0 and 2 are the stochastic rows that downloaded their logits.
        let bytes = [row(91), row(95)].concat();
        let batch = three().with_full_logits(&[0, 2], bytes).unwrap();
        assert_eq!(batch.best, [91, 93, 95]);
        assert!(!batch.has_full_logits());
        assert!(batch.top_two(1).is_err());
        assert!(batch.sample(1, None, TargetSamplingParams::greedy(), 0).is_ok());
        assert!(batch.retain(1).is_err());
        // Row 0 keeps its own bytes even though its packed index differs.
        assert_eq!(batch.top_two(0).unwrap()[0].0, 91);
        assert_eq!(batch.top_two(2).unwrap()[0].0, 95);
        assert_eq!(batch.retain(0).unwrap().select(None).unwrap(), 91);
        assert_eq!(batch.retain(2).unwrap().select(None).unwrap(), 95);
        // A duplicated download selection is rejected.
        assert!(three().with_full_logits(&[0, 0], [row(91), row(91)].concat()).is_err());
        let one = || SampledTargetRows {
            ids: vec![91],
            scores: vec![4.0],
            status: vec![0],
            status_detail: vec![0],
            logits: ds41rt_ffi::Ds41rtDeviceBuffer::default(),
        };
        let two = || SampledTargetRows {
            ids: vec![91, 93],
            scores: vec![4.0, 3.0],
            status: vec![0, 0],
            status_detail: vec![0, 0],
            logits: ds41rt_ffi::Ds41rtDeviceBuffer::default(),
        };
        // A shuffled download selection is rejected.
        assert!(two().with_full_logits(&[1, 0], [row(91), row(93)].concat()).is_err());
        assert!(one().with_full_logits(&[0, 0], [row(91), row(91)].concat()).is_err());
        // A non-finite score on a greedy row is rejected; a stochastic row's
        // placeholder score is not read.
        let mut bad = SampledTargetRows {
            ids: vec![91],
            scores: vec![f32::NAN],
            status: vec![0],
            status_detail: vec![0],
            logits: ds41rt_ffi::Ds41rtDeviceBuffer::default(),
        };
        assert!(BatchScores::from_sampled(&bad, &[true]).is_err());
        assert!(BatchScores::from_sampled(&bad, &[false]).is_ok());
        bad.ids = vec![VOCAB as u32];
        assert!(BatchScores::from_sampled(&bad, &[false]).is_err());
    }

    /// The device status channel maps onto exactly the errors the CPU path
    /// produces today (design D1), including `EmptyCandidates` keeping its
    /// worker-error shape.
    /// P0-A retention: a constrained greedy row's device id is the **masked**
    /// argmax, so retention must validate it against the row's grammar mask.
    /// Recomputing a bare argmax reports a spurious GPU/CPU mismatch (or, on a
    /// batch with no bytes, the "requires full logits" error).
    #[test]
    fn masked_retention_validates_against_the_grammar_mask() {
        // VOCAB is 129_280 for this crate; keep the mask words consistent.
        let words = VOCAB.div_ceil(32);
        // Row with a single clear maximum at token 5 and a second at token 1.
        let mut bytes = vec![0u8; ROW_BYTES];
        let mut logits = vec![-8.0f32; VOCAB];
        logits[1] = 3.0;
        logits[5] = 9.0;
        for (index, value) in logits.iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
        }
        // The grammar allows only token 1, so the masked argmax is 1.
        let mut mask = vec![0u32; words];
        mask[1 / 32] |= 1 << (1 % 32);
        let constrained = BatchScores {
            best: vec![1],
            bytes: bytes.clone(),
            packs: vec![0],
        };
        // With the mask, the recorded id is reproduced exactly.
        let retained = constrained.retain_downloaded_with(0, &bytes, Some(&mask)).unwrap();
        assert_eq!(retained.best, 1);
        // Without it, the unmasked argmax (token 5) is not the recorded id, so
        // the cross-check must fire rather than silently accept the row.
        let error = match constrained.retain_downloaded(0, &bytes) {
            Ok(_) => panic!("an unmasked recomputation must not match the masked id"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("greedy selection differ"), "{error}");
        // An already-resident row is reused without a second copy or check.
        let reused = constrained.retain(0).unwrap();
        assert_eq!(reused.best, 1);
        assert_eq!(&reused.bytes[..], &bytes[..]);
    }

    #[test]
    fn sampled_status_matches_the_cpu_error_strings() {
        let rows = |status: Vec<u32>, detail: Vec<u32>| SampledTargetRows {
            ids: vec![0; status.len()],
            scores: vec![0.0; status.len()],
            status,
            status_detail: detail,
            logits: ds41rt_ffi::Ds41rtDeviceBuffer::default(),
        };
        rows(vec![0], vec![0]).check_status(&[0]).unwrap();
        let empty = rows(vec![ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES], vec![0]);
        assert_eq!(empty.check_status(&[0]).unwrap_err().to_string(), "grammar allows no target token");
        let nonfinite =
            rows(vec![ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT], vec![7]);
        assert!(nonfinite.check_status(&[0]).unwrap_err().to_string().contains("token 7"));
        let temperature =
            rows(vec![ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_INVALID_TEMPERATURE], vec![0]);
        assert_eq!(
            temperature.check_status(&[0]).unwrap_err().to_string(),
            "invalid target sampling parameter: temperature"
        );
        let width = rows(vec![ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_MASK_WIDTH], vec![9]);
        assert!(width.check_status(&[0]).unwrap_err().to_string().contains('9'));
        assert!(empty.check_status(&[1]).is_err());
    }
}
