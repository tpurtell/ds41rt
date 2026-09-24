//! One lane's learned index query and retained source-20 candidates.
use crate::v41_attention_query::AttentionQueryOutput;
use crate::v41_backbone_cache::{CacheAttention, CachePlacement};
use crate::v41_memory::device::{Device, DeviceOwner};
mod placement;
use crate::v41_index_query::{IndexQueryWave, IndexQueryWeights};
use crate::v41_index_selection::{IndexSelectionOutput, IndexSelectionWave};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::OfficialV41Catalog;

const LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
pub(crate) struct IndexLaneWeights<'a> {
    library: &'a NativeLibrary,
    weights: Vec<DeviceOwner<'a, IndexQueryWeights<'a>>>,
    placement: Option<CachePlacement>,
}
impl<'a> IndexLaneWeights<'a> {
    pub fn device_bytes(library: &NativeLibrary, catalog: &OfficialV41Catalog) -> Result<usize> {
        LAYERS.into_iter().try_fold(0usize, |total, layer| {
            total
                .checked_add(IndexQueryWeights::device_bytes(library, catalog, layer)?)
                .context("index weight budget overflow")
        })
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog)? <= budget,
            "index lane weights exceed budget"
        );
        Self::load_placed(library, catalog, staging, None)
    }
}

pub(crate) struct IndexLane<'w, 'a> {
    weights: &'w IndexLaneWeights<'a>,
    source: IndexSelectionWave<'a>,
    reindex: Option<IndexSelectionWave<'a>>,
    query: IndexQueryWave<'w, 'a>,
    next: usize,
    device: Option<i32>,
    ready: Option<usize>,
    invalid: bool,
}
impl<'w, 'a> IndexLane<'w, 'a> {
    pub fn workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<[usize; 3]> {
        Ok([
            IndexQueryWave::device_bytes(library, capacity)?,
            IndexSelectionWave::device_bytes(capacity as usize)?,
            IndexSelectionWave::device_bytes(capacity as usize)?,
        ])
    }
    pub fn new(weights: &'w IndexLaneWeights<'a>, capacity: u32, budget: usize) -> Result<Self> {
        Self::new_inner(weights, capacity, budget, None)
    }
    fn new_inner(weights: &'w IndexLaneWeights<'a>, capacity: u32, budget: usize,
        device: Option<i32>) -> Result<Self> {
        let bytes = match device {
            Some(gpu) => Self::placed_workspace_bytes(weights.library,
                weights.placement.context("placed index weights absent")?,capacity,gpu as usize)?,
            None => Self::workspace_bytes(weights.library, capacity)?,
        };
        let total = bytes.iter().try_fold(0usize, |n, &b| {
            n.checked_add(b).context("index workspace budget overflow")
        })?;
        ensure!(total <= budget, "index lane workspace exceeds budget");
        ensure!(
            weights.weights.len() == 8,
            "index lane requires all eight weight owners"
        );
        ensure!(weights.placement.is_some() == device.is_some(), "index workspace placement differs from weights");
        let first = Self::next_owned(weights, device, 0);
        ensure!(first < LAYERS.len(), "GPU has no index producers");
        let query = weights.weights[first].wave(capacity, bytes[0])?;
        let mut source = IndexSelectionWave::new(weights.library, capacity as usize, bytes[1])?;
        let reindex = if bytes[2] == 0 { None } else if device.is_some() {
            Some(IndexSelectionWave::sharing_scratch(&mut source, bytes[2])?)
        } else { Some(IndexSelectionWave::new(weights.library, capacity as usize, bytes[2])?) };
        Ok(Self {
            weights, query, source, reindex,
            next: first,
            device,
            ready: None,
            invalid: false,
        })
    }
    fn next_owned(weights: &IndexLaneWeights<'_>, device: Option<i32>, start: usize) -> usize {
        if device.is_none() { return start; }
        (start..LAYERS.len()).find(|&i| device.is_none_or(|id| weights.weights[i].device.id == id))
            .unwrap_or(LAYERS.len())
    }
    fn advance(&mut self) {
        self.next = Self::next_owned(self.weights, self.device, self.next + 1);
    }
    /// Begin a new batch after all consumers finish; also recovers failed work.
    pub fn enable_small_graph_shapes(&mut self) {
        self.query.enable_small_graph_shapes();
        self.source.enable_small_graph_shapes();
        if let Some(reindex) = &mut self.reindex { reindex.enable_small_graph_shapes(); }
    }
    pub fn restart(&mut self) -> Result<()> {
        self.restart_at(0)
    }
    /// Replay begins at decoder source 20; earlier encoder selections are absent.
    pub fn restart_decoder(&mut self) -> Result<()> {
        self.restart_at(3)
    }
    fn restart_at(&mut self, first: usize) -> Result<()> {
        self.invalid = true;
        self.ready = None;
        self.source.restart()?;
        if let Some(reindex) = &mut self.reindex { reindex.restart()?; }
        let first = Self::next_owned(self.weights, self.device, first);
        if first < LAYERS.len() { self.query.rebind(&self.weights.weights[first])?; }
        self.next = first;
        self.invalid = false;
        Ok(())
    }
    /// # Safety
    /// Query and cache belong to the same admitted batch. All producers have
    /// completed and no external writes race these owners. Call once per index
    /// producer in layer order; intermediate attention layers reuse output().
    pub unsafe fn select(
        &mut self,
        query: &AttentionQueryOutput<'_>,
        cache: &CacheAttention<'_>,
    ) -> Result<()> {
        if crate::v41_memory::chain::active() {
            // Stream-ordered: the queued projection and selection finish on the
            // chain immediately, so the host never waits here.
            unsafe { self.enqueue_projection(query)?; }
            ensure!(self.poll_projection()?, "chained index projection still pending");
            unsafe { self.enqueue_selection(cache)?; }
            ensure!(self.poll_selection()?, "chained index selection still pending");
            return Ok(());
        }
        let valid = !std::mem::replace(&mut self.invalid, true);
        self.ready = None;
        ensure!(
            valid && LAYERS.get(self.next) == Some(&query.layer)
                && self.device.is_none_or(|id| query.hidden.device_id == id),
            "index producer order differs; restart lane"
        );
        // The direct index path drains its own streams and uses legacy-stream
        // uploads; order it after any chained producers on the host.
        crate::v41_memory::chain::settle(self.weights.library)?;
        self.query.rebind(&self.weights.weights[self.next])?;
        let requests = cache.selection_requests()?;
        let projected = unsafe { self.query.execute_attention(query)? };
        if query.layer <= 20 {
            unsafe {
                self.source.execute(&projected, &requests, None)?;
            }
        } else {
            let candidates = self.source.output()?;
            unsafe {
                self.reindex.as_mut().context("decoder reindex workspace absent")?
                    .execute(&projected, &requests, Some(&candidates))?;
            }
        }
        self.ready = Some(query.layer);
        self.advance();
        self.invalid = false;
        Ok(())
    }
    /// # Safety
    /// Retain main query/cache inputs until selection completes or abort drains.
    pub unsafe fn enqueue_projection(&mut self, query: &AttentionQueryOutput<'_>) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        self.ready = None;
        ensure!(valid && LAYERS.get(self.next) == Some(&query.layer)
                && self.device.is_none_or(|id| query.hidden.device_id == id), "queued index order differs");
        self.query.rebind(&self.weights.weights[self.next])?;
        unsafe { self.query.enqueue_attention(query) }
    }
    pub fn poll_projection(&mut self) -> Result<bool> { self.query.poll_pending() }
    /// # Safety
    /// Preserve completed projection and cache proposal storage through selection.
    pub unsafe fn enqueue_selection(&mut self, cache: &CacheAttention<'_>) -> Result<()> {
        ensure!(self.invalid, "queued index projection absent");
        let query = self.query.output()?;
        ensure!(LAYERS.get(self.next) == Some(&query.layer), "queued selection layer differs");
        let requests = cache.selection_requests()?;
        if query.layer <= 20 {
            unsafe { self.source.enqueue_selection(&query, &requests, None) }
        } else {
            let shared = self.source.output()?;
            unsafe { self.reindex.as_mut().context("decoder reindex workspace absent")?.enqueue_selection(&query, &requests, Some(&shared)) }
        }
    }
    pub fn poll_selection(&mut self) -> Result<bool> {
        ensure!(self.invalid, "queued index selection absent");
        let layer = *LAYERS.get(self.next).context("queued index order exhausted")?;
        let ready = if layer <= 20 { self.source.poll_pending()? } else { self.reindex.as_mut().context("decoder reindex workspace absent")?.poll_pending()? };
        if ready { self.ready = Some(layer); self.advance(); self.invalid = false; }
        Ok(ready)
    }
    pub fn abort_pending(&mut self) -> Result<()> {
        // Drain consumers before their projected inputs can be recycled.
        let reindex = self.reindex.as_mut().map_or(Ok(()), |wave| wave.abort_pending());
        let result = reindex.and(self.source.abort_pending()).and(self.query.abort_pending());
        self.invalid = true; self.ready = None;
        result
    }
    /// Revalidate exact proposal snapshots and row order for the attention layer.
    pub fn output(
        &self,
        layer: usize,
        cache: &CacheAttention<'_>,
    ) -> Result<IndexSelectionOutput<'_>> {
        ensure!(!self.invalid, "index lane requires restart");
        let producer = self.ready.context("index lane output unavailable")?;
        let output = if producer <= 20 {
            self.source.output()?
        } else {
            self.reindex.as_ref().context("decoder reindex workspace absent")?.output()?
        };
        let requests = cache.selection_requests()?;
        let bindings = requests
            .iter()
            .flat_map(|r| r.positions.iter().map(|&p| (r.proposal.binding(), p)))
            .collect::<Vec<_>>();
        output.validate_attention(layer, &bindings)?;
        Ok(output)
    }
}

impl Drop for IndexLane<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.abort_pending() { tracing::error!(%error, "draining index lane"); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_index_lane_budgets_reject_before_allocation() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_LANE_PLAN_LIBRARY") else {
            eprintln!("skip index planning test: DS41RT_LANE_PLAN_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_LANE_PLAN_MODEL")
            .context("DS41RT_LANE_PLAN_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&model),
        )?;
        let bytes = IndexLaneWeights::device_bytes(&library, &catalog)?;
        assert_eq!(bytes, 45_916_160);
        assert!(
            IndexLaneWeights::load(&library, &catalog, bytes - 1, 1024 * 1024)
                .err()
                .unwrap()
                .to_string()
                .contains("weights exceed budget")
        );
        let empty = IndexLaneWeights {
            library: &library,
            weights: Vec::new(),
            placement: None,
        };
        for capacity in [1, 80, 4096] {
            let groups = IndexLane::workspace_bytes(&library, capacity)?;
            let total = groups.iter().sum::<usize>();
            assert!(IndexLane::new(&empty, capacity, total - 1)
                .err()
                .unwrap()
                .to_string()
                .contains("workspace exceeds budget"));
            assert!(IndexLane::new(&empty, capacity, total)
                .err()
                .unwrap()
                .to_string()
                .contains("all eight weight owners"));
            eprintln!("index lane capacity={capacity} workspace_groups={groups:?} total={total}");
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(IndexLane::workspace_bytes(&library, capacity).is_err());
        }
        Ok(())
    }
}
