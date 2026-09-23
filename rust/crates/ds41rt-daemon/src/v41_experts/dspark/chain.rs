//! Draft transformer chain with optional shared embedding and terminal heads.
use super::{DsparkStage, DsparkTerminal, DsparkWeights};
use crate::v41_dspark_cache::{DsparkWindow, WindowLease, WindowRead};
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::{ensure, Context, Result};
use ds41rt_core::DsparkRng;
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41AttentionOps};
use std::ffi::c_void;
mod distributed;
pub(crate) use distributed::DistributedDsparkChain;

pub(crate) struct DsparkChain<'weights, 'library> {
    stream: LoadStream<'library>,
    /// Live draft width; storage and the maximum are fixed at load time.
    width: usize,
    maximum: usize,
    stages: [DsparkStage<'weights, 'library>; 3],
    /// Captured complete drafts keyed by (request count, width).
    graphs: std::collections::BTreeMap<(usize, usize), (*mut c_void, [u64; 3])>,
    ready: Option<usize>,
    pending: Option<PendingDraft<'library>>,
    download: HostAllocation<'library>,
    embedding: Option<&'weights NativeRtxTensors<'library>>,
    tokens: DeviceAllocation<'library>,
    token_staging: HostAllocation<'library>,
    token_count: Option<usize>,
    terminal: Option<DsparkTerminal<'weights, 'library>>,
    ops: V41AttentionOps<'library>,
}
/// Own reservations before the first upload. Errors and unwinding drain before
/// pinned inputs or cache ring slots may be reused. Successful polls disarm it.
struct PendingDraft<'a> {
    count: usize,
    reads: [WindowRead; 3],
    warming: bool,
    armed: bool,
    library: &'a ds41rt_ffi::NativeLibrary,
    stream: *mut c_void,
}
impl Drop for PendingDraft<'_> {
    fn drop(&mut self) {
        if self.armed {
            if let Err(error) = unsafe { self.library.cuda_stream_synchronize(self.stream) } {
                tracing::error!(%error, "draining pending draft before releasing cache readers");
            }
        }
    }
}
impl<'library> DsparkWeights<'library> {
    pub fn draft_bytes(&self, requests: u32) -> Result<usize> {
        self.chain_bytes(requests)?
            .checked_add(DsparkTerminal::device_bytes_with_width(requests as usize, self.draft_width)?)
            .context("dSpark complete draft budget overflow")
    }
    /// Complete seed embedding, three-stage transformer and terminal proposal
    /// graph. Shared coordinator embedding and vocabulary owners are borrowed.
    pub fn draft<'weights>(
        &'weights self,
        embedding: &'weights NativeRtxTensors<'library>,
        head: &'weights VocabularyHead<'library>,
        requests: u32,
        budget: usize,
    ) -> Result<DsparkChain<'weights, 'library>> {
        ensure!(
            self.draft_bytes(requests)? <= budget,
            "dSpark complete draft exceeds budget"
        );
        let mut chain = self.embedded_chain(embedding, requests, self.chain_bytes(requests)?)?;
        chain.terminal = Some(self.terminal(
            head,
            requests as usize,
            DsparkTerminal::device_bytes_with_width(requests as usize, self.draft_width)?,
        )?);
        Ok(chain)
    }

    /// Borrow the coordinator's existing shared embedding table, never duplicate it.
    pub fn embedded_chain<'weights>(
        &'weights self,
        embedding: &'weights NativeRtxTensors<'library>,
        requests: u32,
        budget: usize,
    ) -> Result<DsparkChain<'weights, 'library>> {
        ensure!(
            embedding.get("embed.weight")?.bytes == 129280 * 5120 * 2,
            "invalid shared embedding extent"
        );
        let mut chain = self.chain(requests, budget)?;
        let table_device = embedding.get("embed.weight")?.device_id;
        if table_device != chain.tokens.buffer.device_id {
            // Plan peer access once. Draft graphs only read the shared table;
            // tokens, residuals and all transformer work stay on this device.
            chain.stream.library.cuda_enable_peer(table_device)?;
        }
        chain.embedding = Some(embedding);
        Ok(chain)
    }

    pub fn chain_bytes(&self, requests: u32) -> Result<usize> {
        self.stage_bytes(requests)?
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(requests as usize * 4))
            .context("dSpark chain budget overflow")
    }
    pub fn chain(&self, requests: u32, budget: usize) -> Result<DsparkChain<'_, 'library>> {
        ensure!(
            self.chain_bytes(requests)? <= budget,
            "dSpark chain exceeds budget"
        );
        let library = self.library;
        let bytes = self.stage_bytes(requests)?;
        Ok(DsparkChain {
            width: self.draft_width,
            maximum: self.draft_width,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            stages: [
                self.stage(0, requests, bytes)?,
                self.stage(1, requests, bytes)?,
                self.stage(2, requests, bytes)?,
            ],
            graphs: Default::default(),
            ready: None,
            pending: None,
            download: HostAllocation::new(library, requests as usize * (2 * self.draft_width + 1) * 4)?,
            embedding: None,
            tokens: DeviceAllocation::new(library, requests as usize * 4)?,
            token_staging: HostAllocation::new(library, requests as usize * 4)?,
            token_count: None,
            terminal: None,
            ops: library.v41_attention_ops_width(self.draft_width)?,
        })
    }
}
impl DsparkChain<'_, '_> {
    /// Reserve a fresh attempt in packed seed/cache row order. Failed attempts
    /// consume admitted ranges; replay reuses them until prepared again.
    pub fn prepare_sampling(
        &mut self,
        rngs: &mut [&mut DsparkRng],
        temperatures: &[f32],
    ) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        self.terminal
            .as_mut()
            .context("dSpark chain has no terminal")?
            .prepare_sampling(rngs, temperatures)
    }
    pub fn stage_sampling(&mut self, rngs: &mut [&mut DsparkRng], temperatures: &[f32]) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        self.terminal.as_mut().context("dSpark chain has no terminal")?
            .stage_sampling(rngs, temperatures)
    }
    pub fn has_graph(&self, count: usize) -> bool { self.graphs.contains_key(&(count, self.width)) }
    pub fn width(&self) -> usize { self.width }
    /// Select the draft width (5 or 7, within the loaded maximum) for the next
    /// draft. All storage is sized for the maximum; each width keeps its own
    /// captured graphs, so switching is free after both are warm.
    pub fn set_width(&mut self, width: usize) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        ensure!(matches!(width, 5 | 7) && width <= self.maximum, "draft width must be five or seven within the loaded width");
        if width == self.width { return Ok(()); }
        self.invalidate();
        for stage in &mut self.stages { stage.set_width(width, self.maximum)?; }
        if let Some(terminal) = &mut self.terminal { terminal.set_width(width)?; }
        self.ops = self.stream.library.v41_attention_ops_width(width)?;
        self.width = width;
        Ok(())
    }
    /// Borrowed anchor + K tokens [K+1,R], corrected raw logits [K,R,V] and
    /// raw confidence [K,R], live until reuse/drop. No target history is committed.
    pub fn draft_output(&self) -> Result<[Ds41rtDeviceBuffer; 3]> {
        let requests = self.ready.context("dSpark draft output incomplete")?;
        self.terminal
            .as_ref()
            .context("dSpark chain has no terminal")?
            .output_storage(requests)
    }

    /// Seed IDs follow cache-binding row order; invalid input clears publication
    /// and token readiness so a subsequent execution cannot consume old IDs.
    pub fn set_tokens(&mut self, tokens: &[i32]) -> Result<()> {
        self.stage_tokens(tokens)?;
        let uploaded = self.stream.library.copy_h2d(self.tokens.buffer,
            &self.token_staging.bytes_mut()[..tokens.len() * 4]);
        if uploaded.is_err() { self.token_count = None; }
        uploaded
    }
    /// CPU-only seed staging; pending work excludes writes to this pinned owner.
    pub fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        self.token_count = None;
        ensure!(
            self.embedding.is_some(),
            "dSpark chain has no shared embedding binding"
        );
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= 16
                && tokens.len() <= self.tokens.buffer.bytes / 4,
            "invalid dSpark seed count"
        );
        ensure!(
            tokens.iter().all(|&id| (0..129280).contains(&id)),
            "invalid dSpark seed token"
        );
        let bytes = self.token_staging.bytes_mut();
        for (i, id) in tokens.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&id.to_ne_bytes());
        }
        self.token_count = Some(tokens.len());
        Ok(())
    }

    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        self.stages[0].inputs()
    }
    fn invalidate(&mut self) {
        self.ready = None;
        for stage in &mut self.stages {
            stage.invalidate();
        }
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn prepare(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[WindowRead; 3]> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        ensure!(
            self.embedding.is_none() || self.token_count == Some(bindings[0].len()),
            "dSpark seed count differs from cache bindings"
        );
        if let Some(terminal) = &self.terminal {
            terminal.validate_sampling(bindings[0].len())?;
        }
        let reads = [
            self.stages[0].prepare(windows[0], bindings[0])?,
            self.stages[1].prepare(windows[1], bindings[1])?,
            self.stages[2].prepare(windows[2], bindings[2])?,
        ];
        ensure!(
            reads[0].owner != reads[1].owner
                && reads[0].owner != reads[2].owner
                && reads[1].owner != reads[2].owner,
            "dSpark stages require independent cache owners"
        );
        for stage in 1..3 {
            ensure!(
                bindings[stage].len() == bindings[0].len(),
                "dSpark chain stage counts differ"
            );
            for (&(first, end), &(lease, stage_end)) in bindings[0].iter().zip(bindings[stage]) {
                ensure!(
                    end == stage_end
                        && windows[0].request_id(first)? == windows[stage].request_id(lease)?,
                    "dSpark chain request or committed position differs"
                );
            }
        }
        Ok(reads)
    }
    fn upload(
        &mut self,
        reads: &[WindowRead; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<()> {
        for stage in 0..3 {
            self.stages[stage].upload(&reads[stage], bindings[stage])?;
        }
        Ok(())
    }
    unsafe fn enqueue(&mut self, reads: &[WindowRead; 3], requests: usize) -> Result<()> {
        if let Some(embedding) = self.embedding {
            let output = self.inputs();
            unsafe {
                self.ops.embed(
                    embedding.get("embed.weight")?,
                    self.tokens.buffer,
                    output[0],
                    output[1],
                    requests as u32,
                    self.stream.raw,
                )?;
            }
        }
        for stage in 0..3 {
            if stage > 0 {
                let source = self.stages[stage - 1].output_storage();
                let destination = self.stages[stage].inputs();
                unsafe {
                    self.stream.library.copy_d2d_async(
                        destination[0],
                        source[0],
                        requests * self.width * 40960,
                        self.stream.raw,
                    )?;
                    self.stream.library.copy_d2d_async(
                        destination[1],
                        source[1],
                        requests * self.width * 16,
                        self.stream.raw,
                    )?;
                }
            }
            unsafe {
                self.stages[stage].enqueue_on(&reads[stage], requests, self.stream.raw)?;
            }
        }
        if let Some(terminal) = &self.terminal {
            let source = self.stages[2].output_storage();
            let target = terminal.inputs();
            unsafe {
                self.ops.terminal_layout(
                    source[0],
                    source[1],
                    target[0],
                    target[1],
                    requests as u32,
                    self.stream.raw,
                )?;
                self.stream.library.copy_d2d_async(
                    target[2],
                    self.tokens.buffer,
                    requests * 4,
                    self.stream.raw,
                )?;
                terminal.enqueue_on(requests, self.stream.raw)?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// For a plain chain, initialize finite BF16 residuals and FP32 pre-mix in
    /// input row order. An embedded chain initializes these from set_tokens.
    /// Cache bindings across stages describe the same requests, in seed order.
    /// For a complete draft, prepare sampling in the same packed request order.
    /// All buffers are on this device and serialized through graph completion.
    pub unsafe fn execute(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let reads = self.prepare(windows, bindings)?;
        self.upload(&reads, bindings)?;
        let launched = unsafe { self.enqueue(&reads, bindings[0].len()) };
        let drained = self.synchronize();
        if let Err(error) = launched.and(drained) {
            self.invalidate();
            return Err(error);
        }
        self.ready = Some(bindings[0].len());
        self.output()
    }
    /// # Safety
    /// Same contract as execute; captures all three independent window owners.
    pub unsafe fn capture(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        ensure!(!self.has_graph(bindings[0].len()), "dSpark chain count already captured");
        unsafe {
            self.execute(windows, bindings)?;
        }
        let reads = self.prepare(windows, bindings)?;
        unsafe { self.capture_ready(&reads, bindings[0].len()) }
    }
    /// Warmup completed; never suspend while capturing.
    unsafe fn capture_ready(&mut self, reads: &[WindowRead; 3], count: usize) -> Result<()> {
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(reads, count) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        self.invalidate();
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graphs.insert((count, self.width), (graph, reads.each_ref().map(|r| r.owner)));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same contract as execute; owner IDs and request count must match capture.
    pub unsafe fn replay(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let reads = self.prepare(windows, bindings)?;
        let count = bindings[0].len();
        let &(graph, owners) = self.graphs.get(&(count, self.width)).context("dSpark chain count not captured")?;
        ensure!(
            count == bindings[0].len() && owners == reads.each_ref().map(|r| r.owner),
            "dSpark chain capture binding differs"
        );
        self.upload(&reads, bindings)?;
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(count);
        self.output()
    }
    /// Queue warmup or replay plus compact copies. The caller keeps window
    /// owners alive; this owner retains reservations and pinned inputs throughout.
    /// # Safety
    /// Same inputs as replay; no raw consumer may mutate reserved ring slots.
    pub unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        let reads = self.prepare(windows, bindings)?;
        let count = bindings[0].len();
        if let Some(&(_, owners)) = self.graphs.get(&(count, self.width)) {
            ensure!(owners == reads.each_ref().map(|r| r.owner), "dSpark chain capture binding differs");
        }
        self.terminal.as_ref().context("dSpark chain has no terminal")?.output_storage(count)?;
        let pending = PendingDraft { count, reads, warming: !self.has_graph(count), armed: true,
            library: self.stream.library, stream: self.stream.raw };
        // The local guard already owns reservations before any partial upload.
        unsafe {
            self.stream.library.copy_h2d_async(self.tokens.buffer,
                &self.token_staging.bytes_mut()[..count * 4], self.stream.raw)?;
            self.terminal.as_mut().unwrap().upload_sampling_on(self.stream.raw)?;
            for stage in 0..3 {
                self.stages[stage].upload_on(&pending.reads[stage], bindings[stage], self.stream.raw)?;
            }
            if pending.warming { self.enqueue(&pending.reads, count)?; }
            else { self.launch_download(count)?; }
        }
        self.pending = Some(pending);
        Ok(())
    }
    unsafe fn launch_download(&mut self, count: usize) -> Result<()> {
        let graph = self.graphs.get(&(count, self.width)).context("dSpark chain count not captured")?.0;
        let output = self.terminal.as_ref().context("dSpark chain has no terminal")?.output_storage(count)?;
        unsafe {
            self.stream.library.cuda_graph_launch(graph, self.stream.raw)?;
            let bytes = self.download.bytes_mut();
            self.stream.library.copy_d2h_async(&mut bytes[..count * (self.width + 1) * 4], output[0], self.stream.raw)?;
            self.stream.library.copy_d2h_async(&mut bytes[count * (self.width + 1) * 4..count * (2 * self.width + 1) * 4], output[2], self.stream.raw)
        }
    }
    /// Cold warmup and final replay both poll. Capture itself never suspends.
    pub fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        ensure!(self.pending.is_some(), "dSpark replay not pending");
        match unsafe { self.stream.library.cuda_stream_query(self.stream.raw) } {
            Ok(false) => return Ok(None),
            Ok(true) => {},
            Err(error) => { self.pending = None; return Err(error); },
        }
        let mut pending = self.pending.take().unwrap();
        let count = pending.count;
        if pending.warming {
            unsafe {
                self.capture_ready(&pending.reads, count)?;
                self.launch_download(count)?;
            }
            pending.warming = false;
            self.pending = Some(pending);
            return Ok(None);
        }
        pending.armed = false;
        self.ready = Some(count);
        let bytes = self.download.bytes_mut();
        let tokens = bytes[..count * (self.width + 1) * 4].chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap())).collect();
        let confidence = bytes[count * (self.width + 1) * 4..count * (2 * self.width + 1) * 4].chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
        Ok(Some((tokens, confidence)))
    }
    #[cfg(test)]
    pub fn cancel_pending_for_test(&mut self) {
        self.pending = None; // guard drains before releasing reads
        self.invalidate();
    }
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let requests = self.ready.context("dSpark chain output incomplete")?;
        let mut output = self.stages[2].output_storage();
        output[0].bytes = requests * self.width * 40960;
        output[1].bytes = requests * self.width * 16;
        Ok(output)
    }
}
impl Drop for DsparkChain<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining dSpark chain");
        }
        if let Some(pending) = &mut self.pending { pending.armed = false; }
        self.pending = None; // GPU work has drained before read reservations release.
        for (_, (graph, _)) in std::mem::take(&mut self.graphs) {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying dSpark chain graph");
            }
        }
    }
}
