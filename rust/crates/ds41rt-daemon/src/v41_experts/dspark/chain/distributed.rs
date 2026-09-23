//! Complete lane-owned GPU1 draft chain with peer embeddings and TP2 vocabulary.
use super::*;
use crate::v41_experts::dspark::DistributedDsparkTerminal;
use crate::v41_memory::device::{Device, DeviceOwner};
use crate::v41_tensors::VocabularyShard;
#[cfg(test)]
mod tests;

enum Phase<'a> { Transformer(PendingDraft<'a>), Terminal(usize), Download(usize) }

pub(crate) struct DistributedDsparkChain<'w, 'a> {
    chain: DeviceOwner<'a, DsparkChain<'w, 'a>>,
    terminal: DistributedDsparkTerminal<'w, 'a>,
    ready: Option<usize>,
    pending: Option<Phase<'a>>,
}
impl<'w, 'a> DistributedDsparkChain<'w, 'a> {
    pub fn device(&self) -> Device<'a> { self.chain.device }
    pub fn device_bytes(weights: &DsparkWeights<'a>, requests: u32, split: usize) -> Result<[usize; 2]> {
        let mut bytes = DistributedDsparkTerminal::device_bytes_with_width(requests as usize, split, weights.draft_width)?;
        bytes[0] = bytes[0].checked_add(weights.peer_chain_bytes(requests)?).context("distributed draft peer budget overflow")?;
        bytes[1] = bytes[1].checked_add(weights.chain_bytes(requests)?).context("distributed draft budget overflow")?;
        Ok(bytes)
    }
    pub fn new(devices: [Device<'a>; 2], weights: &'w DsparkWeights<'a>,
        embedding: &'w NativeRtxTensors<'a>, shards: [&'w VocabularyShard<'a>; 2],
        requests: u32, budgets: [usize; 2]) -> Result<Self> {
        let required = devices[1].run(|| Self::device_bytes(weights, requests, shards[0].tokens().end))?;
        ensure!(required.iter().zip(budgets).all(|(need, budget)| *need <= budget), "distributed draft exceeds budget");
        ensure!(weights.tensor("mtp.2.norm.weight")?.device_id == devices[1].id, "draft weights must reside on rank 1");
        let terminal_bytes = DistributedDsparkTerminal::device_bytes_with_width(requests as usize, shards[0].tokens().end, weights.draft_width)?;
        Ok(Self {
            chain: devices[1].own(|| weights.embedded_chain(embedding, requests, weights.chain_bytes(requests)?))?,
            terminal: DistributedDsparkTerminal::new(devices, weights, shards, requests as usize, terminal_bytes)?,
            ready: None, pending: None,
        })
    }
    pub fn width(&self) -> usize { self.chain.width }
    pub fn set_width(&mut self, width: usize) -> Result<()> {
        ensure!(self.pending.is_none(), "distributed draft still pending");
        self.ready = None;
        let device = self.chain.device;
        device.run(|| self.chain.get_mut().set_width(width))?;
        self.terminal.set_width(width)
    }
    pub fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> {
        ensure!(self.pending.is_none(), "distributed draft still pending");
        self.ready = None;
        self.chain.stage_tokens(tokens)
    }
    pub fn stage_sampling(&mut self, rngs: &mut [&mut DsparkRng], temperatures: &[f32]) -> Result<()> {
        ensure!(self.pending.is_none(), "distributed draft still pending");
        self.ready = None;
        self.terminal.stage_sampling(rngs, temperatures)
    }
    /// # Safety
    /// Cache windows belong to GPU1 and describe the same packed requests and
    /// committed positions in all three stages. Producers are complete. Borrowed
    /// weights and window storage stay live without conflicting access until
    /// successful polling or cancellation.
    pub async unsafe fn execute(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<[Ds41rtDeviceBuffer; 3]> {
        unsafe { self.begin_replay(windows, bindings)?; }
        struct Cancel<'s, 'w, 'a> { chain: &'s mut DistributedDsparkChain<'w, 'a>, armed: bool }
        impl Drop for Cancel<'_, '_, '_> {
            fn drop(&mut self) { if self.armed { self.chain.cancel(); } }
        }
        let mut pending = Cancel { chain: self, armed: true };
        while !pending.chain.poll_execute()? { tokio::task::yield_now().await; }
        pending.armed = false;
        pending.chain.output()
    }
    /// # Safety
    /// Same cache/input contract as `execute`, retained through polling/cancel.
    pub unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        let device = self.chain.device;
        device.run(|| unsafe { self.begin_on(windows, bindings) })
    }
    unsafe fn begin_on(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        ensure!(self.pending.is_none(), "distributed draft still pending");
        self.ready = None;
        let count = bindings[0].len();
        self.terminal.validate_sampling(count)?;
        let chain = self.chain.get_mut();
        let reads = chain.prepare(windows, bindings)?;
        if let Some(&(_, owners)) = chain.graphs.get(&(count, chain.width)) {
            ensure!(owners == reads.each_ref().map(|read| read.owner), "distributed draft capture owner differs");
        }
        let pending = PendingDraft { count, reads, warming: !chain.has_graph(count), armed: true,
            library: chain.stream.library, stream: chain.stream.raw };
        // The local guard drains partial submission failures under GPU1 before
        // releasing cache readers. Success transfers it into lane-owned state.
        unsafe {
            let lib = chain.stream.library;
            lib.copy_h2d_async(chain.tokens.buffer, &chain.token_staging.bytes_mut()[..count * 4], chain.stream.raw)?;
            for stage in 0..3 { chain.stages[stage].upload_on(&pending.reads[stage], bindings[stage], chain.stream.raw)?; }
            if pending.warming { chain.enqueue(&pending.reads, count)?; }
            else { lib.cuda_graph_launch(chain.graphs[&(count, chain.width)].0, chain.stream.raw)?; }
            let source = chain.stages[2].output_storage();
            let target = self.terminal.inputs();
            chain.ops.terminal_layout(source[0], source[1], target[0], target[1], count as u32, chain.stream.raw)?;
            lib.copy_d2d_async(target[2], chain.tokens.buffer, count * 4, chain.stream.raw)?;
        }
        self.pending = Some(Phase::Transformer(pending));
        Ok(())
    }
    fn poll_execute(&mut self) -> Result<bool> {
        let device = self.chain.device;
        let result = device.run(|| self.poll_on());
        if result.is_err() { self.cancel(); }
        result
    }
    fn poll_on(&mut self) -> Result<bool> {
        ensure!(self.pending.is_some(), "distributed draft not pending");
        loop {
            match self.pending.take().unwrap() {
                Phase::Transformer(mut pending) => {
                    let chain = self.chain.get_mut();
                    if !unsafe { chain.stream.library.cuda_stream_query(chain.stream.raw)? } {
                        self.pending = Some(Phase::Transformer(pending));
                        return Ok(false);
                    }
                    let count = pending.count;
                    if pending.warming { unsafe { chain.capture_ready(&pending.reads, count)?; } }
                    pending.armed = false;
                    drop(pending);
                    unsafe { self.terminal.begin(count)?; }
                    self.pending = Some(Phase::Terminal(count));
                }
                Phase::Terminal(count) => {
                    // Retain the phase before calls that can partially submit.
                    self.pending = Some(Phase::Terminal(count));
                    if !self.terminal.poll()? { return Ok(false); }
                    let output = self.terminal.output()?;
                    let chain = self.chain.get_mut();
                    unsafe {
                        let bytes = chain.download.bytes_mut();
                        chain.stream.library.copy_d2h_async(&mut bytes[..count * (chain.width + 1) * 4], output[0], chain.stream.raw)?;
                        chain.stream.library.copy_d2h_async(&mut bytes[count * (chain.width + 1) * 4..count * (2 * chain.width + 1) * 4], output[2], chain.stream.raw)?;
                    }
                    self.pending = Some(Phase::Download(count));
                }
                Phase::Download(count) => {
                    self.pending = Some(Phase::Download(count));
                    if !unsafe { self.chain.stream.library.cuda_stream_query(self.chain.stream.raw)? } { return Ok(false); }
                    self.pending = None;
                    self.ready = Some(count);
                    return Ok(true);
                }
            }
        }
    }
    pub fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        if !self.poll_execute()? { return Ok(None); }
        let count = self.ready.unwrap();
        let chain = self.chain.get_mut();
        let bytes = chain.download.bytes_mut();
        let tokens = bytes[..count * (chain.width + 1) * 4].chunks_exact(4)
            .map(|v| u32::from_ne_bytes(v.try_into().unwrap())).collect();
        let confidence = bytes[count * (chain.width + 1) * 4..count * (2 * chain.width + 1) * 4].chunks_exact(4)
            .map(|v| f32::from_ne_bytes(v.try_into().unwrap())).collect();
        Ok(Some((tokens, confidence)))
    }
    pub fn cancel(&mut self) {
        self.terminal.cancel();
        let device = self.chain.device;
        if let Err(error) = device.run(|| {
            // Also covers a partial compact download on an error path.
            let drained = self.chain.synchronize();
            if let Some(Phase::Transformer(mut pending)) = self.pending.take() {
                if drained.is_ok() { pending.armed = false; }
                // Even an error-path guard is destroyed in the device scope.
                drop(pending);
            }
            drained
        }) { tracing::error!(%error, "draining cancelled distributed draft"); }
        self.ready = None;
    }
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 3]> {
        ensure!(self.ready.is_some(), "distributed draft output unpublished");
        self.terminal.output()
    }
}
impl Drop for DistributedDsparkChain<'_, '_> {
    fn drop(&mut self) { self.cancel(); }
}
