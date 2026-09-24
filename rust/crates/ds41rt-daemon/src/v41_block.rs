//! Shifted backbone attention/FFN sequencing with exact query-result identity.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_attention_output::AttentionOutput;
use crate::v41_attention_query::{AttentionQueryOutput, AttentionQueryWave};
use crate::v41_hc::HcSublayer;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
mod encoder_suffix;
mod transfer;
pub(crate) use transfer::BlockTransfer;
pub(crate) use encoder_suffix::EncoderSuffix;
#[derive(Clone, Copy)]
enum Phase {
    Idle,
    Prepared(QueryBinding, usize, bool),
    Attention(QueryBinding, usize),
    QueuedFfn(QueryBinding, usize),
    Ffn(QueryBinding, usize),
    Ready(QueryBinding, usize, bool),
}
/// Next-layer input after required engram work, before attention or dSpark taps.
pub(crate) struct PreparedBlockInput<'a> {
    pub residual: Ds41rtDeviceBuffer,
    pub pre: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
    previous: QueryBinding,
}
impl PreparedBlockInput<'_> {
    pub fn previous_binding(&self) -> QueryBinding { self.previous }
}
pub(crate) struct FfnInput<'a> {
    pub residual: Ds41rtDeviceBuffer,
    pub incoming_pre: Ds41rtDeviceBuffer,
    pub values: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
    binding: QueryBinding,
}
impl FfnInput<'_> {
    pub fn binding(&self) -> QueryBinding {
        self.binding
    }
}
pub(crate) struct BlockOutput<'a> {
    binding: QueryBinding,
    pub residual: Ds41rtDeviceBuffer,
    pub pre: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
}
impl BlockOutput<'_> {
    pub fn binding(&self) -> QueryBinding {
        self.binding
    }
}
pub(crate) struct BackboneBlockWave<'w, 'a> {
    library: &'a NativeLibrary,
    layer: usize,
    attention: HcSublayer<'w, 'a>,
    ffn: HcSublayer<'w, 'a>,
    capacity: usize,
    tokens: Vec<u64>,
    phase: Phase,
}
impl<'w, 'a> BackboneBlockWave<'w, 'a> {
    #[cfg(test)]
    pub fn trace_stream(&self) -> *mut std::ffi::c_void { self.ffn.stream_raw() }
    /// The stream carrying each layer's FFN finish (mHC post, next inputs).
    pub(crate) fn finish_stream(&self) -> *mut std::ffi::c_void { self.ffn.stream_raw() }
    /// Reuse this lane's two mHC workspaces for the adjacent layer. Completed
    /// residual/pre values are copied before installing both validated bindings.
    /// No device allocation or graph capture occurs here.
    pub fn advance(
        &mut self,
        next: &'w crate::v41_backbone_hc::BackboneHcWeights<'a>,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            let (binding, rows, copied_next) = match self.phase {
                Phase::Ready(binding, rows, copied_next) => (binding, rows, copied_next),
                _ => anyhow::bail!("block advance requires a completed FFN"),
            };
            ensure!(self.layer < 39 && next.layer() == self.layer + 1
                && binding.layer() == self.layer && self.tokens.len() == rows,
                "block advance layer or identity differs");
            let [attention, ffn] = next.prepare_bindings(&self.attention, &self.ffn)?;
            if !copied_next {
                let copied = unsafe { self.enqueue_next_inputs(rows) };
                let drained = unsafe { self.library.cuda_stream_synchronize(self.ffn.stream_raw()) };
                copied.and(drained)?;
            }
            // Backbone mHC executes on its own drained streams; it has no
            // external captured mHC graphs to invalidate during this rebind.
            unsafe {
                self.attention.install_binding(attention);
                self.ffn.install_binding(ffn);
            }
            self.layer = next.layer();
            self.phase = Phase::Prepared(binding, rows, ![1,14].contains(&self.layer));
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
    /// Start another sequence after completion, or after an explicit reset on
    /// cancellation. Token initialization is required before attention resumes.
    pub fn restart(
        &mut self,
        first: &'w crate::v41_backbone_hc::BackboneHcWeights<'a>,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            ensure!(first.layer() == 0
                && (matches!(self.phase, Phase::Idle)
                    || (self.layer == 39 && matches!(self.phase, Phase::Ready(..)))),
                "block restart requires layer zero and an idle or completed sequence");
            let [attention, ffn] = first.prepare_bindings(&self.attention, &self.ffn)?;
            unsafe {
                self.attention.install_binding(attention);
                self.ffn.install_binding(ffn);
            }
            self.layer = 0;
            self.reset();
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
    /// Rebind an idle lane to decoder layer 20 and restore retained encoder rows.
    pub fn initialize_decoder(&mut self, decoder: &'w crate::v41_backbone_hc::BackboneHcWeights<'a>,
        encoder: &BlockOutput<'_>) -> Result<()> {
        self.reset();
        let result = (|| -> Result<()> {
            ensure!(decoder.layer() == 20 && encoder.layer == 19, "invalid encoder/decoder boundary");
            let [attention, ffn] = decoder.prepare_bindings(&self.attention, &self.ffn)?;
            unsafe { self.attention.install_binding(attention); self.ffn.install_binding(ffn); }
            self.layer = 20;
            self.initialize_previous(encoder)
        })();
        if result.is_err() { self.reset(); }
        result
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Ok(2 * HcSublayer::device_bytes(capacity)?)
    }
    pub(crate) fn new(
        library: &'a NativeLibrary,
        layer: usize,
        attention: HcSublayer<'w, 'a>,
        ffn: HcSublayer<'w, 'a>,
        capacity: usize,
    ) -> Self {
        Self {
            library,
            layer,
            attention,
            ffn,
            capacity,
            tokens: Vec::with_capacity(capacity),
            phase: Phase::Idle,
        }
    }
    /// Residual [capacity,4,5120] and incoming FP32 pre [capacity,4]. Engram
    /// contributions and dSpark tap reads precede begin_attention at the caller.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        self.attention.inputs()
    }
    pub fn reset(&mut self) {
        self.phase = Phase::Idle;
        self.tokens.clear();
        self.attention.invalidate();
        self.ffn.invalidate();
    }
    /// Copy a completed adjacent layer into stable next-block input storage.
    /// Layers 1 and 14 remain unavailable until their engram update completes.
    pub fn initialize_previous(&mut self, previous: &BlockOutput<'_>) -> Result<()> {
        self.reset();
        let rows = previous.tokens.len();
        ensure!(previous.layer < 39 && self.layer == previous.layer + 1
            && previous.binding.layer() == previous.layer
            && rows > 0 && rows <= self.capacity
            && previous.tokens.iter().all(|&p| p < 1048576)
            && previous.residual.bytes == rows * 40960 && previous.pre.bytes == rows * 16,
            "previous block layer, tokens or extents differ");
        let sources = [previous.residual, previous.pre];
        let destinations = self.inputs();
        ensure!(sources.iter().zip(destinations).all(|(s,d)| s.device_id == d.device_id),
            "previous block device differs");
        for (source, destination) in sources.into_iter().zip(destinations) {
            self.library.copy_d2d(destination, source, source.bytes)?;
        }
        self.tokens.extend_from_slice(previous.tokens);
        self.phase = Phase::Prepared(previous.binding, rows, ![1,14].contains(&self.layer));
        Ok(())
    }
    /// # Safety
    /// Gathered engram rows/masks correspond to this block's exact request and
    /// token order, with current history leases and completed producer writes.
    /// Gather and gate storage remain live and exclusive until the call drains.
    pub unsafe fn apply_engram(
        &mut self,
        gate: &mut crate::v41_engram::layer::EngramGate<'_, '_>,
        gathered: &crate::v41_engram::EngramDeviceView,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            let (binding, rows) = match self.phase {
                Phase::Prepared(binding, rows, false) => (binding, rows),
                _ => anyhow::bail!("block is not awaiting engram"),
            };
            ensure!(gate.layer() == self.layer && gathered.rows == rows,
                "block engram layer or rows differ");
            let output = unsafe { gate.execute_captured(self.inputs()[0], gathered) }?;
            self.library.copy_d2d(self.inputs()[0], output, output.bytes)?;
            self.phase = Phase::Prepared(binding, rows, true);
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
    /// # Safety
    /// Keep gathered upload storage and this block exclusive through the gate/copy.
    pub async unsafe fn apply_engram_cooperative(&mut self,
        gate: &mut crate::v41_engram::layer::EngramGate<'_, '_>,
        gathered: &crate::v41_engram::EngramDeviceView) -> Result<()> {
        let Phase::Prepared(binding, rows, false) = std::mem::replace(&mut self.phase, Phase::Idle) else {
            anyhow::bail!("block is not awaiting engram");
        };
        ensure!(gate.layer() == self.layer && gathered.rows == rows, "block engram layer or rows differ");
        unsafe { gate.execute_into_cooperative(self.inputs()[0], gathered).await?; }
        self.phase = Phase::Prepared(binding, rows, true);
        Ok(())
    }
    /// Metadata for gather association before engram makes query inputs ready.
    pub fn pending_engram(&self) -> Result<(usize, &[u64])> {
        ensure!(matches!(self.phase, Phase::Prepared(_, _, false)),
            "block is not awaiting engram");
        Ok((self.layer, &self.tokens))
    }
    pub fn prepared_input(&self) -> Result<PreparedBlockInput<'_>> {
        let (previous, rows) = match self.phase {
            Phase::Prepared(binding, rows, true) => (binding, rows),
            _ => anyhow::bail!("block input is not prepared; engram may be pending"),
        };
        let [mut residual, mut pre] = self.inputs();
        residual.bytes = rows * 40960;
        pre.bytes = rows * 16;
        Ok(PreparedBlockInput { residual, pre, layer: self.layer,
            tokens: &self.tokens, previous })
    }
    /// # Safety
    /// Query and block storage have exclusive use on the same device. Required
    /// dSpark tap reads must finish before this call overwrites any input.
    pub unsafe fn begin_prepared_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
    ) -> Result<AttentionQueryOutput<'q>> {
        let tokens = match self.prepared_input() {
            Ok(input) => input.tokens.to_vec(),
            Err(e) => { self.reset(); return Err(e); }
        };
        unsafe { self.begin_attention(query, &tokens) }
    }
    /// Copy text embeddings into the first block's stable inputs. This leaves
    /// output unpublished. Image replacement belongs to the separate vision path.
    pub fn initialize_embedding(
        &mut self,
        embedding: &crate::v41_target_embedding::TargetEmbedding<'_>,
    ) -> Result<()> {
        self.reset();
        let rows = embedding.positions.len();
        ensure!(self.layer == 0 && rows > 0
            && rows <= self.capacity && embedding.token_ids.len() == rows
            && embedding.residual.bytes == rows * 40960 && embedding.pre.bytes == rows * 16,
            "initial block embedding geometry differs");
        for (destination, source) in self.inputs().into_iter()
            .zip([embedding.residual, embedding.pre]) {
            ensure!(destination.device_id == source.device_id, "initial block device differs");
            self.library.copy_d2d(destination, source, source.bytes)?;
        }
        Ok(())
    }
    /// # Safety
    /// Query and block storage have exclusive use on the embedding device.
    /// Embeddings are finite text inputs; image replacement uses the vision path.
    pub unsafe fn begin_embedded_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
        embedding: &crate::v41_target_embedding::TargetEmbedding<'_>,
    ) -> Result<AttentionQueryOutput<'q>> {
        self.reset();
        ensure!(query.layer() == 0, "initial block query layer differs");
        self.initialize_embedding(embedding)?;
        unsafe { self.begin_attention(query, embedding.positions) }
    }
    /// # Safety
    /// Inputs are finite in token order with all producer writes complete. Query
    /// storage and both boundary owners have exclusive use on this device.
    pub unsafe fn begin_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
        tokens: &[u64],
    ) -> Result<AttentionQueryOutput<'q>> {
        self.reset();
        ensure!(
            query.layer() == self.layer
                && !tokens.is_empty()
                && tokens.len() <= self.capacity
                && tokens.iter().all(|&p| p < 1048576),
            "block attention layer or tokens differ"
        );
        ensure!(
            query.input().device_id == self.inputs()[0].device_id,
            "block query device differs"
        );
        let timing = std::time::Instant::now();
        let out = match unsafe {
            query.execute_tokens_prepared(tokens, |stream, input| {
                self.attention.enqueue_begin(tokens.len(), Some(input), stream)?;
                Ok(())
            })
        } {
            Ok(o) => o,
            Err(e) => {
                self.reset();
                return Err(e);
            }
        };
        self.tokens.extend_from_slice(tokens);
        tracing::debug!(target: "ds41rt::timing", layer=self.layer, rows=tokens.len(), total_us=timing.elapsed().as_micros() as u64, "target query preparation");
        self.phase = Phase::Attention(out.binding()?, tokens.len());
        Ok(out)
    }
    /// # Safety
    /// Producer, block and query storage stay owned through completion. The
    /// producer initializes block inputs on the query stream before mHC/query.
    pub async unsafe fn begin_attention_cooperative<'q>(&mut self,
        query: &'q mut AttentionQueryWave<'_, '_>, tokens: &[u64],
        prepare: impl FnOnce(*mut std::ffi::c_void, [Ds41rtDeviceBuffer; 2]) -> Result<()>)
        -> Result<AttentionQueryOutput<'q>> {
        unsafe { self.begin_attention_with_projection_cooperative(query,tokens,None,prepare).await }
    }
    /// # Safety
    /// Same contract as begin_attention_cooperative; retain the optional TP2
    /// projection owner through completion or drained cancellation.
    pub async unsafe fn begin_attention_with_projection_cooperative<'q>(&mut self,
        query: &'q mut AttentionQueryWave<'_, '_>, tokens: &[u64],
        projection: Option<&mut crate::v41_projection_tp2::Wave<'_, '_>>,
        prepare: impl FnOnce(*mut std::ffi::c_void, [Ds41rtDeviceBuffer; 2]) -> Result<()>)
        -> Result<AttentionQueryOutput<'q>> {
        self.reset();
        ensure!(query.layer() == self.layer && !tokens.is_empty() && tokens.len() <= self.capacity
            && tokens.iter().all(|&p| p < 1048576), "block attention layer or tokens differ");
        ensure!(query.input().device_id == self.inputs()[0].device_id, "block query device differs");
        let prepare_query = |stream, input| unsafe {
            prepare(stream, self.inputs())?;
            self.attention.enqueue_begin(tokens.len(), Some(input), stream)?;
            Ok(())
        };
        let out = unsafe { match projection {
            Some(projection) => query.execute_tokens_tp2_prepared_cooperative(tokens,projection,prepare_query).await?,
            None => query.execute_tokens_prepared_cooperative(tokens,prepare_query).await?,
        } };
        self.tokens.extend_from_slice(tokens);
        self.phase = Phase::Attention(out.binding()?, tokens.len());
        Ok(out)
    }
    /// # Safety
    /// Same contract as begin_prepared_attention, retained across completion waits.
    pub async unsafe fn begin_prepared_attention_cooperative<'q>(&mut self,
        query: &'q mut AttentionQueryWave<'_, '_>) -> Result<AttentionQueryOutput<'q>> {
        unsafe { self.begin_prepared_attention_with_projection_cooperative(query,None).await }
    }
    /// # Safety
    /// Same contract as begin_attention_with_projection_cooperative.
    pub async unsafe fn begin_prepared_attention_with_projection_cooperative<'q>(&mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
        projection: Option<&mut crate::v41_projection_tp2::Wave<'_, '_>>)
        -> Result<AttentionQueryOutput<'q>> {
        let tokens = match self.prepared_input() {
            Ok(input) => input.tokens.to_vec(),
            Err(error) => { self.reset(); return Err(error); }
        };
        unsafe { self.begin_attention_with_projection_cooperative(query, &tokens, projection, |_, _| Ok(())).await }
    }
    /// # Safety
    /// Attention output is complete and immutable; no external writes race the
    /// preserved residual or mixing coefficients. A rejected finish resets phase.
    pub unsafe fn begin_ffn(&mut self, output: &AttentionOutput<'_>) -> Result<FfnInput<'_>> {
        let result = (|| -> Result<(Ds41rtDeviceBuffer, QueryBinding)> {
            let Phase::Attention(binding, rows) = std::mem::replace(&mut self.phase, Phase::Idle)
            else {
                anyhow::bail!("block attention is not pending");
            };
            ensure!(
                output.binding()? == binding
                    && output.layer == self.layer
                    && output.rows == rows
                    && output.projected.device_id == self.inputs()[0].device_id,
                "block attention result differs"
            );
            let stream = self.attention.stream_raw();
            let enqueued = (|| -> Result<Ds41rtDeviceBuffer> {
                unsafe { self.attention.enqueue_finish(Some(output.projected), stream)?; }
                // Keep the existing residual/pre buffers, but copy on the same
                // stream after attention post and before FFN pre/mixing.
                for ((dst, src), bytes) in self.ffn.inputs().into_iter()
                    .zip(self.attention.output_storage())
                    .zip([rows * 40960, rows * 16]) {
                    unsafe { self.library.copy_d2d_async(dst, src, bytes, stream)?; }
                }
                unsafe { self.ffn.enqueue_begin(rows, None, stream) }
            })();
            // Always drain, including partial enqueue failure: both mHC owners
            // must be safe to reset/rebind when this function returns.
            let drained = unsafe { self.library.cuda_stream_synchronize(stream) };
            let normalized = enqueued.and_then(|value| drained.map(|()| value))?;
            unsafe { self.attention.complete()?; }
            self.phase = Phase::Ffn(binding, rows);
            Ok((normalized, binding))
        })();
        match result {
            Ok((values, binding)) => Ok(FfnInput {
                residual: {
                    let mut b = self.ffn.inputs()[0];
                    b.bytes = values.bytes * 4;
                    b
                },
                incoming_pre: {
                    let mut b = self.ffn.inputs()[1];
                    b.bytes = values.bytes / 10240 * 16;
                    b
                },
                values,
                layer: self.layer,
                tokens: &self.tokens,
                binding,
            }),
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }
    /// # Safety
    /// Projection is ordered on stream with the matching query binding. Keep
    /// both mHC owners exclusive and alive until stream drains, including errors.
    /// Only complete_queued_ffn may publish the resulting normalized input.
    pub unsafe fn enqueue_ffn(
        &mut self, binding: QueryBinding, rows: usize, projected: Ds41rtDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<Ds41rtDeviceBuffer> {
        let Phase::Attention(expected, count) = self.phase else {
            anyhow::bail!("block attention is not pending for queued FFN");
        };
        ensure!(binding == expected && rows == count && binding.layer() == self.layer
            && projected.device_id == self.inputs()[0].device_id
            && !projected.ptr.is_null() && projected.bytes == rows * 10240,
            "queued block attention result differs");
        self.phase = Phase::QueuedFfn(binding, rows);
        unsafe { self.attention.enqueue_finish(Some(projected), stream)?; }
        for ((dst, src), bytes) in self.ffn.inputs().into_iter()
            .zip(self.attention.output_storage()).zip([rows * 40960, rows * 16]) {
            unsafe { self.library.copy_d2d_async(dst, src, bytes, stream)?; }
        }
        unsafe { self.ffn.enqueue_begin(rows, None, stream) }
    }
    /// # Safety
    /// A graph with enqueue_ffn's exact operations is about to replay on the
    /// enclosing stream. Its weights/buffers match this block and rows.
    pub unsafe fn prepare_ffn_graph_replay(&mut self, binding: QueryBinding, rows: usize) -> Result<()> {
        ensure!(matches!(self.phase, Phase::Attention(b, n) if b == binding && n == rows),
            "FFN graph replay query differs");
        unsafe { self.attention.graph_post_state(rows)?; self.ffn.graph_begin_state(rows)?; }
        self.phase = Phase::QueuedFfn(binding, rows); Ok(())
    }
    /// # Safety
    /// Only the unpublished warmup of this same graph has completed. Attention
    /// residual and mixing coefficients were read, not overwritten, by warmup.
    pub unsafe fn restore_ffn_graph_warmup(&mut self, binding: QueryBinding, rows: usize) -> Result<()> {
        ensure!(matches!(self.phase, Phase::QueuedFfn(b, n) if b == binding && n == rows),
            "FFN graph warmup query differs");
        unsafe { self.attention.graph_begin_state(rows)?; }
        self.ffn.invalidate(); self.phase = Phase::Attention(binding, rows); Ok(())
    }
    pub fn chain_graph_identity(&self) -> Vec<usize> {
        self.attention.graph_identity().into_iter().chain(self.ffn.graph_identity()).collect()
    }
    pub fn graph_normalized_storage(&self, rows: usize) -> Ds41rtDeviceBuffer {
        self.ffn.normalized_storage(rows)
    }
    /// # Safety
    /// The enclosing chain has successfully drained its stream. Values are the
    /// buffer returned by this block's enqueue_ffn, without intervening reuse.
    pub unsafe fn complete_queued_ffn(&mut self, values: Ds41rtDeviceBuffer) -> Result<FfnInput<'_>> {
        let Phase::QueuedFfn(binding, rows) = self.phase else {
            anyhow::bail!("block has no queued FFN");
        };
        ensure!(values.bytes == rows * 10240 && values.device_id == self.inputs()[0].device_id,
            "queued normalized FFN extent differs");
        unsafe { self.attention.complete()?; }
        self.phase = Phase::Ffn(binding, rows);
        let mut residual = self.ffn.inputs()[0]; residual.bytes = rows * 40960;
        let mut incoming_pre = self.ffn.inputs()[1]; incoming_pre.bytes = rows * 16;
        Ok(FfnInput { residual, incoming_pre, values, layer: self.layer, tokens: &self.tokens, binding })
    }
    /// # Safety
    /// Result is the finite completed shared+routed FFN output for the returned
    /// FfnInput binding, in its token order, on this device. It stays immutable
    /// through the copy. Transport/expert reduction must establish this contract.
    pub unsafe fn finish_ffn(&mut self, binding: QueryBinding, result: Ds41rtDeviceBuffer) -> Result<BlockOutput<'_>> {
        // Inside a stage chain the next-layer input copies share the ordered
        // mHC stream, so advance() needs no separate drained copy.
        let copy_next = crate::v41_memory::chain::active() && self.layer < 39;
        let rows = unsafe { self.enqueue_finish_ffn(binding, result, copy_next)? };
        if let Err(error) = unsafe { crate::v41_memory::chain::finish(self.library, self.ffn.stream_raw()) } {
            self.reset(); return Err(error);
        }
        unsafe { self.publish_finished_ffn(binding, rows, copy_next) }
    }
    /// # Safety
    /// Retain the completed transport result and exclusive block storage through
    /// completion or cancellation drain. Next-layer copies share the mHC stream.
    pub async unsafe fn finish_ffn_cooperative(&mut self, binding: QueryBinding,
        result: Ds41rtDeviceBuffer) -> Result<BlockOutput<'_>> {
        unsafe { self.finish_ffn_with_next_copy(binding,result,self.layer < 39).await }
    }
    /// # Safety
    /// Same completed-result ownership as finish_ffn_cooperative. A peer handoff
    /// consumes the output directly, so no local next-input copy is required.
    pub async unsafe fn finish_ffn_for_handoff_cooperative(&mut self,binding:QueryBinding,
        result:Ds41rtDeviceBuffer)->Result<BlockOutput<'_>> {
        unsafe { self.finish_ffn_with_next_copy(binding,result,false).await }
    }
    async unsafe fn finish_ffn_with_next_copy(&mut self,binding:QueryBinding,
        result:Ds41rtDeviceBuffer,copy_next:bool)->Result<BlockOutput<'_>> {
        let rows = unsafe { self.enqueue_finish_ffn(binding, result, copy_next)? };
        if let Err(error) = self.ffn.wait_chain().await { self.reset(); return Err(error); }
        unsafe { self.publish_finished_ffn(binding, rows, copy_next) }
    }
    unsafe fn enqueue_next_inputs(&self, rows: usize) -> Result<()> {
        for ((source, destination), bytes) in self.ffn.output_storage().into_iter()
            .zip(self.inputs()).zip([rows * 40960, rows * 16]) {
            unsafe { self.library.copy_d2d_async(destination, source, bytes, self.ffn.stream_raw())?; }
        }
        Ok(())
    }
    unsafe fn enqueue_finish_ffn(&mut self, binding: QueryBinding,
        result: Ds41rtDeviceBuffer, copy_next: bool) -> Result<usize> {
        let submitted = (|| -> Result<usize> {
            let Phase::Ffn(expected, rows) = std::mem::replace(&mut self.phase, Phase::Idle) else {
                anyhow::bail!("block FFN is not pending");
            };
            ensure!(binding == expected && result.device_id == self.inputs()[0].device_id
                && result.bytes >= rows * 10240 && !result.ptr.is_null(), "block FFN result binding differs");
            unsafe { crate::v41_memory::chain::join(self.library, self.ffn.stream_raw())?; }
            unsafe { self.ffn.enqueue_finish(Some(result), self.ffn.stream_raw())?; }
            if copy_next { unsafe { self.enqueue_next_inputs(rows)?; } }
            Ok(rows)
        })();
        if submitted.is_err() {
            let drained = unsafe { self.library.cuda_stream_synchronize(self.ffn.stream_raw()) };
            self.reset();
            return submitted.and_then(|rows| drained.map(|()| rows));
        }
        submitted
    }
    unsafe fn publish_finished_ffn(&mut self, binding: QueryBinding,
        rows: usize, copied_next: bool) -> Result<BlockOutput<'_>> {
        if let Err(error) = unsafe { self.ffn.complete() } { self.reset(); return Err(error); }
        self.phase = Phase::Ready(binding, rows, copied_next);
        self.output()
    }
    #[cfg(test)]
    pub async unsafe fn check_queued_finish(&mut self, binding: QueryBinding,
        result: Ds41rtDeviceBuffer) -> Result<()> {
        use std::{future::Future, task::Poll};
        let Phase::Ffn(expected, rows) = self.phase else { anyhow::bail!("test requires pending FFN"); };
        let read = |library: &NativeLibrary, buffers: &[Ds41rtDeviceBuffer]| -> Result<Vec<Vec<u8>>> {
            buffers.iter().map(|&buffer| { let mut out = vec![0; buffer.bytes];
                library.copy_d2h(&mut out, buffer)?; Ok(out) }).collect()
        };
        let library = self.library;
        let reference = { let out = unsafe { self.finish_ffn(binding, result)? };
            read(library, &[out.residual, out.pre])? };
        self.ffn.rearm_finish_for_test(rows); self.phase = Phase::Ffn(expected, rows);
        let cancelled = {
            let mut future = std::pin::pin!(unsafe { self.finish_ffn_cooperative(binding, result) });
            std::future::poll_fn(|cx| Poll::Ready(future.as_mut().poll(cx).is_pending())).await
        };
        for _ in 0..2 {
            self.ffn.rearm_finish_for_test(rows); self.phase = Phase::Ffn(expected, rows);
            let actual = unsafe { self.finish_ffn_cooperative(binding, result).await? };
            assert_eq!(read(library, &[actual.residual, actual.pre])?, reference);
            if self.layer < 39 {
                let mut input = self.inputs(); input[0].bytes = rows * 40960; input[1].bytes = rows * 16;
                assert_eq!(read(library, &input)?, reference, "queued next-layer inputs differ");
            }
        }
        self.ffn.rearm_finish_for_test(rows); self.phase = Phase::Ffn(expected, rows);
        eprintln!("PASS queued layer finish {}: exact mHC/next-input parity, pending cancellation={cancelled}, reuse", self.layer);
        Ok(())
    }
    pub fn output(&self) -> Result<BlockOutput<'_>> {
        let (binding, rows) = match self.phase {
            Phase::Ready(binding, rows, _) => (binding, rows),
            _ => return Err(anyhow::anyhow!("block output unpublished")),
        };
        ensure!(self.tokens.len() == rows, "block token count differs");
        let [residual, pre] = self.ffn.output().context("block FFN output unavailable")?;
        Ok(BlockOutput {
            binding,
            residual,
            pre,
            layer: self.layer,
            tokens: &self.tokens,
        })
    }
}

#[cfg(test)]
impl<'a> BlockOutput<'a> {
    /// # Safety
    /// Test buffers contain complete residual/pre rows and outlive the output.
    pub(crate) unsafe fn from_test_buffers(layer: usize, tokens: &'a [u64],
        residual: Ds41rtDeviceBuffer, pre: Ds41rtDeviceBuffer) -> Result<Self> {
        Ok(Self { binding: QueryBinding::new(layer)?, layer, tokens, residual, pre })
    }
}
