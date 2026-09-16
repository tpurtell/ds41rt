//! Real backbone shared experts, with immutable weights and exclusive captured waves.
#[path = "v41_backbone_shared/tp2.rs"]
pub(crate) mod tp2;
use crate::v41_attention_binding::QueryBinding;
use crate::v41_block::FfnInput;
use crate::v41_layer_graphs::LayerGraphs;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_shared_ffn::SharedFfn;
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::OfficialV41Catalog;
use std::marker::PhantomData;
pub(crate) struct BackboneSharedWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    tensors: NativeRtxTensors<'a>,
    scales: Vec<DeviceAllocation<'a>>,
}
impl<'a> BackboneSharedWeights<'a> {
    fn names(layer: usize) -> Result<Vec<String>> {
        ensure!(layer < 40, "invalid backbone shared FFN layer");
        Ok(["w1", "w3", "w2"]
            .into_iter()
            .flat_map(|n| {
                ["weight", "scale"].map(|s| format!("layers.{layer}.ffn.shared_experts.{n}.{s}"))
            })
            .collect())
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        Ok(NativeRtxTensors::plan(catalog, &Self::names(layer)?)?
            + (2 * library
                .v41_fp8_matrix_info(1, 5120, 2304)?
                .packed_weight_scale_bytes
                + library
                    .v41_fp8_matrix_info(1, 2304, 5120)?
                    .packed_weight_scale_bytes) as usize)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, layer)? <= budget,
            "backbone shared weights exceed budget"
        );
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging)?;
        let mut scales = Vec::with_capacity(3);
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        for (i, (k, n)) in [(5120, 2304), (5120, 2304), (2304, 5120)]
            .into_iter()
            .enumerate()
        {
            let kernel = library.v41_fp8_matrix_kernel(1, k, n)?;
            scales.push(DeviceAllocation::new(
                library,
                kernel.info().packed_weight_scale_bytes as usize,
            )?);
            unsafe {
                kernel.pack_scales(
                    tensors.get(&names[2 * i + 1])?,
                    scales[i].buffer,
                    stream.raw,
                )?;
            }
        }
        unsafe {
            library.cuda_stream_synchronize(stream.raw)?;
        }
        Ok(Self {
            library,
            layer,
            tensors,
            scales,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<BackboneSharedWave<'_, 'a>> {
        ensure!(
            BackboneSharedWave::device_bytes(self.library, capacity)? <= budget,
            "backbone shared wave exceeds budget"
        );
        Ok(BackboneSharedWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            inner: SharedFfn::new(
                self.library,
                &self.tensors,
                &format!("layers.{}.ffn.shared_experts", self.layer),
                &self.scales,
                capacity,
                SharedFfn::device_bytes(self.library, capacity)?,
            )?,
            input: DeviceAllocation::new(self.library, capacity as usize * 10240)?,
            result: DeviceAllocation::new(self.library, capacity as usize * 10240)?,
            layer: self.layer,
            capacity,
            graphs: LayerGraphs::new(self.library),
            weights: self,
            ready: None,
            origin: None,
        })
    }
}
pub(crate) struct SharedOutput<'a> {
    pub layer: usize,
    pub rows: u32,
    pub input: Ds41rtDeviceBuffer,
    pub intermediates: [Ds41rtDeviceBuffer; 3],
    pub values: Ds41rtDeviceBuffer,
    origin: Option<QueryBinding>,
    _owner: PhantomData<&'a ()>,
}
impl SharedOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.origin.context("shared FFN output has no block origin")
    }
}
pub(crate) struct BackboneSharedWave<'w, 'a> {
    stream: LoadStream<'a>,
    inner: SharedFfn<'w, 'a>,
    input: DeviceAllocation<'a>,
    result: DeviceAllocation<'a>,
    layer: usize,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, BackboneSharedWeights<'a>>,
    weights: &'w BackboneSharedWeights<'a>,
    ready: Option<u32>,
    origin: Option<QueryBinding>,
}
impl<'w, 'a> BackboneSharedWave<'w, 'a> {
    /// Reuse this lane's shared-expert storage with another backbone layer.
    pub fn rebind(&mut self, weights: &'w BackboneSharedWeights<'a>) -> Result<()> {
        self.invalidate();
        ensure!(
            std::ptr::eq(self.stream.library, weights.library),
            "shared FFN rebound weight library differs"
        );
        self.stream.require_complete()?;
        // LayerGraphs retains the full owner, including packed scales, for
        // every cached graph. Only this drained stream consumes the workspace.
        unsafe {
            self.inner.rebind(
                &weights.tensors,
                &format!("layers.{}.ffn.shared_experts", weights.layer),
                &weights.scales,
            )?;
        }
        self.weights = weights;
        self.layer = weights.layer;
        Ok(())
    }
}
impl BackboneSharedWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Ok(SharedFfn::device_bytes(library, capacity)? + capacity as usize * 20480)
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.input.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn invalidate(&mut self) {
        self.ready = None;
        self.origin = None;
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "shared FFN rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.inner
                .enqueue(self.input.buffer, self.result.buffer, rows, self.stream.raw)
        }
    }
    /// # Safety
    /// Input is finite, initialized and exclusively owned until the call drains.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<SharedOutput<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same contract as execute. Warmup is drained before capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        ensure!(
            self.graphs.get_shape(self.layer, self.weights, rows).is_none(),
            "shared FFN graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        unsafe { self.capture_ready(rows) }
    }
    // Warmup has completed. Capture never yields with the stream in capture mode.
    unsafe fn capture_ready(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                if let Err(error) =
                    unsafe { self.graphs.insert(self.layer, self.weights, rows, graph) }
                {
                    unsafe {
                        self.stream.library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(error);
                }
                Ok(())
            }
            (Err(e), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same contract as execute; graph live row count must match.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<SharedOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self
            .graphs
            .get_shape(self.layer, self.weights, rows)
            .context("shared FFN graph missing")?;
        ensure!(count == rows, "shared FFN captured rows differ");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// The completed block input stays immutable through the copy. This wave has
    /// exclusive storage; returned values are only the shared contribution.
    pub unsafe fn execute_ffn(&mut self, input: &FfnInput<'_>) -> Result<SharedOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.input.buffer.device_id,
            "shared FFN block input differs"
        );
        self.stream
            .library
            .copy_d2d(self.input.buffer, input.values, input.values.bytes)?;
        let rows = input.tokens.len() as u32;
        if self
            .graphs
            .get_shape(self.layer, self.weights, rows)
            .is_none()
        {
            unsafe {
                self.capture(rows)?;
            }
        }
        unsafe {
            self.replay(rows)?;
        }
        self.origin = Some(input.binding());
        self.output()
    }
    /// # Safety
    /// Same input contract as execute_ffn; the future retains this wave and input
    /// until completion. Cancellation drains before their storage can be reused.
    pub async unsafe fn execute_ffn_cooperative(&mut self, input: &FfnInput<'_>) -> Result<SharedOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.input.buffer.device_id,
            "shared FFN block input differs"
        );
        let rows = input.tokens.len() as u32;
        let cold = self.graphs.get_shape(self.layer, self.weights, rows).is_none();
        let launched = (|| unsafe {
            self.stream.library.copy_d2d_async(self.input.buffer, input.values, input.values.bytes, self.stream.raw)?;
            if cold { self.enqueue(rows) } else {
                let (graph, _) = self.graphs.get_shape(self.layer, self.weights, rows).unwrap();
                self.stream.library.cuda_graph_launch(graph, self.stream.raw)
            }
        })();
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        self.stream.wait().await?;
        if cold {
            // The eager execution above already completed these inputs. Capture
            // records future launches without executing them; publish that result
            // instead of running the same work again on every cache miss.
            unsafe { self.capture_ready(rows)?; }
        }
        self.ready = Some(rows);
        self.origin = Some(input.binding());
        self.output()
    }
    pub fn output(&self) -> Result<SharedOutput<'_>> {
        let rows = self.ready.context("shared FFN output unpublished")?;
        let mut input = self.input.buffer;
        input.bytes = rows as usize * 10240;
        let mut values = self.result.buffer;
        values.bytes = input.bytes;
        let intermediates = self.inner.intermediates().map(|mut b| {
            b.bytes = rows as usize * 4608;
            b
        });
        Ok(SharedOutput {
            layer: self.layer,
            rows,
            input,
            intermediates,
            values,
            origin: self.origin,
            _owner: PhantomData,
        })
    }
    pub fn enable_small_graph_shapes(&mut self) { self.graphs.enable_small_shapes(); }
    /// Evict all shapes for the current layer; other layers remain cached.
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.synchronize()?;
        unsafe {
            self.graphs.remove(self.layer)?;
        }
        Ok(())
    }
}
impl Drop for BackboneSharedWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self
            .synchronize()
            .and_then(|()| unsafe { self.graphs.clear() })
        {
            tracing::error!(%e,"draining backbone shared FFN");
        }
    }
}
