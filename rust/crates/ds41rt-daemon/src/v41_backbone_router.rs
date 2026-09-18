//! Official backbone routing with block-bound inputs and canonical TP4 requests.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_block::FfnInput;
use crate::v41_layer_graphs::LayerGraphs;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41ExpertInputQuantizer, V41Router};
use ds41rt_loader::OfficialV41Catalog;
use std::marker::PhantomData;
pub(crate) struct BackboneRouterWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    tensors: NativeRtxTensors<'a>,
    nvfp4: bool,
}

fn request_hidden_format(nvfp4: bool) -> (ds41rt_transport::ExpertV2Dtype, usize) {
    if nvfp4 {
        (ds41rt_transport::ExpertV2Dtype::Bf16, 10240)
    } else {
        (ds41rt_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32, 5280)
    }
}
impl<'a> BackboneRouterWeights<'a> {
    fn names(layer: usize) -> Result<Vec<String>> {
        ensure!(layer < 40, "invalid backbone router layer");
        Ok(["weight", "bias", "bias_vl"]
            .map(|n| format!("layers.{layer}.ffn.gate.{n}"))
            .to_vec())
    }
    pub fn device_bytes(catalog: &OfficialV41Catalog, layer: usize) -> Result<usize> {
        let bytes = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        ensure!(
            bytes == 3_935_232,
            "unexpected backbone router weight geometry"
        );
        Ok(bytes)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(catalog, layer)? <= budget,
            "backbone router weights exceed budget"
        );
        Ok(Self {
            library,
            layer,
            nvfp4: catalog.nvfp4().is_some(),
            tensors: NativeRtxTensors::load(
                library,
                catalog,
                &Self::names(layer)?,
                budget,
                staging,
            )?,
        })
    }

    pub fn wave(&self, capacity: u32, budget: usize) -> Result<BackboneRouterWave<'_, 'a>> {
        ensure!(
            BackboneRouterWave::device_bytes(capacity)? <= budget,
            "backbone router wave exceeds budget"
        );
        Ok(BackboneRouterWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            kernel: self.library.v41_router()?,
            input_quantizer: self.library.v41_expert_input_quantizer()?,
            buffers: [10240, 1, 1536, 24, 24, 5280]
                .into_iter()
                .map(|n| DeviceAllocation::new(self.library, n * capacity as usize))
                .collect::<Result<Vec<_>>>()?,
            // Decode/verification benefits from batching small DMA transfers.
            // Large prefill retains direct downloads to avoid another host copy.
            request_staging: HostAllocation::new(self.library,
                capacity as usize * (request_hidden_format(self.nvfp4).1 + 48))?,
            weights: self,
            tokens: Vec::new(),
            layer: self.layer,
            capacity,
            graphs: LayerGraphs::new(self.library),
            other_graphs: LayerGraphs::new(self.library),
            full_request: true,
            ready: None,
            origin: None,
        })
    }
}
pub(crate) struct RouterOutput<'a> {
    pub layer: usize,
    pub rows: u32,
    pub input: Ds41rtDeviceBuffer,
    pub expert_input: Ds41rtDeviceBuffer,
    request_nvfp4: bool,
    pub mask: Ds41rtDeviceBuffer,
    pub scores: Ds41rtDeviceBuffer,
    pub ids: Ds41rtDeviceBuffer,
    pub routing: Ds41rtDeviceBuffer,
    pub tokens: &'a [u64],
    request_staging: ds41rt_ffi::Ds41rtHostBuffer,
    full_request: bool,
    origin: Option<QueryBinding>,
    _owner: PhantomData<&'a ()>,
}
impl RouterOutput<'_> {
    /// # Safety
    /// The completed router output stays live and unchanged through this copy.
    pub unsafe fn capture_route_ids(&self, library: &NativeLibrary, output: &mut Vec<[u32; 6]>) -> Result<()> {
        ensure!(self.rows > 0 && self.rows <= 4096 && self.ids.bytes == self.rows as usize * 24,
            "invalid route capture extent");
        output.resize(self.rows as usize, [0; 6]);
        // Arrays have contiguous u32 layout; every bit pattern is valid. The
        // vector is exclusively borrowed and fully initialized before copying.
        let bytes = unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), self.ids.bytes) };
        let offset = if self.full_request {
            self.rows as usize * request_hidden_format(self.request_nvfp4).1
        } else { 0 };
        ensure!(offset + bytes.len() <= self.request_staging.bytes, "route capture staging extent differs");
        // This execution already drained the graph's route-ID download.
        unsafe { std::ptr::copy_nonoverlapping(self.request_staging.ptr.cast::<u8>().add(offset), bytes.as_mut_ptr(), bytes.len()); }
        let _ = library;
        Ok(())
    }
    pub fn binding(&self) -> Result<QueryBinding> {
        self.origin
            .context("backbone router output has no block origin")
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    use ds41rt_loader::{read_official_v41_catalog, OFFICIAL_V41_MODEL_ID};

    #[test]
    fn expert_request_wire_format_tracks_nvfp4_checkpoint() {
        use ds41rt_transport::ExpertV2Dtype;
        assert_eq!(request_hidden_format(false), (ExpertV2Dtype::Fp8E4m3Ue8m0K32, 5280));
        assert_eq!(request_hidden_format(true), (ExpertV2Dtype::Bf16, 10240));
        for nvfp4 in [false, true] {
            for rows in [1usize, 16, 80, 4096] {
                let (_, width) = request_hidden_format(nvfp4);
                let hidden = rows * width;
                let ids = rows * 24;
                let arena = vec![0xa5u8; rows * (width + 48)];
                let (h, routes) = arena.split_at(hidden);
                let (i, w) = routes.split_at(ids);
                assert_eq!((h.len(), i.len(), w.len()), (rows * width, rows * 24, rows * 24));
            }
        }
    }
    fn bytes(library: &NativeLibrary, buffer: Ds41rtDeviceBuffer) -> Result<Vec<u8>> {
        let mut result = vec![0; buffer.bytes];
        library.copy_d2h(&mut result, buffer)?;
        Ok(result)
    }
    #[test]
    fn local_router_graphs_preserve_ids_without_remote_payload() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_ROUTER_REUSE_LIBRARY") else {
            eprintln!("skip local router GPU test: DS41RT_ROUTER_REUSE_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_ROUTER_REUSE_MODEL").context("DS41RT_ROUTER_REUSE_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, std::path::Path::new(&model))?;
        let weights = (0..2).map(|layer| BackboneRouterWeights::load(&library, &catalog, layer, 1 << 30, 16 << 20))
            .collect::<Result<Vec<_>>>()?;
        for rows in [1, 16, 80, 256] {
            let mut lane = weights[0].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
            let mut handles = std::collections::HashMap::new();
            for (cycle, local) in [false, true, false, true].into_iter().enumerate() {
                for layer in 0..2 {
                    lane.rebind(&weights[layer])?;
                    lane.set_local_mode(local)?;
                    let hidden: Vec<u8> = (0..rows as usize * 5120).flat_map(|i| {
                        let x = ((i + cycle * 3 + layer) % 17) as f32 / 32.;
                        ((x.to_bits() >> 16) as u16).to_ne_bytes()
                    }).collect();
                    let mask: Vec<u8> = (0..rows).map(|i| (i % 2) as u8).collect();
                    let [input, modality] = lane.inputs();
                    library.copy_h2d(input, &hidden)?;
                    library.copy_h2d(modality, &mask)?;
                    let mut fresh = weights[layer].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
                    let [fresh_input, fresh_mask] = fresh.inputs();
                    library.copy_h2d(fresh_input, &hidden)?;
                    library.copy_h2d(fresh_mask, &mask)?;
                    if local {
                        unsafe { std::ptr::write_bytes(lane.request_staging.buffer.ptr, 0xa5, lane.request_staging.buffer.bytes); }
                    }
                    let actual = unsafe { lane.execute_captured(rows)? };
                    let reference = unsafe { fresh.execute(rows)? };
                    if local {
                        let staging = unsafe { std::slice::from_raw_parts(actual.request_staging.ptr.cast::<u8>(), actual.request_staging.bytes) };
                        assert!(staging[rows as usize * 24..].iter().all(|&b| b == 0xa5));
                    }
                    for (a,b) in [(actual.ids,reference.ids),(actual.routing,reference.routing),
                                  (actual.expert_input,reference.expert_input)] {
                        assert_eq!(bytes(&library,a)?,bytes(&library,b)?);
                    }
                    let mut ids = vec![[u32::MAX;6]; rows as usize + 5];
                    unsafe { actual.capture_route_ids(&library, &mut ids)?; }
                    assert_eq!(ids.len(),rows as usize);
                    let captured: Vec<u8> = ids.iter().flatten().flat_map(|v| v.to_ne_bytes()).collect();
                    assert_eq!(captured,bytes(&library,reference.ids)?);
                    assert_eq!(unsafe { actual.download_request(&library) }.is_err(),local);
                    let handle = lane.graphs.get(layer,&weights[layer]).unwrap().0;
                    if let Some(previous) = handles.insert((local,layer),handle) { assert_eq!(previous,handle); }
                    if let Some(other) = handles.get(&(!local,layer)) { assert_ne!(*other,handle); }
                }
            }
        }
        eprintln!("PASS 32 local/remote router graph transitions with exact IDs, routes and FP8 input");
        Ok(())
    }

    #[test]
    fn real_router_rebinding_matches_fresh_owners() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_ROUTER_REUSE_LIBRARY") else {
            eprintln!("skip router reuse GPU test: DS41RT_ROUTER_REUSE_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_ROUTER_REUSE_MODEL")
            .context("DS41RT_ROUTER_REUSE_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog =
            read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, std::path::Path::new(&model))?;
        let weights = (0..40)
            .map(|layer| {
                BackboneRouterWeights::load(&library, &catalog, layer, 3_935_232, 1024 * 1024)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut comparisons = 0;
        for rows in [1u32, 6, 16, 80, 81, 4096] {
            let layers: Vec<usize> = if rows > 80 {
                vec![0, 20, 39]
            } else {
                (0..40).collect()
            };
            let mut lane = weights[0].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
            let pointers = lane.inputs().map(|b| b.ptr);
            let mut handles = std::collections::HashMap::new();
            for cycle in 0..2usize {
                for &layer in &layers {
                    lane.rebind(&weights[layer])?;
                    assert!(lane.output().is_err());
                    assert_eq!(lane.inputs().map(|b| b.ptr), pointers);
                    let hidden: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| {
                            let value =
                                (((i * 7 + cycle * 19 + layer * 3) % 31) as i32 - 15) as f32 / 128.;
                            ((value.to_bits() >> 16) as u16).to_ne_bytes()
                        })
                        .collect();
                    let mask: Vec<u8> = (0..rows as usize)
                        .map(|i| ((i + cycle + layer) % 2) as u8)
                        .collect();
                    let [input, modality] = lane.inputs();
                    library.copy_h2d(input, &hidden)?;
                    library.copy_h2d(modality, &mask)?;
                    let mut fresh =
                        weights[layer].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
                    let [fresh_input, fresh_mask] = fresh.inputs();
                    library.copy_d2d(fresh_input, input, input.bytes)?;
                    library.copy_d2d(fresh_mask, modality, modality.bytes)?;
                    // Poison both GPU inputs and the shared host arena before
                    // staging from an independent completed device input.
                    library.copy_h2d(input, &vec![0xff; input.bytes])?;
                    library.copy_h2d(modality, &vec![0xff; modality.bytes])?;
                    unsafe {
                        std::ptr::write_bytes(
                            lane.request_staging.buffer.ptr,
                            0xa5,
                            lane.request_staging.buffer.bytes,
                        );
                        lane.stage_inputs(fresh_input, &mask)?;
                    }
                    let actual = unsafe { lane.execute_captured(rows)? };
                    let reference = unsafe { fresh.execute(rows)? };
                    assert_eq!(actual.layer, layer);
                    for (a, b) in [
                        (actual.scores, reference.scores),
                        (actual.ids, reference.ids),
                        (actual.routing, reference.routing),
                        (actual.expert_input, reference.expert_input),
                    ] {
                        assert_eq!(bytes(&library, a)?, bytes(&library, b)?);
                    }
                    let (hidden, ids, routing) = unsafe { actual.download_request(&library)? };
                    assert_eq!(hidden, bytes(&library,
                        if reference.request_nvfp4 { reference.input } else { reference.expert_input })?);
                    assert_eq!(ids, bytes(&library, reference.ids)?);
                    assert_eq!(routing, bytes(&library, reference.routing)?);
                    for buffer in [actual.scores, actual.routing] {
                        assert!(bytes(&library, buffer)?
                            .chunks_exact(4)
                            .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite()));
                    }
                    let handle = lane.graphs.get(layer, &weights[layer]).unwrap().0;
                    if cycle == 0 {
                        handles.insert(layer, handle);
                    } else {
                        assert_eq!(handles[&layer], handle);
                    }
                    comparisons += 1;
                }
                eprintln!("PASS real router rows={rows} cycle={cycle} layers={} scores/ids/routing/encoded-input exact; stable inputs and cached handles",layers.len());
            }
            assert!(unsafe { lane.execute_captured(0) }.is_err());
            assert!(lane.output().is_err());
            lane.rebind(&weights[0])?;
            unsafe {
                lane.execute_captured(rows)?;
            }
            lane.clear_graph()?;
            assert!(lane.output().is_err());
        }
        assert_eq!(comparisons, 332);
        eprintln!("PASS 332 real-weight router comparisons and invalid-row recovery");
        Ok(())
    }
}
pub(crate) struct BackboneRouterWave<'w, 'a> {
    stream: LoadStream<'a>,
    kernel: V41Router<'a>,
    input_quantizer: V41ExpertInputQuantizer<'a>,
    weights: &'w BackboneRouterWeights<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    request_staging: HostAllocation<'a>,
    tokens: Vec<u64>,
    layer: usize,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, BackboneRouterWeights<'a>>,
    other_graphs: LayerGraphs<'w, 'a, BackboneRouterWeights<'a>>,
    full_request: bool,
    ready: Option<u32>,
    origin: Option<QueryBinding>,
}
impl<'w, 'a> BackboneRouterWave<'w, 'a> {
    pub fn rebind(&mut self, weights: &'w BackboneRouterWeights<'a>) -> Result<()> {
        self.invalidate();
        ensure!(
            std::ptr::eq(self.stream.library, weights.library),
            "router rebound library differs"
        );
        ensure!(self.weights.nvfp4 == weights.nvfp4,
            "router cannot rebind across expert wire formats");
        for name in BackboneRouterWeights::names(weights.layer)? {
            ensure!(
                weights.tensors.get(&name)?.device_id == self.b(0).device_id,
                "router rebound weight device differs"
            );
        }
        self.stream.require_complete()?;
        self.weights = weights;
        self.layer = weights.layer;
        Ok(())
    }
}
impl BackboneRouterWave<'_, '_> {
    pub fn set_local_mode(&mut self, local: bool) -> Result<()> {
        if self.full_request == !local { return Ok(()); }
        self.stream.require_complete()?;
        self.invalidate();
        std::mem::swap(&mut self.graphs, &mut self.other_graphs);
        self.full_request = !local;
        Ok(())
    }
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid backbone router capacity"
        );
        Ok(capacity as usize * (11825 + 5280))
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        [self.b(0), self.b(1)]
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn invalidate(&mut self) {
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "backbone router rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        let names = BackboneRouterWeights::names(self.layer)?;
        unsafe {
            self.kernel.launch(
                self.b(0),
                self.weights.tensors.get(&names[0])?,
                self.weights.tensors.get(&names[1])?,
                self.weights.tensors.get(&names[2])?,
                Some(self.b(1)),
                self.b(2),
                self.b(3),
                self.b(4),
                rows as usize,
                384,
                self.stream.raw,
            )?;
            // NVFP4 consumes BF16 directly; its FP8 buffer has no consumer.
            if !self.weights.nvfp4 {
                self.input_quantizer
                    .launch(self.b(0), self.b(5), rows, self.stream.raw)?;
            }
            let hidden_row_bytes = request_hidden_format(self.weights.nvfp4).1;
            let bytes = rows as usize * if self.full_request { hidden_row_bytes + 48 } else { 24 };
            if !self.full_request {
                ensure!(bytes <= self.request_staging.buffer.bytes, "local route staging too small");
                let ids = std::slice::from_raw_parts_mut(self.request_staging.buffer.ptr.cast::<u8>(), bytes);
                self.stream.library.copy_d2h_async(ids, self.b(3), self.stream.raw)?;
            } else if bytes <= self.request_staging.buffer.bytes {
                let staging = std::slice::from_raw_parts_mut(
                    self.request_staging.buffer.ptr.cast::<u8>(),
                    bytes,
                );
                let (hidden, routes) = staging.split_at_mut(rows as usize * hidden_row_bytes);
                let (ids, weights) = routes.split_at_mut(rows as usize * 24);
                let library = self.stream.library;
                library.copy_d2h_async(hidden,
                    self.b(if self.weights.nvfp4 { 0 } else { 5 }), self.stream.raw)?;
                library.copy_d2h_async(ids, self.b(3), self.stream.raw)?;
                library.copy_d2h_async(weights, self.b(4), self.stream.raw)?;
            }
            Ok(())
        }
    }
    /// # Safety
    /// Hidden input is finite and initialized; mask bytes are 0 or 1 in the same
    /// row order. Both inputs are exclusively owned until the call drains.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
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
            "backbone router graph already captured"
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
    pub unsafe fn replay(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self
            .graphs
            .get_shape(self.layer, self.weights, rows)
            .context("backbone router graph missing")?;
        ensure!(count == rows, "backbone router captured rows differ");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// Stage completed input on the router stream. The mask uses the beginning
    /// of the pinned request arena; H2D consumes it before graph D2H overwrites it.
    /// The caller must drain even if subsequent graph preparation fails.
    unsafe fn stage_inputs(&mut self, input: Ds41rtDeviceBuffer, mask: &[u8]) -> Result<()> {
        ensure!(
            !mask.is_empty()
                && mask.len() <= self.capacity as usize
                && input.bytes == mask.len() * 10240
                && input.device_id == self.b(0).device_id
                && mask.iter().all(|&v| v <= 1)
                && mask.len() <= self.request_staging.buffer.bytes,
            "invalid router staged input"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                mask.as_ptr(),
                self.request_staging.buffer.ptr.cast::<u8>(),
                mask.len(),
            );
        }
        let staged = (|| unsafe {
            self.stream
                .library
                .copy_d2d_async(self.b(0), input, input.bytes, self.stream.raw)?;
            self.stream.library.copy_host_buffer_h2d_async(
                self.b(1),
                self.request_staging.buffer,
                mask.len(),
                self.stream.raw,
            )
        })();
        if staged.is_err() {
            return staged.and(self.synchronize());
        }
        Ok(())
    }
    /// # Safety
    /// The completed block input stays immutable through the copy. This wave has
    /// exclusive storage; image mask matches those tokens (0=text, 1=image).
    pub unsafe fn execute_ffn(
        &mut self,
        input: &FfnInput<'_>,
        image_mask: &[u8],
    ) -> Result<RouterOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.b(0).device_id
                && image_mask.len() == input.tokens.len()
                && image_mask.iter().all(|&v| v <= 1),
            "backbone router block input differs"
        );
        let rows = input.tokens.len() as u32;
        let executed = (|| unsafe {
            self.stage_inputs(input.values, image_mask)?;
            self.execute_captured(rows).map(|_| ())
        })();
        if executed.is_err() {
            // A staged copy may precede a graph preparation failure. Always
            // drain before permitting the input or pinned arena to be reused.
            return executed.and(self.synchronize()).and_then(|_| self.output());
        }
        self.origin = Some(input.binding());
        self.tokens.extend_from_slice(input.tokens);
        self.output()
    }
    /// # Safety
    /// Same initialized finite hidden and binary-mask contract as execute.
    pub unsafe fn execute_captured(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
        self.invalidate();
        if self
            .graphs
            .get_shape(self.layer, self.weights, rows)
            .is_none()
        {
            unsafe {
                self.capture(rows)?;
            }
        }
        unsafe { self.replay(rows) }
    }
    /// # Safety
    /// Same input contract as execute_ffn; the future retains this wave and input
    /// until completion. Cancellation drains before their storage can be reused.
    pub async unsafe fn execute_ffn_cooperative(&mut self, input: &FfnInput<'_>, image_mask: &[u8]) -> Result<RouterOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.b(0).device_id
                && image_mask.len() == input.tokens.len()
                && image_mask.iter().all(|&v| v <= 1),
            "backbone router block input differs"
        );
        let rows = input.tokens.len() as u32;
        let cold = self.graphs.get_shape(self.layer, self.weights, rows).is_none();
        let launched = (|| unsafe {
            self.stage_inputs(input.values, image_mask)?;
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
        self.tokens.extend_from_slice(input.tokens);
        self.output()
    }
    pub fn output(&self) -> Result<RouterOutput<'_>> {
        let rows = self.ready.context("backbone router output unpublished")?;
        let b = |i: usize, n: usize| {
            let mut b = self.b(i);
            b.bytes = rows as usize * n;
            b
        };
        Ok(RouterOutput {
            layer: self.layer,
            rows,
            input: b(0, 10240),
            expert_input: b(5, 5280),
            request_nvfp4: self.weights.nvfp4,
            mask: b(1, 1),
            scores: b(2, 1536),
            ids: b(3, 24),
            routing: b(4, 24),
            tokens: &self.tokens,
            request_staging: self.request_staging.buffer,
            full_request: self.full_request,
            origin: self.origin,
            _owner: PhantomData,
        })
    }
    /// Clear only this layer; other layers retain their captured shape.
    pub fn enable_small_graph_shapes(&mut self) {
        self.graphs.enable_small_shapes();
        self.other_graphs.enable_small_shapes();
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.synchronize()?;
        unsafe {
            let current = self.graphs.remove(self.layer);
            let other = self.other_graphs.remove(self.layer);
            current.and(other)?;
        }
        Ok(())
    }
}
impl Drop for BackboneRouterWave<'_, '_> {
    fn drop(&mut self) {
        let drained = self.synchronize();
        let current = unsafe { self.graphs.clear() };
        let other = unsafe { self.other_graphs.clear() };
        if let Err(e) = drained.and(current).and(other) {
            tracing::error!(%e,"draining backbone router");
        }
    }
}

/// Scheduler metadata in exactly the block's token order.
pub(crate) struct ExpertRow {
    pub request_id: u64,
    pub position: u64,
    pub kind: ds41rt_transport::ExpertV2SourceKind,
}
/// An immutable host request owns its format-selected input and router result after D2H.
/// The private binding survives asynchronous transport and router-wave reuse.
pub(crate) struct BoundExpertRequest {
    request: ds41rt_transport::ExpertProtocolV2Request,
    binding: QueryBinding,
}
impl BoundExpertRequest {
    pub(crate) fn assign_paired(&mut self, assignment: &mut crate::v41_experts::paired::PairedAssignment,
        profile: &crate::v41_experts::paired::PairedProfile) -> Result<()> {
        let layer = self.binding.layer();
        assignment.encode(&mut self.request, profile.layer(layer)?, layer)?;
        Ok(())
    }

    pub fn request(&self) -> &ds41rt_transport::ExpertProtocolV2Request {
        &self.request
    }
    pub fn binding(&self) -> QueryBinding {
        self.binding
    }
}
impl RouterOutput<'_> {
    pub fn validate_request_rows(&self, rows: &[ExpertRow]) -> Result<QueryBinding> {
        let binding = self.binding()?;
        ensure!(
            binding.layer() == self.layer
                && rows.len() == self.rows as usize
                && self.tokens.len() == rows.len()
                && rows.iter().zip(self.tokens).all(|(r, &p)| r.position == p),
            "router request rows differ from block"
        );
        Ok(binding)
    }
    /// # Safety
    /// The completed output and its owning stream remain exclusively borrowed
    /// until all transfers drain, including after a partial enqueue failure.
    unsafe fn download_request(
        &self,
        library: &NativeLibrary,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        ensure!(self.full_request, "local router output has no remote request payload");
        let hidden_row_bytes = request_hidden_format(self.request_nvfp4).1;
        let bytes = self.rows as usize * (hidden_row_bytes + 48);
        if bytes > self.request_staging.bytes {
            let mut hidden = vec![0; self.rows as usize * hidden_row_bytes];
            let mut ids = vec![0; self.rows as usize * 24];
            let mut weights = vec![0; self.rows as usize * 24];
            library.copy_d2h(&mut hidden,
                if self.request_nvfp4 { self.input } else { self.expert_input })?;
            library.copy_d2h(&mut ids, self.ids)?;
            library.copy_d2h(&mut weights, self.routing)?;
            return Ok((hidden, ids, weights));
        }
        // Successful execution already drained the graph's D2H copies. This
        // borrow prevents another execution from reusing the pinned arena.
        let staging =
            unsafe { std::slice::from_raw_parts(self.request_staging.ptr.cast::<u8>(), bytes) };
        let (hidden, routes) = staging.split_at(self.rows as usize * hidden_row_bytes);
        let (ids, weights) = routes.split_at(self.rows as usize * 24);
        Ok((hidden.to_vec(), ids.to_vec(), weights.to_vec()))
    }

    /// # Safety
    /// Device views still hold this completed router execution with no external
    /// writes. Metadata identifies the actual requests represented by the block.
    /// Calls sharing this wave's pinned staging must not overlap.
    pub unsafe fn expert_request(
        &self,
        library: &NativeLibrary,
        placement: u64,
        rows: &[ExpertRow],
    ) -> Result<BoundExpertRequest> {
        use ds41rt_transport::{
            ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
        };
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let binding = self.validate_request_rows(rows)?;
        let request_id = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .ok()
            .context("native expert request IDs exhausted")?;
        let (hidden, ids, weights) = unsafe { self.download_request(library)? };
        let descriptors = rows
            .iter()
            .enumerate()
            .map(|(i, r)| ExpertProtocolV2RowDescriptor {
                row_id: i as u64,
                source_kind: r.kind,
                source_request_id: r.request_id,
                token_position: r.position,
                route_offset: i as u32 * 6,
                route_count: 6,
            })
            .collect();
        let routes = ids
            .chunks_exact(4)
            .zip(weights.chunks_exact(4))
            .enumerate()
            .map(|(i, (id, w))| ExpertProtocolV2RouteEntry {
                row_index: (i / 6) as u32,
                expert_id: u32::from_ne_bytes(id.try_into().unwrap()),
                gate_weight: f32::from_ne_bytes(w.try_into().unwrap()),
            })
            .collect();
        let mut request = ExpertProtocolV2Request::new(
            request_id,
            placement,
            self.layer as u32,
            5120,
            request_hidden_format(self.request_nvfp4).0,
            descriptors,
            routes,
            hidden,
        )?;
        request.header.flags |=
            ds41rt_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        // Prove the same complete-batch contract used by every Spark receiver.
        ds41rt_transport::v41_expert::V41BackboneRequest::validate_owned(&request, self.rows)?;
        Ok(BoundExpertRequest { request, binding })
    }
}
