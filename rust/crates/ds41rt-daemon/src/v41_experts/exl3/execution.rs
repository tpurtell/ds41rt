//! Bound EXL3 launch tables. Allocations and name resolution happen at setup;
//! each launch only substitutes live inputs/row bounds and enqueues GPU work.
use super::Exl3Weights;
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Exl3Kernel, V41Exl3Routes};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ffi::c_void, path::Path, rc::Rc};

#[derive(Deserialize)]
struct Buffer {
    bytes: usize,
    dtype: String,
    allocation: String,
    zero_on_create: bool,
}
#[derive(Deserialize)]
struct Object {
    label: String,
    pointer_slots: Vec<String>,
    scalar_slots: Vec<String>,
}
#[derive(Deserialize)]
struct Asset {
    file: String,
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
struct RouteManifest {
    manifest: String,
    sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    schema: String,
    output_dtype: String,
    sparkinfer_revision: String,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    capacity: usize,
    top_k: usize,
    bits: Vec<usize>,
    swiglu_limit: f32,
    direct: bool,
    sms: usize,
    blocks_per_sm: usize,
    buffers: BTreeMap<String, Buffer>,
    objects: Vec<Object>,
    trellis_lut: Asset,
    route_preparation: Option<RouteManifest>,
}
impl Manifest {
    fn workspace_bytes(&self, format: Exl3InputFormat) -> Result<usize> {
        let mut bytes = self.trellis_lut.bytes.max(16);
        for (name, buffer) in &self.buffers {
            if name == &buffer.allocation {
                bytes = bytes
                    .checked_add(buffer.bytes.max(16))
                    .context("EXL3 workspace budget overflow")?;
            }
        }
        if format == Exl3InputFormat::Fp8K32 {
            bytes = bytes
                .checked_add(
                    self.capacity
                        .checked_mul(5120 * 2)
                        .context("EXL3 wire workspace overflow")?,
                )
                .context("EXL3 workspace budget overflow")?;
        }
        Ok(bytes)
    }
}

struct Table {
    pointers: Vec<*mut c_void>,
    scalars: Vec<i32>,
    live: usize,
    inputs: Vec<(usize, usize)>,
}
impl Table {
    fn build(
        object: &Object,
        pointers: &BTreeMap<String, Ds41rtDeviceBuffer>,
        values: &BTreeMap<String, i32>,
    ) -> Result<Self> {
        let mut inputs = Vec::new();
        let mut table = Vec::new();
        for (index, name) in object.pointer_slots.iter().enumerate() {
            let dynamic = match name.as_str() {
                "rotation_input_ptr" => Some(0),
                "raw_topk_ids" | "route_expert_ids_ptr" => Some(1),
                "topk_weights_ptr" => Some(2),
                _ => None,
            };
            if let Some(slot) = dynamic {
                inputs.push((index, slot));
                table.push(std::ptr::null_mut());
            } else {
                table.push(
                    pointers
                        .get(name)
                        .with_context(|| format!("missing EXL3 pointer {name}"))?
                        .ptr,
                );
            }
        }
        let scalars = object
            .scalar_slots
            .iter()
            .map(|name| {
                values
                    .get(name)
                    .copied()
                    .with_context(|| format!("missing EXL3 scalar {name}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let live = object
            .scalar_slots
            .iter()
            .position(|s| s == "active_m")
            .context("missing EXL3 live bound")?;
        Ok(Self {
            pointers: table,
            scalars,
            live,
            inputs,
        })
    }
    fn bind(&mut self, inputs: &[Ds41rtDeviceBuffer; 3], rows: usize) {
        for &(index, slot) in &self.inputs {
            self.pointers[index] = inputs[slot].ptr;
        }
        self.scalars[self.live] = rows as i32;
    }
}

struct LayerBinding {
    core: Table,
    sum: Table,
    expert_map: Ds41rtDeviceBuffer,
    output_slot: usize,
}

pub(crate) struct Exl3Execution<'a> {
    kernel: V41Exl3Kernel,
    routes: Option<V41Exl3Routes>,
    wire: Option<(ds41rt_ffi::V41Exl3Wire<'a>, DeviceAllocation<'a>)>,
    _storage: Vec<DeviceAllocation<'a>>,
    // Keep every prebound pointer alive through the last graph replay.
    _weights: Rc<Vec<Exl3Weights<'a>>>,
    layers: Vec<LayerBinding>,
    route_pointers: [*mut c_void; 7],
    route_bytes: [u64; 7],
    output: Ds41rtDeviceBuffer,
    output_element_bytes: usize,
    device: i32,
    capacity: usize,
    topk: usize,
    library: &'a NativeLibrary,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exl3InputFormat {
    Bf16,
    Fp8K32,
}

impl<'a> Exl3Execution<'a> {
    /// Device allocation payload per lane, including aliases and optional wire
    /// reconstruction. Weight storage and CUDA module/driver reserve are separate.
    pub(crate) fn plan(directory: &Path, format: Exl3InputFormat) -> Result<usize> {
        let meta: Manifest =
            serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        meta.workspace_bytes(format)
    }

    /// Trusted build artifacts only. Each owner is exclusive to one decode lane;
    /// callers drain work and destroy graphs before releasing it or its weights.
    pub(crate) unsafe fn new(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
    ) -> Result<Self> {
        Self::with_input_format(library, weights, directory, Exl3InputFormat::Bf16)
    }

    pub(crate) unsafe fn with_input_format(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        format: Exl3InputFormat,
    ) -> Result<Self> {
        let meta: Manifest =
            serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        let lock: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../../third_party/sparkinfer.lock.json"
        ))?;
        ensure!(
            meta.schema == "ds41rt.v41-exl3-aot.v1"
                && meta.sparkinfer_revision == lock["revision"].as_str().unwrap_or(""),
            "EXL3 export/source mismatch"
        );
        ensure!(
            !weights.is_empty(),
            "EXL3 execution requires resident layers"
        );
        let device = library.cuda_get_device()?;
        for weight in weights.iter() {
            ensure!(
                weight.layout.world == weights[0].layout.world
                    && weight.layout.rank == weights[0].layout.rank
                    && weight.buffer("global_to_combined")?.device_id == device,
                "EXL3 resident layers must share one device and TP rank"
            );
            ensure!(
                meta.hidden == 5120
                    && meta.intermediate == weight.layout.intermediate
                    && meta.experts == weight.layout.experts
                    && meta.bits == weight.layout.tiers
                    && meta.swiglu_limit == 10.0,
                "EXL3 export/residency geometry mismatch"
            );
            let expected_topk =
                if matches!(weight.layout.layer, ds41rt_loader::V41Exl3Layer::Dspark(_)) {
                    3
                } else {
                    6
                };
            ensure!(meta.top_k == expected_topk, "EXL3 expert top-k mismatch");
        }
        let kernel = V41Exl3Kernel::load(directory.join("libds41rt_exl3.so"))?;
        let info = kernel.info();
        ensure!(
            info.hidden == meta.hidden
                && info.intermediate == meta.intermediate
                && info.experts == meta.experts
                && info.capacity == meta.capacity
                && info.topk == meta.top_k
                && info.bits[..info.tier_count]
                    .iter()
                    .map(|b| *b as usize)
                    .eq(meta.bits.iter().copied()),
            "EXL3 binary/manifest mismatch"
        );
        let output_dtype = match info.output_element_bytes {
            2 => ("bf16", "torch.bfloat16"),
            4 => ("fp32", "torch.float32"),
            _ => anyhow::bail!("unsupported EXL3 output precision"),
        };
        let output_spec = meta.buffers.get("output").context("missing EXL3 output")?;
        ensure!(
            meta.output_dtype == output_dtype.0
                && output_spec.dtype == output_dtype.1
                && output_spec.bytes == meta.capacity * meta.hidden * info.output_element_bytes,
            "EXL3 output precision/size mismatch"
        );
        let mut storage = Vec::new();
        let mut pointers = BTreeMap::new();
        for (name, spec) in &meta.buffers {
            if name != &spec.allocation {
                continue;
            }
            let allocation = DeviceAllocation::new(library, spec.bytes.max(16))?;
            if spec.zero_on_create {
                library.copy_h2d(allocation.buffer, &vec![0; spec.bytes])?;
            }
            pointers.insert(name.clone(), allocation.buffer);
            storage.push(allocation);
        }
        for (name, spec) in &meta.buffers {
            let mut buffer = *pointers
                .get(&spec.allocation)
                .context("missing EXL3 workspace allocation")?;
            ensure!(
                spec.bytes <= buffer.bytes,
                "EXL3 workspace alias exceeds owner"
            );
            buffer.bytes = spec.bytes;
            pointers.insert(name.clone(), buffer);
        }
        let lut = std::fs::read(directory.join(&meta.trellis_lut.file))?;
        ensure!(
            lut.len() == meta.trellis_lut.bytes
                && format!("{:x}", Sha256::digest(&lut)) == meta.trellis_lut.sha256,
            "EXL3 LUT artifact mismatch"
        );
        let lut_buffer = DeviceAllocation::new(library, lut.len())?;
        library.copy_h2d(lut_buffer.buffer, &lut)?;
        pointers.insert("trellis_lut_ptr".into(), lut_buffer.buffer);
        storage.push(lut_buffer);
        let mut layers = Vec::with_capacity(weights.len());
        for weight in weights.iter() {
            let mut pointers = pointers.clone();
            for (target, source) in [
                ("descriptor_map_ptr", "descriptor_map"),
                ("global_to_combined_ptr", "global_to_combined"),
                ("expert_map_ptr", "global_to_combined"),
                ("intermediate_rotations_ptr", "intermediate_rotations"),
                ("gate_suh_ptr", "gate_suh"),
                ("up_suh_ptr", "up_suh"),
                ("svh_ptr", "down_svh"),
            ] {
                pointers.insert(target.into(), weight.buffer(source)?);
            }
            for (target, source) in [("fc2_ptr", "fc2"), ("output_ptr", "output")] {
                pointers.insert(
                    target.into(),
                    *pointers
                        .get(source)
                        .context("missing EXL3 output workspace")?,
                );
            }
            let mut values = BTreeMap::from([
                ("active_m".into(), 1),
                (
                    "grid_x".into(),
                    i32::try_from(meta.sms * meta.blocks_per_sm)?,
                ),
                ("route_num_experts".into(), meta.experts as i32),
                (
                    "weight_num_experts".into(),
                    (meta.experts * meta.bits.len()) as i32,
                ),
            ]);
            for (tier, counts) in weight.layout.projection_counts.iter().enumerate() {
                for (field, source) in [
                    ("w13", format!("tier{tier}_w13")),
                    ("w2", format!("tier{tier}_w2")),
                    ("w13_scales", "dummy_scales".into()),
                    ("w2_scales", "dummy_scales".into()),
                    ("w13_global", "unit_scales".into()),
                    ("w2_global", "unit_scales".into()),
                ] {
                    pointers.insert(format!("t{tier}_{field}_ptr"), weight.buffer(&source)?);
                }
                for (name, value) in [
                    ("num_experts", meta.experts),
                    ("fc2_experts", counts[2]),
                    ("gate_experts", counts[0]),
                    ("up_experts", counts[1]),
                ] {
                    values.insert(format!("tier{tier}_{name}"), value as i32);
                }
            }
            ensure!(
                meta.objects.len() == 2
                    && meta.objects[0].label == "v41_exl3_core"
                    && meta.objects[1].label == "v41_exl3_sum",
                "EXL3 object ordering mismatch"
            );
            let core = Table::build(&meta.objects[0], &pointers, &values)?;
            let sum = Table::build(&meta.objects[1], &pointers, &values)?;
            ensure!(
                core.pointers.len() == info.core_pointers
                    && core.scalars.len() == info.core_scalars
                    && sum.pointers.len() == info.sum_pointers
                    && sum.scalars.len() == info.sum_scalars,
                "EXL3 native argument lengths disagree"
            );
            layers.push(LayerBinding {
                core,
                sum,
                expert_map: weight.buffer("global_to_combined")?,
                output_slot: meta.objects[1]
                    .pointer_slots
                    .iter()
                    .position(|name| name == "output_ptr")
                    .context("missing EXL3 output pointer slot")?,
            });
        }
        let mut route_pointers = [std::ptr::null_mut(); 7];
        let mut route_bytes = [0; 7];
        let routes = if meta.direct {
            None
        } else {
            let route = meta
                .route_preparation
                .context("missing EXL3 packed route export")?;
            let path = directory.join(route.manifest);
            let bytes = std::fs::read(&path)?;
            ensure!(
                format!("{:x}", Sha256::digest(&bytes)) == route.sha256,
                "EXL3 route export hash mismatch"
            );
            for (index, name) in [
                "global_to_combined_ptr",
                "packed_route_indices",
                "block_expert_ids",
                "packed_route_count",
                "expert_offsets",
                "expert_counts",
            ]
            .iter()
            .enumerate()
            {
                let buffer = if index == 0 {
                    &layers[0].expert_map
                } else {
                    pointers
                        .get(*name)
                        .context("missing EXL3 packed route buffer")?
                };
                route_pointers[index + 1] = buffer.ptr;
                route_bytes[index + 1] = buffer.bytes as u64;
            }
            Some(V41Exl3Routes::load(
                path.parent().unwrap().join("libv41_exl3_routes.so"),
            )?)
        };
        let wire = if format == Exl3InputFormat::Fp8K32 {
            // The router's 5120-value wire row is replicated across TP ranks;
            // only intermediate expert weights are sliced. Local RTX TP1/TP2
            // therefore use the same decoder as Spark TP4.
            Some((
                library.v41_exl3_wire()?,
                DeviceAllocation::new(library, meta.capacity * 5120 * 2)?,
            ))
        } else {
            None
        };
        Ok(Self {
            kernel,
            routes,
            wire,
            _storage: storage,
            _weights: weights,
            layers,
            route_pointers,
            route_bytes,
            output_element_bytes: info.output_element_bytes,
            output: *pointers.get("output").context("missing EXL3 output")?,
            device,
            capacity: meta.capacity,
            topk: meta.top_k,
            library,
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
    pub(crate) fn output_element_bytes(&self) -> usize {
        self.output_element_bytes
    }

    /// Allocated workspace payload per lane, independent of resident layer count.
    pub(crate) fn workspace_bytes(&self) -> usize {
        self._storage.iter().map(|b| b.buffer.bytes).sum::<usize>()
            + self.wire.as_ref().map_or(0, |(_, b)| b.buffer.bytes)
    }

    /// # Safety
    /// Same stream, graph and buffer contract as `launch_layer`.
    pub(crate) unsafe fn launch(
        &mut self,
        inputs: [Ds41rtDeviceBuffer; 3],
        rows: usize,
        stream: *mut c_void,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.launch_layer(0, inputs, rows, stream)
    }

    /// # Safety
    /// Inputs are BF16[rows,5120] or FP8 wire[rows,5280] as selected at setup,
    /// int32[rows,topk], FP32[rows,topk], contiguous
    /// on this owner device and live through completion. No overlapping input /
    /// workspace storage or concurrent use of this lane, including graph replay.
    pub(crate) unsafe fn launch_layer(
        &mut self,
        layer: usize,
        inputs: [Ds41rtDeviceBuffer; 3],
        rows: usize,
        stream: *mut c_void,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.launch_layer_into(layer, inputs, rows, stream, self.output)
    }

    /// # Safety
    /// Same contract as `launch_layer`; output is an exclusive, aligned GPU or
    /// mapped-host allocation on this device, live through execution/graph replay.
    pub(crate) unsafe fn launch_layer_into(
        &mut self,
        layer: usize,
        mut inputs: [Ds41rtDeviceBuffer; 3],
        rows: usize,
        stream: *mut c_void,
        mut output: Ds41rtDeviceBuffer,
    ) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "EXL3 live rows exceed capacity"
        );
        ensure!(
            self.library.cuda_get_device()? == self.device,
            "EXL3 execution on wrong device"
        );
        for (buffer, bytes) in inputs.iter().zip([
            rows * if self.wire.is_some() { 5280 } else { 5120 * 2 },
            rows * self.topk * 4,
            rows * self.topk * 4,
        ]) {
            ensure!(
                !buffer.ptr.is_null()
                    && buffer.ptr as usize % 16 == 0
                    && buffer.bytes >= bytes
                    && buffer.device_id == self.device,
                "EXL3 input buffer contract mismatch"
            );
        }
        ensure!(
            !output.ptr.is_null()
                && output.ptr as usize % 16 == 0
                && output.device_id == self.device
                && output.bytes >= rows * 5120 * self.output_element_bytes,
            "EXL3 output buffer contract mismatch"
        );
        let binding = self
            .layers
            .get_mut(layer)
            .context("EXL3 layer is not bound")?;
        if let Some((wire, decoded)) = &self.wire {
            wire.decode(inputs[0], decoded.buffer, rows, stream)?;
            inputs[0] = decoded.buffer;
        }
        if let Some(routes) = &self.routes {
            self.route_pointers[0] = inputs[1].ptr;
            self.route_pointers[1] = binding.expert_map.ptr;
            self.route_bytes[1] = binding.expert_map.bytes as u64;
            // Router views may expose only live IDs; metadata scratch retains
            // its compiled capacity while all input reads are live-row bounded.
            self.route_bytes[0] = inputs[1].bytes as u64;
            routes.launch(&self.route_pointers, &self.route_bytes, rows as i32, stream)?;
        }
        binding.core.bind(&inputs, rows);
        binding.sum.bind(&inputs, rows);
        binding.sum.pointers[binding.output_slot] = output.ptr;
        self.kernel
            .launch_core(&binding.core.pointers, &binding.core.scalars, stream)?;
        self.kernel
            .launch_sum(&binding.sum.pointers, &binding.sum.scalars, stream)?;
        output.bytes = rows * 5120 * self.output_element_bytes;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{v41_experts::ExpertLayer, v41_memory::LoadStream};

    struct Fixture<'a> {
        inputs: Vec<DeviceAllocation<'a>>,
        original_input: Vec<u8>,
        expected: Vec<u8>,
    }
    impl Fixture<'_> {
        fn inputs(&self) -> [Ds41rtDeviceBuffer; 3] {
            std::array::from_fn(|i| self.inputs[i].buffer)
        }
        fn verify(&self, library: &NativeLibrary, output: Ds41rtDeviceBuffer) -> Result<()> {
            let mut actual = vec![0; output.bytes];
            library.copy_d2h(&mut actual, output)?;
            ensure!(
                actual == self.expected[..actual.len()],
                "EXL3 layer output differs from B12x"
            );
            Ok(())
        }
    }
    struct Graph<'a> {
        library: &'a NativeLibrary,
        exec: *mut c_void,
        stream: *mut c_void,
    }
    impl Drop for Graph<'_> {
        fn drop(&mut self) {
            unsafe {
                let _ = self.library.cuda_stream_synchronize(self.stream);
                let _ = self.library.cuda_graph_exec_destroy(self.exec);
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA, DS41RT_NATIVE_LIB, DS41RT_EXL3_SNAPSHOT, DS41RT_EXL3_AOT and DS41RT_EXL3_FIXTURE; optional DS41RT_EXL3_SECOND_FIXTURE exercises layer 1"]
    fn native_resident_layer_matches_b12x_fixture() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let snapshot = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_SNAPSHOT")?);
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            &snapshot,
        )?;
        let aot = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_AOT")?);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(aot.join("v41_exl3.json"))?)?;
        let mut paths = vec![std::path::PathBuf::from(std::env::var(
            "DS41RT_EXL3_FIXTURE",
        )?)];
        if let Some(second) = std::env::var_os("DS41RT_EXL3_SECOND_FIXTURE") {
            paths.push(second.into());
        }
        let mut fixtures = Vec::new();
        let mut resident = Vec::new();
        let mut input_format = None;
        for (layer, path) in paths.iter().enumerate() {
            let info: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path.join("fixture.json"))?)?;
            for key in ["direct", "tile", "output_dtype"] {
                ensure!(
                    !info[key].is_null() && info[key] == manifest[key],
                    "EXL3 fixture policy mismatch: {key}"
                );
            }
            ensure!(
                info["slice_start"] == 1280
                    && info["width"] == 512
                    && info["topk"] == 6
                    && info["capacity"] == 16
                    && info["layer"] == format!("layers.{layer}")
                    && info["snapshot_revision"].as_str()
                        == snapshot.file_name().and_then(|s| s.to_str()),
                "unexpected EXL3 fixture geometry or snapshot"
            );
            let format = match info["input_format"].as_str() {
                Some("bf16") => Exl3InputFormat::Bf16,
                Some("fp8_k32") => Exl3InputFormat::Fp8K32,
                _ => anyhow::bail!("invalid fixture input format"),
            };
            ensure!(
                input_format.is_none_or(|old| old == format),
                "fixture input format mismatch"
            );
            input_format = Some(format);
            let layer = ExpertLayer::Backbone { layer, rank: 2 };
            let budget = Exl3Weights::plan(&catalog, layer)?;
            let (free, _) = library.cuda_memory_info()?;
            ensure!(
                free > budget.resident_bytes + 256 * 1024 * 1024,
                "insufficient GPU headroom"
            );
            resident.push(Exl3Weights::load(
                &library,
                &catalog,
                layer,
                budget.resident_bytes,
            )?);
            let mut fixture = Fixture {
                inputs: Vec::new(),
                original_input: Vec::new(),
                expected: Vec::new(),
            };
            for name in ["input", "ids", "weights", "expected"] {
                let bytes = std::fs::read(path.join(format!("{name}.bin")))?;
                ensure!(
                    Some(bytes.len() as u64) == info["artifacts"][name]["bytes"].as_u64()
                        && Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
                            == info["artifacts"][name]["sha256"].as_str(),
                    "corrupt EXL3 fixture {name}"
                );
                if name == "expected" {
                    fixture.expected = bytes;
                } else {
                    let allocation = DeviceAllocation::new(&library, bytes.len())?;
                    library.copy_h2d(allocation.buffer, &bytes)?;
                    fixture.inputs.push(allocation);
                    if name == "input" {
                        fixture.original_input = bytes;
                    }
                }
            }
            fixtures.push(fixture);
        }
        if fixtures.len() == 2 {
            ensure!(
                fixtures[0].expected != fixtures[1].expected,
                "layer fixtures must differ"
            );
        }
        let weights = Rc::new(resident);
        let mut execution = unsafe {
            Exl3Execution::with_input_format(
                &library,
                weights.clone(),
                &aot,
                input_format.unwrap(),
            )?
        };
        let workspace_bytes = execution.workspace_bytes();
        ensure!(
            workspace_bytes == Exl3Execution::plan(&aot, input_format.unwrap())?,
            "EXL3 workspace plan disagrees with allocations"
        );
        let mut second_lane = unsafe {
            Exl3Execution::with_input_format(&library, weights, &aot, input_format.unwrap())?
        };
        ensure!(
            second_lane.workspace_bytes() == workspace_bytes
                && second_lane.output.ptr != execution.output.ptr,
            "lanes must own separate equal-sized workspace"
        );
        let stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        let other_stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        // Different layer orders on two lanes; enqueue both before either host wait.
        for rows in [16, 3, 1, 16] {
            for layer in 0..fixtures.len() {
                let other = fixtures.len() - 1 - layer;
                let output = unsafe {
                    execution.launch_layer(layer, fixtures[layer].inputs(), rows, stream.raw)?
                };
                let other_output = unsafe {
                    second_lane.launch_layer(
                        other,
                        fixtures[other].inputs(),
                        rows,
                        other_stream.raw,
                    )?
                };
                unsafe {
                    library.cuda_stream_synchronize(stream.raw)?;
                    library.cuda_stream_synchronize(other_stream.raw)?;
                }
                fixtures[layer].verify(&library, output)?;
                fixtures[other].verify(&library, other_output)?;
            }
        }
        ensure!(
            unsafe { execution.launch_layer(fixtures.len(), fixtures[0].inputs(), 1, stream.raw) }
                .is_err(),
            "invalid layer must reject before launch"
        );
        let mut graphs = Vec::new();
        for (layer, fixture) in fixtures.iter().enumerate() {
            unsafe {
                library.cuda_graph_begin_capture(stream.raw)?;
            }
            let captured =
                unsafe { execution.launch_layer(layer, fixture.inputs(), 3, stream.raw) };
            let graph = Graph {
                library: &library,
                exec: unsafe { library.cuda_graph_end_capture(stream.raw)? },
                stream: stream.raw,
            };
            graphs.push((graph, captured?));
        }
        // Replay graphs after another layer has used the shared workspace.
        for layer in (0..fixtures.len()).chain((0..fixtures.len()).rev()) {
            let fixture = &fixtures[layer];
            let (graph, output) = &graphs[layer];
            library.copy_h2d(
                fixture.inputs[0].buffer,
                &vec![0; fixture.original_input.len()],
            )?;
            library.copy_h2d(*output, &vec![0xff; output.bytes])?;
            unsafe {
                library.cuda_graph_launch(graph.exec, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            let mut actual = vec![0; output.bytes];
            library.copy_d2h(&mut actual, *output)?;
            ensure!(
                actual.iter().all(|b| *b == 0),
                "EXL3 graph did not consume changed zero input"
            );
            library.copy_h2d(fixture.inputs[0].buffer, &fixture.original_input)?;
            library.copy_h2d(*output, &vec![0xff; output.bytes])?;
            unsafe {
                library.cuda_graph_launch(graph.exec, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            fixture.verify(&library, *output)?;
        }
        println!("Rust EXL3: {} layers x 384 resident experts, TP4 rank 2; two independent lanes, {} workspace bytes each; alternating layer rows=16/3/1/16 and changed/restored-input graph replay bitwise equal to six-expert B12x references", fixtures.len(), workspace_bytes);
        Ok(())
    }
}
