//! Owned compressed EXL3 device residency. The native execution adapter binds
//! these buffers only after loading completes and must retain them for graphs.
use super::{ExpertLayer, ExpertLoadBudget, EXPERT_READ_LANES};
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::{OfficialV41Catalog, V41Exl3Layer, V41Exl3Residency};

const JOBS_PER_EXPERT: usize = 9;
const BANKS: usize = 2;
pub(crate) mod execution;
pub(crate) mod worker;

pub(crate) struct Exl3Weights<'a> {
    buffers: Vec<WeightBuffer>,
    _arena: DeviceAllocation<'a>,
    pub(crate) layout: V41Exl3Residency,
    pub(crate) budget: ExpertLoadBudget,
}

struct WeightBuffer { buffer: Ds41rtDeviceBuffer }

/// One explicitly sized allocation avoids independent CUDA allocation padding
/// for every projection/rotation table. All kernel pointers remain 256B aligned.
fn arena_layout(plan: &V41Exl3Residency) -> Result<(Vec<usize>, usize)> {
    let mut bytes = 0usize;
    let mut offsets = Vec::with_capacity(plan.buffers.len());
    for buffer in &plan.buffers {
        bytes = bytes.checked_add(255).context("EXL3 arena alignment overflow")? & !255;
        offsets.push(bytes);
        bytes = bytes.checked_add(buffer.bytes.max(16)).context("EXL3 arena size overflow")?;
    }
    let page = 2 * 1024 * 1024;
    bytes = bytes.checked_add(page-1).context("EXL3 arena page alignment overflow")? / page * page;
    Ok((offsets, bytes))
}

fn layout(catalog: &OfficialV41Catalog, layer: ExpertLayer) -> Result<V41Exl3Residency> {
    let (layer, world, rank) = match layer {
        ExpertLayer::Backbone { layer, rank } => (V41Exl3Layer::Backbone(layer), 4, rank),
        ExpertLayer::BackboneFull { layer } => (V41Exl3Layer::Backbone(layer), 1, 0),
        ExpertLayer::BackboneTp2 { layer, rank } => (V41Exl3Layer::Backbone(layer), 2, rank),
        ExpertLayer::Dspark { stage } => (V41Exl3Layer::Dspark(stage), 1, 0),
    };
    catalog
        .exl3()
        .context("EXL3 residency requires a routed EXL3 checkpoint")?
        .residency(layer, world, rank)
}

fn budget(catalog: &OfficialV41Catalog, plan: &V41Exl3Residency) -> Result<ExpertLoadBudget> {
    ensure!(
        plan.loads.len() == plan.experts * JOBS_PER_EXPERT,
        "EXL3 staging inventory mismatch"
    );
    let staging = plan
        .loads
        .chunks(JOBS_PER_EXPERT)
        .map(|jobs| jobs.iter().map(|j| j.bytes).sum::<usize>())
        .max()
        .unwrap_or(0);
    let init_bytes = plan
        .buffers
        .iter()
        .map(|b| b.initial_words.len() * 4)
        .max()
        .unwrap_or(0);
    // Batch 64 source rows per pread, matching the native FP4 loader policy.
    let scratch = plan
        .scratch_bytes(catalog)?
        .checked_mul(64)
        .context("EXL3 scratch overflow")?;
    Ok(ExpertLoadBudget {
        resident_bytes: arena_layout(plan)?.1,
        device_staging_bytes: 0,
        pinned_host_bytes: staging.max(init_bytes) * EXPERT_READ_LANES * BANKS,
        read_scratch_bytes: scratch * EXPERT_READ_LANES * BANKS,
    })
}

impl<'a> Exl3Weights<'a> {
    pub(crate) fn plan(
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<ExpertLoadBudget> {
        budget(catalog, &layout(catalog, layer)?)
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<Ds41rtDeviceBuffer> {
        let index = self
            .layout
            .buffers
            .iter()
            .position(|b| b.name == name)
            .with_context(|| format!("missing EXL3 device buffer {name}"))?;
        Ok(self.buffers[index].buffer)
    }

    pub(crate) fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<Self> {
        let layout = layout(catalog, layer)?;
        let budget = budget(catalog, &layout)?;
        ensure!(
            budget.peak_device_bytes()? <= available_device_bytes,
            "EXL3 layer needs {} device bytes, budget is {available_device_bytes}",
            budget.resident_bytes
        );
        let (offsets, bytes) = arena_layout(&layout)?;
        let arena = DeviceAllocation::new(library, bytes)?;
        let buffers: Vec<_> = layout.buffers.iter().zip(offsets).map(|(spec, offset)| {
            let mut buffer = arena.buffer;
            buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
            buffer.bytes = spec.bytes.max(16);
            WeightBuffer { buffer }
        }).collect();
        let per_host = budget.pinned_host_bytes / (EXPERT_READ_LANES * BANKS);
        let per_scratch = budget.read_scratch_bytes / (EXPERT_READ_LANES * BANKS);
        let mut hosts = (0..BANKS)
            .map(|_| {
                (0..EXPERT_READ_LANES)
                    .map(|_| HostAllocation::new(library, per_host))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scratch = (0..BANKS)
            .map(|_| {
                (0..EXPERT_READ_LANES)
                    .map(|_| vec![0; per_scratch])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // Declared after every buffer owner: drain queued copies first on errors.
        let streams = (0..BANKS)
            .map(|_| {
                Ok(LoadStream {
                    library,
                    raw: library.cuda_stream_create()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for (spec, allocation) in layout.buffers.iter().zip(&buffers) {
            if spec.initial_words.is_empty() {
                continue;
            }
            let host = &mut hosts[0][0];
            for (dst, word) in host
                .bytes_mut()
                .chunks_exact_mut(4)
                .zip(&spec.initial_words)
            {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            unsafe {
                library.copy_host_buffer_h2d_async(
                    allocation.buffer,
                    host.buffer,
                    spec.initial_words.len() * 4,
                    streams[0].raw,
                )?;
                library.cuda_stream_synchronize(streams[0].raw)?;
            }
        }
        for (group, first) in (0..layout.experts).step_by(EXPERT_READ_LANES).enumerate() {
            let bank = group % BANKS;
            let count = (layout.experts - first).min(EXPERT_READ_LANES);
            // Other bank uploads can continue while these CPU readers run.
            unsafe {
                library.cuda_stream_synchronize(streams[bank].raw)?;
            }
            std::thread::scope(|scope| -> Result<()> {
                let mut readers = Vec::with_capacity(count);
                for (lane, (host, scratch)) in hosts[bank]
                    .iter_mut()
                    .zip(&mut scratch[bank])
                    .take(count)
                    .enumerate()
                {
                    let plan = &layout;
                    let bytes = host.bytes_mut();
                    readers.push(scope.spawn(move || -> Result<()> {
                        let mut offset = 0;
                        for job in
                            (first + lane) * JOBS_PER_EXPERT..(first + lane + 1) * JOBS_PER_EXPERT
                        {
                            let size = plan.loads[job].bytes;
                            plan.read_into(
                                catalog,
                                job,
                                &mut bytes[offset..offset + size],
                                scratch,
                            )?;
                            offset += size;
                        }
                        Ok(())
                    }));
                }
                for reader in readers {
                    reader
                        .join()
                        .map_err(|_| anyhow::anyhow!("EXL3 reader panicked"))??;
                }
                Ok(())
            })?;
            for (lane, host) in hosts[bank].iter().take(count).enumerate() {
                let mut source_offset = 0;
                for job in &layout.loads
                    [(first + lane) * JOBS_PER_EXPERT..(first + lane + 1) * JOBS_PER_EXPERT]
                {
                    for &(buffer, offset) in &job.destinations {
                        ensure!(
                            offset
                                .checked_add(job.bytes)
                                .is_some_and(|end| end <= buffers[buffer].buffer.bytes),
                            "EXL3 upload exceeds destination"
                        );
                        unsafe {
                            let mut source = host.buffer;
                            source.ptr = source.ptr.cast::<u8>().add(source_offset).cast();
                            source.bytes = job.bytes;
                            let mut destination = buffers[buffer].buffer;
                            destination.ptr = destination.ptr.cast::<u8>().add(offset).cast();
                            destination.bytes = job.bytes;
                            library.copy_host_buffer_h2d_async(
                                destination,
                                source,
                                job.bytes,
                                streams[bank].raw,
                            )?;
                        }
                    }
                    source_offset += job.bytes;
                }
            }
        }
        for stream in &streams {
            unsafe {
                library.cuda_stream_synchronize(stream.raw)?;
            }
        }
        Ok(Self {
            buffers,
            _arena: arena,
            layout,
            budget,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_EXL3_SNAPSHOT and CUDA memory for one TP4 layer"]
    fn compressed_layer_uploads_match_staged_bytes() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_EXL3_SNAPSHOT")?),
        )?;
        let rank = std::env::var("DS41RT_EXL3_TEST_RANK").unwrap_or_else(|_| "2".into()).parse()?;
        let layer = ExpertLayer::Backbone { layer: 0, rank };
        let planned = Exl3Weights::plan(&catalog, layer)?;
        let (free, _) = library.cuda_memory_info()?;
        ensure!(
            free > planned.resident_bytes + 256 * 1024 * 1024,
            "insufficient free GPU memory for bounded residency test: free={free}, payload={}", planned.resident_bytes
        );
        let owned = Exl3Weights::load(&library, &catalog, layer, planned.resident_bytes)?;
        assert_eq!(owned.budget.device_staging_bytes, 0);
        let mut staging = vec![0; owned.layout.staging_bytes()];
        let mut scratch = vec![0; owned.layout.scratch_bytes(&catalog)? * 64];
        let mut checked = 0;
        // Sample first/middle/last experts, all projections, all duplicated destinations.
        for expert in [0, owned.layout.experts / 2, owned.layout.experts - 1] {
            for index in expert * JOBS_PER_EXPERT..(expert + 1) * JOBS_PER_EXPERT {
                let size = owned
                    .layout
                    .read_into(&catalog, index, &mut staging, &mut scratch)?;
                for &(buffer, offset) in &owned.layout.loads[index].destinations {
                    let mut view = owned.buffers[buffer].buffer;
                    view.ptr = unsafe { view.ptr.cast::<u8>().add(offset).cast() };
                    view.bytes = size;
                    let mut actual = vec![0; size];
                    library.copy_d2h(&mut actual, view)?;
                    ensure!(actual == staging[..size], "uploaded EXL3 bytes differ for {}", owned.layout.loads[index].tensor);
                    checked += 1;
                }
            }
        }
        for spec in &owned.layout.buffers {
            if spec.initial_words.is_empty() {
                continue;
            }
            let mut actual = vec![0; spec.bytes];
            library.copy_d2h(&mut actual, owned.buffer(&spec.name)?)?;
            let expected: Vec<_> = spec
                .initial_words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            assert_eq!(actual, expected);
        }
        println!(
            "EXL3 GPU residency: rank={rank}, width={}, bytes={}, sampled_destinations={checked}, metadata=exact",
            owned.layout.intermediate, owned.budget.resident_bytes
        );
        Ok(())
    }
}
