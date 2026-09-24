//! Owned native gathered-row upload and BF16 engram embedding production.
pub(crate) mod layer;
pub(crate) mod placement;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::{
    EngramGatherPoll, EngramGatherTicket, EngramGatherView, EngramPipeline, EngramWave,
};

pub(crate) struct EngramDeviceRows<'a> {
    stream: LoadStream<'a>,
    staging: HostAllocation<'a>,
    weights: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    embeddings: DeviceAllocation<'a>,
    text_mask: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    capacity: usize,
    ready: Option<(usize, usize)>,
}
/// Borrowed buffers: never free or retain across owner reuse/drop.
pub(crate) struct EngramDeviceView {
    pub embeddings: Ds41rtDeviceBuffer,
    pub text_mask: Ds41rtDeviceBuffer,
    pub rows: usize,
    pub layer_index: usize,
}
pub(crate) enum EngramUploadPoll {
    Pending,
    Cancelled,
    Ready(EngramDeviceView),
}
impl<'a> EngramDeviceRows<'a> {
    pub(crate) fn library(&self) -> &'a ds41rt_ffi::NativeLibrary { self.stream.library }
    /// Validate current request generations before consuming an early gather.
    pub fn poll_wave(
        &mut self,
        pipeline: &EngramPipeline,
        wave: &mut EngramWave,
        histories: &[&ds41rt_core::EngramHistory],
        layer: usize,
    ) -> Result<EngramUploadPoll> {
        self.ready = None;
        Ok(match pipeline.poll(wave, histories, layer)? {
            EngramGatherPoll::Pending => {
                tracing::debug!(target: "ds41rt::timing", layer, upload_owner=self as *const Self as usize, "engram IO pending");
                EngramUploadPoll::Pending
            }
            EngramGatherPoll::Cancelled => EngramUploadPoll::Cancelled,
            EngramGatherPoll::Ready(lease) => {
                let gathered = lease.view()?;
                if let Some(timing) = lease.timing() {
                    tracing::debug!(target: "ds41rt::timing", layer, rows=gathered.rows,
                        upload_owner=self as *const Self as usize,
                        queue_us=timing.queued.as_micros() as u64,
                        gather_us=timing.gather.as_micros() as u64,
                        ready_age_us=timing.completed.elapsed().as_micros() as u64,
                        minor_faults=timing.minor_faults, major_faults=timing.major_faults,
                        input_blocks=timing.input_blocks, "engram IO delivery");
                }
                EngramUploadPoll::Ready(self.upload(&gathered)?)
            }
        })
    }
    /// Poll on the CUDA-owning thread and recycle ready staging after upload.
    /// The scheduler must cancel tickets whose request history has been invalidated.
    pub fn poll_upload(&mut self, ticket: &mut EngramGatherTicket) -> Result<EngramUploadPoll> {
        self.ready = None;
        Ok(match ticket.poll()? {
            EngramGatherPoll::Pending => EngramUploadPoll::Pending,
            EngramGatherPoll::Cancelled => EngramUploadPoll::Cancelled,
            EngramGatherPoll::Ready(lease) => {
                let output = self.upload(&lease.view()?)?;
                // upload synchronizes before the lease returns its storage to the pool.
                EngramUploadPoll::Ready(output)
            }
        })
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram device capacity"
        );
        capacity
            .checked_mul(24 * (256 + 8 + 512) + 1)
            .context("engram device budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        capacity: usize,
        available_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= available_bytes,
            "engram rows exceed device budget"
        );
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            staging: HostAllocation::new(library, capacity * (24 * (256 + 8) + 1))?,
            weights: DeviceAllocation::new(library, capacity * 24 * 256)?,
            scales: DeviceAllocation::new(library, capacity * 24 * 8)?,
            embeddings: DeviceAllocation::new(library, capacity * 24 * 512)?,
            text_mask: DeviceAllocation::new(library, capacity)?,
            library,
            capacity,
            ready: None,
        })
    }
    fn gathered_buffers(&self, gathered: &EngramGatherView<'_>) -> (Ds41rtDeviceBuffer, Ds41rtDeviceBuffer) {
        let mut scales = self.scales.buffer;
        if gathered.encoding == ds41rt_loader::EngramEncoding::Nvfp4 {
            // Compact FP4 rows leave enough room for their scales in the existing
            // weight allocation, so FP8 and FP4 need identical GPU budgets.
            scales = self.weights.buffer;
            scales.ptr = unsafe { scales.ptr.cast::<u8>().add(gathered.weights.len()).cast() };
            scales.bytes = gathered.scales.len();
        }
        let mut weights = self.weights.buffer;
        weights.bytes = gathered.weights.len();
        (weights, scales)
    }
    unsafe fn dequantize(&self, gathered: &EngramGatherView<'_>) -> Result<()> {
        let (weights, scales) = self.gathered_buffers(gathered);
        let rows = i32::try_from(gathered.rows * 24)?;
        match gathered.encoding {
            ds41rt_loader::EngramEncoding::Fp8 => unsafe {
                self.library.cuda_engram_dequant_bf16_async(weights, scales, self.embeddings.buffer, rows, self.stream.raw)
            },
            ds41rt_loader::EngramEncoding::Nvfp4 => unsafe {
                self.library.cuda_engram_nvfp4_dequant_bf16_async(weights, scales, gathered.global_scale,
                    self.embeddings.buffer, rows, self.stream.raw)
            },
        }
    }
    /// Consume a completed I/O-worker gather; this method never reads mapped tables.
    /// Output is contiguous BF16 [token,24,256] plus a byte-per-token text mask.
    pub fn upload(&mut self, gathered: &EngramGatherView<'_>) -> Result<EngramDeviceView> {
        self.ready = None;
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity && gathered.layer_index < 2,
            "invalid gathered engram dimensions"
        );
        let hash_rows = gathered.rows * 24;
        ensure!(
            gathered.weights.len() == hash_rows * gathered.encoding.weight_bytes()
                && gathered.scales.len() == hash_rows * gathered.encoding.scale_bytes()
                && gathered.text_mask.len() == gathered.rows
                && gathered.text_mask.iter().all(|value| *value <= 1),
            "invalid gathered engram storage"
        );
        self.synchronize()?;
        let (weights, scales) = self.gathered_buffers(gathered);
        self.library.copy_h2d(weights, gathered.weights)?;
        self.library.copy_h2d(scales, gathered.scales)?;
        self.library
            .copy_h2d(self.text_mask.buffer, gathered.text_mask)?;
        unsafe {
            self.dequantize(gathered)?;
        }
        self.synchronize()?;
        self.ready = Some((gathered.rows, gathered.layer_index));
        self.view()
    }
    /// Pinned staging and this upload owner remain exclusive through completion;
    /// cancellation drains before either can be reused. No request owner is borrowed.
    pub async fn upload_cooperative(&mut self, gathered: &EngramGatherView<'_>) -> Result<EngramDeviceView> {
        self.ready = None;
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity && gathered.layer_index < 2,
            "invalid gathered engram dimensions"
        );
        let hash_rows = gathered.rows * 24;
        ensure!(
            gathered.weights.len() == hash_rows * gathered.encoding.weight_bytes()
                && gathered.scales.len() == hash_rows * gathered.encoding.scale_bytes()
                && gathered.text_mask.len() == gathered.rows
                && gathered.text_mask.iter().all(|value| *value <= 1),
            "invalid gathered engram storage"
        );
        let mut offset = 0;
        for bytes in [gathered.weights, gathered.scales, gathered.text_mask] {
            self.staging.bytes_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        }
        let launched = (|| -> Result<()> {
            let mut offset = 0;
            let (weights, scales) = self.gathered_buffers(gathered);
            for (destination, bytes) in [(weights, gathered.weights.len()),
                (scales, gathered.scales.len()), (self.text_mask.buffer, gathered.text_mask.len())] {
                let mut host = self.staging.buffer;
                host.ptr = unsafe { host.ptr.cast::<u8>().add(offset).cast() }; host.bytes = bytes;
                unsafe { self.library.copy_host_buffer_h2d_async(destination, host, bytes, self.stream.raw)?; }
                offset += bytes;
            }
            unsafe { self.dequantize(gathered) }
        })();
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        self.stream.wait().await?;
        self.ready = Some((gathered.rows, gathered.layer_index));
        self.view()
    }
    #[cfg(test)]
    pub async fn check_cooperative(&mut self, gathered: &EngramGatherView<'_>) -> Result<()> {
        use std::{future::Future, task::Poll};
        let read = |lib: &NativeLibrary, view: EngramDeviceView| -> Result<Vec<Vec<u8>>> {
            [view.embeddings, view.text_mask].into_iter().map(|buffer| {
                let mut bytes = vec![0; buffer.bytes]; lib.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
            }).collect()
        };
        let lib = self.library;
        let expected = read(lib, self.upload(gathered)?)?;
        let cancelled = {
            let mut work = std::pin::pin!(self.upload_cooperative(gathered));
            std::future::poll_fn(|cx| Poll::Ready(work.as_mut().poll(cx).is_pending())).await
        };
        if cancelled { assert!(self.view().is_err()); }
        for _ in 0..2 { assert_eq!(read(lib, self.upload_cooperative(gathered).await?)?, expected); }
        eprintln!("PASS queued Engram upload {}: exact dequant/mask parity, pending cancellation={cancelled}, reuse", gathered.layer_index);
        Ok(())
    }
    pub fn view(&self) -> Result<EngramDeviceView> {
        let (rows, layer_index) = self.ready.context("engram device rows are not complete")?;
        let mut embeddings = self.embeddings.buffer;
        embeddings.bytes = rows * 24 * 512;
        let mut text_mask = self.text_mask.buffer;
        text_mask.bytes = rows;
        Ok(EngramDeviceView {
            embeddings,
            text_mask,
            rows,
            layer_index,
        })
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
impl Drop for EngramDeviceRows<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native engram row production");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA native library and real packed PLE fixture with CPU reference"]
    fn nvfp4_ple_upload_matches_cpu_reference() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let root = std::path::PathBuf::from(std::env::var("DS41RT_PLE_FIXTURE")?);
        let fixtures: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for device in 0..2 {
            library.cuda_set_device(device)?;
            let mut owner = EngramDeviceRows::new(&library, 16, EngramDeviceRows::device_bytes(16)?)?;
            let pointers = (owner.weights.buffer.ptr, owner.scales.buffer.ptr, owner.embeddings.buffer.ptr);
            for _ in 0..2 {
                for (index, fixture) in fixtures.as_array().context("fixture array")?.iter().enumerate() {
                    let layer = fixture["layer"].as_u64().context("layer")?;
                    let weights = std::fs::read(root.join(format!("layer{layer}-weights.bin")))?;
                    let scales = std::fs::read(root.join(format!("layer{layer}-scales.bin")))?;
                    let expected = std::fs::read(root.join(format!("layer{layer}-expected.bin")))?;
                    let mask: Vec<u8> = fixture["text_mask"].as_array().context("mask")?.iter()
                        .map(|v| v.as_u64().unwrap() as u8).collect();
                    // Reuse the owner at smaller and larger live sizes; FP4 scale
                    // placement depends on live rows, not allocation capacity.
                    for rows in [1, mask.len(), 3, mask.len()] {
                        let gathered = EngramGatherView {
                            weights: &weights[..rows*24*128], scales: &scales[..rows*24*16],
                            text_mask: &mask[..rows], rows, layer_index: index,
                            encoding: ds41rt_loader::EngramEncoding::Nvfp4,
                            global_scale: fixture["global_scale"].as_f64().context("global scale")? as f32,
                        };
                        for cooperative in [false, true] {
                            let view = if cooperative { runtime.block_on(owner.upload_cooperative(&gathered))? }
                                else { owner.upload(&gathered)? };
                            let mut actual = vec![0; view.embeddings.bytes];
                            library.copy_d2h(&mut actual, view.embeddings)?;
                            assert_eq!(actual, expected[..rows*24*512], "GPU {device}, layer {layer}, rows {rows}");
                            let mut actual_mask = vec![0; rows];
                            library.copy_d2h(&mut actual_mask, view.text_mask)?;
                            assert_eq!(actual_mask, mask[..rows]);
                        }
                        runtime.block_on(owner.check_cooperative(&gathered))?;
                        assert_eq!(pointers, (owner.weights.buffer.ptr, owner.scales.buffer.ptr, owner.embeddings.buffer.ptr));
                    }
                    // Reuse the same allocation with the original FP8 format.
                    let weights = vec![0x38; 24*256];
                    let scales = vec![127; 24*8];
                    let gathered = EngramGatherView {
                        weights: &weights, scales: &scales, text_mask: &[1], rows: 1,
                        layer_index: index, encoding: ds41rt_loader::EngramEncoding::Fp8,
                        global_scale: 1.0,
                    };
                    let view = runtime.block_on(owner.upload_cooperative(&gathered))?;
                    let mut actual = vec![0; view.embeddings.bytes];
                    library.copy_d2h(&mut actual, view.embeddings)?;
                    assert!(actual.chunks_exact(2).all(|v| v == [0x80, 0x3f]));
                }
            }
        }
        Ok(())
    }
}
