//! Bounded background page-in for paired native engram weight and scale tables.
use crate::MappedRows;
use anyhow::{ensure, Context, Result};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::thread::{self, JoinHandle};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngramEncoding { Fp8, Nvfp4 }
impl EngramEncoding {
    pub fn weight_bytes(self) -> usize { match self { Self::Fp8 => 256, Self::Nvfp4 => 128 } }
    pub fn scale_bytes(self) -> usize { match self { Self::Fp8 => 8, Self::Nvfp4 => 16 } }
}

pub struct EngramTable {
    weights: MappedRows,
    scales: MappedRows,
    encoding: EngramEncoding,
    global_scale: f32,
}

impl EngramTable {
    pub fn encoding(&self) -> EngramEncoding { self.encoding }
    pub fn global_scale(&self) -> f32 { self.global_scale }
    pub fn new_nvfp4(weights: MappedRows, scales: MappedRows, global_scale: f32) -> Result<Self> {
        ensure!(weights.rows() == scales.rows() && weights.row_bytes() == 128 && scales.row_bytes() == 16,
            "NVFP4 engram requires paired 128-byte weight and 16-byte scale rows");
        ensure!(global_scale.is_finite() && global_scale > 0.0, "invalid NVFP4 engram global scale");
        Ok(Self { weights, scales, encoding: EngramEncoding::Nvfp4, global_scale })
    }
    pub fn weights(&self) -> &MappedRows {
        &self.weights
    }
    pub fn scales(&self) -> &MappedRows {
        &self.scales
    }

    /// Map the official embedding and per-row scale tensors for one engram layer.
    ///
    /// # Safety
    /// All referenced checkpoint files must remain immutable until this table is dropped.
    pub unsafe fn from_catalog(catalog: &ds41rt_core::TensorCatalog, layer: u32) -> Result<Self> {
        use ds41rt_core::DType;
        use std::path::{Component, Path};
        let rows = match layer {
            1 => 384_006_168_usize,
            14 => 384_016_682_usize,
            _ => anyhow::bail!("V4.1 Flash has no engram table at layer {layer}"),
        };
        let map = |suffix: &str, dtype: DType, width: usize| -> Result<MappedRows> {
            let name = format!("layers.{layer}.engram.embed.{suffix}");
            let mut matching = catalog.tensors.iter().filter(|tensor| tensor.name == name);
            let tensor = matching
                .next()
                .with_context(|| format!("missing engram tensor {name}"))?;
            ensure!(matching.next().is_none(), "duplicate engram tensor {name}");
            ensure!(
                tensor.dtype == dtype && tensor.shape == [rows, width],
                "invalid native engram representation for {name}"
            );
            ensure!(
                tensor.byte_length == rows as u64 * width as u64,
                "invalid engram payload length for {name}"
            );
            ensure!(
                !tensor.file.is_empty()
                    && Path::new(&tensor.file)
                        .components()
                        .all(|part| matches!(part, Component::Normal(_))),
                "invalid engram checkpoint shard path"
            );
            unsafe {
                MappedRows::open(
                    &Path::new(&catalog.snapshot_path).join(&tensor.file),
                    tensor.byte_offset,
                    rows as u64,
                    width,
                )
            }
        };
        Self::new(
            map("weight", DType::F8E4M3, 256)?,
            map("scale", DType::F8E8M0, 8)?,
        )
    }
    pub fn new(weights: MappedRows, scales: MappedRows) -> Result<Self> {
        ensure!(
            weights.rows() == scales.rows(),
            "engram weight/scale row counts differ"
        );
        ensure!(
            weights.row_bytes() == 256 && scales.row_bytes() == 8,
            "native engram rows require 256 FP8 bytes and eight UE8M0 scale bytes"
        );
        Ok(Self { weights, scales, encoding: EngramEncoding::Fp8, global_scale: 1.0 })
    }
}

struct Job {
    table: Arc<EngramTable>,
    rows: Vec<u64>,
    cancelled: Arc<AtomicBool>,
    completion: mpsc::SyncSender<Result<PrefetchOutcome>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchOutcome {
    Cancelled,
    Advised {
        weight_pages: usize,
        scale_pages: usize,
    },
}

/// Request-owned completion, independent of recycled scheduler slot numbers.
/// Dropping it cancels queued work; advice already issued to the OS is harmless.
pub struct PrefetchTicket {
    cancelled: Arc<AtomicBool>,
    completion: mpsc::Receiver<Result<PrefetchOutcome>>,
}

impl PrefetchTicket {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn wait(&self) -> Result<PrefetchOutcome> {
        self.completion
            .recv()
            .context("engram prefetch worker stopped")?
    }
}

impl Drop for PrefetchTicket {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// One worker and a bounded queue; submission never waits for disk or queue space.
/// The row cap bounds job memory independently of page deduplication.
pub struct EngramPrefetcher {
    sender: Option<mpsc::SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
    max_rows: usize,
}

impl EngramPrefetcher {
    pub fn new(queue_depth: usize, max_rows: usize, max_pages_per_table: usize) -> Result<Self> {
        ensure!(
            queue_depth > 0 && max_rows > 0 && max_pages_per_table > 0,
            "engram prefetch capacities must be nonzero"
        );
        let (sender, receiver) = mpsc::sync_channel::<Job>(queue_depth);
        let worker = thread::Builder::new()
            .name("engram-prefetch".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let result = (|| {
                        if job.cancelled.load(Ordering::Acquire) {
                            return Ok(PrefetchOutcome::Cancelled);
                        }
                        let weight_pages =
                            job.table.weights.prefetch(&job.rows, max_pages_per_table)?;
                        if job.cancelled.load(Ordering::Acquire) {
                            return Ok(PrefetchOutcome::Cancelled);
                        }
                        let scale_pages =
                            job.table.scales.prefetch(&job.rows, max_pages_per_table)?;
                        Ok(PrefetchOutcome::Advised {
                            weight_pages,
                            scale_pages,
                        })
                    })();
                    let _ = job.completion.send(result);
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            max_rows,
        })
    }

    /// Submit the address batch prepared from decode, prefill, or verification IDs.
    /// Image rows are omitted, and the layer's official table identity is checked.
    pub fn try_submit_batch(
        &self,
        table: Arc<EngramTable>,
        batch: &ds41rt_core::EngramBatch,
        layer_index: usize,
    ) -> Result<Option<PrefetchTicket>> {
        let expected_rows = ds41rt_core::ENGRAM_ROWS
            .get(layer_index)
            .context("invalid engram layer index")?;
        ensure!(
            table.weights.rows() == *expected_rows,
            "prefetch table belongs to another engram layer"
        );
        let text_rows = (0..batch.hashes().len())
            .filter(|&row| batch.is_image(row) == Some(false))
            .count();
        ensure!(
            text_rows <= self.max_rows / 24,
            "engram hash batch exceeds prefetch row capacity"
        );
        self.try_submit(table, &batch.prefetch_rows(layer_index)?)
    }

    /// None means backpressure: the caller may gather on demand or retry later.
    pub fn try_submit(
        &self,
        table: Arc<EngramTable>,
        rows: &[u64],
    ) -> Result<Option<PrefetchTicket>> {
        ensure!(
            rows.len() <= self.max_rows,
            "engram prefetch exceeds row capacity"
        );
        ensure!(
            rows.iter().all(|&row| row < table.weights.rows()),
            "engram row out of range"
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let (completion, receiver) = mpsc::sync_channel(1);
        let job = Job {
            table,
            rows: rows.to_vec(),
            cancelled: cancelled.clone(),
            completion,
        };
        match self
            .sender
            .as_ref()
            .context("engram prefetcher stopped")?
            .try_send(job)
        {
            Ok(()) => Ok(Some(PrefetchTicket {
                cancelled,
                completion: receiver,
            })),
            Err(mpsc::TrySendError::Full(_)) => Ok(None),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                anyhow::bail!("engram prefetch worker disconnected")
            }
        }
    }
}

impl Drop for EngramPrefetcher {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn official_size_catalog_maps_sparse_shard_and_rejects_eager_loading() -> Result<()> {
        use ds41rt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};
        use std::io::{Seek, SeekFrom};
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("engram.safetensors");
        let mut shard = std::fs::File::create(&path)?;
        let rows = 384_006_168_usize;
        let weight_offset = 13_u64;
        let scale_offset = weight_offset + rows as u64 * 256;
        shard.set_len(scale_offset + rows as u64 * 8)?;
        shard.seek(SeekFrom::Start(scale_offset - 256))?;
        shard.write_all(&[0x38; 256])?;
        shard.seek(SeekFrom::Start(scale_offset + (rows as u64 - 1) * 8))?;
        shard.write_all(&[127; 8])?;
        let mut catalog = TensorCatalog {
            model_id: "deepseek-ai/DeepSeek-V4.1-Flash".into(),
            snapshot_path: directory.path().to_str().unwrap().into(),
            facts: ModelFacts::default(),
            tensors: [
                ("weight", DType::F8E4M3, 256, weight_offset),
                ("scale", DType::F8E8M0, 8, scale_offset),
            ]
            .into_iter()
            .map(|(suffix, dtype, width, offset)| TensorInfo {
                name: format!("layers.1.engram.embed.{suffix}"),
                file: "engram.safetensors".into(),
                dtype,
                shape: vec![rows, width],
                byte_offset: offset,
                byte_length: rows as u64 * width as u64,
                role: TensorRole::Other,
                layer_id: Some(1),
                expert_id: None,
                is_quantization_metadata: suffix == "scale",
            })
            .collect(),
        };
        let table = unsafe { EngramTable::from_catalog(&catalog, 1)? };
        let mut weights = [0; 256];
        let mut scales = [0; 8];
        table
            .weights()
            .gather_into(&[rows as u64 - 1], &mut weights)?;
        table
            .scales()
            .gather_into(&[rows as u64 - 1], &mut scales)?;
        assert_eq!(weights, [0x38; 256]);
        assert_eq!(scales, [127; 8]);
        assert!(crate::load_tensor_bytes(&catalog, "layers.1.engram.embed.weight").is_err());
        assert!(
            crate::read_tensor_bytes_into(&catalog, "layers.1.engram.embed.scale", &mut [])
                .is_err()
        );
        assert!(unsafe { EngramTable::from_catalog(&catalog, 2) }.is_err());
        catalog.tensors[1].dtype = DType::U8;
        assert!(unsafe { EngramTable::from_catalog(&catalog, 1) }.is_err());
        Ok(())
    }

    #[test]
    fn background_prefetch_owns_mapping_and_advises_both_tables() -> Result<()> {
        let mut weights = tempfile::NamedTempFile::new()?;
        let mut scales = tempfile::NamedTempFile::new()?;
        weights.write_all(&[0x38; 512])?;
        scales.write_all(&[127; 16])?;
        let table = Arc::new(EngramTable::new(
            unsafe { MappedRows::open(weights.path(), 0, 2, 256)? },
            unsafe { MappedRows::open(scales.path(), 0, 2, 8)? },
        )?);
        let worker = EngramPrefetcher::new(1, 16, 2)?;
        assert!(worker.try_submit(table.clone(), &[2]).is_err());
        assert!(worker.try_submit(table.clone(), &[0; 17]).is_err());
        let ticket = worker.try_submit(table.clone(), &[1, 0, 1])?.unwrap();
        drop(table);
        assert_eq!(
            ticket.wait()?,
            PrefetchOutcome::Advised {
                weight_pages: 1,
                scale_pages: 1
            }
        );
        Ok(())
    }
}
