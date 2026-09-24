//! Private dSpark inputs from the target pass, in exact flattened request order.
use crate::v41_backbone_cache::CacheBatch;
use crate::v41_backbone_router::ExpertRow;
use crate::v41_block::PreparedBlockInput;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps};

#[derive(Default)]
struct Progress {
    batch: Option<u64>,
    next: usize,
}
impl Progress {
    fn check(&self, batch: u64, layer: usize) -> Result<()> {
        ensure!(
            self.batch == Some(batch) && self.next < 3 && layer == 37 + self.next,
            "target taps require this batch's layers 37, 38, 39 in order"
        );
        Ok(())
    }
    fn ready(&self, batch: u64) -> Result<()> {
        ensure!(
            self.batch == Some(batch) && self.next == 3,
            "target taps incomplete or foreign"
        );
        Ok(())
    }
}

pub(crate) struct TargetTapWave<'a> {
    stream: LoadStream<'a>,
    values: DeviceAllocation<'a>,
    ops: V41AttentionOps<'a>,
    capacity: usize,
    rows: Vec<ExpertRow>,
    progress: Progress,
}
/// Borrowed proposed main context. Consume it before committing or discarding
/// the target pass; raw device views must not outlive this borrow.
pub(crate) struct TargetTaps<'a> {
    values: Ds41rtDeviceBuffer,
    rows: &'a [ExpertRow],
    batch: u64,
}
impl TargetTaps<'_> {
    pub fn values(&self) -> Ds41rtDeviceBuffer {
        self.values
    }
    pub fn rows(&self) -> &[ExpertRow] {
        self.rows
    }
    pub fn batch_identity(&self) -> u64 {
        self.batch
    }
}
impl<'a> TargetTapWave<'a> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid target tap capacity"
        );
        Ok(capacity * 30720)
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, budget: usize) -> Result<Self> {
        let bytes = Self::device_bytes(capacity)?;
        ensure!(bytes <= budget, "target taps exceed device budget");
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            values: DeviceAllocation::new(library, bytes)?,
            ops: library.v41_attention_ops()?,
            capacity,
            rows: Vec::with_capacity(capacity),
            progress: Progress::default(),
        })
    }
    pub(super) fn reset(&mut self) {
        self.progress = Progress::default();
        self.rows.clear();
    }
    pub(super) fn begin(&mut self, batch: &CacheBatch) -> Result<()> {
        self.reset();
        let rows = batch.expert_rows();
        ensure!(
            !rows.is_empty() && rows.len() <= self.capacity,
            "target tap batch exceeds capacity"
        );
        self.rows = rows;
        self.progress.batch = Some(batch.identity());
        Ok(())
    }
    /// # Safety
    /// Input belongs to the target driver's current batch and is fully produced.
    /// This method drains its read before attention can overwrite the residual.
    pub(super) unsafe fn capture(
        &mut self,
        batch: &CacheBatch,
        input: &PreparedBlockInput<'_>,
    ) -> Result<()> {
        let queued = unsafe { self.enqueue_capture(batch, input) };
        let drained = if queued.is_err() { unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) } }
            else { unsafe { crate::v41_memory::chain::finish(self.stream.library, self.stream.raw) } };
        if let Err(error) = queued.and(drained) { self.reset(); return Err(error); }
        self.progress.next += 1;
        Ok(())
    }
    /// # Safety
    /// Retain prepared input until the tap read completes or cancellation drains.
    pub(super) async unsafe fn capture_cooperative(&mut self, batch: &CacheBatch,
        input: &PreparedBlockInput<'_>) -> Result<()> {
        unsafe { self.enqueue_capture(batch, input)?; }
        if let Err(error) = unsafe { crate::v41_memory::chain::finish_cooperative(&self.stream).await } {
            self.reset(); return Err(error);
        }
        self.progress.next += 1;
        Ok(())
    }
    unsafe fn enqueue_capture(&mut self, batch: &CacheBatch, input: &PreparedBlockInput<'_>) -> Result<()> {
        let result = (|| -> Result<()> {
            self.progress.check(batch.identity(), input.layer)?;
            ensure!(
                input.previous_binding().layer() + 1 == input.layer
                    && input.tokens.len() == self.rows.len()
                    && input
                        .tokens
                        .iter()
                        .zip(&self.rows)
                        .all(|(&p, row)| p == row.position)
                    && input.residual.bytes == self.rows.len() * 40960
                    && input.residual.device_id == self.values.buffer.device_id,
                "target tap prepared input differs from its batch"
            );
            #[cfg(test)]
            if let Some(dir) = std::env::var_os("DS41RT_TARGET_PASS_OUTPUT") {
                let dir = std::path::PathBuf::from(dir);
                std::fs::create_dir_all(&dir)?;
                let mut bytes = vec![0; input.residual.bytes];
                self.stream.library.copy_d2h(&mut bytes, input.residual)?;
                std::fs::write(
                    dir.join(format!(
                        "batch{}-layer{}-tap-input.bin",
                        batch.identity(),
                        input.layer
                    )),
                    bytes,
                )?;
            }
            let launched = unsafe {
                crate::v41_memory::chain::join(self.stream.library, self.stream.raw)?;
                self.ops.tap(
                    input.residual,
                    self.values.buffer,
                    self.rows.len() as u32,
                    input.layer as u32,
                    self.stream.raw,
                )
            };
            if let Err(error) = launched {
                unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw)?; }
                return Err(error);
            }
            Ok(())
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }
    pub(super) fn output(&self, batch: &CacheBatch) -> Result<TargetTaps<'_>> {
        self.progress.ready(batch.identity())?;
        let mut values = self.values.buffer;
        values.bytes = self.rows.len() * 30720;
        Ok(TargetTaps {
            values,
            rows: &self.rows,
            batch: batch.identity(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Progress;
    #[test]
    fn taps_require_complete_ordered_batch() {
        let mut progress = Progress {
            batch: Some(7),
            next: 0,
        };
        assert!(progress.ready(7).is_err());
        assert!(progress.check(8, 37).is_err());
        assert!(progress.check(7, 38).is_err());
        for layer in 37..40 {
            progress.check(7, layer).unwrap();
            progress.next += 1;
        }
        progress.ready(7).unwrap();
        assert!(progress.ready(8).is_err());
        assert!(progress.check(7, 39).is_err());
        progress = Progress::default();
        assert!(progress.ready(7).is_err());
    }
}
