//! Target mHC/norm on RTX1 followed by the per-lane split vocabulary head.
use super::*;
use super::distributed::DistributedVocabularyWave;
use crate::v41_memory::device::{Device, DeviceOwner, Stream};
use crate::v41_tensors::VocabularyShard;

const INPUT_STRIDES: [usize; 4] = [40960, 16, 10240, 10240];
struct Normalize<'w, 'a> {
    stream: LoadStream<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    weights: &'w TargetHeadWeights<'a>,
    hc: V41Hc<'a>,
    graphs: [Option<*mut c_void>; 80],
}
fn part(buffer: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Result<Ds41rtDeviceBuffer> {
    ensure!(offset <= buffer.bytes && bytes <= buffer.bytes - offset, "target head slice exceeds storage");
    Ok(Ds41rtDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer })
}
impl<'w, 'a> Normalize<'w, 'a> {
    fn new(weights: &'w TargetHeadWeights<'a>, capacity: usize) -> Result<Self> {
        let lib = weights.library;
        let buffers = INPUT_STRIDES.into_iter().map(|stride| DeviceAllocation::new(lib, stride * capacity))
            .collect::<Result<Vec<_>>>()?;
        ensure!(weights.norm.get("norm.weight")?.device_id == buffers[0].buffer.device_id, "target norm device differs");
        Ok(Self { stream: LoadStream { library: lib, raw: lib.cuda_stream_create()? },
            buffers, weights, hc: lib.v41_hc()?, graphs: [None; 80] })
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.hc.pre(self.buffers[0].buffer, self.buffers[1].buffer, self.buffers[2].buffer, rows, self.stream.raw)?;
            self.weights.library.cuda_ds4_rmsnorm_bf16_rne_async(self.buffers[2].buffer,
                self.weights.norm.get("norm.weight")?, self.buffers[3].buffer, rows as i32,
                5120, 1e-20, self.stream.raw)
        }
    }
    async unsafe fn execute(&mut self, block: &BlockOutput<'_>, selected: &[usize]) -> Result<()> {
        let rows = selected.len();
        let queued = (|| -> Result<()> { unsafe {
            let mut first = 0;
            while first < rows {
                let mut count = 1;
                while first + count < rows && selected[first + count] == selected[first] + count { count += 1; }
                for (i, source, stride) in [(0, block.residual, 40960), (1, block.pre, 16)] {
                    self.weights.library.copy_d2d_async(part(self.buffers[i].buffer, first * stride, count * stride)?,
                        part(source, selected[first] * stride, count * stride)?, count * stride, self.stream.raw)?;
                }
                first += count;
            }
            if let Some(graph) = self.graphs[rows - 1] { self.weights.library.cuda_graph_launch(graph, self.stream.raw) }
            else { self.enqueue(rows) }
        } })();
        let drained = self.stream.wait().await;
        queued.and(drained)?;
        if self.graphs[rows - 1].is_none() {
            let lib = self.weights.library;
            unsafe { lib.cuda_graph_begin_capture(self.stream.raw)?; }
            let queued = unsafe { self.enqueue(rows) };
            let captured = unsafe { lib.cuda_graph_end_capture(self.stream.raw) };
            match (queued, captured) {
                (Ok(()), Ok(graph)) => self.graphs[rows - 1] = Some(graph),
                (Err(error), Ok(graph)) => { unsafe { lib.cuda_graph_exec_destroy(graph)?; } return Err(error); }
                (Err(error), Err(_)) | (Ok(()), Err(error)) => return Err(error),
            }
        }
        Ok(())
    }
}
impl Drop for Normalize<'_, '_> {
    fn drop(&mut self) {
        let lib = self.weights.library;
        if let Err(error) = unsafe { lib.cuda_stream_synchronize(self.stream.raw) } { tracing::error!(%error, "draining target norm"); }
        for graph in self.graphs.iter_mut().filter_map(Option::take) {
            if let Err(error) = unsafe { lib.cuda_graph_exec_destroy(graph) } { tracing::error!(%error, "destroying target norm graph"); }
        }
    }
}

pub(crate) struct DistributedTargetLogits<'a> {
    pub rows: usize,
    pub collapsed: Ds41rtDeviceBuffer,
    pub normalized: Ds41rtDeviceBuffer,
    pub logits: [Ds41rtDeviceBuffer; 2],
    pub selected_rows: &'a [usize],
    pub token_positions: &'a [u64],
    pub binding: QueryBinding,
}

/// Device sampling for the split head: both vocabulary halves are assembled
/// into full rows on the head GPU, then the single-GPU target sampler runs.
struct HeadSampler<'a> {
    assembled: DeviceOwner<'a, DeviceAllocation<'a>>,
    wave: DeviceOwner<'a, TargetSamplingWave<'a>>,
    stream: Stream<'a>,
    download: DeviceOwner<'a, crate::v41_memory::RowDownload<'a>>,
}

pub(crate) struct DistributedTargetHead<'w, 'a> {
    sampler: HeadSampler<'a>,
    normalize: DeviceOwner<'a, Normalize<'w, 'a>>,
    vocabulary: DistributedVocabularyWave<'w, 'a>,
    download: Stream<'a>,
    staging: HostAllocation<'a>,
    rank_downloads: [DeviceOwner<'a, crate::v41_memory::RowDownload<'a>>; 2],
    capacity: usize,
    binding: Option<QueryBinding>,
    selected: Vec<usize>,
    tokens: Vec<u64>,
    greedy: Vec<(u32, f32)>,
}
impl<'w, 'a> DistributedTargetHead<'w, 'a> {
    pub fn device_bytes(capacity: usize, split: usize) -> Result<[usize; 2]> {
        let mut bytes = DistributedVocabularyWave::device_bytes(capacity, split)?;
        bytes[1] += capacity * INPUT_STRIDES.iter().sum::<usize>()
            + capacity * 129280 * 4 + TargetSamplingWave::device_bytes(capacity);
        Ok(bytes)
    }
    pub fn new(devices: [Device<'a>; 2], weights: &'w TargetHeadWeights<'a>,
        vocabulary: [&'w VocabularyShard<'a>; 2], capacity: usize, budgets: [usize; 2]) -> Result<Self> {
        let bytes = Self::device_bytes(capacity, vocabulary[0].tokens().end)?;
        ensure!(bytes.iter().zip(budgets).all(|(n, budget)| *n <= budget), "distributed target head exceeds budget");
        let rank_downloads = [
            devices[0].own(|| crate::v41_memory::RowDownload::new(devices[0].library, capacity * vocabulary[0].tokens().len() * 4))?,
            devices[1].own(|| crate::v41_memory::RowDownload::new(devices[1].library, capacity * vocabulary[1].tokens().len() * 4))?,
        ];
        let normalize = devices[1].own(|| Normalize::new(weights, capacity))?;
        let sampler_bytes = capacity * 129280 * 4 + TargetSamplingWave::device_bytes(capacity);
        let vocabulary = DistributedVocabularyWave::new(devices, vocabulary, capacity,
            [budgets[0], budgets[1] - capacity * INPUT_STRIDES.iter().sum::<usize>() - sampler_bytes])?;
        let sampler = HeadSampler {
            assembled: devices[1].own(|| DeviceAllocation::new(devices[1].library, capacity * 129280 * 4))?,
            wave: devices[1].own(|| TargetSamplingWave::new(devices[1].library, capacity))?,
            stream: Stream::new(devices[1])?,
            download: devices[1].own(|| crate::v41_memory::RowDownload::new(devices[1].library, capacity * 129280 * 4))?,
        };
        Ok(Self { sampler, normalize, vocabulary, download: Stream::new(devices[1])?,
            staging: HostAllocation::new(devices[1].library, capacity * 8)?, rank_downloads, capacity,
            binding: None, selected: Vec::with_capacity(capacity), tokens: Vec::with_capacity(capacity),
            greedy: Vec::with_capacity(capacity) })
    }
    /// # Safety
    /// The final block is complete on RTX1 and immutable through return or
    /// cancellation. Selected rows are unique and define compact output order.
    pub async unsafe fn execute_block(&mut self, block: &BlockOutput<'_>, selected: &[usize]) -> Result<()> {
        unsafe { self.execute_block_with_greedy(block, selected, true).await }
    }
    /// # Safety
    /// Same complete-block ownership contract as execute_block. With greedy
    /// false, retain device logits without downloading compact candidates.
    pub async unsafe fn execute_block_with_greedy(&mut self, block: &BlockOutput<'_>,
        selected: &[usize], greedy: bool) -> Result<()> {
        self.binding = None;
        self.selected.clear(); self.tokens.clear(); self.greedy.clear();
        let device = self.normalize.device;
        ensure!(block.layer == 39 && block.binding().layer() == 39 && block.tokens.len() <= 4096
            && !selected.is_empty() && selected.len() <= self.capacity
            && selected.iter().enumerate().all(|(i, &row)| row < block.tokens.len() && !selected[..i].contains(&row))
            && block.residual.bytes == block.tokens.len() * 40960 && block.pre.bytes == block.tokens.len() * 16
            && block.residual.device_id == device.id && block.pre.device_id == device.id,
            "distributed target head block or selected rows differ");
        // The TP2 vocabulary projection moves rows between GPUs with host-ordered
        // copies; settle a chained final layer first (one wait per pass).
        crate::v41_memory::chain::settle(device.library)?;
        device.future(unsafe { self.normalize.get_mut().execute(block, selected) }).await?;
        let normalized = self.normalize.buffers[3].buffer;
        unsafe { self.vocabulary.execute(normalized, selected.len()).await?; }
        let rows = selected.len();
        if greedy {
            let (ids, scores) = self.vocabulary.greedy()?;
            let host = self.staging.bytes_mut();
            let queued = device.run(|| unsafe {
                device.library.copy_d2h_async(&mut host[..rows * 4], ids, self.download.raw)?;
                device.library.copy_d2h_async(&mut host[rows * 4..rows * 8], scores, self.download.raw)
            });
            let drained = self.download.wait().await;
            queued.and(drained)?;
            for i in 0..rows {
                self.greedy.push((u32::from_ne_bytes(host[i*4..i*4+4].try_into().unwrap()),
                    f32::from_ne_bytes(host[rows*4+i*4..rows*4+i*4+4].try_into().unwrap())));
            }
        }
        self.selected.extend_from_slice(selected);
        self.tokens.extend(selected.iter().map(|&i| block.tokens[i]));
        self.binding = Some(block.binding());
        Ok(())
    }
    pub fn device(&self) -> Device<'a> { self.normalize.device }
    /// Select every row of the completed (non-greedy) head pass on the device:
    /// assemble the two vocabulary halves into full rows on the head GPU and run
    /// the target sampler there, downloading only ids/scores/status.
    /// # Safety
    /// `execute_block_with_greedy(.., false)` completed for these rows; the
    /// head's logits stay immutable until this returns.
    pub async unsafe fn sample(&mut self, requests: &[TargetSamplingRowRequest], masks: Option<&[u32]>,
        mask_words: usize, ordered_rows: bool) -> Result<SampledTargetRows> {
        let rows = self.selected.len();
        ensure!(self.binding.is_some() && rows > 0 && requests.len() == rows,
            "distributed sampling requests must match the head rows");
        let shards = self.vocabulary.logits()?;
        let widths = [shards[0].bytes / rows / 4, shards[1].bytes / rows / 4];
        ensure!(widths[0] + widths[1] == 129280 && shards[1].device_id == self.sampler.stream.device.id,
            "distributed vocabulary halves differ from the sampler layout");
        let device = self.sampler.stream.device;
        let assembled = part(self.sampler.assembled.buffer, 0, rows * 129280 * 4)?;
        let stream = self.sampler.stream.raw;
        let wave = self.sampler.wave.get_mut();
        let queued = device.run(|| unsafe {
            for row in 0..rows {
                device.library.copy_peer_async(part(assembled, row * 129280 * 4, widths[0] * 4)?,
                    part(shards[0], row * widths[0] * 4, widths[0] * 4)?, widths[0] * 4, stream)?;
                device.library.copy_d2d_async(part(assembled, (row * 129280 + widths[0]) * 4, widths[1] * 4)?,
                    part(shards[1], row * widths[1] * 4, widths[1] * 4)?, widths[1] * 4, stream)?;
            }
            wave.upload(requests, masks, mask_words, stream)?;
            wave.launch(assembled, rows, ordered_rows, stream)
        });
        let drained = self.sampler.stream.wait().await;
        queued.and(drained)?;
        device.run(|| self.sampler.wave.get_mut().output(assembled, rows))
    }
    /// Full logits of selected sampled rows, from the assembled head rows.
    pub async fn download_sampled(&mut self, rows: &SampledTargetRows, selection: &[usize]) -> Result<Vec<u8>> {
        ensure!(!selection.is_empty(), "sampled row download selection is empty");
        let download = &mut self.sampler.download;
        let device = download.device;
        device.future(unsafe { download.get_mut().rows(rows.logits, 129280 * 4, selection) }).await
    }
    pub fn output(&self) -> Result<DistributedTargetLogits<'_>> {
        let binding = self.binding()?;
        let rows = self.selected.len();
        Ok(DistributedTargetLogits { rows,
            collapsed: part(self.normalize.buffers[2].buffer, 0, rows * 10240)?,
            normalized: part(self.normalize.buffers[3].buffer, 0, rows * 10240)?,
            logits: self.vocabulary.logits()?, selected_rows: &self.selected,
            token_positions: &self.tokens, binding })
    }
    /// Explicit sampling/constrained readback, ordered by compact output rows.
    /// Greedy execution does not call this or download full vocabulary logits.
    pub async fn download_rows(&mut self, rows: &[usize]) -> Result<Vec<u8>> {
        let logits = self.logits()?;
        let widths = [logits[0].bytes / self.selected.len(), logits[1].bytes / self.selected.len()];
        let [first, second] = &mut self.rank_downloads;
        let d0 = first.device;
        let d1 = second.device;
        let (a, b) = tokio::join!(
            d0.future(unsafe { first.get_mut().rows(logits[0], widths[0], rows) }),
            d1.future(unsafe { second.get_mut().rows(logits[1], widths[1], rows) }));
        let (a, b) = (a?, b?);
        let mut output = Vec::with_capacity(rows.len() * 129280 * 4);
        for row in 0..rows.len() {
            output.extend_from_slice(&a[row*widths[0]..(row+1)*widths[0]]);
            output.extend_from_slice(&b[row*widths[1]..(row+1)*widths[1]]);
        }
        Ok(output)
    }
    pub fn greedy_output(&self) -> Result<&[(u32, f32)]> {
        ensure!(self.binding.is_some(), "distributed target head unpublished");
        ensure!(!self.greedy.is_empty(), "distributed greedy output not requested");
        Ok(&self.greedy)
    }
    pub fn logits(&self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        ensure!(self.binding.is_some(), "distributed target head unpublished");
        self.vocabulary.logits()
    }
    pub fn selected_rows(&self) -> Result<&[usize]> {
        ensure!(self.binding.is_some(), "distributed target head unpublished"); Ok(&self.selected)
    }
    pub fn token_positions(&self) -> Result<&[u64]> {
        ensure!(self.binding.is_some(), "distributed target head unpublished"); Ok(&self.tokens)
    }
    pub fn binding(&self) -> Result<QueryBinding> { self.binding.context("distributed target head unpublished") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_memory::device::Allocation;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn distributed_target_head_matches_full_selection_and_publication() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let norm = devices[1].own(|| TargetHeadWeights::load(&lib, &catalog, 10240, 10240))?;
        let a = devices[0].own(|| VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?;
        let b = devices[1].own(|| VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?;
        let full = devices[1].own(|| VocabularyHead::load(&lib, &catalog, 2 << 30, 16 << 20))?;
        let mut baseline = devices[1].own(|| norm.wave(&full, 6, TargetHeadWave::device_bytes(6)?))?;
        let mut head = DistributedTargetHead::new(devices, &norm, [&a, &b], 6,
            DistributedTargetHead::device_bytes(6, 64640)?)?;
        let residual = Allocation::new(devices[1], 6 * 40960)?;
        let pre = Allocation::new(devices[1], 6 * 16)?;
        let data: Vec<u8> = (0..6 * 20480).flat_map(|i| {
            let x = ((i * 13 % 131) as f32 - 65.) / 64.;
            ((x.to_bits() >> 16) as u16).to_ne_bytes()
        }).collect();
        let pre_data: Vec<u8> = (0..24).flat_map(|_| 0.25f32.to_ne_bytes()).collect();
        devices[1].run(|| { lib.copy_h2d(residual.buffer, &data)?; lib.copy_h2d(pre.buffer, &pre_data) })?;
        let tokens = [10, 11, 12, 13, 14, 15];
        let block = unsafe { BlockOutput::from_test_buffers(39, &tokens, residual.buffer, pre.buffer)? };
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for selected in [vec![5], vec![4, 1, 3], vec![0, 1, 2, 3, 4, 5], vec![3, 0, 5]] {
            runtime.block_on(devices[1].future(unsafe { baseline.get_mut().execute_block_greedy(&block, &selected, true) }))?;
            runtime.block_on(unsafe { head.execute_block(&block, &selected) })?;
            let expected_greedy = baseline.greedy_output()?;
            assert_eq!(head.greedy_output()?, expected_greedy.as_slice());
            assert_eq!(head.selected_rows()?, selected.as_slice());
            assert_eq!(head.token_positions()?, selected.iter().map(|&i| tokens[i]).collect::<Vec<_>>());
            assert_eq!(head.binding()?, block.binding());
            let expected_buffer = baseline.output()?.logits;
            let mut expected = vec![0u8; selected.len() * 129280 * 4];
            devices[1].run(|| lib.copy_d2h(&mut expected, expected_buffer))?;
            for (rank, buffer) in head.logits()?.into_iter().enumerate() {
                let mut actual = vec![0u8; buffer.bytes];
                devices[rank].run(|| lib.copy_d2h(&mut actual, buffer))?;
                for row in 0..selected.len() {
                    let start = (row * 129280 + rank * 64640) * 4;
                    ensure!(actual[row*64640*4..(row+1)*64640*4] == expected[start..start+64640*4],
                        "complete target head logits differ at rank {rank}, row {row}");
                }
            }
            let chosen = [selected.len() - 1, 0];
            let downloaded = runtime.block_on(head.download_rows(&chosen))?;
            let expected_download: Vec<u8> = chosen.into_iter().flat_map(|row|
                expected[row*129280*4..(row+1)*129280*4].iter().copied()).collect();
            ensure!(downloaded == expected_download, "distributed head compact row download differs");
            assert!(runtime.block_on(head.download_rows(&[selected.len()])).is_err());
            eprintln!("PASS complete target head selection={selected:?}: logits, greedy and metadata exact");
        }
        runtime.block_on(unsafe { head.execute_block_with_greedy(&block, &[2, 0], false) })?;
        assert!(head.greedy_output().is_err());
        assert!(head.output().is_ok());
        assert_eq!(runtime.block_on(head.download_rows(&[1]))?.len(), 129280 * 4);
        for bad in [vec![], vec![1, 1], vec![6]] {
            assert!(runtime.block_on(unsafe { head.execute_block(&block, &bad) }).is_err());
            assert!(head.greedy_output().is_err());
            assert!(head.logits().is_err());
        }
        runtime.block_on(unsafe { head.execute_block(&block, &[2]) })?;
        assert!(head.greedy_output().is_ok());
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!("PASS target head invalid selection unpublishes output and allows reuse");
        Ok(())
    }
}
