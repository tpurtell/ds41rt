//! One request lane's distributed backbone execution and transaction progress.
use super::placement::PendingPlacedProduction;
use super::*;

pub(crate) struct DistributedExecution<'w, 'a> {
    weights: &'w CacheProducerWeights<'a>,
    pub producers: PlacedProducerWaves<'w, 'a>,
    progress: PassProgress,
    produced: Option<Produced>,
    publication: Option<(u64, usize)>,
}
pub(crate) struct PendingDistributedProduction<'p, 'w, 'i, 'a> {
    pending: PendingPlacedProduction<'p, 'w, 'i, 'a>,
    produced: &'p mut Option<Produced>,
    batch: u64,
    layer: usize,
    indexed: bool,
    started: std::time::Instant,
}
impl PendingDistributedProduction<'_, '_, '_, '_> {
    /// # Safety
    /// Retain the original query/cache owners until completion or drained drop.
    pub unsafe fn poll(&mut self, bank: &BackboneCache<'_>, batch: &CacheBatch) -> Result<bool> {
        if !unsafe { self.pending.poll(bank, batch)? } {
            return Ok(false);
        }
        self.finish();
        Ok(true)
    }
    /// # Safety
    /// As for poll; additionally abort queued publication before request release.
    pub unsafe fn poll_encoder(
        &mut self,
        bank: &mut BackboneCache<'_>,
        batch: &CacheBatch,
    ) -> Result<bool> {
        if !unsafe { self.pending.poll_encoder(bank, batch)? } {
            return Ok(false);
        }
        self.finish();
        Ok(true)
    }
    fn finish(&mut self) {
        *self.produced = Some(Produced {
            batch: self.batch,
            layer: self.layer,
            started: self.started,
            producer_us: self.started.elapsed().as_micros() as u64,
            indexed: self.indexed,
        });
    }
}
/// Scope every poll and cancellation of attention/FFN work to this layer's GPU.
pub(crate) struct PlacedPreparedLayer<'l, 'w, 'a> {
    device: Device<'a>,
    prepared: Option<PreparedLayer<'l, 'w, 'a>>,
}
impl PlacedPreparedLayer<'_, '_, '_> {
    #[cfg(test)]
    pub async unsafe fn trace_ffn_input(mut self, destination: ds41rt_ffi::Ds41rtDeviceBuffer, stream: *mut std::ffi::c_void)
        -> Result<(Self, (u64, usize, usize))> {
        let prepared = self.prepared.take().context("trace owner absent")?;
        let (prepared, record) = self.device.future(unsafe {
            prepared.trace_ffn_input(self.device.library, destination, stream)
        }).await?;
        self.prepared = Some(prepared);
        Ok((self, record))
    }
    /// # Safety
    /// Retain this batch's producers/index, lane, transport and modality mask
    /// through completion or drained cancellation, as for PreparedLayer::execute.
    pub async unsafe fn execute<'t>(
        mut self,
        transport: &'t mut NativeTp4Wave<'_>,
        placement: u64,
        image_mask: &[u8],
    ) -> Result<CompletedLayer<'t>> {
        self.device
            .future(unsafe {
                self.prepared
                    .take()
                    .expect("prepared layer present")
                    .execute(transport, placement, image_mask)
            })
            .await
    }
}
impl Drop for PlacedPreparedLayer<'_, '_, '_> {
    fn drop(&mut self) {
        if let Some(prepared) = self.prepared.take() {
            if let Err(error) = self.device.run(|| {
                drop(prepared);
                Ok(())
            }) {
                tracing::error!(%error,"draining unexecuted placed attention");
            }
        }
    }
}
impl<'w, 'a> DistributedExecution<'w, 'a> {
    pub fn new(
        weights: &'w CacheProducerWeights<'a>,
        capacity: u32,
        budgets: [usize; 2],
    ) -> Result<Self> {
        Ok(Self {
            weights,
            producers: PlacedProducerWaves::new(weights, capacity, budgets)?,
            progress: PassProgress::default(),
            produced: None,
            publication: None,
        })
    }
    /// All external readers/writers have completed or drained before restarting.
    pub fn restart_for(&mut self, stage: CacheStage) {
        self.progress = PassProgress::for_stage(stage);
        self.produced = None;
    }
    /// # Safety
    /// Query is complete. Its owner and this batch's bank slots remain immutable
    /// through production and subsequent attention completion/cancellation.
    pub unsafe fn enqueue_production<'p, 'i>(
        &'p mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        lane: &BackboneLane<'_, '_>,
        index: Option<&'p mut DeviceOwner<'a, IndexLane<'i, 'a>>>,
    ) -> Result<PendingDistributedProduction<'p, 'w, 'i, 'a>> {
        ensure!(
            self.produced.is_none() && batch.stage() == self.progress.stage,
            "distributed production phase or pending result differs"
        );
        ensure!(
            self.publication.is_none(),
            "encoder publication still pending"
        );
        let query = lane.query_output()?;
        let layer = query.layer;
        if batch.stage() == CacheStage::Encoder && layer == 20 {
            self.progress.begin_decoder_source(batch.identity())?;
        } else {
            self.progress.begin(batch.identity(), layer)?;
        }
        let indexed = index.is_some();
        ensure!(
            indexed == (INDEX.contains(&layer) && batch.stage().windows().contains(&layer)),
            "distributed index producer presence differs"
        );
        let started = std::time::Instant::now();
        let pending = unsafe {
            match index {
                Some(index) => self
                    .producers
                    .enqueue_production_and_index(bank, batch, &query, index)?,
                None => self.producers.enqueue_production(bank, batch, &query)?,
            }
        };
        Ok(PendingDistributedProduction {
            pending,
            produced: &mut self.produced,
            batch: batch.identity(),
            layer,
            indexed,
            started,
        })
    }
    /// # Safety
    /// Completed cache/index proposals are retained until this prepared layer's
    /// attention consumer completes or drains. The bank borrow ends on return.
    pub unsafe fn prepare_layer<'l, 'lw>(
        &mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        lane: &'l mut DeviceOwner<'a, BackboneLane<'lw, 'a>>,
        index: Option<&IndexLane<'_, '_>>,
    ) -> Result<PlacedPreparedLayer<'l, 'lw, 'a>> {
        let produced = self
            .produced
            .take()
            .context("distributed layer production incomplete")?;
        ensure!(
            self.progress.invalid
                && self.progress.batch == Some(batch.identity())
                && self.progress.next == produced.layer
                && produced.batch == batch.identity()
                && self.progress.stage == batch.stage()
                && batch.stage().windows().contains(&produced.layer),
            "distributed prepared layer identity differs"
        );
        let layer = produced.layer;
        let device = lane.device;
        ensure!(
            device.id == bank.attention_device(layer)?.id,
            "distributed lane and cache GPUs differ"
        );
        let source = SOURCES
            .iter()
            .rposition(|&l| l <= layer)
            .filter(|&i| !batch.stage().reuses_sources() && !batch.is_reserved())
            .map(|i| self.producers.sources[i].get());
        let cache = bank.attention(batch, layer, &self.producers.windows[layer], source)?;
        let indexed_us = produced.started.elapsed().as_micros() as u64;
        let sink = self.weights.sink_views[layer];
        let rows = batch.expert_rows();
        let pending = device.run(|| unsafe {
            lane.get_mut().enqueue_attention_replicated_ffn(sink,bank,&cache,index)
        })?;
        Ok(PlacedPreparedLayer {
            device,
            prepared: Some(PreparedLayer {
                ffn: PreparedFfn::Pending(pending),
                rows,
                batch: batch.identity(),
                layer,
                started: produced.started,
                produced_us: produced.producer_us,
                indexed_us,
                attended_us: indexed_us,
            }),
        })
    }
    /// # Safety
    /// The completed transport and lane describe this exact layer/batch. Retain
    /// both until the final mHC producer completes or drains on cancellation.
    pub async unsafe fn complete_layer(
        &mut self,
        batch: &CacheBatch,
        lane: &mut DeviceOwner<'a, BackboneLane<'_, 'a>>,
        completed: CompletedLayer<'_>,
    ) -> Result<()> {
        ensure!(
            self.progress.invalid
                && self.progress.batch == Some(completed.batch)
                && batch.identity() == completed.batch
                && self.progress.next == completed.layer
                && self.progress.stage == batch.stage(),
            "distributed completion identity differs"
        );
        let device = lane.device;
        ensure!(
            completed.result.values.device_id == device.id,
            "distributed FFN result GPU differs"
        );
        device
            .future(unsafe {
                lane.get_mut()
                    .finish_ffn_cooperative(completed.result.binding(), completed.result.values)
            })
            .await?;
        completed.log_cost(lane);
        tracing::debug!(target: "ds41rt::timing", layer=completed.layer, rows=completed.rows,
            production_and_index_us=completed.indexed_us,
            attention_us=completed.attended_us-completed.indexed_us,
            experts_us=completed.experts_us-completed.attended_us,
            finish_us=completed.started.elapsed().as_micros() as u64-completed.experts_us,
            "distributed target layer");
        self.progress.finish();
        Ok(())
    }
    pub fn finish_decoder_source(&mut self, batch: &CacheBatch) -> Result<()> {
        let produced = self
            .produced
            .take()
            .context("distributed encoder boundary incomplete")?;
        ensure!(
            self.progress.invalid
                && self.progress.next == 20
                && self.progress.stage == CacheStage::Encoder
                && produced.layer == 20
                && produced.batch == batch.identity()
                && self.progress.batch == Some(batch.identity()),
            "distributed encoder boundary identity differs"
        );
        self.progress.finish_decoder_source();
        Ok(())
    }
    /// Queue publication only after this layer's attention/FFN consumers drain.
    /// # Safety
    /// Retain bank and producer owners until publication or explicit abort.
    pub unsafe fn enqueue_encoder_publication(
        &mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        layer: usize,
    ) -> Result<()> {
        ensure!(
            batch.is_reserved()
                && batch.stage() == CacheStage::Encoder
                && self.publication.is_none()
                && !self.progress.invalid
                && self.progress.batch == Some(batch.identity())
                && layer < 20
                && self.progress.next == layer + 1,
            "encoder publication requires this completed layer"
        );
        self.publication = Some((batch.identity(), layer));
        if layer < 20 {
            unsafe {
                bank.enqueue_encoder_window(batch, layer, &mut self.producers.windows[layer])?;
            }
        }
        Ok(())
    }
    pub fn poll_encoder_publication(&self) -> Result<bool> {
        use crate::v41_backbone_cache::CacheWave;
        let (_, layer) = self.publication.context("encoder publication absent")?;
        if layer < 20 && !self.producers.windows[layer].on_device(|wave| wave.poll_commit())? {
            return Ok(false);
        }
        Ok(true)
    }
    pub fn finish_encoder_publication(
        &mut self,
        bank: &mut BackboneCache<'_>,
        batch: &CacheBatch,
    ) -> Result<()> {
        let (identity, layer) = self.publication.context("encoder publication absent")?;
        ensure!(
            identity == batch.identity() && self.poll_encoder_publication()?,
            "encoder publication identity differs or writes pending"
        );
        if layer < 20 {
            bank.publish_encoder_window(batch, layer, &mut self.producers.windows[layer])?;
        }
        self.publication = None;
        Ok(())
    }
    /// # Safety
    /// Retain cache bank and lane owners until publication or explicit abort.
    pub unsafe fn enqueue_cache_commit(
        &mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        accepted: &[u32],
    ) -> Result<()> {
        ensure!(
            self.publication.is_none(),
            "encoder publication still pending"
        );
        self.progress.commit(batch.identity())?;
        unsafe { self.producers.enqueue_cache_commit(bank, batch, accepted) }
    }
    pub fn poll_cache_commit(&self) -> Result<bool> {
        self.producers.poll_cache_commit()
    }
    pub fn finish_cache_commit(
        &mut self,
        bank: &mut BackboneCache<'_>,
        batch: &CacheBatch,
        accepted: &[u32],
    ) -> Result<()> {
        self.producers.finish_cache_commit(bank, batch, accepted)?;
        self.restart_for(batch.stage());
        Ok(())
    }
    pub fn abort_cache_commit(&mut self, bank: &mut BackboneCache<'_>) -> Result<()> {
        self.progress.invalid = true;
        self.produced = None;
        let result = self.producers.abort_cache_commit(bank);
        self.publication = None;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_backbone_cache::CacheWork;
    use crate::v41_backbone_lane::BackboneLaneWeights;
    use crate::v41_backbone_shared::tp2::Weights as SharedWeights;
    use crate::v41_experts::{tp2::RankWeights, tp2_ffn};
    use crate::v41_index_lane::IndexLaneWeights;
    use crate::v41_target_embedding::TargetEmbeddingWave;
    use ds41rt_transport::{ExpertV2SourceKind, TcpTransportConfig, v41_expert::V41Tp4Roce};
    use std::rc::Rc;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn distributed_layer_zero_tp2_matches_single_gpu_execution_and_handoff() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let devices = [
            Device {
                library: &lib,
                id: 0,
            },
            Device {
                library: &lib,
                id: 1,
            },
        ];
        // Exercise the entire layer on GPU1 and the next-layer handoff to GPU0.
        let map = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer == 0)))?;
        let weights = BackboneLaneWeights::load_distributed(
            &lib,
            &catalog,
            map,
            BackboneLaneWeights::distributed_device_bytes(&lib, &catalog, map)?,
            1024 * 1024,
        )?;
        let cache_weights = CacheProducerWeights::load_distributed(
            &lib,
            &catalog,
            map,
            CacheProducerWeights::distributed_device_bytes(&lib, &catalog, map)?,
            1024 * 1024,
        )?;
        let single_weights = BackboneLaneWeights::load(
            &lib,
            &catalog,
            BackboneLaneWeights::device_bytes(&lib, &catalog)?,
            1024 * 1024,
        )?;
        let single_cache_weights = CacheProducerWeights::load(
            &lib,
            &catalog,
            CacheProducerWeights::device_bytes(&lib, &catalog)?,
            1024 * 1024,
        )?;
        let index_weights = IndexLaneWeights::load(
            &lib,
            &catalog,
            IndexLaneWeights::device_bytes(&lib, &catalog)?,
            1024 * 1024,
        )?;
        let names = ["embed.weight".to_string()];
        let tables = devices
            .iter()
            .map(|d| {
                d.own(|| {
                    NativeRtxTensors::load(
                        &lib,
                        &catalog,
                        &names,
                        NativeRtxTensors::plan(&catalog, &names)?,
                        1024 * 1024,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut embeddings = tables
            .iter()
            .map(|t| {
                t.device.own(|| {
                    TargetEmbeddingWave::new(&lib, t, 16, TargetEmbeddingWave::device_bytes(16)?)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut lane = BackboneLane::new_on_device(
            &weights,
            16,
            BackboneLane::placed_workspace_bytes(&lib, 16)?,
            1,
        )?;
        let mut next = BackboneLane::new_on_device(
            &weights,
            16,
            BackboneLane::placed_workspace_bytes(&lib, 16)?,
            0,
        )?;
        let mut single = BackboneLane::new(
            &single_weights,
            16,
            BackboneLane::workspace_bytes(&lib, 16)?.iter().sum(),
        )?;
        let mut index = IndexLane::new(
            &index_weights,
            16,
            IndexLane::workspace_bytes(&lib, 16)?.iter().sum(),
        )?;
        let mut execution = DistributedExecution::new(
            &cache_weights,
            16,
            PlacedProducerWaves::device_bytes(&lib, map, 16)?,
        )?;
        let mut reference = BackboneExecution::new(
            &single_cache_weights,
            16,
            BackboneExecution::workspace_bytes(&lib, 16)?,
        )?;
        let pages = [2, 2, 2, 4];
        let mut bank = BackboneCache::new_distributed(
            &lib,
            map,
            1,
            pages,
            BackboneCache::distributed_device_bytes(map, 1, pages)?,
        )?;
        let mut single_bank =
            BackboneCache::new(&lib, 1, pages, BackboneCache::device_bytes(1, pages)?)?;
        let lease = bank.begin_request(0, 1)?;
        let single_lease = single_bank.begin_request(0, 1)?;
        let routed = [
            Rc::new(RankWeights::load(devices[0], &catalog, 1, 4_000_000_000)?),
            Rc::new(RankWeights::load(devices[1], &catalog, 1, 4_000_000_000)?),
        ];
        let shared = [
            Rc::new(vec![SharedWeights::load(
                devices[0],
                &catalog,
                0,
                SharedWeights::load_peak_device_bytes(),
            )?]),
            Rc::new(vec![SharedWeights::load(
                devices[1],
                &catalog,
                0,
                SharedWeights::load_peak_device_bytes(),
            )?]),
        ];
        let peers = [1u16, 2, 3, 4].map(|port| std::net::SocketAddr::from(([127, 0, 0, 1], port)));
        let roce = V41Tp4Roce::new(
            peers,
            [1, 2, 3, 4],
            16,
            TcpTransportConfig { timing: crate::v41_native_serve::protocol_v2_timing(),
                timeout: std::time::Duration::from_secs(1),
                max_frame_bytes: 2 * 1024 * 1024,
            },
        )?;
        let mut transport = NativeTp4Wave::new(&lib, roce, NativeTp4Wave::device_bytes(16)?)?;
        transport.install_tp2(tp2_ffn::Wave::new(routed, shared, 1, 16)?)?;
        let mut handoff = crate::v41_block::BlockTransfer::new(devices[1], devices[0])?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        for rows in [1usize, 16, 1] {
            execution.restart_for(CacheStage::Full);
            reference.restart();
            devices[1].run(|| lane.restart())?;
            single.restart()?;
            index.restart()?;
            let batch = bank.plan(&[CacheWork {
                lease,
                tokens: rows as u32,
                kind: ExpertV2SourceKind::Prefill,
            }])?;
            let rb = single_bank.plan(&[CacheWork {
                lease: single_lease,
                tokens: rows as u32,
                kind: ExpertV2SourceKind::Prefill,
            }])?;
            let tokens: Vec<u32> = (0..rows).map(|i| 100 + i as u32).collect();
            let positions: Vec<u64> = (0..rows as u64).collect();
            for gpu in 0..2 {
                devices[gpu].run(|| {
                    let embedded = embeddings[gpu].execute(&tokens, &positions)?;
                    if gpu == 0 {
                        unsafe {
                            single.begin_embedded(&embedded)?;
                        }
                    } else {
                        unsafe {
                            lane.begin_embedded(&embedded)?;
                        }
                    }
                    Ok(())
                })?;
            }
            let mut pending = unsafe { execution.enqueue_production(&bank, &batch, &lane, None)? };
            runtime.block_on(async {
                while !unsafe { pending.poll(&bank, &batch)? } {
                    tokio::task::yield_now().await;
                }
                Ok::<_, anyhow::Error>(())
            })?;
            drop(pending);
            let prepared = unsafe { execution.prepare_layer(&bank, &batch, &mut lane, None)? };
            let mask = vec![0; rows];
            let prepared = if rows == 1 {
                // Unpolled execution still owns queued attention. Its drop must
                // scope cleanup to GPU1 even though the caller is on GPU0.
                drop(unsafe { prepared.execute(&mut transport, 0, &mask) });
                assert_eq!(lib.cuda_get_device()?, 0);
                execution.restart_for(CacheStage::Full);
                devices[1].run(|| {
                    lane.restart()?;
                    let embedded = embeddings[1].output()?;
                    unsafe {
                        lane.begin_embedded(&embedded)?;
                    }
                    Ok(())
                })?;
                let mut pending =
                    unsafe { execution.enqueue_production(&bank, &batch, &lane, None)? };
                runtime.block_on(async {
                    while !unsafe { pending.poll(&bank, &batch)? } {
                        tokio::task::yield_now().await;
                    }
                    Ok::<_, anyhow::Error>(())
                })?;
                drop(pending);
                unsafe { execution.prepare_layer(&bank, &batch, &mut lane, None)? }
            } else {
                prepared
            };

            let completed =
                runtime.block_on(unsafe { prepared.execute(&mut transport, 0, &mask) })?;
            runtime.block_on(unsafe { execution.complete_layer(&batch, &mut lane, completed) })?;
            let prepared = unsafe {
                reference.prepare_layer_cooperative(&single_bank, &rb, &mut single, &mut index)?
            };
            let completed =
                runtime.block_on(unsafe { prepared.execute(&mut transport, 0, &mask) })?;
            runtime.block_on(unsafe {
                reference.complete_layer_cooperative(&rb, &mut single, completed)
            })?;
            let actual = lane.output()?;
            let expected = single.output()?;
            for (a, b) in [
                (actual.residual, expected.residual),
                (actual.pre, expected.pre),
            ] {
                let mut left = vec![0; a.bytes];
                let mut right = vec![0; b.bytes];
                devices[1].run(|| lib.copy_d2h(&mut left, a))?;
                lib.copy_d2h(&mut right, b)?;
                assert_eq!(left, right, "distributed layer-zero TP2 result differs");
            }
            runtime.block_on(unsafe {
                next.get_mut()
                    .import_previous_cooperative(&actual, &mut handoff)
            })?;
            assert_eq!(next.pending_engram()?.0, 1);
            assert!(
                next.prepared_input().is_err(),
                "handoff bypassed layer-one Engram"
            );
            assert!(
                unsafe { execution.enqueue_cache_commit(&bank, &batch, &[rows as u32]) }.is_err(),
                "partial pass published a cache commit"
            );
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
}
