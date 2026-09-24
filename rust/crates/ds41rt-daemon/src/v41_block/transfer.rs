//! Move completed adjacent-layer state directly into the next GPU's workspace.
use super::*;
use crate::v41_backbone_hc::BackboneHcWeights;
use crate::v41_memory::device::{Device, Stream};

/// One directed transfer stream per request lane. No intermediate device or
/// host buffer is needed: both copies land in the destination block inputs.
pub(crate) struct BlockTransfer<'a> {
    source: Device<'a>,
    destination: Stream<'a>,
}
impl<'a> BlockTransfer<'a> {
    pub fn new(source: Device<'a>, destination: Device<'a>) -> Result<Self> {
        ensure!(
            source.id != destination.id,
            "block transfer requires different GPUs"
        );
        source.run(|| source.library.cuda_enable_peer(destination.id))?;
        destination.run(|| destination.library.cuda_enable_peer(source.id))?;
        Ok(Self {
            source,
            destination: Stream::new(destination)?,
        })
    }
    /// # Safety
    /// Sources are complete and all buffers remain exclusively owned until the
    /// copy finishes or cancellation drains. Never queue behind an unready
    /// producer event: doing so can stall otherwise independent peer DMA work.
    async unsafe fn copy(
        &mut self,
        sources: [Ds41rtDeviceBuffer; 2],
        destinations: [Ds41rtDeviceBuffer; 2],
    ) -> Result<()> {
        ensure!(
            sources
                .iter()
                .zip(destinations)
                .all(|(s, d)| s.device_id == self.source.id
                    && d.device_id == self.destination.device.id
                    && s.bytes <= d.bytes),
            "block transfer devices or extents differ"
        );
        let device = self.destination.device;
        // Peer DMA never waits on an unresolved event; settle chained producers.
        crate::v41_memory::chain::settle(device.library)?;
        let queued = device.run(|| {
            for (source, destination) in sources.into_iter().zip(destinations) {
                unsafe {
                    device.library.copy_peer_async(
                        destination,
                        source,
                        source.bytes,
                        self.destination.raw,
                    )?;
                }
            }
            Ok(())
        });
        if let Err(error) = queued {
            self.destination.drain()?;
            return Err(error);
        }
        // Stream::wait retains and drains the stream on failure/cancellation.
        self.destination.wait().await
    }
}

impl<'w, 'a> BackboneBlockWave<'w, 'a> {
    /// Rebind this GPU's reusable block and import a completed adjacent layer.
    /// Publication occurs only after both peer copies finish. Engram boundaries
    /// remain gated exactly as in the ordinary adjacent-layer path.
    /// # Safety
    /// The previous output is complete; its buffers and this destination have
    /// no conflicting users. Retain their owners until completion/cancellation.
    pub async unsafe fn import_previous_cooperative(
        &mut self,
        next: &'w BackboneHcWeights<'a>,
        previous: &BlockOutput<'_>,
        transfer: &mut BlockTransfer<'a>,
    ) -> Result<()> {
        self.reset();
        let rows = previous.tokens.len();
        ensure!(
            previous.layer < 39
                && next.layer() == previous.layer + 1
                && previous.binding.layer() == previous.layer
                && rows > 0
                && rows <= self.capacity
                && previous.tokens.iter().all(|&p| p < 1048576)
                && previous.residual.bytes == rows * 40960
                && previous.pre.bytes == rows * 16,
            "imported block layer, tokens or extents differ: previous={}, next={}, binding={}, rows={rows}, capacity={}, residual_bytes={}, pre_bytes={}",
            previous.layer, next.layer(), previous.binding.layer(), self.capacity,
            previous.residual.bytes, previous.pre.bytes
        );
        let device = transfer.destination.device;
        ensure!(
            self.inputs().iter().all(|b| b.device_id == device.id),
            "block transfer destination differs"
        );
        device.run(|| {
            let [attention, ffn] = next.prepare_bindings(&self.attention, &self.ffn)?;
            unsafe {
                self.attention.install_binding(attention);
                self.ffn.install_binding(ffn);
            }
            Ok(())
        })?;
        unsafe {
            transfer
                .copy([previous.residual, previous.pre], self.inputs())
                .await?;
        }
        self.layer = next.layer();
        self.tokens.extend_from_slice(previous.tokens);
        self.phase = Phase::Prepared(previous.binding, rows, ![1, 14].contains(&self.layer));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_attention_query::{AttentionQueryWave, AttentionQueryWeights};
    use crate::v41_memory::device::Allocation;

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn imported_block_state_matches_direct_query_and_cancellation_reuses_storage() -> Result<()> {
        use std::{
            future::Future,
            sync::atomic::{AtomicBool, Ordering},
            task::{Context as TaskContext, Poll, Waker},
        };
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
        let placement =
            crate::v41_backbone_cache::CachePlacement::new(std::array::from_fn(|layer| {
                usize::from(layer >= 14)
            }))?;
        let lane_weights = crate::v41_backbone_lane::BackboneLaneWeights::load_distributed(
            &lib,
            &catalog,
            placement,
            crate::v41_backbone_lane::BackboneLaneWeights::distributed_device_bytes(
                &lib, &catalog, placement,
            )?,
            1024 * 1024,
        )?;
        let lane_bytes = crate::v41_backbone_lane::BackboneLane::placed_workspace_bytes(&lib, 16)?;
        let mut placed_lane = crate::v41_backbone_lane::BackboneLane::new_on_device(
            &lane_weights,
            16,
            lane_bytes,
            1,
        )?;
        let placed_engram_weights = crate::v41_engram::placement::PlacedEngramWeights::load(
            &lib,
            &catalog,
            placement,
            crate::v41_engram::placement::PlacedEngramWeights::device_bytes(
                &lib, &catalog, placement,
            )?,
            1024 * 1024,
        )?;
        let mut placed_engram = crate::v41_engram::placement::PlacedEngram::new(
            &placed_engram_weights,
            16,
            crate::v41_engram::placement::PlacedEngram::device_bytes(&lib, placement, 16)?,
        )?;
        let reference_engram_weights = devices[1].own(|| {
            crate::v41_engram::layer::EngramLayerWeights::load(
                &lib,
                &catalog,
                1,
                crate::v41_engram::layer::EngramLayerWeights::device_bytes(&lib, &catalog, 1)?,
                1024 * 1024,
            )
        })?;
        let mut reference_gate = devices[1].own(|| {
            crate::v41_engram::layer::EngramGate::new(
                &reference_engram_weights,
                16,
                crate::v41_engram::layer::EngramGate::device_bytes(&lib, 16)?,
            )
        })?;
        let mut reference_upload = devices[1].own(|| {
            crate::v41_engram::EngramDeviceRows::new(
                &lib,
                16,
                crate::v41_engram::EngramDeviceRows::device_bytes(16)?,
            )
        })?;
        let hc = devices
            .iter()
            .map(|d| {
                d.own(|| {
                    BackboneHcWeights::load(
                        &lib,
                        &catalog,
                        20,
                        BackboneHcWeights::device_bytes(&catalog, 20)?,
                        1024 * 1024,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let engram_hc = devices[1].own(|| {
            BackboneHcWeights::load(
                &lib,
                &catalog,
                14,
                BackboneHcWeights::device_bytes(&catalog, 14)?,
                1024 * 1024,
            )
        })?;
        let qw = devices
            .iter()
            .map(|d| {
                d.own(|| {
                    AttentionQueryWeights::load(
                        &lib,
                        &catalog,
                        20,
                        AttentionQueryWeights::device_bytes(&lib, &catalog, 20)?,
                        1024 * 1024,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut blocks = hc
            .iter()
            .map(|w| {
                w.device
                    .own(|| w.block(16, BackboneBlockWave::device_bytes(16)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut queries = qw
            .iter()
            .map(|w| {
                w.device
                    .own(|| w.wave(16, AttentionQueryWave::device_bytes(&lib, 16)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let residual = Allocation::new(devices[0], 16 * 40960)?;
        let pre = Allocation::new(devices[0], 16 * 16)?;
        let mut transfer = BlockTransfer::new(devices[0], devices[1])?;
        let addresses = blocks[1].inputs().map(|b| b.ptr);
        let token_capacity = blocks[1].tokens.capacity();
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // Deterministically hold the transfer stream to exercise a pending drop.
        unsafe extern "C" fn hold(data: *mut std::ffi::c_void) {
            let release = unsafe { &*data.cast::<AtomicBool>() };
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        struct Runtime(*mut std::ffi::c_void);
        impl Drop for Runtime {
            fn drop(&mut self) {
                unsafe {
                    libc::dlclose(self.0);
                }
            }
        }
        let cuda = Runtime(unsafe { libc::dlopen(c"libcudart.so.13".as_ptr(), libc::RTLD_NOW) });
        ensure!(!cuda.0.is_null(), "CUDA runtime unavailable");
        let symbol = unsafe { libc::dlsym(cuda.0, c"cudaLaunchHostFunc".as_ptr()) };
        ensure!(!symbol.is_null(), "CUDA host callback API unavailable");
        let launch: unsafe extern "C" fn(
            *mut std::ffi::c_void,
            unsafe extern "C" fn(*mut std::ffi::c_void),
            *mut std::ffi::c_void,
        ) -> i32 = unsafe { std::mem::transmute(symbol) };
        for (seed, rows) in [(0, 16usize), (7, 1), (11, 16)] {
            let host: Vec<u8> = (0..rows * 20480)
                .flat_map(|i| {
                    let value = ((i + seed) % 31) as f32 / 32.0 - 0.5;
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                })
                .collect();
            lib.copy_h2d(residual.buffer, &host)?;
            lib.copy_h2d(pre.buffer, &0.25f32.to_ne_bytes().repeat(rows * 4))?;
            let mut r = residual.buffer;
            r.bytes = rows * 40960;
            let mut p = pre.buffer;
            p.bytes = rows * 16;
            let tokens: Vec<_> = (100..100 + rows as u64).collect();
            let mut previous = BlockOutput {
                binding: QueryBinding::new(19)?,
                residual: r,
                pre: p,
                layer: 19,
                tokens: &tokens,
            };
            if seed == 0 {
                let release = AtomicBool::new(false);
                struct Release<'a>(&'a AtomicBool);
                impl Drop for Release<'_> {
                    fn drop(&mut self) {
                        self.0.store(true, Ordering::Release);
                    }
                }
                let release_guard = Release(&release);
                devices[1].run(|| {
                    ensure!(
                        unsafe {
                            launch(
                                transfer.destination.raw,
                                hold,
                                (&release as *const AtomicBool).cast_mut().cast(),
                            )
                        } == 0,
                        "queueing transfer gate failed"
                    );
                    Ok(())
                })?;
                let mut work = Box::pin(unsafe {
                    blocks[1].get_mut().import_previous_cooperative(
                        &hc[1],
                        &previous,
                        &mut transfer,
                    )
                });
                let mut context = TaskContext::from_waker(Waker::noop());
                let pending = matches!(work.as_mut().poll(&mut context), Poll::Pending);
                drop(release_guard); // Release before any assertion/drop can drain.
                drop(work);
                assert!(pending, "transfer did not suspend behind its gate");
                assert!(blocks[1].prepared_input().is_err());
                assert_eq!(lib.cuda_get_device()?, 0);
            }
            runtime.block_on(unsafe {
                blocks[1]
                    .get_mut()
                    .import_previous_cooperative(&hc[1], &previous, &mut transfer)
            })?;
            assert_eq!(
                blocks[1].prepared_input()?.previous_binding(),
                previous.binding
            );
            assert_eq!(blocks[1].inputs().map(|b| b.ptr), addresses);
            assert_eq!(blocks[1].tokens.capacity(), token_capacity);
            devices[0].run(|| blocks[0].initialize_previous(&previous))?;
            let mut outputs = Vec::new();
            for gpu in 0..2 {
                outputs.push(devices[gpu].run(|| {
                    let output =
                        unsafe { blocks[gpu].begin_prepared_attention(&mut queries[gpu])? };
                    [
                        output.hidden,
                        output.raw_rank,
                        output.normalized_rank,
                        output.projected,
                        output.rotated,
                    ]
                    .into_iter()
                    .map(|b| {
                        let mut bytes = vec![0; b.bytes];
                        lib.copy_d2h(&mut bytes, b)?;
                        Ok(bytes)
                    })
                    .collect::<Result<Vec<_>>>()
                })?);
                assert_eq!(lib.cuda_get_device()?, 0);
            }
            assert!(
                outputs[0] == outputs[1],
                "imported residual produces different attention query"
            );
            runtime.block_on(unsafe {
                placed_lane
                    .get_mut()
                    .import_previous_cooperative(&previous, &mut transfer)
            })?;
            let placed_query = runtime.block_on(
                devices[1].future(unsafe { placed_lane.get_mut().begin_prepared_cooperative() }),
            )?;
            let placed_output = devices[1].run(|| {
                [
                    placed_query.hidden,
                    placed_query.raw_rank,
                    placed_query.normalized_rank,
                    placed_query.projected,
                    placed_query.rotated,
                ]
                .into_iter()
                .map(|b| {
                    let mut bytes = vec![0; b.bytes];
                    lib.copy_d2h(&mut bytes, b)?;
                    Ok(bytes)
                })
                .collect::<Result<Vec<_>>>()
            })?;
            assert!(
                outputs[0] == placed_output,
                "placed backbone lane handoff query differs"
            );
            previous.layer = 13;
            previous.binding = QueryBinding::new(13)?;
            runtime.block_on(unsafe {
                blocks[1].get_mut().import_previous_cooperative(
                    &engram_hc,
                    &previous,
                    &mut transfer,
                )
            })?;
            assert!(
                blocks[1].prepared_input().is_err(),
                "layer 14 bypassed required Engram"
            );
            assert_eq!(blocks[1].pending_engram()?.0, 14);
            runtime.block_on(unsafe {
                placed_lane
                    .get_mut()
                    .import_previous_cooperative(&previous, &mut transfer)
            })?;
            let gathered_weights = vec![0x38; rows * 24 * 256];
            let gathered_scales = vec![127; rows * 24 * 8];
            let mask: Vec<u8> = (0..rows).map(|i| (i % 2) as u8).collect();
            let gathered = ds41rt_loader::EngramGatherView {
                encoding: ds41rt_loader::EngramEncoding::Fp8, global_scale: 1.0,
                weights: &gathered_weights,
                scales: &gathered_scales,
                text_mask: &mask,
                rows,
                layer_index: 1,
            };
            runtime.block_on(unsafe { placed_engram.apply(&mut placed_lane, &gathered) })?;
            runtime.block_on(devices[1].future(async {
                let uploaded = reference_upload.upload_cooperative(&gathered).await?;
                unsafe {
                    blocks[1]
                        .apply_engram_cooperative(&mut reference_gate, &uploaded)
                        .await
                }
            }))?;
            let actual = placed_lane.prepared_input()?;
            let expected = blocks[1].prepared_input()?;
            for (a, b) in [
                (actual.residual, expected.residual),
                (actual.pre, expected.pre),
            ] {
                let mut left = vec![0; a.bytes];
                let mut right = vec![0; b.bytes];
                devices[1].run(|| {
                    lib.copy_d2h(&mut left, a)?;
                    lib.copy_d2h(&mut right, b)
                })?;
                assert_eq!(
                    left, right,
                    "placed Engram residual differs from ordinary gate"
                );
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}
