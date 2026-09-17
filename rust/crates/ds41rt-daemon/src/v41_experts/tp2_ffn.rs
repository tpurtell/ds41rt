//! One serving lane's TP2 encoder FFN, including peer inputs and final addition.
use super::tp2::{ExpertWave, RankInputs, RankWeights};
use crate::v41_backbone_shared::tp2::{Wave as SharedWave, Weights as SharedWeights};
use crate::v41_memory::device::{Allocation, Device, Stream};
use anyhow::{Result, ensure};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Bf16Add};
use std::rc::Rc;

struct PeerInputs<'a> {
    values: Allocation<'a>,
    wire: Allocation<'a>,
    ids: Allocation<'a>,
    routing: Allocation<'a>,
}
impl<'a> PeerInputs<'a> {
    fn new(device: Device<'a>, capacity: u32) -> Result<Self> {
        let rows = capacity as usize;
        Ok(Self {
            values: Allocation::new(device, rows * 10240)?,
            wire: Allocation::new(device, rows * 5280)?,
            ids: Allocation::new(device, rows * 24)?,
            routing: Allocation::new(device, rows * 24)?,
        })
    }
    fn buffers(&self) -> [Ds41rtDeviceBuffer; 4] {
        [
            self.values.buffer,
            self.wire.buffer,
            self.ids.buffer,
            self.routing.buffer,
        ]
    }
}

pub(crate) struct Wave<'a> {
    // Drain uploads and final addition before any referenced allocation drops.
    streams: [Stream<'a>; 2],
    peers: [PeerInputs<'a>; 2],
    shared: SharedWave<'a>,
    routed: ExpertWave<'a>,
    output: [Allocation<'a>; 2],
    add: V41Bf16Add<'a>,
    layers: usize,
    capacity: u32,
}
impl<'a> Wave<'a> {
    pub fn exl3_device_bytes(directory: &std::path::Path, library: &ds41rt_ffi::NativeLibrary,
        capacity: u32) -> Result<usize> {
        ExpertWave::exl3_device_bytes(directory, capacity)?
            .checked_add(SharedWave::device_bytes(library, capacity)?)
            .and_then(|bytes| bytes.checked_add(capacity as usize * (10240 + 5280 + 24 + 24 + 10240)))
            .ok_or_else(|| anyhow::anyhow!("TP2 EXL3 FFN workspace overflow"))
    }
    /// Per GPU, per lane. Includes both expert workspaces, peer inputs, and
    /// final output; weights, CUDA modules/streams, and headroom are separate.
    pub fn device_bytes(library: &ds41rt_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        ExpertWave::device_bytes(library, capacity)?
            .checked_add(SharedWave::device_bytes(library, capacity)?)
            .and_then(|bytes| {
                bytes.checked_add(capacity as usize * (10240 + 5280 + 24 + 24 + 10240))
            })
            .ok_or_else(|| anyhow::anyhow!("TP2 FFN workspace overflow"))
    }
    pub fn new(
        routed: [Rc<RankWeights<'a>>; 2],
        shared: [Rc<Vec<SharedWeights<'a>>>; 2],
        layers: usize,
        capacity: u32,
    ) -> Result<Self> {
        let devices = [routed[0].device(), routed[1].device()];
        ensure!(
            devices[0].id == 0
                && devices[1].id == 1
                && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid TP2 FFN devices"
        );
        ensure!(
            (1..=40).contains(&layers) && (1..=4096).contains(&capacity),
            "invalid TP2 FFN layer count/capacity"
        );
        for rank in 0..2 {
            ensure!(
                routed[rank].layers() >= layers
                    && shared[rank].len() >= layers
                    && shared[rank][0].device().id == devices[rank].id
                    && std::ptr::eq(shared[rank][0].device().library, devices[rank].library),
                "TP2 FFN weights do not cover the requested devices/layers"
            );
        }
        for rank in 0..2 {
            devices[rank].run(|| devices[rank].library.cuda_enable_peer(1 - rank as i32))?;
        }
        Ok(Self {
            streams: [Stream::new(devices[0])?, Stream::new(devices[1])?],
            peers: [
                PeerInputs::new(devices[0], capacity)?,
                PeerInputs::new(devices[1], capacity)?,
            ],
            shared: SharedWave::new(shared, capacity)?,
            routed: ExpertWave::new(routed, capacity)?,
            output: [
                Allocation::new(devices[0], capacity as usize * 10240)?,
                Allocation::new(devices[1], capacity as usize * 10240)?,
            ],
            add: devices[0].library.v41_bf16_add()?,
            layers,
            capacity,
        })
    }
    pub fn contains(&self, layer: usize) -> bool {
        layer < self.layers
    }

    pub fn contains_shared(&self, layer: usize) -> bool {
        self.shared.contains(layer)
    }

    /// Execute only the shared TP2 contribution while routed experts
    /// run on Sparks. Reuses the encoder FFN's lane-local input/shared storage.
    /// # Safety
    /// Completed normalized input remains immutable through completion or drain.
    pub async unsafe fn execute_shared(
        &mut self,
        layer: usize,
        rows: u32,
        values: Ds41rtDeviceBuffer,
    ) -> Result<Ds41rtDeviceBuffer> {
        unsafe { self.execute_shared_on(layer, rows, values, values.device_id as usize).await }
    }

    /// Same input lifetime as `execute_shared`; select the reduction owner so
    /// Spark collection can remain on its transport GPU for encoder layers too.
    /// # Safety
    /// Input storage remains complete, immutable and live through completion or drain.
    pub async unsafe fn execute_shared_on(&mut self, layer: usize, rows: u32,
        values: Ds41rtDeviceBuffer, destination: usize) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            self.contains_shared(layer)
                && destination < 2
                && rows > 0
                && rows <= self.capacity
                && matches!(values.device_id, 0 | 1)
                && !values.ptr.is_null()
                && values.bytes >= rows as usize * 10240,
            "TP2 shared input layer/device/extent differs"
        );
        let local = values.device_id as usize;
        let remote = 1 - local;
        let upload = &self.streams[remote];
        struct Drain<'s, 'a> {
            stream: &'s Stream<'a>,
            complete: bool,
        }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    if let Err(error) = self.stream.drain() {
                        tracing::error!(%error,"draining TP2 shared input upload");
                    }
                }
            }
        }
        let mut guard = Drain {
            stream: upload,
            complete: false,
        };
        upload.wait().await?;
        let peer = self.peers[remote].values.buffer;
        upload.device.run(|| unsafe {
            upload
                .device
                .library
                .copy_peer_async(peer, values, rows as usize * 10240, upload.raw)
        })?;
        let mut inputs = [values; 2];
        inputs[remote] = peer;
        let output = unsafe {
            self.shared
                .execute(
                    layer,
                    rows,
                    destination,
                    inputs,
                    [&self.streams[0], &self.streams[1]],
                )
                .await?
        };
        guard.complete = true;
        Ok(output)
    }

    /// Return a completed remote-expert reduction to its block's GPU using
    /// lane-owned scratch. Stream::wait drains this copy on cancellation.
    /// # Safety
    /// Source storage remains immutable and live until the transfer completes or drains.
    pub async unsafe fn return_result(&mut self, source: Ds41rtDeviceBuffer,
        destination: usize, rows: u32) -> Result<Ds41rtDeviceBuffer> {
        ensure!(destination < 2 && rows > 0 && rows <= self.capacity
            && source.bytes >= rows as usize * 10240, "invalid TP2 result transfer");
        let stream = &self.streams[destination];
        let mut output = self.output[destination].buffer;
        output.bytes = rows as usize * 10240;
        let queued = stream.device.run(|| unsafe {
            stream.device.library.copy_peer_async(output, source, output.bytes, stream.raw)
        });
        let drained = stream.wait().await;
        queued.and(drained)?;
        Ok(output)
    }

    /// # Safety
    /// Local inputs are complete, immutable, and retained through return or
    /// cancellation. They all represent the same normalized FFN rows. Each lane
    /// owns a distinct Wave. Returned storage is valid until its next execute.
    pub async unsafe fn execute(
        &mut self,
        layer: usize,
        rows: u32,
        values: Ds41rtDeviceBuffer,
        wire: Ds41rtDeviceBuffer,
        ids: Ds41rtDeviceBuffer,
        routing: Ds41rtDeviceBuffer,
    ) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            self.contains(layer) && rows > 0 && rows <= self.capacity,
            "TP2 FFN layer/rows unavailable"
        );
        ensure!(
            matches!(values.device_id, 0 | 1),
            "invalid TP2 FFN input device"
        );
        let local = values.device_id as usize;
        let remote = 1 - local;
        let inputs = [values, wire, ids, routing];
        let widths = [10240, 5280, 24, 24];
        for (buffer, width) in inputs.into_iter().zip(widths) {
            ensure!(
                buffer.device_id == local as i32
                    && !buffer.ptr.is_null()
                    && buffer.bytes >= rows as usize * width,
                "TP2 FFN input extent/device mismatch"
            );
        }
        struct Drain<'s, 'a> {
            stream: &'s Stream<'a>,
            complete: bool,
        }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    if let Err(error) = self.stream.drain() {
                        tracing::error!(%error, "draining TP2 FFN input upload");
                    }
                }
            }
        }
        let upload = &self.streams[remote];
        let mut guard = Drain {
            stream: upload,
            complete: false,
        };
        // Do not submit DMA behind an unresolved stream dependency: a blocked
        // copy packet can hold up independent lanes on the shared copy engine.
        upload.wait().await?;
        let peer = self.peers[remote].buffers();
        upload.device.run(|| {
            for ((source, destination), width) in inputs.into_iter().zip(peer).zip(widths) {
                unsafe {
                    upload.device.library.copy_peer_async(
                        destination,
                        source,
                        rows as usize * width,
                        upload.raw,
                    )?;
                }
            }
            Ok(())
        })?;
        let mut ranks = [inputs; 2];
        ranks[remote] = peer;
        // Each backend records its own producer event after the queued copies;
        // local compute can begin before peer input transfers finish.
        let (routed, shared) = tokio::try_join!(
            unsafe {
                self.routed.execute(
                    layer,
                    rows,
                    local,
                    std::array::from_fn(|rank| RankInputs {
                        wire: ranks[rank][1],
                        ids: ranks[rank][2],
                        routing: ranks[rank][3],
                        producer: &self.streams[rank],
                    }),
                )
            },
            unsafe {
                self.shared.execute(
                    layer,
                    rows,
                    local,
                    [ranks[0][0], ranks[1][0]],
                    [&self.streams[0], &self.streams[1]],
                )
            }
        )?;
        guard.complete = true;
        let stream = &self.streams[local];
        let output = self.output[local].buffer;
        let queued = stream.device.run(|| unsafe {
            self.add
                .launch(routed, shared, output, rows as usize * 5120, stream.raw)
        });
        let drained = stream.wait().await;
        queued.and(drained)?;
        let mut result = output;
        result.bytes = rows as usize * 10240;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires combined TP2 DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT, two GPUs"]
    fn real_tp2_ffn_broadcast_and_combine_independent_lanes() -> Result<()> {
        let lib = unsafe { ds41rt_ffi::NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
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
        let routed = [
            Rc::new(RankWeights::load(devices[0], &catalog, 1, 4_000_000_000)?),
            Rc::new(RankWeights::load(devices[1], &catalog, 1, 4_000_000_000)?),
        ];
        let shared: [Rc<Vec<SharedWeights<'_>>>; 2] = [0, 1]
            .map(|rank| {
                (0..40)
                    .map(|layer| {
                        SharedWeights::load(
                            devices[rank],
                            &catalog,
                            layer,
                            SharedWeights::load_peak_device_bytes(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(Rc::new)
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .ok()
            .expect("two shared ranks");
        let mut lanes = [
            Wave::new(routed.clone(), shared.clone(), 1, 16)?,
            Wave::new(routed.clone(), shared.clone(), 1, 16)?,
        ];
        let mut reference_routed = ExpertWave::new(routed, 16)?;
        let mut reference_shared = SharedWave::new(shared, 16)?;
        let inputs = [
            PeerInputs::new(devices[0], 16)?,
            PeerInputs::new(devices[1], 16)?,
        ];
        let reference_inputs = [
            PeerInputs::new(devices[0], 16)?,
            PeerInputs::new(devices[1], 16)?,
        ];
        let producers = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        assert!(lanes[0].contains_shared(39));
        assert!(!lanes[0].contains_shared(40));
        assert!(!lanes[0].contains(20));
        for layer in [20, 39] {
            for rows in [1u32, 16, 1] {
                for local in 0..2 {
                    let host: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| {
                            let value = ((i + layer + local) % 17) as f32 / 32.0 - 0.25;
                            ((value.to_bits() >> 16) as u16).to_ne_bytes()
                        })
                        .collect();
                    for rank in 0..2 {
                        devices[rank].run(|| {
                            devices[rank]
                                .library
                                .copy_h2d(reference_inputs[rank].values.buffer, &host)
                        })?;
                    }
                    devices[local].run(|| {
                        devices[local]
                            .library
                            .copy_h2d(inputs[local].values.buffer, &host)
                    })?;
                    let actual = runtime.block_on(unsafe {
                        lanes[local].execute_shared(layer, rows, inputs[local].values.buffer)
                    })?;
                    let expected = runtime.block_on(unsafe {
                        reference_shared.execute(
                            layer,
                            rows,
                            local,
                            [
                                reference_inputs[0].values.buffer,
                                reference_inputs[1].values.buffer,
                            ],
                            [&producers[0], &producers[1]],
                        )
                    })?;
                    devices[local].run(|| {
                        let mut left = vec![0; actual.bytes];
                        let mut right = vec![0; expected.bytes];
                        devices[local].library.copy_d2h(&mut left, actual)?;
                        devices[local].library.copy_d2h(&mut right, expected)?;
                        assert_eq!(
                            left, right,
                            "decoder shared TP2 broadcast/reduction differs"
                        );
                        Ok(())
                    })?;
                    let returned = runtime.block_on(async {
                        let peer = unsafe { lanes[local].execute_shared_on(layer, rows,
                            inputs[local].values.buffer, 1-local).await? };
                        unsafe { lanes[local].return_result(peer, local, rows).await }
                    })?;
                    devices[local].run(|| {
                        let mut left = vec![0; returned.bytes];
                        let mut right = vec![0; expected.bytes];
                        lib.copy_d2h(&mut left, returned)?;
                        lib.copy_d2h(&mut right, expected)?;
                        ensure!(left == right, "peer-owned shared reduction and return differs");
                        Ok(())
                    })?;
                    assert_eq!(lib.cuda_get_device()?, 0);
                }
            }
        }
        for (rows, bf16, fp8) in [
            (1, 0u16, 0u8),
            (16, 0x3f80, 0x38),
            (1, 0x3f00, 0x30),
            (16, 0, 0),
        ] {
            let frames: [[Vec<u8>; 4]; 2] = std::array::from_fn(|rank| {
                let negative = rank == 1 && bf16 != 0;
                let values: Vec<u8> = (bf16 ^ if negative { 0x8000 } else { 0 })
                    .to_ne_bytes()
                    .into_iter()
                    .cycle()
                    .take(16 * 10240)
                    .collect();
                let mut wire = vec![0; 16 * 5280];
                for row in wire.chunks_exact_mut(5280) {
                    row[..5120].fill(fp8 ^ if negative { 0x80 } else { 0 });
                    row[5120..].fill(127);
                }
                let ids: Vec<u8> = (0..96i32)
                    .flat_map(|i| (i % 6 + rank as i32 * 6).to_ne_bytes())
                    .collect();
                let routing: Vec<u8> = (0..96)
                    .flat_map(|i| {
                        ([0.1f32, 0.15, 0.2, 0.25, 0.1, 0.2][(i + rank) % 6]).to_ne_bytes()
                    })
                    .collect();
                [values, wire, ids, routing]
            });
            for rank in 0..2 {
                devices[rank].run(|| {
                    for (buffer, bytes) in inputs[rank].buffers().into_iter().zip(&frames[rank]) {
                        lib.copy_h2d(buffer, bytes)?;
                    }
                    Ok(())
                })?;
            }
            let [left, right] = &mut lanes;
            let a = inputs[0].buffers();
            let b = inputs[1].buffers();
            let (x, y) = runtime.block_on(async {
                tokio::join!(
                    unsafe { left.execute(0, rows, a[0], a[1], a[2], a[3]) },
                    unsafe { right.execute(0, rows, b[0], b[1], b[2], b[3]) }
                )
            });
            let mut actual = [
                vec![0; rows as usize * 10240],
                vec![0; rows as usize * 10240],
            ];
            for (rank, buffer) in [x?, y?].into_iter().enumerate() {
                devices[rank].run(|| lib.copy_d2h(&mut actual[rank], buffer))?;
            }
            assert_eq!(
                actual[0] != actual[1],
                bf16 != 0,
                "different lane inputs must remain independent"
            );
            for lane in 0..2 {
                for rank in 0..2 {
                    devices[rank].run(|| {
                        for (buffer, bytes) in reference_inputs[rank]
                            .buffers()
                            .into_iter()
                            .zip(&frames[lane])
                        {
                            lib.copy_h2d(buffer, bytes)?;
                        }
                        Ok(())
                    })?;
                }
                let route = runtime.block_on(unsafe {
                    reference_routed.execute(
                        0,
                        rows,
                        0,
                        std::array::from_fn(|rank| RankInputs {
                            wire: reference_inputs[rank].wire.buffer,
                            ids: reference_inputs[rank].ids.buffer,
                            routing: reference_inputs[rank].routing.buffer,
                            producer: &producers[rank],
                        }),
                    )
                })?;
                let shared = runtime.block_on(unsafe {
                    reference_shared.execute(
                        0,
                        rows,
                        0,
                        [
                            reference_inputs[0].values.buffer,
                            reference_inputs[1].values.buffer,
                        ],
                        [&producers[0], &producers[1]],
                    )
                })?;
                let mut route_bytes = vec![0; actual[0].len()];
                let mut shared_bytes = vec![0; actual[0].len()];
                lib.copy_d2h(&mut route_bytes, route)?;
                lib.copy_d2h(&mut shared_bytes, shared)?;
                for ((a, b), out) in route_bytes
                    .chunks_exact(2)
                    .zip(shared_bytes.chunks_exact(2))
                    .zip(actual[lane].chunks_exact(2))
                {
                    let decode =
                        |v: &[u8]| f32::from_bits((u16::from_ne_bytes([v[0], v[1]]) as u32) << 16);
                    let sum = decode(a) + decode(b);
                    assert!(sum.is_finite());
                    let bits = sum.to_bits();
                    let expected = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
                    assert_eq!(u16::from_ne_bytes([out[0], out[1]]), expected);
                }
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        for capacity in [1, 16, 80, 256, 1024, 4096] {
            eprintln!(
                "TP2 FFN capacity={capacity} workspace_bytes_per_gpu_per_lane={}",
                Wave::device_bytes(&lib, capacity)?
            );
        }
        // Deliberately hold one lane's upload while the other completes real
        // kernels. A bounded release avoids hanging even if a join regresses.
        use std::ffi::c_void;
        use std::{
            future::Future,
            sync::atomic::{AtomicBool, Ordering},
            task::{Context, Poll, Waker},
        };
        unsafe extern "C" fn hold(data: *mut c_void) {
            let release = unsafe { &*data.cast::<AtomicBool>() };
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        struct CudaLibrary(*mut c_void);
        impl Drop for CudaLibrary {
            fn drop(&mut self) {
                unsafe {
                    libc::dlclose(self.0);
                }
            }
        }
        let cuda =
            CudaLibrary(unsafe { libc::dlopen(c"libcudart.so.13".as_ptr(), libc::RTLD_NOW) });
        ensure!(!cuda.0.is_null(), "CUDA runtime unavailable");
        let symbol = unsafe { libc::dlsym(cuda.0, c"cudaLaunchHostFunc".as_ptr()) };
        ensure!(!symbol.is_null(), "CUDA callback API unavailable");
        let launch: unsafe extern "C" fn(
            *mut c_void,
            unsafe extern "C" fn(*mut c_void),
            *mut c_void,
        ) -> i32 = unsafe { std::mem::transmute(symbol) };
        let release = AtomicBool::new(false);
        struct Release<'a>(&'a AtomicBool);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _release = Release(&release);
        let [left, right] = &mut lanes;
        devices[1].run(|| {
            ensure!(
                unsafe {
                    launch(
                        left.streams[1].raw,
                        hold,
                        (&release as *const AtomicBool).cast_mut().cast(),
                    )
                } == 0,
                "could not stall TP2 upload"
            );
            Ok(())
        })?;
        let a = inputs[0].buffers();
        let b = inputs[1].buffers();
        let _entered = runtime.enter();
        let mut blocked = Box::pin(unsafe { left.execute(0, 16, a[0], a[1], a[2], a[3]) });
        let _unwind_release = Release(&release);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(blocked.as_mut().poll(&mut context), Poll::Pending));
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(500));
                release.store(true, Ordering::Release);
            });
            for rank in 0..2 {
                devices[rank].run(|| unsafe {
                    lib.copy_peer_async(
                        right.output[rank].buffer,
                        inputs[1 - rank].values.buffer,
                        10240,
                        right.streams[rank].raw,
                    )
                })?;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            for rank in 0..2 {
                assert!(
                    devices[rank]
                        .run(|| unsafe { lib.cuda_stream_query(right.streams[rank].raw) })?,
                    "unrelated peer copy stalled on GPU {rank}"
                );
            }
            let started = std::time::Instant::now();
            let mut other = Box::pin(unsafe { right.execute(0, 16, b[0], b[1], b[2], b[3]) });
            let initial = other.as_mut().poll(&mut context);
            eprintln!(
                "independent lane first poll {:?}, pending={}",
                started.elapsed(),
                initial.is_pending()
            );
            let result = match initial {
                Poll::Ready(result) => result,
                Poll::Pending => runtime.block_on(other),
            };
            eprintln!("independent lane completed {:?}", started.elapsed());
            let independent = !release.load(Ordering::Acquire);
            drop(blocked); // Cancellation must drain before releasing input owners.
            assert!(release.load(Ordering::Acquire));
            result?;
            assert!(
                independent,
                "one lane's stalled input upload held the other lane"
            );
            Ok(())
        })?;
        // Reuse the cancelled lane's same streams, events, and scratch.
        runtime.block_on(unsafe { left.execute(0, 16, a[0], a[1], a[2], a[3]) })?;
        release.store(false, Ordering::Release);
        devices[1].run(|| {
            ensure!(
                unsafe {
                    launch(
                        left.streams[1].raw,
                        hold,
                        (&release as *const AtomicBool).cast_mut().cast(),
                    )
                } == 0,
                "stalling shared upload failed"
            );
            Ok(())
        })?;
        let mut blocked_shared = Box::pin(unsafe { left.execute_shared(20, 16, a[0]) });
        let shared_unwind_release = Release(&release);
        assert!(matches!(
            blocked_shared.as_mut().poll(&mut context),
            Poll::Pending
        ));
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(500));
                release.store(true, Ordering::Release);
            });
            let result = runtime.block_on(unsafe { right.execute_shared(39, 1, b[0]) });
            let independent = !release.load(Ordering::Acquire);
            drop(blocked_shared);
            result?;
            assert!(independent, "decoder shared work joined the other lane");
            Ok(())
        })?;
        drop(shared_unwind_release);
        runtime.block_on(unsafe { left.execute_shared(20, 16, a[0]) })?;
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}
