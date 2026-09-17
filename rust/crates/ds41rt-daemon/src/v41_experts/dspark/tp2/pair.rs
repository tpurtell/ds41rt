//! Capture-compatible draft expert fork/join; transformer owner stays on RTX1.
use super::rank::{Wave, Weights};
use crate::v41_memory::device::{Allocation, Device, Event, Stream};
use anyhow::{ensure, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41ExpertInputQuantizer, V41PeerCopy, V41RouteReducer,
};
use std::{ffi::c_void, rc::Rc};

pub(super) struct Pair<'a> {
    // External root stream and its graphs must be drained/destroyed by caller
    // first. Our peer stream drains before any of its referenced allocations.
    peer: Stream<'a>,
    ready: Event<'a>,
    done: Event<'a>,
    ranks: [Wave<'a>; 2],
    wire: Allocation<'a>,
    peer_inputs: [Allocation<'a>; 3],
    peer_output: Allocation<'a>,
    output: Allocation<'a>,
    copies: [V41PeerCopy<'a>; 2],
    quantizer: V41ExpertInputQuantizer<'a>,
    reducer: V41RouteReducer<'a>,
    owner: Device<'a>,
    capacity: u32,
}
impl<'a> Pair<'a> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<[usize; 2]> {
        let scratch = Wave::device_bytes(library, capacity)?;
        let rows = capacity as usize;
        Ok([
            scratch + rows * (5280 + 12 + 12),
            scratch + rows * (5280 + 3 * 5120 * 4 + 5120 * 2),
        ])
    }
    pub fn new(weights: [Rc<Weights<'a>>; 2], capacity: u32) -> Result<Self> {
        let devices = weights.each_ref().map(|w| w.device);
        ensure!(
            devices[0].id == 0
                && devices[1].id == 1
                && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid draft TP2 weights"
        );
        let owner = devices[1];
        for device in devices {
            device.run(|| device.library.cuda_enable_peer(1 - device.id))?;
        }
        let copies = [
            devices[0].run(|| owner.library.v41_peer_copy())?,
            owner.run(|| owner.library.v41_peer_copy())?,
        ];
        let quantizer = owner.run(|| owner.library.v41_expert_input_quantizer())?;
        let reducer = owner.library.v41_route_reducer()?;
        let [left, right] = weights;
        let ranks = [Wave::new(left, capacity)?, Wave::new(right, capacity)?];
        let rows = capacity as usize;
        Ok(Self {
            peer: Stream::new(devices[0])?,
            ready: Event::new(owner)?,
            done: Event::new(devices[0])?,
            ranks,
            wire: Allocation::new(owner, rows * 5280)?,
            peer_inputs: [
                Allocation::new(devices[0], rows * 5280)?,
                Allocation::new(devices[0], rows * 12)?,
                Allocation::new(devices[0], rows * 12)?,
            ],
            peer_output: Allocation::new(owner, rows * 3 * 5120 * 4)?,
            output: Allocation::new(owner, rows * 5120 * 2)?,
            copies,
            quantizer,
            reducer,
            owner,
            capacity,
        })
    }
    pub fn output(&self) -> Ds41rtDeviceBuffer {
        self.output.buffer
    }
    /// Quantize once on RTX1, fork RTX0, then join its FP32 route contributions
    /// before ordered top-3 reduction. No allocations or host waits here.
    ///
    /// # Safety
    /// Caller owns the RTX1 stream and all inputs/shared through completion and
    /// graph lifetime. Prior use of this pair must precede this enqueue. On any
    /// failure, end/abort capture as appropriate and drain both streams before
    /// freeing or reusing any inputs. Graph destruction precedes dropping Pair.
    pub unsafe fn enqueue(
        &mut self,
        stage: usize,
        rows: u32,
        inputs: [Ds41rtDeviceBuffer; 3],
        shared: Option<Ds41rtDeviceBuffer>,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            stage < 3 && rows > 0 && rows <= self.capacity,
            "invalid draft TP2 stage/rows"
        );
        for (buffer, width) in inputs.iter().zip([10240, 12, 12]) {
            ensure!(
                buffer.device_id == 1 && buffer.bytes >= rows as usize * width,
                "draft TP2 root input device or size differs"
            );
        }
        if let Some(buffer) = shared {
            ensure!(
                buffer.device_id == 1 && buffer.bytes >= rows as usize * 10240,
                "draft TP2 shared output device or size differs"
            );
        }
        let library = self.owner.library;
        self.owner.run(|| unsafe {
            self.quantizer
                .launch(inputs[0], self.wire.buffer, rows, stream)?;
            library.cuda_event_record(self.ready.raw, stream)
        })?;
        let encoded = [self.wire.buffer, inputs[1], inputs[2]];
        let peer_result = self.peer.device.run(|| unsafe {
            library.cuda_stream_wait_event(self.peer.raw, self.ready.raw)?;
            for ((destination, source), width) in
                self.peer_inputs.iter().zip(encoded).zip([5280, 12, 12])
            {
                self.copies[0].launch(
                    destination.buffer,
                    source,
                    rows as usize * width,
                    self.peer.raw,
                )?;
            }
            let output = self.ranks[0].enqueue(
                stage,
                rows,
                self.peer_inputs.each_ref().map(|v| v.buffer),
                self.peer.raw,
            )?;
            library.cuda_event_record(self.done.raw, self.peer.raw)?;
            Ok(output)
        })?;
        self.owner.run(|| unsafe {
            let local = self.ranks[1].enqueue(stage, rows, encoded, stream)?;
            library.cuda_stream_wait_event(stream, self.done.raw)?;
            self.copies[1].launch(
                self.peer_output.buffer,
                peer_result,
                rows as usize * 3 * 5120 * 4,
                stream,
            )?;
            self.reducer.launch(
                [
                    self.peer_output.buffer.ptr.cast(),
                    local.ptr.cast(),
                    std::ptr::null(),
                    std::ptr::null(),
                ],
                shared.map_or(std::ptr::null(), |v| v.ptr.cast()),
                self.output.buffer.ptr.cast(),
                rows,
                2,
                3,
                stream,
            )
        })
    }
    /// Used after ending a failed capture or after root completion on error.
    pub fn drain_peer(&self) -> Result<()> {
        self.peer.drain()
    }
}
