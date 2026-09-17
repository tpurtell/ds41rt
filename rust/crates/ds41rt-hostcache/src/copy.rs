//! Copy engine (packet HC-3): the only thing the cache asks of the GPU. Two streams (store and
//! restore) carry byte copies between device ranges and pinned host ranges; events mark
//! positions on a stream and are polled or waited on with a budget. The daemon implements it
//! over CUDA; [`StubCopyEngine`] implements it on a virtual clock with a bandwidth and latency
//! model and fake memories, so the suites can check both timing and content.
//!
//! Ordering contract: copies on one stream complete in issue order; the two streams are
//! independent; an event completes when every copy issued on its stream before `record` has
//! completed. A copy reads its source when it *executes*, not when it is issued: the caller must
//! keep source memory alive and unchanged until the event after it completes (the stub models
//! this by copying at completion time, so a violated hold shows up as wrong bytes).
use crate::pool::{HostChunk, HostRange, PinnedMemory};
use anyhow::{anyhow, bail, Result};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Stream {
    Store,
    Restore,
}

/// A device byte range: an address the engine understands (a device pointer under CUDA, an
/// offset into the fake device under the stub) and a length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct DeviceRange {
    pub addr: u64,
    pub bytes: usize,
}

/// A position on a stream. Valid until `completed` returns true or `wait` succeeds; querying a
/// stale event is a logic error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Event(pub u64);

pub trait CopyEngine: PinnedMemory {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> anyhow::Result<()>;
    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> anyhow::Result<()>;
    /// Issue every `(device, host)` copy of `copies` as one batch on `stream`. The default loops
    /// the 1D calls, so an engine that only implements `d2h` stays correct; engines with a real
    /// batch entry point override this. Semantics are the batch's extents issued back-to-back in
    /// order: either every extent is accepted or the error leaves the engine exactly as the 1D
    /// loop's error would (extents accepted before the failure stay issued, the rest do not).
    /// An empty `copies` issues nothing and always succeeds.
    fn d2h_many(
        &mut self,
        stream: Stream,
        copies: &[(DeviceRange, HostRange)],
    ) -> anyhow::Result<()> {
        for &(src, dst) in copies {
            self.d2h(stream, src, dst)?;
        }
        Ok(())
    }
    /// The restore direction of [`CopyEngine::d2h_many`]: `(host, device)` pairs.
    fn h2d_many(
        &mut self,
        stream: Stream,
        copies: &[(HostRange, DeviceRange)],
    ) -> anyhow::Result<()> {
        for &(src, dst) in copies {
            self.h2d(stream, src, dst)?;
        }
        Ok(())
    }
    /// Engine submissions issued so far: one per 1D call and one per non-empty `*_many` batch.
    /// Engines without submission instrumentation return 0.
    fn submission_count(&self) -> u64 {
        0
    }
    fn record(&mut self, stream: Stream) -> anyhow::Result<Event>;
    /// Non-blocking: has everything before `event` completed?
    fn completed(&mut self, event: Event) -> anyhow::Result<bool>;
    /// Block up to `budget_ns`; true if the event completed within the budget.
    fn wait(&mut self, event: Event, budget_ns: u64) -> anyhow::Result<bool>;
    /// The clock every budget is measured against (monotonic; virtual under the stub).
    fn now_ns(&self) -> u64;
}

/// Bandwidths and latencies the stub models. Defaults are the design's assumptions (25 GB/s
/// each way, 10 µs per submission), replaced by measurements from the fleet when they exist.
///
/// A `*_many` batch is charged as one submission: a single `per_copy_latency_ns` plus the
/// transfer time of the summed extents at the direction's bandwidth, with every extent of the
/// batch completing at that one time. Each 1D call is charged its own latency and transfer and
/// counts as its own submission (see [`CopyEngine::submission_count`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CopyModel {
    pub d2h_bytes_per_ns: f64,
    pub h2d_bytes_per_ns: f64,
    pub per_copy_latency_ns: u64,
}

impl Default for CopyModel {
    fn default() -> Self {
        Self {
            d2h_bytes_per_ns: 25.0,
            h2d_bytes_per_ns: 25.0,
            per_copy_latency_ns: 10_000,
        }
    }
}

/// A fault the stub can arm for the next matching operation (exactly once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyFault {
    /// The next submission on the stream fails: the next 1D issue, or a whole `*_many` batch
    /// (an empty batch is not a submission and does not consume the fault).
    IssueFails(Stream),
    /// The next event on the stream wedges it: that event and every later copy and event on the
    /// stream never complete (the stub's model of a wedged stream); `wait` times out. The other
    /// stream is unaffected.
    StreamStalls(Stream),
}

/// Which way a copy moves bytes; selects the bandwidth the model charges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    D2h,
    H2d,
}

/// One end of a copy: a device range or a pinned host range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemRange {
    Device(DeviceRange),
    Host(HostRange),
}

impl MemRange {
    /// The length of the range, whichever memory it names.
    fn bytes(self) -> usize {
        match self {
            MemRange::Device(range) => range.bytes,
            MemRange::Host(range) => range.bytes,
        }
    }
}

/// One planned copy in its direction's pair order: `(device, host)` for D2H, `(host, device)`
/// for H2D. Lets the batch issue path validate and enqueue either direction through one
/// implementation without allocating.
trait Extent: Copy {
    fn src(self) -> MemRange;
    fn dst(self) -> MemRange;
}

impl Extent for (DeviceRange, HostRange) {
    fn src(self) -> MemRange {
        MemRange::Device(self.0)
    }
    fn dst(self) -> MemRange {
        MemRange::Host(self.1)
    }
}

impl Extent for (HostRange, DeviceRange) {
    fn src(self) -> MemRange {
        MemRange::Host(self.0)
    }
    fn dst(self) -> MemRange {
        MemRange::Device(self.1)
    }
}

/// A copy waiting for its completion time on the virtual clock. The map key's `seq` breaks ties
/// between the two streams so execution order is total and deterministic.
struct PendingCopy {
    stream: Stream,
    direction: Direction,
    src: MemRange,
    dst: MemRange,
    completion_ns: u64,
}

/// The virtual-clock engine. Fake device memory is a flat byte array of `device_bytes`; fake
/// host chunks are allocated on demand up to `host_bytes`. `advance` moves the clock and
/// executes every copy whose completion time has passed, in completion order, ties broken by
/// issue order, moving bytes between the fake memories; `wait` advances the clock itself up to
/// the budget. A stream wedged by [`CopyFault::StreamStalls`] accepts later copies but never
/// executes them and never completes later events.
pub struct StubCopyEngine {
    model: CopyModel,
    device: Vec<u8>,
    host_bytes: usize,
    host_used: usize,
    chunks: Vec<Option<Vec<u8>>>,
    now_ns: u64,
    /// Completion time of the last copy issued on each stream, or zero before the first.
    last_completion: [u64; 2],
    /// Pending copies keyed by `(completion_ns, seq)`, so the first entry is the next to execute.
    pending: BTreeMap<(u64, u64), PendingCopy>,
    pending_count: [usize; 2],
    /// Copies accepted on a wedged stream: counted as pending, never inserted into `pending`.
    held_count: [usize; 2],
    /// Completion time of each recorded event; `None` is a stalled event that never completes.
    events: Vec<Option<u64>>,
    issue_fault: [bool; 2],
    stall_fault: [bool; 2],
    /// A stream wedged by a fired `StreamStalls` fault: later copies and events never complete.
    stalled: [bool; 2],
    seq: u64,
    /// Engine submissions issued and accepted: one per 1D issue, one per non-empty batch.
    submissions: u64,
}

impl StubCopyEngine {
    /// A stub over `device_bytes` of fake device memory and `host_bytes` of fake pinned host
    /// memory, with the clock at zero and nothing in flight.
    pub fn new(model: CopyModel, device_bytes: usize, host_bytes: usize) -> Self {
        Self {
            model,
            device: vec![0; device_bytes],
            host_bytes,
            host_used: 0,
            chunks: Vec::new(),
            now_ns: 0,
            last_completion: [0; 2],
            pending: BTreeMap::new(),
            pending_count: [0; 2],
            held_count: [0; 2],
            events: Vec::new(),
            issue_fault: [false; 2],
            stall_fault: [false; 2],
            stalled: [false; 2],
            seq: 0,
            submissions: 0,
        }
    }

    /// Move the virtual clock forward by `nanos` and execute every copy whose completion time
    /// has been reached, in completion order. The clock never moves backwards.
    pub fn advance(&mut self, nanos: u64) {
        self.now_ns = self.now_ns.saturating_add(nanos);
        self.execute_due();
    }

    /// Overwrite `range` of the fake device memory with `bytes`; panics if the range is out of
    /// bounds or its length disagrees with `bytes`.
    pub fn write_device(&mut self, range: DeviceRange, bytes: &[u8]) {
        assert_eq!(range.bytes, bytes.len(), "device write length mismatch");
        let start = range.addr as usize;
        let Some(end) = start.checked_add(range.bytes) else {
            panic!("device write range overflows");
        };
        assert!(end <= self.device.len(), "device write out of bounds");
        self.device[start..end].copy_from_slice(bytes);
    }

    /// The bytes currently in `range` of the fake device memory; panics if out of bounds.
    pub fn read_device(&self, range: DeviceRange) -> Vec<u8> {
        let start = range.addr as usize;
        let Some(end) = start.checked_add(range.bytes) else {
            panic!("device read range overflows");
        };
        assert!(end <= self.device.len(), "device read out of bounds");
        self.device[start..end].to_vec()
    }

    /// The bytes currently in `range` of a fake host chunk; panics if the chunk is not allocated
    /// or the range is out of bounds.
    pub fn read_host(&self, range: HostRange) -> Vec<u8> {
        let Some(chunk) = self
            .chunks
            .get(range.chunk as usize)
            .and_then(Option::as_ref)
        else {
            panic!("host chunk {} is not allocated", range.chunk);
        };
        let Some(end) = range.offset.checked_add(range.bytes) else {
            panic!("host read range overflows");
        };
        assert!(end <= chunk.len(), "host read out of bounds");
        chunk[range.offset..end].to_vec()
    }

    /// Arm `fault`; it fires on the next matching operation and is then disarmed. An issue that
    /// fails validation (length or range) is not a matching operation and does not consume an
    /// armed fault.
    pub fn inject(&mut self, fault: CopyFault) {
        match fault {
            CopyFault::IssueFails(stream) => self.issue_fault[stream_index(stream)] = true,
            CopyFault::StreamStalls(stream) => self.stall_fault[stream_index(stream)] = true,
        }
    }

    /// Copies issued but not yet executed, per stream (for the suites' invariants). Copies held
    /// on a wedged stream are counted: they were issued and will never execute.
    pub fn pending(&self, stream: Stream) -> usize {
        let index = stream_index(stream);
        self.pending_count[index] + self.held_count[index]
    }

    /// Queue one copy: validate it, charge the model, advance the stream's tail, and remember the
    /// source and destination so the bytes move when the copy executes. Validation runs before the
    /// issue fault is consumed, so an invalid issue leaves the engine (and any armed fault)
    /// untouched. On a wedged stream the copy is accepted and counted as pending but never queued.
    fn issue(
        &mut self,
        stream: Stream,
        direction: Direction,
        src: MemRange,
        dst: MemRange,
    ) -> Result<()> {
        let index = stream_index(stream);
        if src.bytes() != dst.bytes() {
            bail!(
                "copy length mismatch: source {} bytes, destination {} bytes",
                src.bytes(),
                dst.bytes()
            );
        }
        self.check_range(src)?;
        self.check_range(dst)?;
        if self.issue_fault[index] {
            self.issue_fault[index] = false;
            bail!("injected issue fault on {stream:?}");
        }
        // An accepted issue is a submission, held ones included: the engine told the caller the
        // copy was queued even though a wedged stream will never run it.
        self.submissions += 1;
        if self.stalled[index] {
            self.held_count[index] += 1;
            return Ok(());
        }
        let rate = match direction {
            Direction::D2h => self.model.d2h_bytes_per_ns,
            Direction::H2d => self.model.h2d_bytes_per_ns,
        };
        let start = self.last_completion[index].max(self.now_ns);
        let completion_ns = start
            .saturating_add(self.model.per_copy_latency_ns)
            .saturating_add(transfer_ns(src.bytes(), rate));
        self.last_completion[index] = completion_ns;
        self.pending_count[index] += 1;
        self.seq += 1;
        self.pending.insert(
            (completion_ns, self.seq),
            PendingCopy {
                stream,
                direction,
                src,
                dst,
                completion_ns,
            },
        );
        Ok(())
    }

    /// Queue a whole batch: validate every extent first (an invalid batch leaves the engine —
    /// and any armed fault — untouched, exactly like an invalid 1D issue), then charge the
    /// batch model once — one `per_copy_latency_ns` plus the transfer of the summed extents —
    /// and enqueue every extent at that one completion time, in issue order. On a wedged
    /// stream the batch is accepted and held; an empty batch issues nothing, charges nothing
    /// and does not consume an armed fault.
    fn issue_many<E: Extent>(
        &mut self,
        stream: Stream,
        direction: Direction,
        copies: &[E],
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let index = stream_index(stream);
        for &copy in copies {
            let (src, dst) = (copy.src(), copy.dst());
            if src.bytes() != dst.bytes() {
                bail!(
                    "copy length mismatch: source {} bytes, destination {} bytes",
                    src.bytes(),
                    dst.bytes()
                );
            }
            self.check_range(src)?;
            self.check_range(dst)?;
        }
        if self.issue_fault[index] {
            self.issue_fault[index] = false;
            bail!("injected issue fault on {stream:?}");
        }
        self.submissions += 1;
        if self.stalled[index] {
            self.held_count[index] += copies.len();
            return Ok(());
        }
        let rate = match direction {
            Direction::D2h => self.model.d2h_bytes_per_ns,
            Direction::H2d => self.model.h2d_bytes_per_ns,
        };
        let total_bytes: usize = copies.iter().map(|&copy| copy.src().bytes()).sum();
        let start = self.last_completion[index].max(self.now_ns);
        let completion_ns = start
            .saturating_add(self.model.per_copy_latency_ns)
            .saturating_add(transfer_ns(total_bytes, rate));
        self.last_completion[index] = completion_ns;
        for &copy in copies {
            self.pending_count[index] += 1;
            self.seq += 1;
            self.pending.insert(
                (completion_ns, self.seq),
                PendingCopy {
                    stream,
                    direction,
                    src: copy.src(),
                    dst: copy.dst(),
                    completion_ns,
                },
            );
        }
        Ok(())
    }

    /// Reject a range that names memory the stub does not have.
    fn check_range(&self, range: MemRange) -> Result<()> {
        match range {
            MemRange::Device(r) => {
                let end = (r.addr as usize)
                    .checked_add(r.bytes)
                    .ok_or_else(|| anyhow!("device range overflows"))?;
                if end > self.device.len() {
                    bail!(
                        "device range {}..{} exceeds {} bytes",
                        r.addr,
                        end,
                        self.device.len()
                    );
                }
            }
            MemRange::Host(r) => {
                let chunk = self
                    .chunks
                    .get(r.chunk as usize)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| anyhow!("host chunk {} is not allocated", r.chunk))?;
                let end = r
                    .offset
                    .checked_add(r.bytes)
                    .ok_or_else(|| anyhow!("host range overflows"))?;
                if end > chunk.len() {
                    bail!(
                        "host range {}..{} exceeds chunk {} of {} bytes",
                        r.offset,
                        end,
                        r.chunk,
                        chunk.len()
                    );
                }
            }
        }
        Ok(())
    }

    /// Execute every pending copy whose completion time the clock has reached, earliest first.
    /// The next entry is peeked before it is removed, so a copy that is not due yet stays put
    /// instead of being popped and reinserted on every advance.
    fn execute_due(&mut self) {
        while let Some((_, copy)) = self.pending.first_key_value() {
            if copy.completion_ns > self.now_ns {
                break;
            }
            let Some((_, copy)) = self.pending.pop_first() else {
                break;
            };
            self.pending_count[stream_index(copy.stream)] -= 1;
            self.execute(copy);
        }
    }

    /// Move one copy's bytes at execution time. A chunk released before execution is skipped:
    /// the caller broke the hold contract, and the stub has nowhere to write.
    fn execute(&mut self, copy: PendingCopy) {
        match copy.direction {
            Direction::D2h => {
                let (MemRange::Device(src), MemRange::Host(dst)) = (copy.src, copy.dst) else {
                    return;
                };
                let Some(chunk) = self
                    .chunks
                    .get_mut(dst.chunk as usize)
                    .and_then(Option::as_mut)
                else {
                    return;
                };
                let n = src.bytes;
                let start = src.addr as usize;
                chunk[dst.offset..dst.offset + n].copy_from_slice(&self.device[start..start + n]);
            }
            Direction::H2d => {
                let (MemRange::Host(src), MemRange::Device(dst)) = (copy.src, copy.dst) else {
                    return;
                };
                let Some(chunk) = self.chunks.get(src.chunk as usize).and_then(Option::as_ref)
                else {
                    return;
                };
                let n = src.bytes;
                let start = dst.addr as usize;
                self.device[start..start + n].copy_from_slice(&chunk[src.offset..src.offset + n]);
            }
        }
    }

    /// The completion time of a recorded event, or `None` for a stalled one.
    fn event_completion(&self, event: Event) -> Result<Option<u64>> {
        self.events
            .get(event.0 as usize)
            .copied()
            .ok_or_else(|| anyhow!("unknown event {}", event.0))
    }
}

impl PinnedMemory for StubCopyEngine {
    fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<crate::pool::HostChunk> {
        let used = self
            .host_used
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("host allocation overflows"))?;
        if used > self.host_bytes {
            bail!(
                "host memory exhausted: {} of {} bytes in use, {} requested",
                self.host_used,
                self.host_bytes,
                bytes
            );
        }
        let id = u32::try_from(self.chunks.len()).map_err(|_| anyhow!("too many host chunks"))?;
        self.chunks.push(Some(vec![0; bytes]));
        self.host_used = used;
        Ok(HostChunk { id, bytes })
    }

    fn release_chunk(&mut self, chunk: crate::pool::HostChunk) -> anyhow::Result<()> {
        let slot = self
            .chunks
            .get_mut(chunk.id as usize)
            .ok_or_else(|| anyhow!("unknown host chunk {}", chunk.id))?;
        let bytes = slot
            .take()
            .ok_or_else(|| anyhow!("host chunk {} already released", chunk.id))?;
        self.host_used -= bytes.len();
        Ok(())
    }
}

impl CopyEngine for StubCopyEngine {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> anyhow::Result<()> {
        self.issue(
            stream,
            Direction::D2h,
            MemRange::Device(src),
            MemRange::Host(dst),
        )
    }

    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> anyhow::Result<()> {
        self.issue(
            stream,
            Direction::H2d,
            MemRange::Host(src),
            MemRange::Device(dst),
        )
    }

    fn d2h_many(
        &mut self,
        stream: Stream,
        copies: &[(DeviceRange, HostRange)],
    ) -> anyhow::Result<()> {
        self.issue_many(stream, Direction::D2h, copies)
    }

    fn h2d_many(
        &mut self,
        stream: Stream,
        copies: &[(HostRange, DeviceRange)],
    ) -> anyhow::Result<()> {
        self.issue_many(stream, Direction::H2d, copies)
    }

    fn submission_count(&self) -> u64 {
        self.submissions
    }

    fn record(&mut self, stream: Stream) -> anyhow::Result<Event> {
        let index = stream_index(stream);
        let completion = if self.stall_fault[index] {
            self.stall_fault[index] = false;
            self.stalled[index] = true;
            None
        } else if self.stalled[index] {
            None
        } else {
            Some(self.last_completion[index])
        };
        let id = self.events.len() as u64;
        self.events.push(completion);
        Ok(Event(id))
    }

    fn completed(&mut self, event: Event) -> anyhow::Result<bool> {
        Ok(match self.event_completion(event)? {
            Some(completion) => self.now_ns >= completion,
            None => false,
        })
    }

    fn wait(&mut self, event: Event, budget_ns: u64) -> anyhow::Result<bool> {
        let completion = self.event_completion(event)?;
        let target = match completion {
            Some(completion) => completion.min(self.now_ns.saturating_add(budget_ns)),
            None => self.now_ns.saturating_add(budget_ns),
        };
        if target > self.now_ns {
            self.now_ns = target;
            self.execute_due();
        }
        Ok(matches!(completion, Some(completion) if self.now_ns >= completion))
    }

    fn now_ns(&self) -> u64 {
        self.now_ns
    }
}

/// The slot a stream occupies in the per-stream arrays.
fn stream_index(stream: Stream) -> usize {
    match stream {
        Stream::Store => 0,
        Stream::Restore => 1,
    }
}

/// Nanoseconds to move `bytes` at `bytes_per_ns`, rounded up so a non-empty copy always costs
/// at least one nanosecond.
fn transfer_ns(bytes: usize, bytes_per_ns: f64) -> u64 {
    (bytes as f64 / bytes_per_ns).ceil() as u64
}

/// One endpoint of a planned copy. Coalescing merges two planned copies exactly when both
/// endpoints are adjacent, so the merged copy moves the same bytes to the same places.
trait Coalescible: Copy {
    fn bytes(&self) -> usize;
    /// This endpoint ends exactly where `next` begins.
    fn touches(&self, next: &Self) -> bool;
    /// Lengthen this endpoint by `next`, which must touch it.
    fn absorb(&mut self, next: &Self);
}

impl Coalescible for DeviceRange {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn touches(&self, next: &Self) -> bool {
        self.addr
            .checked_add(self.bytes as u64)
            .is_some_and(|end| end == next.addr)
    }
    fn absorb(&mut self, next: &Self) {
        self.bytes += next.bytes;
    }
}

impl Coalescible for HostRange {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn touches(&self, next: &Self) -> bool {
        self.chunk == next.chunk
            && self
                .offset
                .checked_add(self.bytes)
                .is_some_and(|end| end == next.offset)
    }
    fn absorb(&mut self, next: &Self) {
        self.bytes += next.bytes;
    }
}

/// Merge adjacent planned copies into single copies: consecutive entries whose device ranges
/// are address-contiguous and whose host ranges are offset-contiguous within one chunk merge
/// into one copy of the summed bytes. Invariants: the ordered byte-mapping of the plan is
/// unchanged (same source bytes land in the same destination addresses, in the same order),
/// the result is never longer than the input, and no two results touch on both sides. A pair
/// whose two lengths disagree is malformed (the engine rejects it); it never merges, neither
/// into nor with another pair, so coalescing cannot hide that error behind a longer copy.
fn coalesce_pairs<A: Coalescible, B: Coalescible>(copies: &[(A, B)]) -> Vec<(A, B)> {
    let mut merged: Vec<(A, B)> = Vec::with_capacity(copies.len());
    for &(a, b) in copies {
        match merged.last_mut() {
            Some((last_a, last_b))
                if last_a.bytes() == last_b.bytes()
                    && a.bytes() == b.bytes()
                    && last_a.touches(&a)
                    && last_b.touches(&b) =>
            {
                last_a.absorb(&a);
                last_b.absorb(&b);
            }
            _ => merged.push((a, b)),
        }
    }
    merged
}

/// The store orientation of [`coalesce_pairs`]: `(device, host)` pairs, as `d2h_many` issues.
pub fn coalesce(copies: &[(DeviceRange, HostRange)]) -> Vec<(DeviceRange, HostRange)> {
    coalesce_pairs(copies)
}

/// The restore orientation of [`coalesce_pairs`]: `(host, device)` pairs, as `h2d_many` issues.
pub fn coalesce_restore(copies: &[(HostRange, DeviceRange)]) -> Vec<(HostRange, DeviceRange)> {
    coalesce_pairs(copies)
}

/// Test support shared by the unit and integration suites: an independent shadow model of the
/// stub's documented behaviour. It lives in the crate (not in `tests/`) so both suites exercise
/// one implementation instead of duplicating it through `include!`/`#[path]`; it is hidden from
/// the public docs.
#[doc(hidden)]
pub mod testing {
    use super::{stream_index, transfer_ns, CopyModel, Stream};

    /// A copy the shadow model has queued.
    struct ShadowCopy {
        stream: Stream,
        d2h: bool,
        src: usize,
        dst: usize,
        bytes: usize,
        completion: u64,
        seq: u64,
    }

    /// An independent model of the stub: the same completion formula, the same execution-time
    /// byte movement, and the same global completion order. It exists to disagree with the engine
    /// when the engine is wrong.
    pub struct Shadow {
        pub device: Vec<u8>,
        pub host: Vec<u8>,
        pub now: u64,
        pub events: Vec<Option<u64>>,
        last: [u64; 2],
        pending: Vec<ShadowCopy>,
        seq: u64,
        model: CopyModel,
    }

    impl Shadow {
        pub fn new(model: CopyModel, device_bytes: usize, host_bytes: usize) -> Self {
            Self {
                device: vec![0; device_bytes],
                host: vec![0; host_bytes],
                now: 0,
                events: Vec::new(),
                last: [0; 2],
                pending: Vec::new(),
                seq: 0,
                model,
            }
        }

        pub fn write_device(&mut self, addr: usize, bytes: &[u8]) {
            self.device[addr..addr + bytes.len()].copy_from_slice(bytes);
        }

        pub fn issue(&mut self, stream: Stream, d2h: bool, src: usize, dst: usize, bytes: usize) {
            let index = stream_index(stream);
            let rate = if d2h {
                self.model.d2h_bytes_per_ns
            } else {
                self.model.h2d_bytes_per_ns
            };
            let start = self.last[index].max(self.now);
            let completion = start
                .saturating_add(self.model.per_copy_latency_ns)
                .saturating_add(transfer_ns(bytes, rate));
            self.last[index] = completion;
            self.seq += 1;
            self.pending.push(ShadowCopy {
                stream,
                d2h,
                src,
                dst,
                bytes,
                completion,
                seq: self.seq,
            });
        }

        /// The batch issue: one `per_copy_latency_ns` plus the transfer of the summed extents,
        /// every extent completing at that one time, in issue order — the same model the stub
        /// charges a `*_many` call.
        pub fn issue_many(&mut self, stream: Stream, d2h: bool, copies: &[(usize, usize, usize)]) {
            if copies.is_empty() {
                return;
            }
            let index = stream_index(stream);
            let rate = if d2h {
                self.model.d2h_bytes_per_ns
            } else {
                self.model.h2d_bytes_per_ns
            };
            let total_bytes: usize = copies.iter().map(|&(_, _, bytes)| bytes).sum();
            let start = self.last[index].max(self.now);
            let completion = start
                .saturating_add(self.model.per_copy_latency_ns)
                .saturating_add(transfer_ns(total_bytes, rate));
            self.last[index] = completion;
            for &(src, dst, bytes) in copies {
                self.seq += 1;
                self.pending.push(ShadowCopy {
                    stream,
                    d2h,
                    src,
                    dst,
                    bytes,
                    completion,
                    seq: self.seq,
                });
            }
        }

        pub fn record(&mut self, stream: Stream) -> usize {
            self.events.push(Some(self.last[stream_index(stream)]));
            self.events.len() - 1
        }

        pub fn advance(&mut self, nanos: u64) {
            self.now = self.now.saturating_add(nanos);
            self.execute_due();
        }

        pub fn wait(&mut self, event: usize, budget: u64) -> bool {
            let completion = self.events[event];
            let target = match completion {
                Some(completion) => completion.min(self.now.saturating_add(budget)),
                None => self.now.saturating_add(budget),
            };
            if target > self.now {
                self.now = target;
                self.execute_due();
            }
            matches!(completion, Some(completion) if self.now >= completion)
        }

        pub fn pending(&self, stream: Stream) -> usize {
            self.pending
                .iter()
                .filter(|copy| copy.stream == stream)
                .count()
        }

        fn execute_due(&mut self) {
            self.pending.sort_by_key(|copy| (copy.completion, copy.seq));
            let mut i = 0;
            while i < self.pending.len() {
                if self.pending[i].completion <= self.now {
                    let copy = self.pending.remove(i);
                    if copy.d2h {
                        self.host[copy.dst..copy.dst + copy.bytes]
                            .copy_from_slice(&self.device[copy.src..copy.src + copy.bytes]);
                    } else {
                        self.device[copy.dst..copy.dst + copy.bytes]
                            .copy_from_slice(&self.host[copy.src..copy.src + copy.bytes]);
                    }
                } else {
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_rounds_up() {
        assert_eq!(transfer_ns(0, 25.0), 0);
        assert_eq!(transfer_ns(1, 25.0), 1);
        assert_eq!(transfer_ns(25, 25.0), 1);
        assert_eq!(transfer_ns(26, 25.0), 2);
        assert_eq!(transfer_ns(crate::PAGE_BYTES, 25.0), 3_646);
    }

    #[test]
    fn stream_slots_are_distinct() {
        assert_ne!(stream_index(Stream::Store), stream_index(Stream::Restore));
    }

    #[test]
    fn default_model_page_cost_is_latency_plus_transfer() {
        let model = CopyModel::default();
        let cost =
            model.per_copy_latency_ns + transfer_ns(crate::PAGE_BYTES, model.d2h_bytes_per_ns);
        assert_eq!(cost, 13_646);
    }

    #[test]
    fn mem_range_reports_its_length() {
        assert_eq!(
            MemRange::Device(DeviceRange { addr: 7, bytes: 11 }).bytes(),
            11
        );
        assert_eq!(
            MemRange::Host(HostRange {
                chunk: 0,
                offset: 3,
                bytes: 5
            })
            .bytes(),
            5
        );
    }

    /// An issue that fails validation must leave the engine exactly as it was: no pending copy,
    /// no moved stream tail, and any armed fault still armed.
    #[test]
    fn invalid_issue_leaves_no_trace() {
        type IssueCase = (&'static str, fn(&mut StubCopyEngine, HostChunk));
        let mut engine = StubCopyEngine::new(CopyModel::default(), 100, 100);
        let chunk = engine.allocate_chunk(100).unwrap();
        let cases: [IssueCase; 4] = [
            ("device range out of bounds", |engine, chunk| {
                assert!(engine
                    .d2h(
                        Stream::Store,
                        DeviceRange {
                            addr: 50,
                            bytes: 100
                        },
                        HostRange {
                            chunk: chunk.id,
                            offset: 0,
                            bytes: 100,
                        },
                    )
                    .is_err());
            }),
            ("host range out of bounds", |engine, chunk| {
                assert!(engine
                    .d2h(
                        Stream::Store,
                        DeviceRange {
                            addr: 0,
                            bytes: 100
                        },
                        HostRange {
                            chunk: chunk.id,
                            offset: 50,
                            bytes: 100,
                        },
                    )
                    .is_err());
            }),
            ("unknown host chunk", |engine, _chunk| {
                assert!(engine
                    .d2h(
                        Stream::Store,
                        DeviceRange {
                            addr: 0,
                            bytes: 100
                        },
                        HostRange {
                            chunk: 99,
                            offset: 0,
                            bytes: 100,
                        },
                    )
                    .is_err());
            }),
            ("length mismatch", |engine, chunk| {
                assert!(engine
                    .d2h(
                        Stream::Store,
                        DeviceRange {
                            addr: 0,
                            bytes: 100
                        },
                        HostRange {
                            chunk: chunk.id,
                            offset: 0,
                            bytes: 50,
                        },
                    )
                    .is_err());
            }),
        ];
        for (name, issue) in cases {
            let tail = engine.last_completion;
            let pending = [
                engine.pending(Stream::Store),
                engine.pending(Stream::Restore),
            ];
            engine.inject(CopyFault::IssueFails(Stream::Store));
            issue(&mut engine, chunk);
            assert_eq!(engine.last_completion, tail, "{name} moved the stream tail");
            assert_eq!(
                [
                    engine.pending(Stream::Store),
                    engine.pending(Stream::Restore)
                ],
                pending,
                "{name} changed the pending counts"
            );
            assert!(
                engine.issue_fault[stream_index(Stream::Store)],
                "{name} consumed the armed issue fault"
            );
            engine.issue_fault[stream_index(Stream::Store)] = false;
        }
    }

    /// A copy that is not due yet must survive an advance untouched and execute at its own time.
    #[test]
    fn execute_due_keeps_not_due_copies() {
        let mut engine = StubCopyEngine::new(CopyModel::default(), 1024, 1024);
        let chunk = engine.allocate_chunk(1024).unwrap();
        engine.write_device(
            DeviceRange {
                addr: 0,
                bytes: 100,
            },
            &[1u8; 100],
        );
        for offset in [0, 100] {
            engine
                .d2h(
                    Stream::Store,
                    DeviceRange {
                        addr: 0,
                        bytes: 100,
                    },
                    HostRange {
                        chunk: chunk.id,
                        offset,
                        bytes: 100,
                    },
                )
                .unwrap();
        }
        let first_key = *engine.pending.first_key_value().unwrap().0;
        engine.advance(10_003);
        assert_eq!(engine.pending(Stream::Store), 2);
        assert_eq!(*engine.pending.first_key_value().unwrap().0, first_key);
        engine.advance(1);
        assert_eq!(engine.pending(Stream::Store), 1);
        assert_eq!(
            engine.read_host(HostRange {
                chunk: chunk.id,
                offset: 0,
                bytes: 100
            }),
            vec![1u8; 100]
        );
        assert_eq!(
            engine.read_host(HostRange {
                chunk: chunk.id,
                offset: 100,
                bytes: 100
            }),
            vec![0u8; 100]
        );
        engine.advance(10_004);
        assert_eq!(engine.pending(Stream::Store), 0);
        assert_eq!(
            engine.read_host(HostRange {
                chunk: chunk.id,
                offset: 100,
                bytes: 100
            }),
            vec![1u8; 100]
        );
    }

    /// A fired stall wedges the stream: later copies are held, not queued, and later events never
    /// complete; the other stream is untouched.
    #[test]
    fn stall_holds_later_copies_and_events() {
        let mut engine = StubCopyEngine::new(CopyModel::default(), 1024, 1024);
        let chunk = engine.allocate_chunk(1024).unwrap();
        engine.write_device(
            DeviceRange {
                addr: 0,
                bytes: 100,
            },
            &[1u8; 100],
        );
        engine.inject(CopyFault::StreamStalls(Stream::Store));
        let stalled = engine.record(Stream::Store).unwrap();
        assert!(!engine.completed(stalled).unwrap());
        engine
            .d2h(
                Stream::Store,
                DeviceRange {
                    addr: 0,
                    bytes: 100,
                },
                HostRange {
                    chunk: chunk.id,
                    offset: 0,
                    bytes: 100,
                },
            )
            .unwrap();
        assert_eq!(engine.held_count[stream_index(Stream::Store)], 1);
        assert!(engine.pending.first_key_value().is_none());
        assert_eq!(engine.pending(Stream::Store), 1);
        let later = engine.record(Stream::Store).unwrap();
        assert!(!engine.wait(later, 1_000_000).unwrap());
        assert_eq!(
            engine.read_host(HostRange {
                chunk: chunk.id,
                offset: 0,
                bytes: 100
            }),
            vec![0u8; 100]
        );
    }

    /// The peeked `execute_due` must still agree with the independent shadow model across partial
    /// advances that leave copies pending.
    #[test]
    fn partial_advances_match_the_shadow() {
        let model = CopyModel::default();
        let mut engine = StubCopyEngine::new(model, 1024, 1024);
        let chunk = engine.allocate_chunk(1024).unwrap();
        let mut shadow = testing::Shadow::new(model, 1024, 1024);
        for step in 0..8u64 {
            let bytes = 1 + (step as usize * 13) % 200;
            let addr = (step as usize * 64) % 512;
            let pattern: Vec<u8> = (0..bytes).map(|i| (i as u8) ^ (step as u8)).collect();
            engine.write_device(
                DeviceRange {
                    addr: addr as u64,
                    bytes,
                },
                &pattern,
            );
            shadow.write_device(addr, &pattern);
            engine
                .d2h(
                    Stream::Store,
                    DeviceRange {
                        addr: addr as u64,
                        bytes,
                    },
                    HostRange {
                        chunk: chunk.id,
                        offset: addr,
                        bytes,
                    },
                )
                .unwrap();
            shadow.issue(Stream::Store, true, addr, addr, bytes);
            let nanos = 1 + (step * 997) % 5_000;
            engine.advance(nanos);
            shadow.advance(nanos);
            assert_eq!(engine.now_ns(), shadow.now);
            assert_eq!(engine.pending(Stream::Store), shadow.pending(Stream::Store));
            assert_eq!(
                engine.read_device(DeviceRange {
                    addr: 0,
                    bytes: 1024
                }),
                shadow.device
            );
            assert_eq!(
                engine.read_host(HostRange {
                    chunk: chunk.id,
                    offset: 0,
                    bytes: 1024
                }),
                shadow.host
            );
        }
    }

    fn pair(addr: u64, chunk: u32, offset: usize, bytes: usize) -> (DeviceRange, HostRange) {
        (
            DeviceRange { addr, bytes },
            HostRange {
                chunk,
                offset,
                bytes,
            },
        )
    }

    /// The merged plan must move exactly the bytes the 1D plan moves, in the same order:
    /// expanding every copy into its unit byte mappings must yield identical sequences, and no
    /// two merged copies may overlap.
    fn assert_same_mapping(
        one_d: &[(DeviceRange, HostRange)],
        merged: &[(DeviceRange, HostRange)],
    ) {
        let expand = |plan: &[(DeviceRange, HostRange)]| -> Vec<(u64, u32, usize)> {
            plan.iter()
                .flat_map(|&(device, host)| {
                    (0..device.bytes as u64)
                        .map(move |i| (device.addr + i, host.chunk, host.offset + i as usize))
                })
                .collect()
        };
        assert_eq!(expand(one_d), expand(merged), "byte mapping changed");
        let mut sorted = expand(merged);
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            expand(merged).len(),
            "merged plan overlaps itself"
        );
    }

    #[test]
    fn coalesce_merges_only_both_sides_adjacent() {
        // Device-contiguous only: pages in one chunk but a host gap must not merge.
        let plan = vec![pair(0, 0, 0, 100), pair(100, 0, 200, 100)];
        assert_eq!(coalesce(&plan), plan);
        // Host-contiguous only: a device gap must not merge.
        let plan = vec![pair(0, 0, 0, 100), pair(300, 0, 100, 100)];
        assert_eq!(coalesce(&plan), plan);
        // A chunk boundary on the host side must not merge.
        let plan = vec![pair(0, 0, 0, 100), pair(100, 1, 0, 100)];
        assert_eq!(coalesce(&plan), plan);
        // Both sides adjacent: two copies become one of the summed bytes.
        let plan = vec![
            pair(0, 0, 0, 100),
            pair(100, 0, 100, 50),
            pair(150, 0, 150, 70),
        ];
        assert_eq!(coalesce(&plan), vec![pair(0, 0, 0, 220)]);
        // The byte mapping is preserved by every case above.
        for plan in [
            vec![pair(0, 0, 0, 100), pair(100, 0, 200, 100)],
            vec![pair(0, 0, 0, 100), pair(300, 0, 100, 100)],
            vec![pair(0, 0, 0, 100), pair(100, 1, 0, 100)],
            vec![
                pair(0, 0, 0, 100),
                pair(100, 0, 100, 50),
                pair(150, 0, 150, 70),
            ],
        ] {
            assert_same_mapping(&plan, &coalesce(&plan));
        }
    }

    #[test]
    fn coalesce_preserves_malformed_pairs() {
        // A length-mismatched pair never merges, so the engine still sees and rejects it.
        let malformed = (
            DeviceRange { addr: 0, bytes: 10 },
            HostRange {
                chunk: 0,
                offset: 0,
                bytes: 20,
            },
        );
        let touching = pair(10, 0, 20, 10);
        let plan = vec![malformed, touching];
        assert_eq!(coalesce(&plan), plan);
        // Even when the following pair would make the combined lengths agree again.
        let malformed = (
            DeviceRange { addr: 0, bytes: 10 },
            HostRange {
                chunk: 0,
                offset: 0,
                bytes: 20,
            },
        );
        let touching = pair(10, 0, 20, 5);
        let plan = vec![malformed, touching];
        assert_eq!(coalesce(&plan), plan);
    }

    #[test]
    fn coalesce_restore_mirrors_coalesce() {
        let plan: Vec<(HostRange, DeviceRange)> = vec![
            (
                HostRange {
                    chunk: 0,
                    offset: 0,
                    bytes: 64,
                },
                DeviceRange { addr: 8, bytes: 64 },
            ),
            (
                HostRange {
                    chunk: 0,
                    offset: 64,
                    bytes: 64,
                },
                DeviceRange {
                    addr: 8 + 64,
                    bytes: 64,
                },
            ),
        ];
        let merged = coalesce_restore(&plan);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].0.bytes, 128);
        assert_eq!(merged[0].1.bytes, 128);
    }

    #[test]
    fn coalesce_of_empty_is_empty() {
        assert!(coalesce(&[]).is_empty());
        assert!(coalesce_restore(&[]).is_empty());
    }

    /// A batch moves the same bytes as the 1D loop, completes at one latency plus the summed
    /// transfer, and counts as one submission.
    #[test]
    fn batch_moves_bytes_and_charges_one_submission() {
        let model = CopyModel::default();
        let mut engine = StubCopyEngine::new(model, 4096, 4096);
        let chunk = engine.allocate_chunk(4096).unwrap();
        let copies = [pair(0, chunk.id, 0, 100), pair(100, chunk.id, 100, 200)];
        engine.write_device(
            DeviceRange {
                addr: 0,
                bytes: 300,
            },
            &(0..300u16).map(|i| i as u8).collect::<Vec<_>>(),
        );
        engine
            .d2h_many(Stream::Store, &copies)
            .expect("batch issues");
        assert_eq!(engine.submission_count(), 1);
        // One latency plus the transfer of 300 bytes at 25 bytes/ns.
        let expected = model.per_copy_latency_ns + transfer_ns(300, model.d2h_bytes_per_ns);
        let event = engine.record(Stream::Store).expect("record");
        engine.advance(expected);
        assert!(engine.completed(event).expect("completed"));
        let expect: Vec<u8> = (0..300u16).map(|i| i as u8).collect();
        assert_eq!(
            engine.read_host(HostRange {
                chunk: chunk.id,
                offset: 0,
                bytes: 300
            }),
            expect
        );
    }

    /// An invalid extent anywhere in the batch rejects the whole batch before anything is
    /// issued, and the armed fault is not consumed.
    #[test]
    fn invalid_batch_leaves_no_trace() {
        let mut engine = StubCopyEngine::new(CopyModel::default(), 1024, 1024);
        let chunk = engine.allocate_chunk(1024).unwrap();
        let good = pair(0, chunk.id, 0, 100);
        let out_of_bounds = pair(0, chunk.id, 0, 10_000);
        let mismatched = (
            DeviceRange {
                addr: 0,
                bytes: 100,
            },
            HostRange {
                chunk: chunk.id,
                offset: 0,
                bytes: 50,
            },
        );
        for batch in [vec![good, out_of_bounds], vec![good, mismatched]] {
            let tail = engine.last_completion;
            let pending = [
                engine.pending(Stream::Store),
                engine.pending(Stream::Restore),
            ];
            let submissions = engine.submission_count();
            engine.inject(CopyFault::IssueFails(Stream::Store));
            assert!(engine.d2h_many(Stream::Store, &batch).is_err());
            assert_eq!(engine.last_completion, tail, "batch moved the stream tail");
            assert_eq!(
                [
                    engine.pending(Stream::Store),
                    engine.pending(Stream::Restore)
                ],
                pending,
                "batch changed the pending counts"
            );
            assert_eq!(engine.submission_count(), submissions, "batch was counted");
            assert!(
                engine.issue_fault[stream_index(Stream::Store)],
                "batch consumed the armed issue fault"
            );
        }
    }

    /// The issue fault fires on the whole batch, after validation: the batch fails with
    /// nothing enqueued, and an empty batch neither fails nor consumes the fault.
    #[test]
    fn issue_fault_fires_on_the_whole_batch() {
        let mut engine = StubCopyEngine::new(CopyModel::default(), 1024, 1024);
        let chunk = engine.allocate_chunk(1024).unwrap();
        let copies = [pair(0, chunk.id, 0, 100), pair(100, chunk.id, 100, 100)];
        engine.inject(CopyFault::IssueFails(Stream::Store));
        // An empty batch is not a submission and does not consume the fault.
        let empty: [(DeviceRange, HostRange); 0] = [];
        assert!(engine.d2h_many(Stream::Store, &empty).is_ok());
        assert_eq!(engine.submission_count(), 0);
        assert!(
            engine.issue_fault[stream_index(Stream::Store)],
            "empty batch consumed the fault"
        );
        assert!(engine.d2h_many(Stream::Store, &copies).is_err());
        assert_eq!(engine.pending(Stream::Store), 0);
        assert_eq!(
            engine.submission_count(),
            0,
            "a failed batch is not a submission"
        );
    }

    /// A wedged stream holds the whole batch: every extent is counted as pending, none is
    /// queued, and the one submission is counted.
    #[test]
    fn stall_holds_the_whole_batch() {
        let mut engine = StubCopyEngine::new(CopyModel::default(), 1024, 1024);
        let chunk = engine.allocate_chunk(1024).unwrap();
        let copies = [pair(0, chunk.id, 0, 100), pair(100, chunk.id, 100, 100)];
        engine.inject(CopyFault::StreamStalls(Stream::Store));
        let event = engine.record(Stream::Store).expect("record");
        engine
            .d2h_many(Stream::Store, &copies)
            .expect("batch accepted on the wedged stream");
        assert_eq!(engine.held_count[stream_index(Stream::Store)], 2);
        assert_eq!(engine.pending(Stream::Store), 2);
        assert_eq!(engine.submission_count(), 1);
        assert!(!engine.wait(event, 1_000_000).expect("wait"));
        assert_eq!(
            engine.read_host(HostRange {
                chunk: chunk.id,
                offset: 0,
                bytes: 200
            }),
            vec![0u8; 200]
        );
    }

    /// Batches and 1D issues chain on the stream by the batch model, and the shadow agrees
    /// with the engine across mixed issues and partial advances.
    #[test]
    fn batches_match_the_shadow() {
        type Batch = Vec<(usize, usize, usize)>;
        let model = CopyModel::default();
        let mut engine = StubCopyEngine::new(model, 4096, 4096);
        let chunk = engine.allocate_chunk(4096).unwrap();
        let mut shadow = testing::Shadow::new(model, 4096, 4096);
        let schedule: [(Stream, bool, Batch, u64); 5] = [
            (
                Stream::Store,
                true,
                vec![(0, 0, 100), (100, 100, 200)],
                5_000,
            ),
            (Stream::Store, true, vec![(300, 300, 100)], 1),
            (
                Stream::Restore,
                false,
                vec![(0, 512, 64), (64, 576, 64)],
                50_000,
            ),
            (Stream::Store, true, vec![], 10),
            (Stream::Restore, false, vec![(128, 640, 128)], 1_000_000),
        ];
        for (step, (stream, d2h, copies, nanos)) in schedule.iter().enumerate() {
            let copies: Vec<(DeviceRange, HostRange)> = copies
                .iter()
                .map(|&(src, dst, bytes)| pair(src as u64, chunk.id, dst, bytes))
                .collect();
            if *d2h {
                engine.d2h_many(*stream, &copies).expect("batch issues");
            } else {
                let restore: Vec<(HostRange, DeviceRange)> =
                    copies.iter().map(|&(d, h)| (h, d)).collect();
                engine.h2d_many(*stream, &restore).expect("batch issues");
            }
            let flat: Vec<(usize, usize, usize)> = copies
                .iter()
                .map(|&(d, h)| (d.addr as usize, h.offset, d.bytes))
                .collect();
            shadow.issue_many(*stream, *d2h, &flat);
            engine.advance(*nanos);
            shadow.advance(*nanos);
            assert_eq!(engine.now_ns(), shadow.now, "step {step}: clock diverged");
            assert_eq!(
                engine.pending(*stream),
                shadow.pending(*stream),
                "step {step}: pending diverged"
            );
            assert_eq!(
                engine.read_device(DeviceRange {
                    addr: 0,
                    bytes: 4096
                }),
                shadow.device,
                "step {step}: device diverged"
            );
            assert_eq!(
                engine.read_host(HostRange {
                    chunk: chunk.id,
                    offset: 0,
                    bytes: 4096
                }),
                shadow.host,
                "step {step}: host diverged"
            );
        }
    }
}
