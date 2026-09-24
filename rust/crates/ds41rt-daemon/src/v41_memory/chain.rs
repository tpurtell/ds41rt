//! Device-side ordering for the stages of one target pass.
//!
//! Each layer used to drain every stage on the host before the next stage was
//! submitted (mHC finish, query, window KV, attention, shared FFN, reduction),
//! leaving the RTX idle while the host issued the following launch. Inside a
//! chain scope, a stage instead joins the chain's CUDA event before its first
//! enqueue and re-records that event after its last one. Consecutive stages on
//! different streams are therefore ordered on the device, and the host waits
//! only where it must read results (routes before dispatch, the head output).
//!
//! Safety argument: every stage of a scoped pass joins the chain, so the chain
//! is a total order over the pass's GPU work. The per-layer route download is
//! a host wait on a stream joined after all earlier stages, hence every stage
//! submitted before it, including pinned-staging uploads, has completed when
//! the host reuses that staging in the next layer. Work outside a scope keeps
//! its original synchronous behaviour.
//!
//! The scope is a thread-local set only while polling the pass future (the
//! same pattern as the device-scoped futures), so interleaved lanes on one
//! executor thread keep independent chains.
use super::LoadStream;
use anyhow::Result;
use ds41rt_ffi::NativeLibrary;
use std::cell::Cell;
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

#[derive(Clone)]
struct Current {
    /// One ordering event per device the pass runs on; (device, event).
    events: Rc<[(i32, *mut c_void)]>,
    /// Index of the event holding the chain head, if any stage finished.
    head: Rc<Cell<Option<usize>>>,
}

thread_local! {
    static CURRENT: std::cell::RefCell<Option<Current>> = const { std::cell::RefCell::new(None) };
}

/// Ordering events for one pass owner, one per participating device.
pub(crate) struct StageChain<'a> {
    library: &'a NativeLibrary,
    events: Rc<[(i32, *mut c_void)]>,
    head: Rc<Cell<Option<usize>>>,
}
impl<'a> StageChain<'a> {
    /// A chain for the current device only.
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        let device = library.cuda_get_device()?;
        Self::on_devices(library, &[device])
    }
    /// A chain spanning `devices`; each event is created on its own device.
    pub fn on_devices(library: &'a NativeLibrary, devices: &[i32]) -> Result<Self> {
        let previous = library.cuda_get_device()?;
        let mut events = Vec::with_capacity(devices.len());
        let created = (|| -> Result<()> {
            for &device in devices {
                library.cuda_set_device(device)?;
                events.push((device, library.cuda_event_create_ordering()?));
            }
            Ok(())
        })();
        library.cuda_set_device(previous)?;
        if let Err(error) = created {
            for &(_, event) in &events { let _ = unsafe { library.cuda_event_destroy(event) }; }
            return Err(error);
        }
        Ok(Self { library, events: events.into(), head: Rc::new(Cell::new(None)) })
    }
    /// An owned handle that can wrap a future borrowing the chain's owner.
    pub fn handle(&self) -> ChainHandle {
        ChainHandle(Current { events: self.events.clone(), head: self.head.clone() })
    }
    /// Host wait for everything recorded so far, then forget the head. Call
    /// after the pass (or an aborted pass) before any unscoped consumer.
    pub fn drain(&self) -> Result<()> {
        if let Some(head) = self.head.replace(None) {
            unsafe { self.library.cuda_event_synchronize(self.events[head].1)?; }
        }
        Ok(())
    }
}
impl Drop for StageChain<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            tracing::error!(%error, "draining target stage chain");
        }
        for &(_, event) in self.events.iter() {
            if let Err(error) = unsafe { self.library.cuda_event_destroy(event) } {
                tracing::error!(%error, "destroying target stage chain event");
            }
        }
    }
}

pub(crate) struct ChainHandle(Current);
impl ChainHandle {
    /// Poll `future` with this chain installed as the current scope.
    pub fn scope<F: Future>(self, future: F) -> ChainScope<F> {
        ChainScope { current: self.0, future }
    }
}

/// Whether target passes order their stages on the device (default) or drain
/// each stage on the host (`DS41RT_STAGE_CHAIN=0`, the pre-v14 behaviour).
pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("DS41RT_STAGE_CHAIN").map_or(true, |v| v != "0"))
}

pub(crate) struct ChainScope<F> {
    current: Current,
    future: F,
}
impl<F: Future> Future for ChainScope<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<F::Output> {
        // The future is never moved after pinning; only borrowed in place.
        let this = unsafe { self.get_unchecked_mut() };
        let previous = CURRENT.with(|c| c.replace(Some(this.current.clone())));
        struct Restore(Option<Current>);
        impl Drop for Restore {
            fn drop(&mut self) { let previous = self.0.take(); CURRENT.with(|c| *c.borrow_mut() = previous); }
        }
        let _restore = Restore(previous);
        unsafe { Pin::new_unchecked(&mut this.future) }.poll(context)
    }
}

fn current() -> Option<Current> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether stages are currently device-ordered instead of host-drained.
pub(crate) fn active() -> bool {
    CURRENT.with(|c| c.borrow().is_some())
}

/// Order `stream` after the chain head. Call before a stage's first enqueue.
/// # Safety
/// `stream` is a live stream on the chain's device.
pub(crate) unsafe fn join(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let Some(head) = current.head.get() else { return Ok(()) };
    // Cross-device event waits are permitted; the event lives on its own device.
    unsafe { library.cuda_stream_wait_event(stream, current.events[head].1) }
}

/// Complete a stage: record the chain head on `stream` inside a scope,
/// otherwise drain `stream` on the host exactly as before.
/// # Safety
/// `stream` holds this stage's queued work on the chain's device.
pub(crate) unsafe fn finish(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return unsafe { library.cuda_stream_synchronize(stream) } };
    let device = library.cuda_get_device()?;
    let Some(index) = current.events.iter().position(|&(d, _)| d == device) else {
        // A device outside this chain: complete the stage on the host.
        return unsafe { library.cuda_stream_synchronize(stream) };
    };
    // Merge: the stream first waits for the previous head, so parallel branches
    // (window, compressor, index projection) all precede the new head.
    if let Some(head) = current.head.get() {
        unsafe { library.cuda_stream_wait_event(stream, current.events[head].1)?; }
    }
    unsafe { library.cuda_event_record(current.events[index].1, stream)?; }
    current.head.set(Some(index));
    Ok(())
}

/// Cooperative form of [`finish`]: record inside a scope, otherwise yield
/// until the stream completes.
/// # Safety
/// Same as [`finish`].
pub(crate) async unsafe fn finish_cooperative(stream: &LoadStream<'_>) -> Result<()> {
    if active() {
        unsafe { finish(stream.library, stream.raw) }
    } else {
        stream.wait().await
    }
}

/// Host wait for all chained work before a host-synchronous operation (legacy
/// stream copies do not order with the non-blocking stage streams).
pub(crate) fn settle(library: &NativeLibrary) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    if let Some(head) = current.head.get() {
        unsafe { library.cuda_event_synchronize(current.events[head].1)?; }
    }
    Ok(())
}
