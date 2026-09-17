//! Handle over the CUDA runtime library already loaded in the daemon's process image,
//! exposing `cudaMemcpyBatchAsync` (CUDA ≥ 12.8) for the host snapshot cache's copy engine.
//!
//! The daemon links the native library against `CUDA::cudart`, so `libcudart` is always in the
//! process address space before this runs. [`CudaRuntime::load`] therefore opens the runtime
//! with `RTLD_NOW | RTLD_NOLOAD` under each known soname in turn (`libcudart.so.13`, then
//! `libcudart.so.12`): under glibc — the fleet target — `RTLD_NOLOAD` guarantees we only ever
//! bind to the runtime the native library is already using; a fresh dlopen of an unlinked
//! runtime would carry its own context state and must never happen. Absence of the library or
//! the symbol is not an error: it selects the merged-1D fallback, once, at engine construction.
//!
//! `cudaMemcpyBatchAsync` changed arity between CUDA 12.8 and 13.x, so the extern signature is
//! selected from the loaded runtime's own version (see [`select_copy_mechanism`]); calling with
//! the wrong arity is undefined behaviour.
//!
//! Struct layouts verified against both toolchains' headers (identical):
//! `driver_types.h` CUDA 13.2 lines 2430–2437 and CUDA 12.8 lines 2337–2342:
//! `cudaMemcpyAttributes { cudaMemcpySrcAccessOrder srcAccessOrder; cudaMemLocation
//! srcLocHint; cudaMemLocation dstLocHint; unsigned flags; }` — offsets: srcAccessOrder @0
//! (4 bytes), srcLocHint @4 (8), dstLocHint @12 (8), flags @20 (4); sizeof 24, alignof 4.
//! `cudaMemLocation` is `{ enum type; int id; }`
//! (8 bytes). `cudaMemcpySrcAccessOrderStream == 1` (13.2 line 2402, 12.8 line 2309) and is the
//! value passed, preserving the engine's "source read in stream order at execution time"
//! contract.
use anyhow::{bail, ensure, Result};
use libloading::os::unix::Library;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::Arc;

/// `RTLD_NOW` from `dlfcn.h` (glibc and macOS both use 2): resolve every symbol before
/// `dlopen` returns, so a missing entry point fails the probe instead of crashing later.
const RTLD_NOW: c_int = 2;

/// `RTLD_NOLOAD` from glibc's `dlfcn.h`: look the library up in the process image without
/// loading it. The no-second-runtime guarantee this value carries is glibc-specific (the
/// fleet target): musl defines the constant but its `dlopen` ignores flags, and other
/// platforms disagree (FreeBSD uses 0x20000). macOS is kept correct for local development
/// but is not a fleet target.
#[cfg(not(target_os = "macos"))]
const RTLD_NOLOAD: c_int = 0x4;
#[cfg(target_os = "macos")]
const RTLD_NOLOAD: c_int = 0x10;

/// Sonames probed, newest first: the fleet image ships `libcudart.so.13`, the build toolchain
/// `libcudart.so.12`.
const SONAMES: [&str; 2] = ["libcudart.so.13", "libcudart.so.12"];

/// `cudaMemcpySrcAccessOrderStream`: the source is accessed in stream order, matching how the
/// engine already holds source memory alive until the event after the copy.
const CUDA_MEMCPY_SRC_ACCESS_ORDER_STREAM: c_uint = 1;

/// `cudaError_t` is a C enum; `cudaGetErrorString` maps it to text.
type GetErrorStringFn = unsafe extern "C" fn(c_int) -> *const c_char;
type RuntimeGetVersionFn = unsafe extern "C" fn(*mut c_int) -> c_int;
/// CUDA 13.x: no `failIdx` (fleet header, `cuda_runtime_api.h:6540`).
type MemcpyBatchThirteenFn = unsafe extern "C" fn(
    *const *mut c_void,
    *const *const c_void,
    *const usize,
    usize,
    *mut CudaMemcpyAttributes,
    *mut usize,
    usize,
    *mut c_void,
) -> c_int;
/// CUDA 12.8/12.9: trailing `size_t *failIdx` (CUDA 12.8 header, `cuda_runtime_api.h:7334`).
type MemcpyBatchTwelveFn = unsafe extern "C" fn(
    *const *mut c_void,
    *const *const c_void,
    *const usize,
    usize,
    *mut CudaMemcpyAttributes,
    *mut usize,
    usize,
    *mut usize,
    *mut c_void,
) -> c_int;

/// `struct cudaMemLocation` (driver_types.h, both versions): `{ enum cudaMemLocationType type;
/// int id; }`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaMemLocation {
    loc_type: c_uint,
    id: c_int,
}

/// `struct cudaMemcpyAttributes`; see the module doc for the layout evidence. Zeroed hints are
/// "no hint", which is what the engine means: both operands are ordinary device/pinned memory.
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaMemcpyAttributes {
    src_access_order: c_uint,
    src_loc_hint: CudaMemLocation,
    dst_loc_hint: CudaMemLocation,
    flags: c_uint,
}

impl CudaMemcpyAttributes {
    /// The single attribute entry every batch passes: stream-ordered source access, no
    /// location hints, no flags. The runtime rejects a batch whose copies have no attribute
    /// entry (`attrs == NULL` / `numAttrs == 0` is `cudaErrorInvalidValue`), so the entry is
    /// always present and applies to the whole batch.
    fn stream_order() -> Self {
        Self {
            src_access_order: CUDA_MEMCPY_SRC_ACCESS_ORDER_STREAM,
            src_loc_hint: CudaMemLocation { loc_type: 0, id: 0 },
            dst_loc_hint: CudaMemLocation { loc_type: 0, id: 0 },
            flags: 0,
        }
    }
}

/// The copy mechanism the host cache should use, selected once from the runtime version the
/// process actually has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyMechanism {
    /// One `cudaMemcpyBatchAsync` call per coalesced snapshot copy list.
    MemcpyBatch,
    /// The runtime has no usable batch entry point: issue the coalesced extents as 1D copies.
    Merged1d,
}

impl CopyMechanism {
    /// The mechanism's short name for the boot log.
    pub fn name(self) -> &'static str {
        match self {
            CopyMechanism::MemcpyBatch => "cuda-memcpy-batch",
            CopyMechanism::Merged1d => "merged-1d",
        }
    }
}

/// Select the copy mechanism for a CUDA runtime reported at `version` (`None` when no runtime
/// or no `cudaMemcpyBatchAsync` symbol is loaded). Version ranges per the two arities:
/// ≥ 13000 the 13.x signature, 12080–12999 the 12.x signature (both exported since 12.8),
/// anything older or absent the merged-1D fallback. Pure so the fallback paths are testable
/// without a GPU.
pub fn select_copy_mechanism(runtime_version: Option<i32>) -> CopyMechanism {
    match runtime_version {
        Some(version) if version >= 13_000 || (12_080..13_000).contains(&version) => {
            CopyMechanism::MemcpyBatch
        }
        _ => CopyMechanism::Merged1d,
    }
}

/// The `cudaMemcpyBatchAsync` entry point, with the arity the loaded runtime's version
/// selects; plain function pointers, so `CudaRuntime` stays `Send + Sync`.
#[derive(Clone, Copy)]
enum BatchFn {
    Thirteen(MemcpyBatchThirteenFn),
    Twelve(MemcpyBatchTwelveFn),
}

/// The CUDA runtime already loaded in this process, with the version-selected batch entry
/// point. Cheap to clone: the library handle is shared, the function pointers copied.
#[derive(Clone)]
pub struct CudaRuntime {
    _library: Arc<Library>, // keepalive: the batch fn pointers stay valid only while the runtime stays loaded
    version: i32,
    batch: BatchFn,
    get_error_string: GetErrorStringFn,
}

// The handle binds function pointers of a library the process already loaded; it loads
// nothing and owns no device state, so it is safe to move and share like the library itself.
unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

impl CudaRuntime {
    /// Bind the already-loaded CUDA runtime, or `None` when no soname resolves or the runtime
    /// predates `cudaMemcpyBatchAsync` (the merged-1D fallback). Never loads a runtime that is
    /// not already in the process image (`RTLD_NOLOAD`).
    pub fn load() -> Option<CudaRuntime> {
        SONAMES.iter().find_map(|soname| Self::probe(soname))
    }

    /// The runtime's own version (`major*1000 + minor*10`), from `cudaRuntimeGetVersion`.
    pub fn version(&self) -> i32 {
        self.version
    }

    /// Enqueue one batch of `count` copies on `stream`: entry `i` moves `sizes[i]` bytes from
    /// `srcs[i]` to `dsts[i]`. The batch executes in stream order as a whole; the copies are
    /// disjoint (the cache coalesces a non-overlapping plan), so intra-batch order is
    /// irrelevant.
    ///
    /// # Safety
    /// The pointers must name valid memory of at least `sizes[i]` bytes on the stream's
    /// device, stay alive until the batch completes, and the destinations must be mutually
    /// disjoint. `stream` must be a created (non-legacy) stream.
    pub unsafe fn memcpy_batch_async(
        &self,
        dsts: &[*mut c_void],
        srcs: &[*const c_void],
        sizes: &[usize],
        stream: *mut c_void,
    ) -> Result<()> {
        let count = dsts.len();
        ensure!(
            srcs.len() == count && sizes.len() == count,
            "cudaMemcpyBatchAsync arrays disagree: {} dsts, {} srcs, {} sizes",
            dsts.len(),
            srcs.len(),
            sizes.len()
        );
        ensure!(
            !stream.is_null(),
            "cudaMemcpyBatchAsync needs a created stream"
        );
        // Every copy must have a valid attribute entry; one entry applying to the whole batch
        // satisfies the contract (see CudaMemcpyAttributes::stream_order).
        let mut attrs = [CudaMemcpyAttributes::stream_order()];
        let mut attrs_idxs = [0usize];
        let status = match self.batch {
            BatchFn::Thirteen(batch) => unsafe {
                batch(
                    dsts.as_ptr(),
                    srcs.as_ptr(),
                    sizes.as_ptr(),
                    count,
                    attrs.as_mut_ptr(),
                    attrs_idxs.as_mut_ptr(),
                    1,
                    stream,
                )
            },
            BatchFn::Twelve(batch) => {
                let mut fail_idx = usize::MAX;
                unsafe {
                    batch(
                        dsts.as_ptr(),
                        srcs.as_ptr(),
                        sizes.as_ptr(),
                        count,
                        attrs.as_mut_ptr(),
                        attrs_idxs.as_mut_ptr(),
                        1,
                        &mut fail_idx,
                        stream,
                    )
                }
            }
        };
        if status != 0 {
            bail!(
                "cudaMemcpyBatchAsync failed with status {}: {}",
                status,
                self.error_string(status)
            );
        }
        Ok(())
    }

    /// The runtime's message for a `cudaError_t`.
    fn error_string(&self, status: c_int) -> String {
        let raw = unsafe { (self.get_error_string)(status) };
        if raw.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned()
    }

    /// Bind one soname with `RTLD_NOLOAD`; `None` on any absence, including a runtime too old
    /// to carry the batch entry point.
    fn probe(soname: &str) -> Option<CudaRuntime> {
        // SAFETY: RTLD_NOLOAD only looks up a library already mapped into the process, so this
        // can neither load a second runtime nor run any new initialization.
        let _library = unsafe { Library::open(Some(soname), RTLD_NOW | RTLD_NOLOAD) }
            .ok()
            .map(Arc::new)?;
        let get_version: RuntimeGetVersionFn =
            unsafe { *_library.get(b"cudaRuntimeGetVersion").ok()? };
        let mut version = 0;
        // SAFETY: `version` is a valid out-pointer for the duration of the call.
        if unsafe { get_version(&mut version) } != 0 {
            return None;
        }
        let get_error_string: GetErrorStringFn =
            unsafe { *_library.get(b"cudaGetErrorString").ok()? };
        let batch = match select_copy_mechanism(Some(version)) {
            CopyMechanism::MemcpyBatch if version >= 13_000 => {
                BatchFn::Thirteen(unsafe { *_library.get(b"cudaMemcpyBatchAsync").ok()? })
            }
            CopyMechanism::MemcpyBatch => {
                BatchFn::Twelve(unsafe { *_library.get(b"cudaMemcpyBatchAsync").ok()? })
            }
            CopyMechanism::Merged1d => return None,
        };
        Some(CudaRuntime {
            _library,
            version,
            batch,
            get_error_string,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mechanism_selection_follows_the_version_ranges() {
        // Absent runtime, and runtimes older than 12.8 (no cudaMemcpyBatchAsync): fallback.
        assert_eq!(select_copy_mechanism(None), CopyMechanism::Merged1d);
        for version in [0, 11_080, 12_000, 12_079] {
            assert_eq!(
                select_copy_mechanism(Some(version)),
                CopyMechanism::Merged1d,
                "version {version} must fall back"
            );
        }
        // 12.8+ through 12.9: the 12.x arity with failIdx.
        for version in [12_080, 12_500, 12_999] {
            assert_eq!(
                select_copy_mechanism(Some(version)),
                CopyMechanism::MemcpyBatch,
                "version {version} must batch"
            );
        }
        // 13.x: the 13.x arity without failIdx.
        for version in [13_000, 13_020, 99_999] {
            assert_eq!(
                select_copy_mechanism(Some(version)),
                CopyMechanism::MemcpyBatch,
                "version {version} must batch"
            );
        }
    }

    #[test]
    fn attributes_layout_matches_the_headers() {
        // The batch call hands this struct to the runtime by pointer, so its size and field
        // offsets are ABI. See the module doc for the header line numbers both versions agree
        // on.
        assert_eq!(std::mem::size_of::<CudaMemcpyAttributes>(), 24);
        assert_eq!(std::mem::size_of::<CudaMemLocation>(), 8);
        let attrs = CudaMemcpyAttributes::stream_order();
        assert_eq!(attrs.src_access_order, 1);
        assert_eq!(attrs.flags, 0);
    }

    #[test]
    fn mechanism_names_are_stable() {
        assert_eq!(CopyMechanism::MemcpyBatch.name(), "cuda-memcpy-batch");
        assert_eq!(CopyMechanism::Merged1d.name(), "merged-1d");
    }

    #[test]
    fn probe_declines_a_soname_that_is_not_loaded() {
        // RTLD_NOLOAD must refuse to map a library that is not already in the process image,
        // so an absent runtime selects the merged-1D fallback. The soname is chosen to not
        // exist on any system, which keeps this true even on a GPU host where a real cudart
        // is loaded: `load()` probes fixed sonames and would find that one instead.
        assert!(CudaRuntime::probe("libds41rt-probe-definitely-absent.so.1").is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn probe_declines_a_loaded_library_without_the_runtime_symbols() {
        // glibc is always mapped into a Linux test process and carries no CUDA entry points,
        // so the probe walks its found-library / absent-symbol path — the same path an
        // already-loaded but pre-12.8 runtime takes — and returns None.
        assert!(CudaRuntime::probe("libc.so.6").is_none());
    }
}
