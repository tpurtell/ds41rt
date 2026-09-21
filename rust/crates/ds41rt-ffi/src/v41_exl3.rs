//! Owned mixed EXL3 AOT module. Captured graphs and borrowed device buffers must
//! be released/drained before dropping this owner on its original CUDA thread.
use anyhow::{ensure, Context, Result};
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41Exl3Layout {
    Disjoint,
    PairedFirst,
    PairedLast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41Exl3Info {
    pub layout: V41Exl3Layout,
    /// Bytes per output element: BF16 (2) or FP32 (4).
    pub output_element_bytes: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub experts: usize,
    pub capacity: usize,
    pub topk: usize,
    pub tier_count: usize,
    pub bits: [u32; 4],
    pub core_pointers: usize,
    pub core_scalars: usize,
    pub sum_pointers: usize,
    pub sum_scalars: usize,
}

impl V41Exl3Info {
    fn from_words(words: [u32; 16]) -> Result<Self> {
        ensure!(
            words[0] == 2
                && matches!(words[15], 2 | 4)
                && words[1] == 5120
                // Published rank shard widths: 2304 full (RTX local), 1152 for
                // the implicit TP2 group, 768 for the implicit TP3 group, and
                // 640/512 for TP4, where the first two ranks own one extra
                // whole H128 block.
                && matches!(words[2], 512 | 640 | 768 | 1152 | 2304)
                && words[3] <= 384
                && words[3] >= words[5]
                && words[4] > 0
                && matches!(words[5], 3 | 6)
                && matches!(words[6], 2..=4),
            "invalid EXL3 native geometry: {words:?}"
        );
        let tier_count = words[6] as usize;
        let bits: [u32; 4] = words[11..15].try_into().unwrap();
        ensure!(
            bits[..tier_count].iter().all(|bit| (2..=5).contains(bit))
                && bits[tier_count..].iter().all(|bit| *bit == 0)
                && bits[..tier_count]
                    .iter()
                    .enumerate()
                    .all(|(i, bit)| !bits[..i].contains(bit)),
            "invalid EXL3 native tier bits"
        );
        ensure!(
            words[7..11].iter().all(|count| *count > 0 && *count <= 64),
            "invalid EXL3 pointer/scalar ABI"
        );
        Ok(Self {
            layout: V41Exl3Layout::Disjoint,
            output_element_bytes: words[15] as usize,
            hidden: words[1] as usize,
            intermediate: words[2] as usize,
            experts: words[3] as usize,
            capacity: words[4] as usize,
            topk: words[5] as usize,
            tier_count,
            bits,
            core_pointers: words[7] as usize,
            core_scalars: words[8] as usize,
            sum_pointers: words[9] as usize,
            sum_scalars: words[10] as usize,
        })
    }

    fn from_paired_words(words: [u32; 18]) -> Result<Self> {
        ensure!(
            words[0] == 3
                && words[2] == 640
                && words[5] == 6
                && words[6] == 2
                && matches!(words[16], 1 | 2)
                && words[17] == 4,
            "invalid paired EXL3 native contract: {words:?}"
        );
        let mut original: [u32; 16] = words[..16].try_into().unwrap();
        original[0] = 2;
        let mut info = Self::from_words(original)?;
        info.layout = if words[16] == 1 {
            V41Exl3Layout::PairedFirst
        } else {
            V41Exl3Layout::PairedLast
        };
        Ok(info)
    }

    pub fn require_layout(&self, expected: V41Exl3Layout) -> Result<()> {
        ensure!(
            self.layout == expected,
            "EXL3 native layout mismatch: loaded {:?}, expected {:?}",
            self.layout,
            expected
        );
        Ok(())
    }
}

type Launch = unsafe extern "C" fn(*mut c_void, *const *mut c_void, *const i32, *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);

pub struct V41Exl3Kernel {
    // Function addresses and context are only valid while the module is loaded.
    _library: Library,
    context: NonNull<c_void>,
    info: V41Exl3Info,
    core: Launch,
    sum: Launch,
    destroy: Destroy,
    _owner_thread: PhantomData<Rc<()>>,
}

impl V41Exl3Kernel {
    /// # Safety
    /// Load only trusted generated code, with a current CUDA device. Caller must
    /// keep that device current during launches and drop, and drain every stream
    /// and destroy graphs referencing this module before dropping it.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_layout(path, V41Exl3Layout::Disjoint)
    }

    /// # Safety
    /// Same device/thread and lifetime requirements as `load`. The caller must
    /// use the paired resident weights and four-row descriptor when requested.
    pub unsafe fn load_with_layout(
        path: impl AsRef<Path>,
        expected: V41Exl3Layout,
    ) -> Result<Self> {
        let library = Library::new(path.as_ref())?;
        let query =
            *library.get::<unsafe extern "C" fn(*mut u32, u32) -> i32>(b"ds41rt_exl3_info")?;
        let create =
            *library.get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"ds41rt_exl3_create")?;
        let core = *library.get::<Launch>(b"ds41rt_exl3_core")?;
        let sum = *library.get::<Launch>(b"ds41rt_exl3_sum")?;
        let destroy = *library.get::<Destroy>(b"ds41rt_exl3_destroy")?;
        let info = if let Ok(paired_query) =
            library.get::<unsafe extern "C" fn(*mut u32, u32) -> i32>(b"ds41rt_exl3_paired_info")
        {
            let mut words = [0; 18];
            ensure!(
                paired_query(words.as_mut_ptr(), 18) == 0,
                "paired EXL3 native info query failed"
            );
            V41Exl3Info::from_paired_words(words)?
        } else {
            let mut words = [0; 16];
            ensure!(
                query(words.as_mut_ptr(), 16) == 0,
                "EXL3 native info query failed"
            );
            V41Exl3Info::from_words(words)?
        };
        info.require_layout(expected)?;
        let mut context = std::ptr::null_mut();
        let status = create(&mut context);
        ensure!(
            status == 0,
            "EXL3 native initialization failed with CUDA status {status}"
        );
        let context = NonNull::new(context).context("EXL3 initialized a null context")?;
        Ok(Self {
            _library: library,
            context,
            info,
            core,
            sum,
            destroy,
            _owner_thread: PhantomData,
        })
    }

    pub fn info(&self) -> V41Exl3Info {
        self.info
    }

    /// # Safety
    /// Pointer/scalar order must match the verified export manifest. Device
    /// storage must meet its sizes, types, alignment and non-aliasing contract
    /// and remain alive through stream completion or the last graph replay.
    pub unsafe fn launch_core(
        &self,
        pointers: &[*mut c_void],
        scalars: &[i32],
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            pointers.len() == self.info.core_pointers && scalars.len() == self.info.core_scalars,
            "EXL3 core argument count mismatch"
        );
        let status = (self.core)(
            self.context.as_ptr(),
            pointers.as_ptr(),
            scalars.as_ptr(),
            stream,
        );
        ensure!(status == 0, "EXL3 core failed with CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Same buffer lifetime/order requirements as `launch_core`; the epilogue
    /// must follow the producing core on an ordered stream.
    pub unsafe fn launch_sum(
        &self,
        pointers: &[*mut c_void],
        scalars: &[i32],
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            pointers.len() == self.info.sum_pointers && scalars.len() == self.info.sum_scalars,
            "EXL3 sum argument count mismatch"
        );
        let status = (self.sum)(
            self.context.as_ptr(),
            pointers.as_ptr(),
            scalars.as_ptr(),
            stream,
        );
        ensure!(status == 0, "EXL3 sum failed with CUDA status {status}");
        Ok(())
    }
}

impl Drop for V41Exl3Kernel {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context.as_ptr());
        }
    }
}

/// Native route preparation, owned on the creating CUDA thread/context.
pub struct V41Exl3Routes {
    _library: Library,
    context: NonNull<c_void>,
    launch:
        unsafe extern "C" fn(*mut c_void, *const *mut c_void, *const u64, i32, *mut c_void) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void) -> i32,
    _owner_thread: PhantomData<Rc<()>>,
}
impl V41Exl3Routes {
    /// # Safety
    /// Trusted module only; keep its owning context current and drain streams /
    /// destroy referencing graphs before dropping this handle.
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        let library = Library::new(path.as_ref())?;
        let create = *library
            .get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"ds41rt_exl3_routes_create")?;
        let launch = *library.get(b"ds41rt_exl3_routes_launch")?;
        let destroy = *library.get(b"ds41rt_exl3_routes_destroy")?;
        let mut context = std::ptr::null_mut();
        let status = create(&mut context);
        ensure!(status == 0, "EXL3 route initialization failed: {status}");
        Ok(Self {
            _library: library,
            context: NonNull::new(context).context("null EXL3 route context")?,
            launch,
            destroy,
            _owner_thread: PhantomData,
        })
    }
    /// # Safety
    /// Seven int32 buffers in manifest order, correctly sized/aligned, distinct,
    /// on the owning device and alive through all uses of this stream/graph.
    pub unsafe fn launch(
        &self,
        pointers: &[*mut c_void; 7],
        bytes: &[u64; 7],
        rows: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        let status = (self.launch)(
            self.context.as_ptr(),
            pointers.as_ptr(),
            bytes.as_ptr(),
            rows,
            stream,
        );
        ensure!(status == 0, "EXL3 route launch failed: {status}");
        Ok(())
    }
}
impl Drop for V41Exl3Routes {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.context.as_ptr());
        }
    }
}

#[cfg(test)]
mod info_tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU, DS41RT_NATIVE_LIB and four-tier DS41RT_EXL3_AOT"]
    fn native_four_tier_library_loads_through_rust_owner() -> Result<()> {
        let native = unsafe { crate::NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        native.cuda_set_device(0)?;
        let path =
            std::path::PathBuf::from(std::env::var("DS41RT_EXL3_AOT")?).join("libds41rt_exl3.so");
        let first = unsafe { V41Exl3Kernel::load(&path)? };
        let second = unsafe { V41Exl3Kernel::load(&path)? };
        assert_eq!(first.info().tier_count, 4);
        assert_eq!(first.info().bits, [2, 3, 4, 5]);
        assert_eq!(first.info(), second.info());
        drop(first);
        drop(second);
        let reloaded = unsafe { V41Exl3Kernel::load(&path)? };
        assert_eq!(reloaded.info().tier_count, 4);
        Ok(())
    }

    #[test]
    fn paired_native_info_requires_explicit_matching_layout() {
        let valid = [3, 5120, 640, 6, 16, 6, 2, 32, 12, 7, 3, 3, 4, 0, 0, 2, 1, 4];
        for (boundary, layout) in [
            (1, V41Exl3Layout::PairedFirst),
            (2, V41Exl3Layout::PairedLast),
        ] {
            let mut words = valid;
            words[16] = boundary;
            let info = V41Exl3Info::from_paired_words(words).unwrap();
            assert!(info.require_layout(layout).is_ok());
            assert!(info.require_layout(V41Exl3Layout::Disjoint).is_err());
            assert!(info
                .require_layout(if boundary == 1 {
                    V41Exl3Layout::PairedLast
                } else {
                    V41Exl3Layout::PairedFirst
                })
                .is_err());
        }
        for (index, value) in [
            (0, 2),
            (2, 512),
            (5, 3),
            (6, 3),
            (16, 0),
            (16, 3),
            (17, 3),
            (11, 1),
        ] {
            let mut words = valid;
            words[index] = value;
            assert!(V41Exl3Info::from_paired_words(words).is_err());
        }
        let mut original: [u32; 16] = valid[..16].try_into().unwrap();
        assert!(V41Exl3Info::from_words(original).is_err());
        original[0] = 2;
        let disjoint = V41Exl3Info::from_words(original).unwrap();
        assert!(disjoint.require_layout(V41Exl3Layout::Disjoint).is_ok());
        assert!(disjoint.require_layout(V41Exl3Layout::PairedFirst).is_err());
    }

    #[test]
    fn native_info_accepts_two_through_four_distinct_tiers() {
        for bits in [&[2, 5][..], &[2, 3, 5], &[2, 3, 4, 5]] {
            let mut words = [
                2,
                5120,
                512,
                6,
                16,
                6,
                bits.len() as u32,
                44,
                18,
                7,
                3,
                0,
                0,
                0,
                0,
                2,
            ];
            words[11..11 + bits.len()].copy_from_slice(bits);
            let info = V41Exl3Info::from_words(words).unwrap();
            assert_eq!(info.tier_count, bits.len());
            assert_eq!(&info.bits[..bits.len()], bits);
        }
    }

    /// Every published rank shard width is a distinct accepted geometry. The
    /// implicit three-rank EXL3 group exports 768 (18 whole H128 blocks split
    /// three ways), which must not be confused with the native FP8 shard widths
    /// 384/576 that no EXL3 publication ever produces.
    #[test]
    fn native_info_accepts_every_published_shard_width() {
        let base = [2u32, 5120, 768, 384, 16, 6, 2, 44, 18, 7, 3, 2, 3, 0, 0, 2];
        for intermediate in [512u32, 640, 768, 1152, 2304] {
            let mut words = base;
            words[2] = intermediate;
            let info = V41Exl3Info::from_words(words)
                .unwrap_or_else(|error| panic!("shard width {intermediate} must be accepted: {error}"));
            assert_eq!(info.intermediate, intermediate as usize);
        }
        for intermediate in [0u32, 384, 576, 896, 1024, 1280] {
            let mut words = base;
            words[2] = intermediate;
            assert!(
                V41Exl3Info::from_words(words).is_err(),
                "shard width {intermediate} must be rejected"
            );
        }
    }

    #[test]
    fn native_info_rejects_invalid_tier_headers_before_slicing() {
        let valid = [2, 5120, 512, 6, 16, 6, 4, 44, 18, 7, 3, 2, 3, 4, 5, 2];
        for count in [0, 1, 5, u32::MAX] {
            let mut words = valid;
            words[6] = count;
            assert!(V41Exl3Info::from_words(words).is_err());
        }
        for (index, value) in [(11, 1), (14, 6), (12, 2), (7, 65)] {
            let mut words = valid;
            words[index] = value;
            assert!(V41Exl3Info::from_words(words).is_err());
        }
        let mut nonzero_padding = valid;
        nonzero_padding[6] = 3;
        assert!(V41Exl3Info::from_words(nonzero_padding).is_err());
    }
}
