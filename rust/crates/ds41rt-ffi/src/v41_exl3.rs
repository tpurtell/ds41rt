//! Owned mixed EXL3 AOT module. Captured graphs and borrowed device buffers must
//! be released/drained before dropping this owner on its original CUDA thread.
use anyhow::{ensure, Context, Result};
use libloading::Library;
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41Exl3Info {
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
        let library = Library::new(path.as_ref())?;
        let query =
            *library.get::<unsafe extern "C" fn(*mut u32, u32) -> i32>(b"ds41rt_exl3_info")?;
        let create =
            *library.get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"ds41rt_exl3_create")?;
        let core = *library.get::<Launch>(b"ds41rt_exl3_core")?;
        let sum = *library.get::<Launch>(b"ds41rt_exl3_sum")?;
        let destroy = *library.get::<Destroy>(b"ds41rt_exl3_destroy")?;
        let mut words = [0; 15];
        ensure!(
            query(words.as_mut_ptr(), 15) == 0,
            "EXL3 native info query failed"
        );
        ensure!(
            words[0] == 1
                && words[1] == 5120
                && matches!(words[2], 512 | 640 | 1152 | 2304)
                && words[3] <= 384
                && words[3] >= words[5]
                && words[4] > 0
                && matches!(words[5], 3 | 6)
                && matches!(words[6], 2 | 3),
            "invalid EXL3 native geometry: {words:?}"
        );
        let tier_count = words[6] as usize;
        let bits: [u32; 4] = words[11..15].try_into().unwrap();
        ensure!(
            bits[..tier_count].iter().all(|bit| (2..=5).contains(bit))
                && bits[tier_count..].iter().all(|bit| *bit == 0),
            "invalid EXL3 native tier bits"
        );
        ensure!(
            words[7..11].iter().all(|count| *count > 0 && *count <= 64),
            "invalid EXL3 pointer/scalar ABI"
        );
        let info = V41Exl3Info {
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
        };
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
