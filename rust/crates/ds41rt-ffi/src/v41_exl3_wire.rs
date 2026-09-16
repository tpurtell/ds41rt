use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use std::{ffi::c_void, marker::PhantomData, ptr::NonNull, rc::Rc};

type Decode = unsafe extern "C" fn(*mut c_void, *const u8, u64, *mut u16, u64, u32, *mut c_void) -> i32;
pub struct V41Exl3Wire<'a> {
    _library: &'a NativeLibrary,
    context: NonNull<c_void>,
    decode: Decode,
    destroy: unsafe extern "C" fn(*mut c_void),
    _owner_thread: PhantomData<Rc<()>>,
}
impl NativeLibrary {
    pub fn v41_exl3_wire(&self) -> Result<V41Exl3Wire<'_>> {
        let initialize = unsafe { *self.lib.get::<unsafe extern "C" fn(*mut *mut c_void) -> i32>(b"ds41rt_v41_exl3_wire_initialize")? };
        let decode = unsafe { *self.lib.get(b"ds41rt_v41_exl3_wire_decode")? };
        let destroy = unsafe { *self.lib.get(b"ds41rt_v41_exl3_wire_destroy")? };
        let mut context = std::ptr::null_mut();
        let status = unsafe { initialize(&mut context) };
        ensure!(status == 0, "EXL3 wire initialization failed: {status}");
        Ok(V41Exl3Wire { _library: self, context: NonNull::new(context).context("null EXL3 wire context")?,
            decode, destroy, _owner_thread: PhantomData })
    }
}
impl V41Exl3Wire<'_> {
    /// # Safety
    /// Input/output are distinct on the owning current device and must stay
    /// valid through stream completion. Destroy referencing graphs before this
    /// handle. Input is E4M3+UE8M0 K32 in 5280-byte rows; output is BF16.
    pub unsafe fn decode(&self, input: Ds41rtDeviceBuffer, output: Ds41rtDeviceBuffer, rows: usize, stream: *mut c_void) -> Result<()> {
        ensure!((1..=4096).contains(&rows) && input.device_id == output.device_id, "invalid EXL3 wire extent/device");
        let status = (self.decode)(self.context.as_ptr(), input.ptr.cast(), input.bytes as u64,
            output.ptr.cast(), output.bytes as u64, rows as u32, stream);
        ensure!(status == 0, "EXL3 wire decoding failed: {status}");
        Ok(())
    }
}
impl Drop for V41Exl3Wire<'_> {
    fn drop(&mut self) { unsafe { (self.destroy)(self.context.as_ptr()); } }
}
