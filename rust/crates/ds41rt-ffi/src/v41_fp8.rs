//! Library-borrowing native V4.1 block FP8 launch handle.
//! Geometry (32768,8192) is grouped WO-A: eight independent [4096,1024] projections.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::{ffi::c_void, ptr::NonNull};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct V41Fp8Info {
    pub abi_version: u32,
    pub capacity_rows: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub scratch_bytes: u64,
    pub values_offset: u64,
    pub row_scales_offset: u64,
    pub mma_scales_offset: u64,
    pub packed_weight_scale_bytes: u64,
}
const _: [(); 56] = [(); std::mem::size_of::<V41Fp8Info>()];

type InfoFn = unsafe extern "C" fn(i32, i32, i32, *mut V41Fp8Info) -> i32;
type InitFn = unsafe extern "C" fn(i32, i32, i32, *mut *mut c_void) -> i32;
type ScratchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut f32, *mut c_void) -> i32;
type PackFn = unsafe extern "C" fn(*const u8, *mut u8, i32, i32, *mut c_void) -> i32;
type LaunchFn = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    *const f32,
    *const u8,
    *const u8,
    *mut c_void,
    u64,
    *const f32,
    *mut u16,
    i32,
    *mut c_void,
) -> i32;
pub struct V41Fp8Kernel<'a> {
    _library: &'a NativeLibrary,
    handle: NonNull<c_void>,
    info: V41Fp8Info,
    scratch: ScratchFn,
    pack: PackFn,
    launch: LaunchFn,
}
impl NativeLibrary {
    pub fn v41_fp8_info(&self, capacity: u32) -> Result<V41Fp8Info> {
        self.v41_fp8_matrix_info(capacity, 6144, 25600)
    }
    pub fn v41_fp8_matrix_info(
        &self,
        capacity: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<V41Fp8Info> {
        ensure!(
            matches!(
                (input_dim, output_dim),
                (6144, 25600)
                    | (5120, 2304)
                    | (2304, 5120)
                    | (5120, 1152)
                    | (1152, 5120)
                    | (15360, 5120)
                    | (5120, 1280)
                    | (1280, 32768)
                    | (1280, 16384)
                    | (1280, 4096)
                    | (5120, 512)
                    | (8192, 5120)
                    | (8192, 2560)
                    | (32768, 8192)
            ),
            "unsupported native FP8 matrix"
        );
        let capacity = i32::try_from(capacity)?;
        let function: InfoFn = unsafe { *self.lib.get(b"ds41rt_v41_fp8_matrix_info")? };
        let mut info = V41Fp8Info::default();
        let status = unsafe { function(capacity, input_dim as i32, output_dim as i32, &mut info) };
        ensure!(
            status == 0,
            "native V4.1 FP8 info failed with CUDA status {status}"
        );
        ensure!(
            info.abi_version == 1
                && info.capacity_rows == capacity as u32
                && info.input_dim == input_dim
                && info.output_dim == output_dim
                && info.packed_weight_scale_bytes
                    == u64::from(input_dim) * u64::from(output_dim) / groups(input_dim, output_dim) / 32,
            "unsupported native V4.1 FP8 geometry/ABI"
        );
        ensure!(
            info.values_offset < info.scratch_bytes
                && info.row_scales_offset < info.scratch_bytes
                && info.mma_scales_offset < info.scratch_bytes,
            "invalid native FP8 scratch offsets"
        );
        Ok(info)
    }
    pub fn v41_fp8_kernel(&self, capacity: u32) -> Result<V41Fp8Kernel<'_>> {
        self.v41_fp8_matrix_kernel(capacity, 6144, 25600)
    }
    pub fn v41_fp8_matrix_kernel(
        &self,
        capacity: u32,
        input_dim: u32,
        output_dim: u32,
    ) -> Result<V41Fp8Kernel<'_>> {
        let info = self.v41_fp8_matrix_info(capacity, input_dim, output_dim)?;
        unsafe {
            let initialize: InitFn = *self.lib.get(b"ds41rt_v41_fp8_matrix_initialize")?;
            let scratch = *self.lib.get(b"ds41rt_v41_fp8_initialize_scratch")?;
            let pack = *self.lib.get(b"ds41rt_v41_fp8_matrix_pack_scales")?;
            let launch = *self.lib.get(b"ds41rt_v41_fp8_launch_rope")?;
            let mut handle = std::ptr::null_mut();
            let status = initialize(
                i32::try_from(capacity)?,
                input_dim as i32,
                output_dim as i32,
                &mut handle,
            );
            ensure!(
                status == 0,
                "native V4.1 FP8 initialization failed with CUDA status {status}"
            );
            let handle = NonNull::new(handle)
                .ok_or_else(|| anyhow::anyhow!("native FP8 returned null handle"))?;
            Ok(V41Fp8Kernel {
                _library: self,
                handle,
                info,
                scratch,
                pack,
                launch,
            })
        }
    }
}
fn require(buffer: Ds41rtDeviceBuffer, bytes: usize) -> Result<()> {
    ensure!(
        !buffer.ptr.is_null() && buffer.bytes >= bytes,
        "native FP8 buffer is null or too small"
    );
    Ok(())
}
impl V41Fp8Kernel<'_> {
    pub fn info(&self) -> V41Fp8Info {
        self.info
    }
    /// # Safety
    /// Scratch and alpha are distinct current-device allocations that remain live
    /// until completion; initialize outside capture before executing/replaying.
    pub unsafe fn initialize_scratch(
        &self,
        scratch: Ds41rtDeviceBuffer,
        alpha: Ds41rtDeviceBuffer,
        stream: *mut c_void,
    ) -> Result<()> {
        require(scratch, usize::try_from(self.info.scratch_bytes)?)?;
        require(alpha, 4)?;
        let status = unsafe {
            (self.scratch)(
                self.handle.as_ptr(),
                scratch.ptr,
                scratch.bytes as u64,
                alpha.ptr.cast(),
                stream,
            )
        };
        ensure!(
            status == 0,
            "native FP8 scratch initialization failed with CUDA status {status}"
        );
        Ok(())
    }
    /// # Safety
    /// Source is native UE8M0 [output_dim/32,input_dim/32], or [8,32,128]
    /// for grouped WO-A. Destination is distinct current-device
    /// storage, with both allocations live and correctly ordered through completion.
    pub unsafe fn pack_scales(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        stream: *mut c_void,
    ) -> Result<()> {
        require(
            source,
            self.info.packed_weight_scale_bytes as usize / 32,
        )?;
        require(
            destination,
            usize::try_from(self.info.packed_weight_scale_bytes)?,
        )?;
        let status = unsafe {
            (self.pack)(
                source.ptr.cast(),
                destination.ptr.cast(),
                self.info.input_dim as i32,
                self.info.output_dim as i32,
                stream,
            )
        };
        ensure!(
            status == 0,
            "native FP8 scale packing failed with CUDA status {status}"
        );
        Ok(())
    }
    /// # Safety
    /// Source/output are BF16, weights native row-major FP8, scales packed by this
    /// kernel and alpha initialized to one. All buffers are distinct on the current
    /// device, correctly ordered on stream, and remain live through completion and
    /// any graph replay; scratch is exclusively owned by this in-flight wave.
    pub unsafe fn launch(
        &self,
        source: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        scratch: Ds41rtDeviceBuffer,
        alpha: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe { self.launch_inner(source, None, weight, scales, scratch, alpha, output, rows, stream) }
    }
    /// Fuse V4.1 inverse RoPE with grouped input quantization.
    /// # Safety
    /// Same ownership as launch; frequencies are initialized FP32 [rows,32,2]
    /// on the current device and remain immutable through stream completion.
    pub unsafe fn launch_rope(
        &self,
        source: Ds41rtDeviceBuffer,
        frequencies: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        scratch: Ds41rtDeviceBuffer,
        alpha: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe { self.launch_inner(source, Some(frequencies), weight, scales, scratch, alpha, output, rows, stream) }
    }
    unsafe fn launch_inner(
        &self,
        source: Ds41rtDeviceBuffer,
        frequencies: Option<Ds41rtDeviceBuffer>,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        scratch: Ds41rtDeviceBuffer,
        alpha: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.info.capacity_rows,
            "native FP8 rows exceed capacity"
        );
        if let Some(f) = frequencies {
            ensure!((self.info.input_dim,self.info.output_dim)==(32768,8192), "inverse RoPE requires grouped WO-A");
            require(f, rows as usize * 256)?;
        }
        require(source, rows as usize * self.info.input_dim as usize * 2)?;
        require(
            weight,
            self.info.packed_weight_scale_bytes as usize * 32,
        )?;
        require(
            scales,
            usize::try_from(self.info.packed_weight_scale_bytes)?,
        )?;
        require(scratch, usize::try_from(self.info.scratch_bytes)?)?;
        require(alpha, 4)?;
        require(output, rows as usize * self.info.output_dim as usize * 2)?;
        let status = unsafe {
            (self.launch)(
                self.handle.as_ptr(),
                source.ptr.cast(),
                frequencies.map_or(std::ptr::null(), |f| f.ptr.cast()),
                weight.ptr.cast(),
                scales.ptr.cast(),
                scratch.ptr,
                scratch.bytes as u64,
                alpha.ptr.cast(),
                output.ptr.cast(),
                i32::try_from(rows)?,
                stream,
            )
        };
        ensure!(
            status == 0,
            "native FP8 projection failed with CUDA status {status}"
        );
        Ok(())
    }
}

type SwiGluFn = unsafe extern "C" fn(*const u16, *const u16, *mut u16, i32, *mut c_void) -> i32;
pub struct V41SharedSwiGlu<'a> {
    _library: &'a NativeLibrary,
    launch: SwiGluFn,
    width: usize,
}
impl NativeLibrary {
    pub fn v41_shared_swiglu(&self) -> Result<V41SharedSwiGlu<'_>> {
        Ok(V41SharedSwiGlu {
            _library: self,
            launch: unsafe { *self.lib.get(b"ds41rt_v41_shared_swiglu")? },
            width: 2304,
        })
    }
    pub fn v41_shared_tp2_swiglu(&self) -> Result<V41SharedSwiGlu<'_>> {
        Ok(V41SharedSwiGlu { _library: self,
            launch: unsafe { *self.lib.get(b"ds41rt_v41_shared_tp2_swiglu")? }, width: 1152 })
    }
}
impl V41SharedSwiGlu<'_> {
    /// # Safety
    /// Inputs are initialized finite BF16 [rows,width] on the stream device.
    /// Output is distinct; all storage remains live and ordered through completion.
    pub unsafe fn launch(
        &self,
        gate: Ds41rtDeviceBuffer,
        up: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid shared SwiGLU rows");
        for buffer in [gate, up, output] {
            require(buffer, rows as usize * self.width * 2)?;
        }
        let status = unsafe {
            (self.launch)(
                gate.ptr.cast(),
                up.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native shared SwiGLU CUDA status {status}");
        Ok(())
    }
}

// The official WO-A projection is block diagonal over eight head groups.
fn groups(input: u32, output: u32) -> u64 {
    if (input, output) == (32768, 8192) { 8 } else { 1 }
}
