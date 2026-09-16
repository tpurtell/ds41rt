use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Mixes = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    i32,
    *mut c_void,
) -> i32;
type MixesWorkspace = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut c_void,
    u64,
    i32,
    *mut c_void,
) -> i32;
type Begin = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    u64,
    i32,
    *mut c_void,
) -> i32;
type Pre = unsafe extern "C" fn(*const u16, *const f32, *mut u16, i32, *mut c_void) -> i32;
type Post = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const f32,
    *const f32,
    *mut u16,
    i32,
    *mut c_void,
) -> i32;
pub struct V41Hc<'a> {
    _library: &'a NativeLibrary,
    pre: Pre,
    post: Post,
    mixes: Mixes,
    mixes_workspace: MixesWorkspace,
    begin: Option<Begin>,
}
impl NativeLibrary {
    pub fn v41_hc(&self) -> Result<V41Hc<'_>> {
        let initialize = unsafe {
            self.lib
                .get::<unsafe extern "C" fn() -> i32>(b"ds41rt_v41_hc_project_initialize")?
        };
        let status = unsafe { initialize() };
        ensure!(status == 0, "mHC AOT initialization CUDA status {status}");
        Ok(V41Hc {
            _library: self,
            begin: unsafe {
                self.lib
                    .get::<Begin>(b"ds41rt_v41_hc_begin")
                    .ok()
                    .map(|symbol| *symbol)
            },
            mixes_workspace: unsafe {
                *self
                    .lib
                    .get::<MixesWorkspace>(b"ds41rt_v41_hc_mixes_workspace")?
            },
            mixes: unsafe { *self.lib.get::<Mixes>(b"ds41rt_v41_hc_mixes")? },
            pre: unsafe { *self.lib.get::<Pre>(b"ds41rt_v41_hc_pre")? },
            post: unsafe { *self.lib.get::<Post>(b"ds41rt_v41_hc_post")? },
        })
    }
}
fn buffers(rows: usize, views: &[(Ds41rtDeviceBuffer, usize)]) -> Result<()> {
    ensure!((1..=4096).contains(&rows), "invalid mHC rows");
    for (view, stride) in views {
        ensure!(
            !view.ptr.is_null() && view.bytes >= rows * stride,
            "invalid mHC buffer"
        );
    }
    Ok(())
}
impl V41Hc<'_> {
    /// # Safety
    /// Inputs are initialized on this device. Outputs and scratch are disjoint
    /// from every input and each other, and remain live through stream completion.
    /// Returns false only when this library or row count uses the legacy path.
    pub unsafe fn try_begin(
        &self,
        residual: Ds41rtDeviceBuffer,
        projection: Ds41rtDeviceBuffer,
        scale: Ds41rtDeviceBuffer,
        base: Ds41rtDeviceBuffer,
        incoming: Ds41rtDeviceBuffer,
        norm: Ds41rtDeviceBuffer,
        predicted: Ds41rtDeviceBuffer,
        post: Ds41rtDeviceBuffer,
        comb: Ds41rtDeviceBuffer,
        normalized: Ds41rtDeviceBuffer,
        scratch: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<bool> {
        let Some(begin) = self.begin else {
            return Ok(false);
        };
        if rows > 80 {
            return Ok(false);
        }
        buffers(
            rows,
            &[
                (residual, 40960),
                (incoming, 16),
                (predicted, 16),
                (post, 16),
                (comb, 64),
                (normalized, 10240),
                (scratch, 8000),
            ],
        )?;
        buffers(
            1,
            &[
                (projection, 24 * 20480 * 4),
                (scale, 12),
                (base, 96),
                (norm, 10240),
            ],
        )?;
        let status = unsafe {
            begin(
                residual.ptr,
                projection.ptr,
                scale.ptr,
                base.ptr,
                incoming.ptr,
                norm.ptr,
                predicted.ptr,
                post.ptr,
                comb.ptr,
                normalized.ptr,
                scratch.ptr,
                scratch.bytes as u64,
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "mHC begin CUDA status {status}");
        Ok(true)
    }

    /// # Safety
    /// Inputs must be initialized on the stream device and all buffers remain
    /// live through completion; outputs must be mutually disjoint and disjoint
    /// from inputs. Coefficients produced here belong to the following sublayer.
    pub unsafe fn mixes(
        &self,
        residual: Ds41rtDeviceBuffer,
        projection: Ds41rtDeviceBuffer,
        scale: Ds41rtDeviceBuffer,
        base: Ds41rtDeviceBuffer,
        pre: Ds41rtDeviceBuffer,
        post: Ds41rtDeviceBuffer,
        comb: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(
            rows,
            &[(residual, 40960), (pre, 16), (post, 16), (comb, 64)],
        )?;
        buffers(1, &[(projection, 24 * 20480 * 4), (scale, 12), (base, 96)])?;
        let status = unsafe {
            (self.mixes)(
                residual.ptr.cast(),
                projection.ptr.cast(),
                scale.ptr.cast(),
                base.ptr.cast(),
                pre.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "mHC mixes CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Inputs must be initialized on the stream device and all buffers remain
    /// live through completion; outputs must be mutually disjoint and disjoint
    /// from inputs. Scratch must also be disjoint from all inputs and outputs,
    /// and remain live through completion. Coefficients produced here belong to the following sublayer.
    pub unsafe fn mixes_workspace(
        &self,
        residual: Ds41rtDeviceBuffer,
        projection: Ds41rtDeviceBuffer,
        scale: Ds41rtDeviceBuffer,
        base: Ds41rtDeviceBuffer,
        pre: Ds41rtDeviceBuffer,
        post: Ds41rtDeviceBuffer,
        comb: Ds41rtDeviceBuffer,
        scratch: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(
            rows,
            &[(residual, 40960), (pre, 16), (post, 16), (comb, 64)],
        )?;
        buffers(1, &[(projection, 24 * 20480 * 4), (scale, 12), (base, 96)])?;
        if rows <= 16 {
            buffers(rows, &[(scratch, 1536)])?;
        }
        let status = unsafe {
            (self.mixes_workspace)(
                residual.ptr.cast(),
                projection.ptr.cast(),
                scale.ptr.cast(),
                base.ptr.cast(),
                pre.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                scratch.ptr,
                scratch.bytes as u64,
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "mHC mixes CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Inputs must be initialized on the stream's device and all buffers remain
    /// live through completion; output must be disjoint from both inputs.
    pub unsafe fn pre(
        &self,
        residual: Ds41rtDeviceBuffer,
        pre: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(rows, &[(residual, 40960), (pre, 16), (output, 10240)])?;
        let status = unsafe {
            (self.pre)(
                residual.ptr.cast(),
                pre.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "mHC pre CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// Same device/lifetime/output-disjointness contract as pre. Comb is FP32
    /// [rows,source,destination], not destination-major.
    pub unsafe fn post(
        &self,
        sublayer: Ds41rtDeviceBuffer,
        residual: Ds41rtDeviceBuffer,
        post: Ds41rtDeviceBuffer,
        comb: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(
            rows,
            &[
                (sublayer, 10240),
                (residual, 40960),
                (post, 16),
                (comb, 64),
                (output, 40960),
            ],
        )?;
        let status = unsafe {
            (self.post)(
                sublayer.ptr.cast(),
                residual.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "mHC post CUDA status {status}");
        Ok(())
    }
}
