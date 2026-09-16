//! Load trusted EXL3 modules together and inspect their independent native ABIs.
use anyhow::{ensure, Result};
use ds41rt_ffi::V41Exl3Kernel;
fn main() -> Result<()> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(!paths.is_empty(), "usage: v41_exl3_module LIBRARY [LIBRARY ...]");
    let mut kernels = Vec::new();
    for path in paths {
        let kernel = unsafe { V41Exl3Kernel::load(&path)? };
        println!("{}: {:?}", std::path::Path::new(&path).display(), kernel.info());
        unsafe {
            assert!(kernel.launch_core(&[], &[], std::ptr::null_mut()).is_err());
            assert!(kernel.launch_sum(&[], &[], std::ptr::null_mut()).is_err());
        }
        kernels.push(kernel);
    }
    println!("{} modules loaded concurrently; native initialization and argument-count rejection passed", kernels.len());
    Ok(())
}
