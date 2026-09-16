//! Initialize a trusted generated EXL3 module and inspect its native ABI.
use anyhow::{Context, Result};
use ds41rt_ffi::V41Exl3Kernel;
fn main() -> Result<()> {
    let path = std::env::args_os().nth(1).context("usage: v41_exl3_module LIBRARY")?;
    let kernel = unsafe { V41Exl3Kernel::load(path)? };
    println!("{:?}", kernel.info());
    // Reject incomplete argument tables in Rust, before entering the driver.
    unsafe {
        assert!(kernel.launch_core(&[], &[], std::ptr::null_mut()).is_err());
        assert!(kernel.launch_sum(&[], &[], std::ptr::null_mut()).is_err());
    }
    println!("native initialization and argument-count rejection passed");
    Ok(())
}
