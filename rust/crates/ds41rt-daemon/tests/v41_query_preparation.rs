//! Real-weight query-chain parity, graph shape replacement, and error recovery.
#[path = "../src/v41_attention_binding.rs"]
mod v41_attention_binding;
#[path = "../src/v41_attention_query.rs"]
mod v41_attention_query;
#[path = "../src/v41_projection_tp2.rs"]
mod v41_projection_tp2;
#[path = "../src/v41_hc.rs"]
mod v41_hc;
#[path = "../src/v41_layer_graphs.rs"]
mod v41_layer_graphs;
#[path = "../src/v41_memory.rs"]
mod v41_memory;
#[path = "../src/v41_tensors.rs"]
mod v41_tensors;
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::{read_official_v41_catalog, OFFICIAL_V41_MODEL_ID};
fn read(lib: &NativeLibrary, b: Ds41rtDeviceBuffer) -> Result<Vec<u8>> {
    let mut v = vec![0; b.bytes];
    lib.copy_d2h(&mut v, b)?;
    Ok(v)
}
#[test]
#[ignore = "requires the official V4.1 snapshot and a CUDA native library"]
fn query_preparation_preserves_values_and_drains_producer_errors() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    let cat = read_official_v41_catalog(
        OFFICIAL_V41_MODEL_ID,
        std::path::Path::new(&std::env::var("DS41RT_V41_SNAPSHOT")?),
    )?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let mut cases = 0;
    let mut layers = Vec::new();
    for layer in [0, 14, 39] {
        let qw = v41_attention_query::AttentionQueryWeights::load(
            &lib,
            &cat,
            layer,
            usize::MAX,
            1 << 20,
        )?;
        let names = [
            format!("layers.{layer}.hc_attn_fn"),
            format!("layers.{layer}.hc_attn_scale"),
            format!("layers.{layer}.hc_attn_base"),
            format!("layers.{layer}.attn_norm.weight"),
        ];
        let hw = v41_tensors::NativeRtxTensors::load(&lib, &cat, &names, usize::MAX, 1 << 20)?;
        layers.push((layer, qw, hw, names));
    }
    let mut hc = v41_hc::HcSublayer::new(&lib, &layers[0].2, layers[0].3.clone(), 256, usize::MAX)?;
    let mut q = layers[0].1.wave(256, usize::MAX)?;
    let mut alternate = layers[0].1.wave(256, usize::MAX)?;
    for index in [0, 1, 2, 0] {
        let (layer, qw, hw, names) = &layers[index];
        // Every query call drained; discard the unfinished fixture sublayer
        // before rebinding, as the fixture does not execute attention/FFN.
        hc.invalidate();
        let binding = hc.prepare_binding(hw, names.clone())?;
        unsafe {
            hc.install_binding(binding);
        }
        q.rebind(qw)?;
        alternate.rebind(qw)?;
        for rows in [1, 6, 16, 80, 256, 6, 1] {
            for changed in [false, true] {
                let vals = (0..256 * 4 * 5120)
                    .flat_map(|i| {
                        let x =
                            ((i * 17 % 97) as f32 - 48.) / 32. * if changed { -0.5 } else { 1. };
                        ((x.to_bits() >> 16) as u16).to_ne_bytes()
                    })
                    .collect::<Vec<_>>();
                lib.copy_h2d(hc.inputs()[0], &vals)?;
                let pre = (0..256 * 4)
                    .flat_map(|i| (if i % 4 == 0 { 1f32 } else { 0. }).to_ne_bytes())
                    .collect::<Vec<_>>();
                lib.copy_h2d(hc.inputs()[1], &pre)?;
                let tokens = (0..rows)
                    .map(|i| i as u64 + if changed { 777 } else { 129 })
                    .collect::<Vec<_>>();
                let normalized = unsafe { hc.begin(rows)? };
                lib.copy_d2d(q.input(), normalized, normalized.bytes)?;
                let o = unsafe { q.execute_tokens(&tokens)? };
                let baseline = [
                    o.hidden,
                    o.raw_rank,
                    o.normalized_rank,
                    o.projected,
                    o.rotated,
                    o.frequencies,
                ]
                .into_iter()
                .map(|b| read(&lib, b))
                .collect::<Result<Vec<_>>>()?;
                lib.copy_h2d(q.input(), &vec![0; q.input().bytes])?;
                let o = unsafe {
                    runtime.block_on(q.execute_tokens_prepared_cooperative(&tokens, |stream, input| {
                        hc.enqueue_begin(rows, Some(input), stream)?;
                        Ok(())
                    }))?
                };
                for (b, expected) in [
                    o.hidden,
                    o.raw_rank,
                    o.normalized_rank,
                    o.projected,
                    o.rotated,
                    o.frequencies,
                ]
                .into_iter()
                .zip(&baseline)
                {
                    ensure!(
                        &read(&lib, b)? == expected,
                        "query changed layer={layer} rows={rows}"
                    );
                }
                let failed = unsafe {
                    runtime.block_on(q.execute_tokens_prepared_cooperative(&tokens, |stream, input| {
                        hc.enqueue_begin(rows, Some(input), stream)?;
                        anyhow::bail!("injected producer failure")
                    }))
                };
                ensure!(failed.is_err(), "producer error lost");
                ensure!(q.output().is_err(), "failed output published");
                lib.copy_h2d(q.input(), &vec![0; q.input().bytes])?;
                let o = unsafe {
                    runtime.block_on(q.execute_tokens_prepared_cooperative(&tokens, |stream, input| {
                        hc.enqueue_begin(rows, Some(input), stream)?;
                        Ok(())
                    }))?
                };
                ensure!(read(&lib, o.rotated)? == baseline[4], "recovery differs");
                // Switch normalized destinations without changing weights/rows.
                lib.copy_h2d(alternate.input(), &vec![0; alternate.input().bytes])?;
                let alt = unsafe {
                    runtime.block_on(alternate.execute_tokens_prepared_cooperative(&tokens, |stream, input| {
                        hc.enqueue_begin(rows, Some(input), stream).map(|_| ())
                    }))?
                };
                ensure!(
                    read(&lib, alt.rotated)? == baseline[4],
                    "destination switch differs"
                );
                cases += 1;
            }
        }
    }
    println!("{cases} real-weight cases bit-exact; partial-producer failure and reuse passed");
    Ok(())
}
