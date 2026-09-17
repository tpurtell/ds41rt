use super::*;

#[test]
#[ignore = "requires DS41RT_NATIVE_LIB and two CUDA GPUs"]
fn k7_sparse_reservation_preserves_k5_accounting_and_stable_query_storage() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    for gpu in 0..2 {
        crate::v41_memory::device::Device { library: &lib, id: gpu }.run(|| {
            let budget = SparseAttentionWave::device_bytes(80)?;
            let mut wave = SparseAttentionWave::new(&lib, 80, budget)?;
            let used = |wave: &SparseAttentionWave<'_>| wave.query.buffer.bytes + wave.output.buffer.bytes
                + wave.metadata.buffer.bytes + wave.replay_begins.buffer.bytes
                + wave.descriptors.buffer.bytes + wave.split_scratch.buffer.bytes;
            assert_eq!(used(&wave), budget);
            let query = wave.query.buffer.ptr;
            let output = wave.output.buffer.ptr;
            let original_scratch = wave.split_scratch.buffer.ptr;
            wave.reserve_decode_rows(48)?;
            assert_eq!(wave.split_scratch.buffer.ptr, original_scratch);
            assert_eq!(used(&wave), budget);
            wave.reserve_decode_rows(64)?;
            assert_eq!(used(&wave) - budget, 16 * (10 * 64 * 514 * 4 + 120));
            assert_eq!(wave.query.buffer.ptr, query);
            assert_eq!(wave.output.buffer.ptr, output);
            let scratch = wave.split_scratch.buffer.ptr;
            wave.reserve_decode_rows(64)?;
            assert_eq!(wave.split_scratch.buffer.ptr, scratch);
            assert!(wave.reserve_decode_rows(65).is_err());
            wave.enable_small_graph_shapes();
            assert_eq!(wave.graph_limit, 64);
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
#[ignore = "requires DS41RT_NATIVE_LIB and two CUDA GPUs"]
fn committed_prefix_survives_append_but_private_boundary_stays_exact() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    for gpu in 0..2 {
        crate::v41_memory::device::Device {
            library: &lib,
            id: gpu,
        }
        .run(|| {
            let kernel = lib.v41_sparse_attention()?;
            let allocate = |bytes, value| -> Result<DeviceAllocation<'_>> {
                let allocation = DeviceAllocation::new(&lib, bytes)?;
                lib.copy_h2d(allocation.buffer, &vec![value; bytes])?;
                Ok(allocation)
            };
            let query = allocate(65536, 0)?;
            let sink = allocate(256, 0)?;
            let metadata = allocate(80, 0)?;
            let selected = allocate(2048, 255)?;
            let mut ids = vec![-1i32; 512];
            ids[0] = 0;
            lib.copy_h2d(
                selected.buffer,
                &ids.iter().flat_map(|n| n.to_ne_bytes()).collect::<Vec<_>>(),
            )?;
            let window_values = allocate(128 * 512, 0x38)?;
            let window_scales = allocate(128 * 16, 0x38)?;
            let window_end = allocate(8, 0)?;
            let source_values = allocate(256 * V41Kv::COMPRESSED_VALUE_BYTES, 0x22)?;
            let source_scales = allocate(256 * V41Kv::COMPRESSED_SCALE_BYTES, 0x38)?;
            let source_end = allocate(8, 0)?;
            let pages = allocate(4, 0)?;
            let output = allocate(65536, 0)?;
            let scratch = allocate(V41SparseAttention::split_scratch_bytes(1, 10)?, 0)?;
            let window = V41SparseWindow {
                values: window_values.buffer,
                scales: window_scales.buffer,
                proposals: window_values.buffer,
                proposal_scales: window_scales.buffer,
                end: window_end.buffer,
                proposal_capacity: 128,
                replay_begins: None,
            };
            let source = V41SparseSource {
                values: source_values.buffer,
                scales: source_scales.buffer,
                proposals: source_values.buffer,
                proposal_scales: source_scales.buffer,
                pages: pages.buffer,
                end: source_end.buffer,
                capacity: 256,
                proposal_capacity: 1,
                page_stride: 1,
            };
            let stream = LoadStream {
                library: &lib,
                raw: unsafe { lib.cuda_stream_create()? },
            };
            let append_stream = LoadStream {
                library: &lib,
                raw: unsafe { lib.cuda_stream_create()? },
            };
            let appended_values = allocate(V41Kv::COMPRESSED_VALUE_BYTES, 0x66)?;
            let appended_scales = allocate(V41Kv::COMPRESSED_SCALE_BYTES, 0x40)?;
            let destination = allocate(8, 0)?;
            lib.copy_h2d(destination.buffer, &1u64.to_ne_bytes())?;
            let mut append_end = HostAllocation::new(&lib, 8)?;
            append_end.bytes_mut().copy_from_slice(&2u64.to_ne_bytes());
            let outputs = allocate(64 * 65536, 0)?;
            let kv = lib.v41_compressed_kv()?;
            for split in [None, Some((scratch.buffer, 10))] {
                let run = |end: u64, private: u64| -> Result<Vec<u8>> {
                    let values = [0u64, 0, 1, 0, 0, 1, 1, private, 0, 1];
                    lib.copy_h2d(
                        metadata.buffer,
                        &values
                            .iter()
                            .flat_map(|n| n.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    lib.copy_h2d(source_end.buffer, &end.to_ne_bytes())?;
                    unsafe {
                        kernel.launch(
                            query.buffer,
                            sink.buffer,
                            metadata.buffer,
                            Some(selected.buffer),
                            &window,
                            Some(&source),
                            output.buffer,
                            1,
                            0,
                            split,
                            stream.raw,
                        )?;
                        lib.cuda_stream_synchronize(stream.raw)?;
                    }
                    let mut bytes = vec![0; output.buffer.bytes];
                    lib.copy_d2h(&mut bytes, output.buffer)?;
                    Ok(bytes)
                };
                let prefix = run(1, 0)?;
                assert!(
                    prefix.iter().any(|&byte| byte != 0),
                    "valid prefix produced no attention"
                );
                assert_eq!(
                    run(2, 0)?,
                    prefix,
                    "append invalidated a committed causal prefix"
                );
                assert!(
                    run(0, 0)?.iter().all(|&byte| byte == 0),
                    "shortened backing accepted"
                );
                assert!(
                    run(1, 1)?.iter().any(|&byte| byte != 0),
                    "valid private boundary rejected"
                );
                assert!(
                    run(2, 1)?.iter().all(|&byte| byte == 0),
                    "stale private boundary accepted"
                );
                let prefix = run(1, 0)?;
                // A follower writes distinct values to the next physical row
                // and extends the shared length while prefix kernels execute.
                for iteration in 0..64 {
                    let out = Ds41rtDeviceBuffer {
                        ptr: unsafe { outputs.buffer.ptr.cast::<u8>().add(iteration * 65536).cast() },
                        bytes: 65536, ..outputs.buffer
                    };
                    unsafe {
                        kernel.launch(query.buffer, sink.buffer, metadata.buffer,
                            Some(selected.buffer), &window, Some(&source), out,
                            1, 0, split, stream.raw)?;
                        kv.store(appended_values.buffer, appended_scales.buffer,
                            destination.buffer, source_values.buffer, source_scales.buffer,
                            1, 256, append_stream.raw)?;
                        lib.copy_h2d_async(source_end.buffer, append_end.bytes_mut(), append_stream.raw)?;
                    }
                }
                unsafe {
                    lib.cuda_stream_synchronize(stream.raw)?;
                    lib.cuda_stream_synchronize(append_stream.raw)?;
                }
                let mut bytes = vec![0; outputs.buffer.bytes];
                lib.copy_d2h(&mut bytes, outputs.buffer)?;
                for (iteration, actual) in bytes.chunks_exact(65536).enumerate() {
                    assert!(actual == prefix,
                        "concurrent append changed prefix attention: gpu={gpu}, iteration={iteration}, split={}", split.is_some());
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn compact_wave_budget_preserves_full_geometry() -> Result<()> {
    for capacity in [1,48,80,4096] {
        let scratch=V41SparseAttention::split_scratch_bytes(capacity.min(48),10)?;
        let descriptors=V41SparseAttention::batch_descriptor_bytes(capacity.min(48))?;
        assert_eq!(SparseAttentionWave::device_bytes(capacity)?,capacity*131160+scratch+descriptors);
        assert_eq!(CompactSparseAttentionWave::device_bytes(capacity)?,capacity*(65536+88)+scratch/2+descriptors);
    }
    Ok(())
}

#[test]
#[ignore = "requires DS41RT_NATIVE_LIB with compact attention and two CUDA GPUs"]
fn compact_wave_batch_graph_consumes_local_buffers() -> Result<()> {
    let lib=unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    for gpu in 0..2 {
        crate::v41_memory::device::Device { library:&lib,id:gpu }.run(|| {
            let mut wave=CompactSparseAttentionWave::new(&lib,80,CompactSparseAttentionWave::device_bytes(80)?)?;
            let input=wave.input().ptr;
            wave.reserve_decode_rows(64)?;
            assert_eq!(wave.input().ptr,input);
            assert_eq!(wave.split_scratch.buffer.bytes,64*10*32*514*4);
            let allocate=|bytes,value|->Result<DeviceAllocation<'_>> {
                let allocation=DeviceAllocation::new(&lib,bytes)?;
                lib.copy_h2d(allocation.buffer,&vec![value;bytes])?;Ok(allocation)
            };
            let values=allocate(128*512,0x38)?;
            let scales=allocate(128*16,127)?;
            let end=allocate(8,0)?;
            let sink=allocate(32*4,0)?;
            lib.copy_h2d(wave.query.buffer,&vec![0;wave.query.buffer.bytes])?;
            lib.copy_h2d(wave.replay_begins.buffer,&vec![0;wave.replay_begins.buffer.bytes])?;
            let fields:Vec<u64>=[(1,0),(3,0),(3,1),(3,2)].into_iter()
                .flat_map(|(count,pos)|[0,0,count,pos,0,0,0,0,0,0]).collect();
            lib.copy_h2d(slice(wave.metadata.buffer,0,4*80),
                &fields.iter().flat_map(|v|v.to_ne_bytes()).collect::<Vec<_>>())?;
            let window=||V41SparseWindow { values:values.buffer,scales:scales.buffer,
                proposals:values.buffer,proposal_scales:scales.buffer,end:end.buffer,
                proposal_capacity:128,replay_begins:Some(wave.replay_begins.buffer) };
            let windows=[window(),window()];
            let batch=wave.kernel.prepare_batch(wave.query.buffer,sink.buffer,wave.metadata.buffer,None,
                &[(&windows[0],None,1),(&windows[1],None,3)],wave.output.buffer,
                wave.descriptors.buffer,wave.replay_begins.buffer,wave.split_scratch.buffer)?;
            assert_eq!(batch.backend_key(),2);
            lib.copy_h2d(wave.descriptors.buffer,batch.bytes())?;
            let plan=ColdSparse { layer:0,rows:4,sink:sink.buffer,launches:vec![],selected:None,
                batch:Some(batch),fingerprint:vec![],needed:0,tail_identity:None };
            unsafe { wave.enqueue(plan.sink,&plan.launches,None,plan.batch.as_ref())?; }
            wave.synchronize()?;
            let graph=unsafe { wave.capture_plan(&plan,&mut None)? };
            wave.graphs[0].push_back((graph,vec![]));
            for replay in 0..2 {
                if replay==1 { lib.copy_h2d(values.buffer,&vec![0;values.buffer.bytes])?; }
                unsafe { lib.cuda_graph_launch(graph,wave.stream.raw)?; }
                wave.synchronize()?;
                let mut output=vec![0u8;4*32768];
                lib.copy_d2h(&mut output,slice(wave.output.buffer,0,4*32768))?;
                for (row,value) in [0.5f32,0.5,2.0/3.0,0.75].into_iter().enumerate() {
                    let bits=value.to_bits();
                    let expected=if replay==0 { ((bits+0x7fff+((bits>>16)&1))>>16) as u16 } else { 0 };
                    for item in output[row*32768..(row+1)*32768].chunks_exact(2) {
                        assert_eq!(u16::from_ne_bytes([item[0],item[1]]),expected);
                    }
                }
            }
            wave.clear_graph()?;
            Ok(())
        })?;
    }
    Ok(())
}
