use super::*;
mod decode_lanes;
use crate::v41_backbone_cache::BackboneCache;
use crate::v41_backbone_execution::CacheProducerWeights;
use crate::v41_backbone_lane::BackboneLaneWeights;
use crate::v41_engram::{
    layer::{EngramGate, EngramLayerWeights},
    EngramDeviceRows,
};
use crate::v41_index_lane::IndexLaneWeights;
use crate::v41_requests::{RequestTokens, Requests};
use crate::v41_target_embedding::TargetEmbeddingWave;
use crate::v41_target_head::{TargetHeadWave, TargetHeadWeights};
use crate::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::Context;
use ds41rt_ffi::NativeLibrary;
use ds41rt_transport::v41_expert::V41Tp4Roce;
use ds41rt_transport::{ExpertV2SourceKind, TcpTransportConfig};
use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

#[test]
fn real_target_prefill_commit_and_decode() -> Result<()> {
    let Some(path) = std::env::var_os("DS41RT_TARGET_PASS_LIBRARY") else {
        eprintln!("skip distributed layer test: DS41RT_TARGET_PASS_LIBRARY unset");
        return Ok(());
    };
    let model = std::env::var_os("DS41RT_TARGET_PASS_MODEL")
        .context("DS41RT_TARGET_PASS_MODEL required")?;
    let peers: [SocketAddr; 4] = std::env::var("DS41RT_TARGET_PASS_PEERS")?
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("four test peers required"))?;
    let lib = unsafe { NativeLibrary::load(path)? };
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        std::path::Path::new(&model),
    )?;
    let start = Instant::now();
    let weights = BackboneLaneWeights::load(
        &lib,
        &catalog,
        BackboneLaneWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let producers = CacheProducerWeights::load(
        &lib,
        &catalog,
        CacheProducerWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let index_weights = IndexLaneWeights::load(
        &lib,
        &catalog,
        IndexLaneWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let names = ["embed.weight".to_string()];
    let table = NativeRtxTensors::load(
        &lib,
        &catalog,
        &names,
        NativeRtxTensors::plan(&catalog, &names)?,
        16 * 1024 * 1024,
    )?;
    eprintln!(
        "layer fixture official coordinator weights loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let embedding =
        TargetEmbeddingWave::new(&lib, &table, 80, TargetEmbeddingWave::device_bytes(80)?)?;
    let lane = BackboneLane::new(
        &weights,
        80,
        BackboneLane::workspace_bytes(&lib, 80)?.into_iter().sum(),
    )?;
    let index = IndexLane::new(
        &index_weights,
        80,
        IndexLane::workspace_bytes(&lib, 80)?.into_iter().sum(),
    )?;
    let execution = BackboneExecution::new(
        &producers,
        80,
        BackboneExecution::workspace_bytes(&lib, 80)?,
    )?;
    let map = ds41rt_loader::EngramTokenMap::from_file(
        &std::path::Path::new(&model).join("tokenizer.json"),
    )?;
    let pipeline =
        unsafe { ds41rt_loader::EngramPipeline::new(&catalog, map, 80, 4, 8 * 1024 * 1024)? };
    let mut requests = Requests::new(
        &lib,
        pipeline,
        16,
        [16; 4],
        BackboneCache::device_bytes(16, [16; 4])?,
    )?;
    let engram_weights = [0, 1]
        .map(|i| EngramLayerWeights::load(&lib, &catalog, i, 256 * 1024 * 1024, 16 * 1024 * 1024))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let gates = [
        EngramGate::new(&engram_weights[0], 80, 128 * 1024 * 1024)?,
        EngramGate::new(&engram_weights[1], 80, 128 * 1024 * 1024)?,
    ];
    let upload = EngramDeviceRows::new(&lib, 80, EngramDeviceRows::device_bytes(80)?)?;
    let vocabulary = VocabularyHead::load(
        &lib,
        &catalog,
        VocabularyHead::plan(&catalog)?,
        16 * 1024 * 1024,
    )?;
    let head_weights = TargetHeadWeights::load(
        &lib,
        &catalog,
        TargetHeadWeights::device_bytes(&catalog)?,
        16 * 1024 * 1024,
    )?;
    let head = head_weights.wave(&vocabulary, 16, TargetHeadWave::device_bytes(16)?)?;
    let mut pass = TargetPass::new(
        embedding,
        lane,
        index,
        execution,
        upload,
        gates,
        head,
        crate::v41_target_pass::TargetTapWave::new(
            &lib,
            80,
            crate::v41_target_pass::TargetTapWave::device_bytes(80)?,
        )?,
        Duration::from_secs(120),
    )?;
    let roce = V41Tp4Roce::new(
        peers,
        [1, 2, 3, 4],
        80,
        TcpTransportConfig { timing: false,
            timeout: Duration::from_secs(120),
            max_frame_bytes: 2 * 1024 * 1024,
        },
    )?;
    let mut transport = NativeTp4Wave::new(&lib, roce, NativeTp4Wave::device_bytes(80)?)?;
    if std::env::var_os("DS41RT_TARGET_PASS_ENCODER_PAIR").is_some()
        || std::env::var_os("DS41RT_TARGET_PASS_DECODE_LANES").is_some() {
        let mut other = TargetPass::new(
            TargetEmbeddingWave::new(&lib, &table, 80, TargetEmbeddingWave::device_bytes(80)?)?,
            BackboneLane::new(&weights, 80, BackboneLane::workspace_bytes(&lib, 80)?.into_iter().sum())?,
            IndexLane::new(&index_weights, 80, IndexLane::workspace_bytes(&lib, 80)?.into_iter().sum())?,
            BackboneExecution::new(&producers, 80, BackboneExecution::workspace_bytes(&lib, 80)?)?,
            EngramDeviceRows::new(&lib, 80, EngramDeviceRows::device_bytes(80)?)?,
            [EngramGate::new(&engram_weights[0], 80, 128 * 1024 * 1024)?,
             EngramGate::new(&engram_weights[1], 80, 128 * 1024 * 1024)?],
            head_weights.wave(&vocabulary, 16, TargetHeadWave::device_bytes(16)?)?,
            TargetTapWave::new(&lib, 80, TargetTapWave::device_bytes(80)?)?,
            Duration::from_secs(120),
        )?;
        let mut second_transport = NativeTp4Wave::new(&lib,
            V41Tp4Roce::new(peers, [1, 2, 3, 4], 80, TcpTransportConfig { timing: false,
                timeout: Duration::from_secs(120), max_frame_bytes: 2 * 1024 * 1024,
            })?, NativeTp4Wave::device_bytes(80)?)?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        if std::env::var_os("DS41RT_TARGET_PASS_DECODE_LANES").is_some() {
            let draft_weights = crate::v41_experts::dspark::DsparkWeights::load(
                &lib, &catalog, 80, 1, 32 * 1024 * 1024 * 1024, 16 * 1024 * 1024)?;
            let mut draft = crate::v41_native_serve::speculative::DraftRuntime::new(
                &lib, &draft_weights, &table, &vocabulary, 80)?;
            return decode_lanes::qualify(&lib, &runtime, &mut requests,
                &mut pass, &mut other, &mut transport, &mut second_transport, &mut draft);
        }
        // Drop after the first cooperative suspension, with dispatched work and
        // potentially a follower waiting for KV. Both admissions must be revoked
        // and the subsequent comparisons must reuse the same owners successfully.
        {
            use std::future::Future;
            let tokens: Vec<u32> = (0..130).map(|i| ((i * 7919 + 17) % 129280) as u32).collect();
            let lease = requests.admit(0, 6999)?;
            requests.begin_encoder(lease, 130)?;
            let mut first = requests.reserve_encoder(&[RequestTokens { lease, tokens: &tokens[..65],
                image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
            let mut second = requests.reserve_encoder(&[RequestTokens { lease, tokens: &tokens[65..],
                image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
            let mut suffix = EncoderSuffix::new(&lib, 130, EncoderSuffix::device_bytes(130)?)?;
            {
                let mut pending = Box::pin(unsafe { pass.execute_encoder_pair(&mut other, &mut requests,
                    [&mut first, &mut second], [&mut transport, &mut second_transport], &mut suffix) });
                runtime.block_on(std::future::poll_fn(|cx| {
                    assert!(pending.as_mut().poll(cx).is_pending(), "expected a live encoder suspension");
                    std::task::Poll::Ready(())
                }));
            }
            assert!(requests.validate(&first).is_err());
            assert!(requests.validate(&second).is_err());
            assert!(requests.cache().request_id(lease).is_err());
            pass.discard(&mut first)?; other.discard(&mut second)?;
            transport.reset_connections(); second_transport.reset_connections();
            eprintln!("PASS cancelled independent encoder: both batches revoked before owner reuse");
        }
        for (left, right, tail) in [(65usize, 65usize, 0usize), (79, 1, 0), (65, 65, 1), (64, 64, 27)] {
            let tokens: Vec<u32> = (0..left + right + tail).map(|i| ((i * 7919 + 17) % 129280) as u32).collect();
            let mut expected = None;
            for mode in 0..3 {
                let paired = mode == 1;
                let lease = requests.admit(0, 7000 + mode as u64)?;
                let end = tokens.len() as u64;
                requests.begin_encoder(lease, end)?;
                let mut suffix = EncoderSuffix::new(&lib, end, EncoderSuffix::device_bytes(end)?)?;
                if mode == 2 {
                    let chunks: Vec<_> = [&tokens[..left], &tokens[left..left + right], &tokens[left + right..]]
                        .into_iter().filter(|c| !c.is_empty()).collect();
                    runtime.block_on(unsafe { pass.execute_encoder_stream(&mut other, &mut requests,
                        lease, &chunks, [&mut transport, &mut second_transport], &mut suffix, &|| true) })?;
                } else if paired {
                    let mut first = requests.reserve_encoder(&[RequestTokens { lease,
                        tokens: &tokens[..left], image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
                    let mut second = requests.reserve_encoder(&[RequestTokens { lease,
                        tokens: &tokens[left..left + right], image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
                    runtime.block_on(unsafe { pass.execute_encoder_pair(&mut other, &mut requests,
                        [&mut first, &mut second], [&mut transport, &mut second_transport], &mut suffix) })?;
                    pass.commit(&mut requests, &mut first, &[left as u32])?;
                    other.commit(&mut requests, &mut second, &[right as u32])?;
                    if tail != 0 {
                        let mut batch = requests.reserve_encoder(&[RequestTokens { lease,
                            tokens: &tokens[left + right..], image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
                        runtime.block_on(unsafe { pass.execute_reserved_encoder(&mut requests,
                            &mut batch, &mut transport, &mut suffix) })?;
                        pass.commit(&mut requests, &mut batch, &[tail as u32])?;
                    }
                } else {
                    for chunk in [&tokens[..left], &tokens[left..left + right], &tokens[left + right..]] {
                        if chunk.is_empty() { continue; }
                        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
                            image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
                        runtime.block_on(unsafe { pass.execute_encoder(&requests, &mut batch,
                            &mut transport, 0, &mut suffix) })?;
                        pass.commit(&mut requests, &mut batch, &[chunk.len() as u32])?;
                    }
                }
                let output = suffix.output()?;
                let mut bytes = Vec::new();
                for buffer in [output.residual, output.pre] {
                    let mut value = vec![0; buffer.bytes]; lib.copy_d2h(&mut value, buffer)?;
                    bytes.extend_from_slice(&value);
                }
                if let Some(expected) = &expected { assert!(&bytes == expected, "encoder pair differs at {left}+{right}+{tail}"); }
                else { expected = Some(bytes); }
                assert_eq!(requests.cache().committed_end(lease)?, end);
                assert_eq!(requests.begin_decoder_replay(lease)?, end.saturating_sub(128));
                requests.release(lease)?;
            }
            eprintln!("PASS independent encoder chunks {left}+{right}+{tail}: suffix residual/pre byte-identical to serial, ordered commits and decoder replay ready");
        }
        // Five chunks reuse both lanes, crossing more than one former pair boundary.
        // Drop a live stream first, then reuse the same owners for exact comparisons.
        let tokens: Vec<u32> = (0..175).map(|i| ((i * 7919 + 17) % 129280) as u32).collect();
        let chunks = [&tokens[..31], &tokens[31..64], &tokens[64..99], &tokens[99..136], &tokens[136..]];
        {
            use std::future::Future;
            let lease = requests.admit(0, 7100)?;
            requests.begin_encoder(lease, 175)?;
            let mut suffix = EncoderSuffix::new(&lib, 175, EncoderSuffix::device_bytes(175)?)?;
            {
                let mut pending = Box::pin(unsafe { pass.execute_encoder_stream(&mut other,
                    &mut requests, lease, &chunks, [&mut transport, &mut second_transport],
                    &mut suffix, &|| true) });
                runtime.block_on(std::future::poll_fn(|cx| {
                    assert!(pending.as_mut().poll(cx).is_pending());
                    std::task::Poll::Ready(())
                }));
            }
            assert!(requests.cache().request_id(lease).is_err());
            assert_eq!(pass.state, State::Idle);
            assert_eq!(other.state, State::Idle);
            transport.reset_connections(); second_transport.reset_connections();
            eprintln!("PASS cancelled encoder stream revoked admission and reset both owners");
        }
        let mut expected = None;
        for streaming in [false, true] {
            let lease = requests.admit(0, 7101 + streaming as u64)?;
            requests.begin_encoder(lease, 175)?;
            let mut suffix = EncoderSuffix::new(&lib, 175, EncoderSuffix::device_bytes(175)?)?;
            if streaming {
                runtime.block_on(unsafe { pass.execute_encoder_stream(&mut other, &mut requests,
                    lease, &chunks, [&mut transport, &mut second_transport], &mut suffix, &|| true) })?;
            } else {
                for chunk in chunks {
                    let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
                        image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
                    runtime.block_on(unsafe { pass.execute_encoder(&requests, &mut batch,
                        &mut transport, 0, &mut suffix) })?;
                    pass.commit(&mut requests, &mut batch, &[chunk.len() as u32])?;
                }
            }
            let output = suffix.output()?;
            let mut bytes = Vec::new();
            for buffer in [output.residual, output.pre] {
                let mut value = vec![0; buffer.bytes]; lib.copy_d2h(&mut value, buffer)?;
                bytes.extend_from_slice(&value);
            }
            if let Some(expected) = &expected { assert!(&bytes == expected, "five-chunk encoder stream differs"); }
            else { expected = Some(bytes); }
            assert_eq!(requests.cache().committed_end(lease)?, 175);
            assert_eq!(requests.begin_decoder_replay(lease)?, 47);
            requests.release(lease)?;
        }
        eprintln!("PASS five-chunk stream: serial-exact suffix and ordered decoder replay");
        return Ok(());
    }
    let dspark_weights = if std::env::var_os("DS41RT_TARGET_PASS_DSPARK").is_some() {
        let start = Instant::now();
        let weights = crate::v41_experts::dspark::DsparkWeights::load(
            &lib,
            &catalog,
            80,
            1,
            32 * 1024 * 1024 * 1024,
            16 * 1024 * 1024,
        )?;
        eprintln!(
            "real dSpark weights loaded in {:.3}s resident_bytes={}",
            start.elapsed().as_secs_f64(),
            weights.budget().resident_bytes()?
        );
        Some(weights)
    } else {
        None
    };
    let mut main_context = dspark_weights
        .as_ref()
        .map(|weights| {
            weights.main_context(
                80,
                crate::v41_experts::dspark::DsparkMainContext::device_bytes(&lib, 80)?,
            )
        })
        .transpose()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let case = std::env::var_os("DS41RT_TARGET_PASS_CASE")
        .map(|path| -> Result<serde_json::Value> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?;
    let request_count = if case.is_some() { 1 } else { 16 };
    let mut draft_chain = dspark_weights
        .as_ref()
        .map(|weights| {
            weights.draft(
                &table,
                &vocabulary,
                request_count as u32,
                weights.draft_bytes(request_count as u32)?,
            )
        })
        .transpose()?;
    let mut draft_rngs = (0..request_count)
        .map(|i| ds41rt_core::DsparkRng::new(1000 + i as u64))
        .collect::<Vec<_>>();
    let mut tokens: Vec<u32> = match &case {
        Some(case) => serde_json::from_value(case["token_ids"].clone())?,
        None => (0..80u32).map(|i| (i * 7919 + 17) % 129280).collect(),
    };
    ensure!(
        !tokens.is_empty() && tokens.len() <= 80,
        "fixture input must fit 80 rows"
    );
    let prompt_rows = tokens.len() / request_count;
    let cycles = case
        .as_ref()
        .and_then(|c| c["max_new_tokens"].as_u64())
        .unwrap_or(3) as usize;
    ensure!(
        (1..=128).contains(&cycles),
        "fixture generation must be 1..128 tokens"
    );
    let eos = case.as_ref().and_then(|c| c["eos_id"].as_u64());
    let leases = (0..request_count)
        .map(|slot| requests.admit(slot, 1000 + slot as u64))
        .collect::<Result<Vec<_>>>()?;
    let mut draft_windows = if main_context.is_some() {
        (0..3)
            .map(|_| {
                crate::v41_dspark_cache::DsparkWindow::new(
                    &lib,
                    16,
                    80,
                    crate::v41_dspark_cache::DsparkWindow::device_bytes(16, 80)?,
                )
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let draft_leases = draft_windows
        .iter_mut()
        .map(|window| {
            (0..request_count)
                .map(|slot| window.begin_request(slot, 1000 + slot as u64))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut transaction_windows = (0..draft_windows.len())
        .map(|_| {
            crate::v41_dspark_cache::DsparkWindow::new(
                &lib,
                16,
                80,
                crate::v41_dspark_cache::DsparkWindow::device_bytes(16, 80)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let transaction_leases = transaction_windows
        .iter_mut()
        .map(|window| {
            (0..request_count)
                .map(|slot| window.begin_request(slot, 1000 + slot as u64))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut committed = 0u64;
    for cycle in 0..cycles {
        let count = if cycle == 0 { prompt_rows } else { 1 };
        let kind = if cycle == 0 {
            ExpertV2SourceKind::Prefill
        } else {
            ExpertV2SourceKind::Decode
        };
        let work = leases
            .iter()
            .zip(tokens.chunks_exact(count))
            .map(|(&lease, tokens)| RequestTokens {
                lease,
                tokens,
                image_mask: None,
                kind,
            })
            .collect::<Vec<_>>();
        let mut batch = requests.prepare(&work)?;
        let selected = (0..request_count)
            .map(|i| (i + 1) * count - 1)
            .collect::<Vec<_>>();
        let start = Instant::now();
        let logits = runtime.block_on(unsafe {
            pass.execute(&requests, &mut batch, &mut transport, 0, &selected)
        })?;
        assert_eq!(logits.rows, request_count);
        assert_eq!(logits.selected_rows, selected);
        assert!(logits
            .token_positions
            .iter()
            .all(|&p| p == committed + count as u64 - 1));
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(values.iter().all(|v| v.is_finite()));
        tokens = values
            .chunks_exact(129280)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32
            })
            .collect();
        let taps = pass.taps(&batch)?;
        assert_eq!(taps.batch_identity(), batch.cache()?.identity());
        assert_eq!(taps.rows().len(), count * request_count);
        let expected_rows = batch.cache()?.expert_rows();
        assert!(taps
            .rows()
            .iter()
            .zip(&expected_rows)
            .all(|(a, b)| a.request_id == b.request_id
                && a.position == b.position
                && a.kind == b.kind));
        let mut tap_bytes = vec![0; taps.values().bytes];
        lib.copy_d2h(&mut tap_bytes, taps.values())?;
        assert!(tap_bytes.chunks_exact(2).all(|b| f32::from_bits(
            (u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16
        )
        .is_finite()));
        let mut main_bytes = None;
        if let Some(main) = &mut main_context {
            let positions = taps.rows().iter().map(|r| r.position).collect::<Vec<_>>();
            if cycle != 0 {
                let mut invalid = taps.values();
                invalid.bytes -= 2;
                assert!(unsafe { main.execute_input(invalid, &positions) }.is_err());
                assert!(main.output().is_err());
            }
            let mut proposal =
                unsafe { main.execute_rows(taps.values(), taps.batch_identity(), taps.rows())? };
            let output = proposal.output()?;
            let mut bytes = vec![0; output.bytes];
            lib.copy_d2h(&mut bytes, output)?;
            assert!(bytes.chunks_exact(2).all(|b| f32::from_bits(
                (u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16
            )
            .is_finite()));
            main_bytes = Some(bytes);
            for stage in 0..3 {
                let output = proposal.kv_output(stage)?;
                let mut bytes = vec![0; output.bytes];
                lib.copy_d2h(&mut bytes, output)?;
                assert!(bytes.chunks_exact(2).all(|b| f32::from_bits(
                    (u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16
                )
                .is_finite()));
                if let Some(dir) = std::env::var_os("DS41RT_TARGET_PASS_OUTPUT") {
                    std::fs::write(
                        std::path::PathBuf::from(dir)
                            .join(format!("cycle{cycle}-stage{stage}-kv.bin")),
                        bytes,
                    )?;
                }
            }
            let [a, b, c] = draft_windows.as_mut_slice() else {
                unreachable!()
            };
            unsafe {
                proposal.commit(
                    taps.batch_identity(),
                    &mut [a, b, c],
                    [&draft_leases[0], &draft_leases[1], &draft_leases[2]],
                    &vec![count as u32; request_count],
                )?;
            }
            assert!(proposal.output().is_err());
            drop(proposal);
            for (window, leases) in draft_windows.iter().zip(&draft_leases) {
                for &lease in leases {
                    assert_eq!(window.committed_end(lease)?, Some(committed + count as u64));
                }
            }
            if cycle == 0 && request_count == 16 {
                qualify_main_prefixes(&lib, main, &taps, &draft_windows, count)?;
            }
        }
        if let Some(dir) = std::env::var_os("DS41RT_TARGET_PASS_OUTPUT") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join(format!("cycle{cycle}-logits.bin")), &bytes)?;
            if let Some(bytes) = &main_bytes {
                std::fs::write(dir.join(format!("cycle{cycle}-main-context.bin")), bytes)?;
            }
            std::fs::write(
                dir.join(format!("batch{}-taps.bin", taps.batch_identity())),
                &tap_bytes,
            )?;
            std::fs::write(
                dir.join(format!("cycle{cycle}-tap-metadata.json")),
                serde_json::to_vec(
                    &serde_json::json!({"batch":taps.batch_identity(),"rows":taps.rows().len(),"positions":taps.rows().iter().map(|r|r.position).collect::<Vec<_>>(),"request_ids":taps.rows().iter().map(|r|r.request_id).collect::<Vec<_>>()}),
                )?,
            )?;
            std::fs::write(
                dir.join(format!("cycle{cycle}-greedy.json")),
                serde_json::to_vec(&tokens)?,
            )?;
        }
        if let Some(main) = &mut main_context {
            let taps = pass.taps(&batch)?;
            let mut proposal =
                unsafe { main.execute_rows(taps.values(), taps.batch_identity(), taps.rows())? };
            let [a, b, c] = transaction_windows.as_mut_slice() else {
                unreachable!()
            };
            if cycle == 0 {
                assert!(unsafe {
                    pass.commit_with_dspark(
                        &mut requests,
                        &mut batch,
                        &mut proposal,
                        &mut [a, b, c],
                        [
                            &transaction_leases[0],
                            &transaction_leases[1],
                            &transaction_leases[2],
                        ],
                        &vec![count as u32 + 1; request_count],
                    )
                }
                .is_err());
                requests.validate(&batch)?;
                pass.output(&batch)?;
                proposal.output()?;
            }
            if cycle + 1 == cycles
                && std::env::var_os("DS41RT_TARGET_PASS_COMMIT_FAILURE").is_some()
            {
                // Invalidate the completed target producer after successful model
                // execution, forcing a failure after dSpark publication starts.
                pass.execution.restart();
                let accepted = (0..request_count)
                    .map(|i| (i % 2) as u32)
                    .collect::<Vec<_>>();
                assert!(unsafe {
                    pass.commit_with_dspark(
                        &mut requests,
                        &mut batch,
                        &mut proposal,
                        &mut [a, b, c],
                        [
                            &transaction_leases[0],
                            &transaction_leases[1],
                            &transaction_leases[2],
                        ],
                        &accepted,
                    )
                }
                .is_err());
                assert!(batch.cache().is_err());
                assert!(proposal.output().is_err());
                for &lease in &leases {
                    assert!(requests.cache().request_id(lease).is_err());
                }
                for (window, leases) in transaction_windows.iter().zip(&transaction_leases) {
                    for &lease in leases {
                        assert!(window.request_id(lease).is_err());
                    }
                }
                pass.discard(&mut batch)?;
                for slot in 0..request_count {
                    let fresh = requests.admit(slot, 1000 + slot as u64)?;
                    assert_ne!(fresh, leases[slot]);
                    assert_eq!(requests.cache().committed_end(fresh)?, 0);
                    requests.release(fresh)?;
                    for (stage, window) in transaction_windows.iter_mut().enumerate() {
                        let fresh = window.begin_request(slot, 1000 + slot as u64)?;
                        assert!(window.request_id(transaction_leases[stage][slot]).is_err());
                        assert_eq!(window.committed_end(fresh)?, None);
                        window.release(fresh)?;
                    }
                }
                eprintln!("PASS combined commit failure: all target/engram/dSpark admissions revoked including zero acceptance; fresh generations recover");
                return Ok(());
            }
            unsafe {
                pass.commit_with_dspark(
                    &mut requests,
                    &mut batch,
                    &mut proposal,
                    &mut [a, b, c],
                    [
                        &transaction_leases[0],
                        &transaction_leases[1],
                        &transaction_leases[2],
                    ],
                    &vec![count as u32; request_count],
                )?;
            }
            assert!(proposal.output().is_err());
            for (window, leases) in transaction_windows.iter().zip(&transaction_leases) {
                for &lease in leases {
                    assert_eq!(window.committed_end(lease)?, Some(committed + count as u64));
                }
            }
        } else {
            pass.commit(
                &mut requests,
                &mut batch,
                &vec![count as u32; request_count],
            )?;
        }
        committed += count as u64;
        if let Some(chain) = &mut draft_chain {
            let bindings = transaction_leases
                .iter()
                .map(|leases| {
                    leases
                        .iter()
                        .map(|&lease| (lease, committed))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let windows = [
                &transaction_windows[0],
                &transaction_windows[1],
                &transaction_windows[2],
            ];
            let bindings = [&bindings[0][..], &bindings[1][..], &bindings[2][..]];
            chain.set_tokens(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>())?;
            let draft_temperature = if cycle == 1 { 0.7 } else { 0.0 };
            chain.prepare_sampling(
                &mut draft_rngs.iter_mut().collect::<Vec<_>>(),
                &vec![draft_temperature; request_count],
            )?;
            let start = Instant::now();
            unsafe {
                chain.execute(windows, bindings)?;
            }
            let read =
                |chain: &crate::v41_experts::dspark::DsparkChain<'_, '_>| -> Result<Vec<Vec<u8>>> {
                    chain
                        .draft_output()?
                        .iter()
                        .map(|&buffer| {
                            let mut bytes = vec![0; buffer.bytes];
                            lib.copy_d2h(&mut bytes, buffer)?;
                            Ok(bytes)
                        })
                        .collect()
                };
            let expected = read(chain)?;
            let ids = expected[0]
                .chunks_exact(4)
                .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(ids.len(), 6 * request_count);
            assert!(ids.iter().all(|&id| (0..129280).contains(&id)));
            assert!(ids[..request_count]
                .iter()
                .zip(&tokens)
                .all(|(&a, &b)| a as u32 == b));
            let logits = expected[1]
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(logits.len(), 5 * request_count * 129280);
            assert!(logits.iter().all(|v| v.is_finite()));
            for (i, row) in logits.chunks_exact(129280).enumerate().filter(|_| draft_temperature == 0.0) {
                assert_eq!(
                    row[ids[request_count + i] as usize],
                    row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                );
            }
            assert!(expected[2]
                .chunks_exact(4)
                .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite()));
            // Reuse the exact staged RNG reservation from eager execution. Cold
            // warmup, warm replay and cancellation must preserve tokens, logits
            // and confidence byte-for-byte, without reserving additional draws.
            let seeds = tokens.iter().map(|&t| t as i32).collect::<Vec<_>>();
            chain.stage_tokens(&seeds)?;
            unsafe { chain.begin_replay(windows, bindings)?; }
            assert!(chain.stage_tokens(&seeds).is_err());
            assert!(chain.stage_sampling(&mut draft_rngs.iter_mut().collect::<Vec<_>>(),
                &vec![0.0; request_count]).is_err());
            chain.cancel_pending_for_test();
            assert!(chain.draft_output().is_err());
            let was_cold = !chain.has_graph(request_count);
            unsafe { chain.begin_replay(windows, bindings)?; }
            let mut pending_polls = 0usize;
            let (queued_tokens, queued_confidence) = runtime.block_on(async {
                loop {
                    if let Some(output) = chain.poll_replay()? { break Ok::<_, anyhow::Error>(output); }
                    pending_polls += 1;
                    tokio::task::yield_now().await;
                }
            })?;
            assert_eq!(queued_tokens, ids.iter().map(|&id| id as u32).collect::<Vec<_>>());
            assert_eq!(queued_confidence.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(), expected[2]);
            assert_eq!(expected, read(chain)?, "queued cold/warm draft differs from eager");
            assert!(chain.has_graph(request_count));
            unsafe { chain.begin_replay(windows, bindings)?; }
            chain.cancel_pending_for_test();
            unsafe { chain.replay(windows, bindings)?; }
            assert_eq!(expected, read(chain)?, "draft replay after cancellation differs");
            eprintln!("PASS queued draft cold={was_cold} pending_polls={pending_polls} cancellation_reuse=true full_output_exact=true");
            if let Some(dir) = std::env::var_os("DS41RT_TARGET_PASS_OUTPUT") {
                for (i, name) in ["tokens", "logits", "confidence"].iter().enumerate() {
                    std::fs::write(
                        std::path::PathBuf::from(&dir)
                            .join(format!("cycle{cycle}-draft-{name}.bin")),
                        &expected[i],
                    )?;
                }
            }
            eprintln!("PASS real dSpark draft cycle={cycle} requests={request_count} end={committed} finite_logits_confidence=true temperature={draft_temperature} graph_byte_exact=true qualification_seconds={:.3}", start.elapsed().as_secs_f64());
            if std::env::var_os("DS41RT_TARGET_PASS_VERIFY").is_some() && cycle + 1 == cycles {
                ensure!(
                    request_count == 1,
                    "initial verification fixture requires one request"
                );
                let mut input = ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
                if std::env::var_os("DS41RT_TARGET_PASS_VERIFY_CORRUPT").is_some() {
                    input[2] = (input[2] + 1) % 129280;
                }
                let mut verify = requests.prepare(&[RequestTokens {
                    lease: leases[0],
                    tokens: &input,
                    image_mask: None,
                    kind: ExpertV2SourceKind::MtpVerify,
                }])?;
                let selected = (0..input.len()).collect::<Vec<_>>();
                let output = runtime.block_on(unsafe {
                    pass.execute(&requests, &mut verify, &mut transport, 0, &selected)
                })?;
                let mut bytes = vec![0; output.logits.bytes];
                lib.copy_d2h(&mut bytes, output.logits)?;
                let values = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert!(values.iter().all(|v| v.is_finite()));
                let next = values
                    .chunks_exact(129280)
                    .map(|row| {
                        // Keep the lower token ID for ties, matching greedy sampling.
                        row.iter()
                            .enumerate()
                            .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
                                if v > best.1 {
                                    (i, v)
                                } else {
                                    best
                                }
                            })
                            .0 as u32
                    })
                    .collect::<Vec<_>>();
                let acceptance =
                    ds41rt_core::verify_dspark_greedy(&input, &next, eos.unwrap_or(1) as u32, 32)
                        .map_err(anyhow::Error::msg)?;
                let taps = pass.taps(&verify)?;
                let mut proposal = unsafe {
                    main_context.as_mut().unwrap().execute_rows(
                        taps.values(),
                        taps.batch_identity(),
                        taps.rows(),
                    )?
                };
                let [a, b, c] = transaction_windows.as_mut_slice() else {
                    unreachable!()
                };
                unsafe {
                    pass.commit_with_dspark(
                        &mut requests,
                        &mut verify,
                        &mut proposal,
                        &mut [a, b, c],
                        [
                            &transaction_leases[0],
                            &transaction_leases[1],
                            &transaction_leases[2],
                        ],
                        &[acceptance.accepted_inputs],
                    )?;
                }
                committed += u64::from(acceptance.accepted_inputs);
                assert_eq!(requests.cache().committed_end(leases[0])?, committed);
                for (stage, window) in transaction_windows.iter().enumerate() {
                    assert_eq!(
                        window.committed_end(transaction_leases[stage][0])?,
                        Some(committed)
                    );
                }
                drop(proposal);
                let accepted_end = committed;
                let mut resumed_next = None;
                if !acceptance.eos && !acceptance.length_limit {
                    let pending = [*acceptance.emitted.last().unwrap()];
                    let mut resumed = requests.prepare(&[RequestTokens {
                        lease: leases[0],
                        tokens: &pending,
                        image_mask: None,
                        kind: ExpertV2SourceKind::Decode,
                    }])?;
                    let output = runtime.block_on(unsafe {
                        pass.execute(&requests, &mut resumed, &mut transport, 0, &[0])
                    })?;
                    let mut bytes = vec![0; output.logits.bytes];
                    lib.copy_d2h(&mut bytes, output.logits)?;
                    resumed_next = Some(
                        bytes
                            .chunks_exact(4)
                            .enumerate()
                            .map(|(i, b)| (i as u32, f32::from_ne_bytes(b.try_into().unwrap())))
                            .fold((0, f32::NEG_INFINITY), |best, candidate| {
                                if candidate.1 > best.1 {
                                    candidate
                                } else {
                                    best
                                }
                            })
                            .0,
                    );
                    let taps = pass.taps(&resumed)?;
                    let mut proposal = unsafe {
                        main_context.as_mut().unwrap().execute_rows(
                            taps.values(),
                            taps.batch_identity(),
                            taps.rows(),
                        )?
                    };
                    let [a, b, c] = transaction_windows.as_mut_slice() else {
                        unreachable!()
                    };
                    unsafe {
                        pass.commit_with_dspark(
                            &mut requests,
                            &mut resumed,
                            &mut proposal,
                            &mut [a, b, c],
                            [
                                &transaction_leases[0],
                                &transaction_leases[1],
                                &transaction_leases[2],
                            ],
                            &[1],
                        )?;
                    }
                    committed += 1;
                    assert_eq!(requests.cache().committed_end(leases[0])?, committed);
                    for (stage, window) in transaction_windows.iter().enumerate() {
                        assert_eq!(
                            window.committed_end(transaction_leases[stage][0])?,
                            Some(committed)
                        );
                    }
                }
                let report = serde_json::json!({"inputs":input,"target_next":next,"accepted_inputs":acceptance.accepted_inputs,"emitted":acceptance.emitted,"eos":acceptance.eos,"length_limit":acceptance.length_limit,"accepted_end":accepted_end,"committed_end":committed,"resumed_next":resumed_next});
                eprintln!("PASS real target draft verification {report}");
                if let Some(dir) = std::env::var_os("DS41RT_TARGET_PASS_OUTPUT") {
                    std::fs::write(
                        std::path::PathBuf::from(dir).join("verification.json"),
                        serde_json::to_vec_pretty(&report)?,
                    )?;
                }
            }
        }
        assert!(pass.output(&batch).is_err());
        assert!(pass.taps(&batch).is_err());
        for &lease in &leases {
            assert_eq!(requests.cache().committed_end(lease)?, committed);
        }
        eprintln!("PASS full target cycle={cycle} requests={request_count} input_rows={} finite_logits=true cache_and_engram_commit=true seconds={:.3}",count*request_count,start.elapsed().as_secs_f64());
        if request_count == 1 && eos.is_some_and(|eos| u64::from(tokens[0]) == eos) {
            break;
        }
    }
    for lease in leases {
        requests.release(lease)?;
    }
    Ok(())
}

fn qualify_main_prefixes(
    lib: &NativeLibrary,
    main: &mut crate::v41_experts::dspark::DsparkMainContext<'_, '_>,
    taps: &TargetTaps<'_>,
    full: &[crate::v41_dspark_cache::DsparkWindow<'_>],
    count: usize,
) -> Result<()> {
    use crate::v41_dspark_cache::DsparkWindow;
    let mut windows = (0..3)
        .map(|_| DsparkWindow::new(lib, 16, 80, DsparkWindow::device_bytes(16, 80)?))
        .collect::<Result<Vec<_>>>()?;
    let mut leases = windows
        .iter_mut()
        .map(|window| {
            (0..16)
                .map(|slot| window.begin_request(slot, 1000 + slot as u64))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    for window in &windows {
        lib.copy_h2d(
            window.ring_for_test(),
            &vec![0x5a; window.ring_for_test().bytes],
        )?;
    }
    let check_untouched = |windows: &[DsparkWindow<'_>]| -> Result<()> {
        for window in windows {
            let mut bytes = vec![0; window.ring_for_test().bytes];
            lib.copy_d2h(&mut bytes, window.ring_for_test())?;
            assert!(bytes.iter().all(|&b| b == 0x5a));
        }
        Ok(())
    };
    let batch = taps.batch_identity();
    let mut proposal = unsafe { main.execute_rows(taps.values(), batch, taps.rows())? };
    let mut too_many = vec![count as u32; 16];
    too_many[3] += 1;
    let [a, b, c] = windows.as_mut_slice() else {
        unreachable!()
    };
    assert!(unsafe {
        proposal.commit(
            batch,
            &mut [a, b, c],
            [&leases[0], &leases[1], &leases[2]],
            &too_many,
        )
    }
    .is_err());
    let [a, b, c] = windows.as_mut_slice() else {
        unreachable!()
    };
    assert!(unsafe {
        proposal.commit(
            batch + 1,
            &mut [a, b, c],
            [&leases[0], &leases[1], &leases[2]],
            &[0; 16],
        )
    }
    .is_err());
    let mut swapped = leases[1].clone();
    swapped.swap(0, 1);
    let [a, b, c] = windows.as_mut_slice() else {
        unreachable!()
    };
    assert!(unsafe {
        proposal.commit(
            batch,
            &mut [a, b, c],
            [&leases[0], &swapped, &leases[2]],
            &[0; 16],
        )
    }
    .is_err());
    windows[2].release(leases[2][0])?;
    let replacement = windows[2].begin_request(0, 1000)?;
    let [a, b, c] = windows.as_mut_slice() else {
        unreachable!()
    };
    assert!(unsafe {
        proposal.commit(
            batch,
            &mut [a, b, c],
            [&leases[0], &leases[1], &leases[2]],
            &[0; 16],
        )
    }
    .is_err());
    leases[2][0] = replacement;
    proposal.output()?;
    check_untouched(&windows)?;
    let [a, b, c] = windows.as_mut_slice() else {
        unreachable!()
    };
    unsafe {
        proposal.commit(
            batch,
            &mut [a, b, c],
            [&leases[0], &leases[1], &leases[2]],
            &[0; 16],
        )?;
    }
    assert!(proposal.output().is_err());
    drop(proposal);
    check_untouched(&windows)?;
    for (window, leases) in windows.iter().zip(&leases) {
        for &lease in leases {
            assert_eq!(window.committed_end(lease)?, None);
        }
    }
    let accepted = (0..16)
        .map(|i| (i % (count + 1)) as u32)
        .collect::<Vec<_>>();
    let mut proposal = unsafe { main.execute_rows(taps.values(), batch, taps.rows())? };
    let [a, b, c] = windows.as_mut_slice() else {
        unreachable!()
    };
    unsafe {
        proposal.commit(
            batch,
            &mut [a, b, c],
            [&leases[0], &leases[1], &leases[2]],
            &accepted,
        )?;
    }
    let [a, b, c] = windows.as_mut_slice() else {
        unreachable!()
    };
    assert!(unsafe {
        proposal.commit(
            batch,
            &mut [a, b, c],
            [&leases[0], &leases[1], &leases[2]],
            &accepted,
        )
    }
    .is_err());
    drop(proposal);
    let mut discarded = unsafe { main.execute_rows(taps.values(), batch, taps.rows())? };
    let [a, b, c] = windows.as_mut_slice() else {
        unreachable!()
    };
    // Even zero acceptance cannot publish a batch behind committed positions.
    assert!(unsafe {
        discarded.commit(
            batch,
            &mut [a, b, c],
            [&leases[0], &leases[1], &leases[2]],
            &[0; 16],
        )
    }
    .is_err());
    discarded.output()?;
    drop(discarded);
    assert!(main.output().is_err());
    const STRIDE: usize = 128 * 528;
    for stage in 0..3 {
        let mut actual = vec![0; windows[stage].ring_for_test().bytes];
        let mut expected = vec![0; full[stage].ring_for_test().bytes];
        lib.copy_d2h(&mut actual, windows[stage].ring_for_test())?;
        lib.copy_d2h(&mut expected, full[stage].ring_for_test())?;
        for slot in 0..16 {
            let n = accepted[slot] as usize;
            let begin = slot * STRIDE;
            let end = begin + n * 528;
            assert_eq!(&actual[begin..end], &expected[begin..end]);
            assert!(actual[end..begin + STRIDE].iter().all(|&b| b == 0x5a));
            assert_eq!(
                windows[stage].committed_end(leases[stage][slot])?,
                if n == 0 { None } else { Some(n as u64) }
            );
        }
    }
    eprintln!("PASS dSpark accepted prefixes: all three FP8 rings exact; rejected tails untouched; zero/foreign/stale/duplicate-commit guards");
    Ok(())
}
