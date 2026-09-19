//! Cross-host native wire qualification without checkpoint/GPU compute.
//! server <rank 0..3> <bind>, or client <peer0> <peer1> <peer2> <peer3>.
use anyhow::{bail, ensure, Result};
use ds41rt_transport::{v41_expert::*, *};
use std::{sync::Arc, time::Duration};

struct Fixture(u64);
impl ProtocolV2ExpertExecutor for Fixture {
    fn name(&self) -> &'static str {
        "v41-roce-qualification"
    }
    fn execute(&self, _: &ExpertProtocolV2RequestView<'_>) -> Result<ExpertProtocolV2Response> {
        bail!("fixture requires streaming")
    }
    fn execute_streaming_device_payload_with_identity(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        _: ProtocolV2RequestDevicePayload,
        emit: &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        let native = V41BackboneRequest::parse(request.frame_bytes(), 80)?;
        ensure!(
            request.hidden_payload() == input(native.rows(), request.header.request_id),
            "input corrupted"
        );
        for row in 0..native.rows() {
            let bytes = vec![
                marker(request.header.request_id, self.0, row);
                V41_PARTIAL_ROW_BYTES as usize
            ];
            let mut indices = [0];
            emit(ProtocolV2ExecutorResponseRef::Host(native.response_chunk(
                self.0,
                row,
                &bytes,
                &mut indices,
                2 * 1024 * 1024,
            )?))?;
        }
        Ok(())
    }
}
fn marker(id: u64, rank: u64, row: u32) -> u8 {
    (id.wrapping_mul(7) + rank * 13 + row as u64) as u8
}
fn input(rows: u32, id: u64) -> Vec<u8> {
    (0..rows as usize * 5280)
        .map(|i| (i as u64 + id) as u8)
        .collect()
}
fn request(rows: u32, id: u64) -> Result<ExpertProtocolV2Request> {
    let mut r = ExpertProtocolV2Request::new(
        id,
        17,
        0,
        5120,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        (0..rows)
            .map(|r| ExpertProtocolV2RowDescriptor {
                row_id: r as u64,
                source_kind: ExpertV2SourceKind::Decode,
                source_request_id: 100 + r as u64,
                token_position: 7,
                route_offset: r * 6,
                route_count: 6,
            })
            .collect(),
        (0..rows * 6)
            .map(|r| ExpertProtocolV2RouteEntry {
                row_index: r / 6,
                expert_id: (r % 6) * 63,
                gate_weight: 0.125,
            })
            .collect(),
        input(rows, id),
    )?;
    r.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(r)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        match args.get(1).map(String::as_str) {
            Some("server") => {
                let rank: u64 = args
                    .get(2)
                    .ok_or_else(|| anyhow::anyhow!("missing rank"))?
                    .parse()?;
                ensure!(rank < 4, "invalid rank");
                serve_protocol_v2_verbs_host_with_executor(&args[3], Arc::new(Fixture(rank + 1)))
                    .await
            }
            Some("client") => {
                ensure!(args.len() == 6, "four peers required");
                let peers = args[2..]
                    .iter()
                    .map(|x| x.parse())
                    .collect::<std::result::Result<Vec<_>, _>>()?
                    .try_into()
                    .unwrap();
                let mut tp = V41Tp4Roce::new(
                    peers,
                    [1, 2, 3, 4],
                    80,
                    TcpTransportConfig { timing: false,
                        timeout: Duration::from_secs(10),
                        max_frame_bytes: 2 * 1024 * 1024,
                    },
                )?;
                for id in 1..=12 {
                    let rows = [1, 2, 6, 16, 80][(id as usize - 1) % 5];
                    let request = request(rows, id)?;
                    let mut count = 0;
                    let mut held = Vec::new();
                    tp.dispatch(&request).await?.receive_owned(|rank, start, payload| {
                        ensure!(payload.retains_receive_slot() == (start + 1 == rows),
                                "only the final streamed frame should retain its receive slot");
                        let bytes = payload.as_ref();
                        ensure!(
                            bytes.len() == V41_PARTIAL_ROW_BYTES as usize,
                            "unexpected chunk size"
                        );
                        ensure!(
                            bytes
                                .iter()
                                .all(|&b| b == marker(id, rank as u64 + 1, start)),
                            "response corrupted"
                        );
                        count += 1;
                        held.push((rank, start, payload));
                        Ok(())
                    })
                    .await?;
                    ensure!(count == rows * 4, "missing chunks");
                    if id == 3 {
                        ensure!(tp.dispatch(&request).await.is_err(),
                                "dispatch must reject retained slots from the previous wave");
                        // The reset must not unregister storage retained by payloads.
                    }
                    for (rank, start, payload) in &held {
                        ensure!(payload.as_ref().iter().all(|&b| b == marker(id, *rank as u64 + 1, *start)),
                                "retained payload changed before release");
                    }
                    drop(held);
                    eprintln!("PASS id={id} rows={rows} chunks={count}: retained final slots, copied earlier chunks, stable bytes");
                    if id == 5 {
                        let abandoned = request_for_cancel()?;
                        drop(tp.dispatch(&abandoned).await?);
                        eprintln!("cancelled enqueued wave; next request must recover");
                    }
                }
                Ok(())
            }
            _ => bail!("server <rank> <bind> | client <peer0> <peer1> <peer2> <peer3>"),
        }
    })
}
fn request_for_cancel() -> Result<ExpertProtocolV2Request> {
    request(80, 1000)
}
