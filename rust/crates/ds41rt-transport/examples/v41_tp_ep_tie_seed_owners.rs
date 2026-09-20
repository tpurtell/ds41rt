//! Diagnostic-only exporter: real Rust replicated-group owners plus the exact
//! transport-encoded route words and per-rank worker masks for a fixed top-6 M1
//! case. It is a standalone example, separate from the frozen kernel harness and
//! it does not modify any production path.
//!
//! The encoding/decoding uses only the public transport API
//! (`with_native_group_owners`, `V41NativeOwnerRouteWord`, `V41NativeOwnershipBatch`,
//! `parse_native_group`, `copy_native_group_routes_into`), so the consumer never
//! re-implements owner bit-packing.
//!
//! Usage:
//!   cargo run -p ds41rt-transport --example v41_tp_ep_tie_seed_owners -- \
//!     --topology tp2ep2 --layer 20 --experts 0,383,1,2,3,4 --ids 1..64 \
//!     --out owners.json [--verify]
//!
//! `--verify` runs the focused CPU assertions (same-ID determinism, bounded
//! cross-ID owner-difference search, ordinal rank/group/shard mapping for
//! TP2x2 and TP4x1, and invalid owner/shape rejection) and exits non-zero on
//! failure. It is a diagnostic, not a quality gate.

use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use ds41rt_core::{
    replicated_expert_tie_seed, ReplicatedExpertCostModel, ReplicatedExpertScheduleConfig,
    ReplicatedExpertScheduler, INACTIVE_REPLICATED_EXPERT_GROUP,
};
use ds41rt_transport::v41_expert::{
    V41BackboneRequest, V41NativeOwnerRouteWord, V41NativeOwnershipBatch, V41SparkTopology,
    V41_NATIVE_GROUP_REQUEST_FLAG, V41_NATIVE_UNASSIGNED_EXPERT_ID, V41_ROUTED_EXPERTS,
    EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
};
use ds41rt_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind,
};
use serde_json::{json, Value};

const EXPERTS: usize = V41_ROUTED_EXPERTS;
const MAX_ROWS: u32 = 4096;
const HIDDEN_BYTES: usize = 5120 * 2;
const TOPK: usize = 6;

fn parse_topology(name: &str) -> Result<V41SparkTopology> {
    Ok(match name {
        "tp2ep2" => V41SparkTopology::NATIVE_TP2_EP2,
        "tp4ep1" => V41SparkTopology::NATIVE_TP4_EP1,
        "tp3ep2" => V41SparkTopology::NATIVE_TP3_EP2,
        other => anyhow::bail!("unsupported topology {other} (want tp2ep2|tp3ep2|tp4ep1)"),
    })
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn fixed_request(layer: u32, experts: &[u32]) -> Result<ExpertProtocolV2Request> {
    ensure!(
        experts.len() == TOPK && experts.iter().all(|e| (*e as usize) < EXPERTS),
        "expected six in-range experts"
    );
    let descriptors = vec![ExpertProtocolV2RowDescriptor {
        row_id: 0,
        source_kind: ExpertV2SourceKind::Decode,
        source_request_id: 1,
        token_position: 0,
        route_offset: 0,
        route_count: TOPK as u32,
    }];
    let routes: Vec<ExpertProtocolV2RouteEntry> = experts
        .iter()
        .map(|&expert_id| ExpertProtocolV2RouteEntry {
            row_index: 0,
            expert_id,
            gate_weight: 1.0 / TOPK as f32,
        })
        .collect();
    let mut request = ExpertProtocolV2Request::new(
        1,
        17,
        layer,
        5120,
        ExpertV2Dtype::Bf16,
        descriptors,
        routes,
        vec![0u8; HIDDEN_BYTES],
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

fn owners_for(
    counts: &[u32; EXPERTS],
    ep: u8,
    layer: u32,
    request_id: u64,
) -> Result<Vec<u8>> {
    let model = ReplicatedExpertCostModel::new(1, 0, 16);
    let config = ReplicatedExpertScheduleConfig::new(ep, model);
    let mut scheduler = ReplicatedExpertScheduler::new(config, EXPERTS)?;
    let seed = replicated_expert_tie_seed(layer, request_id);
    let plan = scheduler.plan(counts, seed)?;
    Ok(plan.assignment().iter().map(|group| group.encoded()).collect())
}

/// One exported case: owners, encoded words, decoded owners and per-rank masks.
fn case_json(
    request: &ExpertProtocolV2Request,
    experts: &[u32],
    owners: &[u8],
    topology: V41SparkTopology,
    layer: u32,
    request_id: u64,
) -> Result<Value> {
    ensure!(owners.len() == EXPERTS);
    let mut flagged = request.clone();
    // The scheduler seed is derived from this exact header id; the cloned
    // fixture must carry it so the encoded wire is self-consistent.
    flagged.header.request_id = request_id;
    flagged.with_native_group_owners(owners, topology)?;
    ensure!(
        flagged.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0,
        "native group flag must be set"
    );
    let frame = flagged.encode()?;
    let decoded_request = ExpertProtocolV2Request::decode(&frame)?;
    ensure!(
        decoded_request.header.request_id == request_id
            && decoded_request.header.layer_id == layer
            && replicated_expert_tie_seed(
                decoded_request.header.layer_id,
                decoded_request.header.request_id,
            ) == replicated_expert_tie_seed(layer, request_id),
        "encoded header request id / layer must reproduce the scheduler seed"
    );
    let mut batch = V41NativeOwnershipBatch::default();
    let mut encoded_words = Vec::with_capacity(TOPK);
    let mut decoded_owners = Vec::with_capacity(TOPK);
    for route in &flagged.routes {
        let word = route.expert_id;
        encoded_words.push(word);
        let decoded = batch.observe(word, topology.group_count())?;
        decoded_owners.push(decoded.owner);
    }
    let mut rank_mask = Vec::new();
    for rank in 0..topology.world_size() {
        let group = topology.group(rank)?;
        let native = V41BackboneRequest::parse_native_group(&frame, MAX_ROWS, topology)?;
        let mut ids = vec![-1i32; TOPK];
        let mut weights = vec![-1.0f32; TOPK];
        native.copy_native_group_routes_into(&mut ids, &mut weights, group)?;
        rank_mask.push(json!({
            "rank": rank,
            "group": group,
            "tp_rank": topology.tp_rank(rank)?,
            "executor_id": topology.executor_id(rank)?,
            "ids": ids,
            "weights": weights,
        }));
    }
    // Cross-check: the worker mask owner must equal the scheduler owner for the
    // route's true expert (no Python bit-packing anywhere).
    for (slot, &expert) in experts.iter().enumerate() {
        let decoded = V41NativeOwnerRouteWord::decode(encoded_words[slot], topology.group_count())?;
        ensure!(
            decoded.expert_id == expert && decoded.owner == owners[expert as usize],
            "transport owner decode disagrees with the scheduler owner"
        );
    }
    let inactive_ranks = rank_mask
        .iter()
        .filter(|entry| {
            entry["ids"]
                .as_array()
                .map(|ids| {
                    ids.iter()
                        .all(|id| id.as_i64() == Some(V41_NATIVE_UNASSIGNED_EXPERT_ID as i64))
                })
                .unwrap_or(false)
        })
        .count();
    Ok(json!({
        "request_id": request_id,
        "header_request_id": decoded_request.header.request_id,
        "header_layer_id": decoded_request.header.layer_id,
        "seed": replicated_expert_tie_seed(layer, request_id),
        "owners": owners,
        "owners_hash": format!("{:016x}", fnv1a(owners)),
        "encoded_words": encoded_words,
        "decoded_owners": decoded_owners,
        "rank_mask": rank_mask,
        "inactive_rank_count": inactive_ranks,
    }))
}

struct Args {
    topology: V41SparkTopology,
    layer: u32,
    experts: Vec<u32>,
    ids: Vec<u64>,
    out: Option<PathBuf>,
    verify: bool,
}

fn parse_args() -> Result<Args> {
    let mut topology = "tp2ep2".to_string();
    let mut layer = 20u32;
    let mut experts = vec![0u32, 383, 1, 2, 3, 4];
    let mut ids: Vec<u64> = (1..=64).collect();
    let mut out = None;
    let mut verify = false;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--topology" => topology = argv.next().context("--topology needs a value")?,
            "--layer" => layer = argv.next().context("--layer needs a value")?.parse()?,
            "--experts" => {
                experts = argv
                    .next()
                    .context("--experts needs a value")?
                    .split(',')
                    .map(|value| value.parse::<u32>())
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            "--ids" => {
                let value = argv.next().context("--ids needs a value")?;
                ids = if let Some((lo, hi)) = value.split_once("..") {
                    (lo.parse::<u64>()?..=hi.parse::<u64>()?).collect()
                } else {
                    value
                        .split(',')
                        .map(|item| item.parse::<u64>())
                        .collect::<std::result::Result<Vec<_>, _>>()?
                };
            }
            "--out" => out = Some(PathBuf::from(argv.next().context("--out needs a value")?)),
            "--verify" => verify = true,
            other => anyhow::bail!("unknown flag {other}"),
        }
    }
    ensure!(!ids.is_empty(), "at least one request id is required");
    Ok(Args {
        topology: parse_topology(&topology)?,
        layer,
        experts,
        ids,
        out,
        verify,
    })
}

fn verify_focused(args: &Args, request: &ExpertProtocolV2Request, counts: &[u32; EXPERTS]) -> Result<()> {
    let topology = args.topology;
    // 1. Same request id is deterministic.
    let first = owners_for(counts, topology.group_count(), args.layer, args.ids[0])?;
    let again = owners_for(counts, topology.group_count(), args.layer, args.ids[0])?;
    ensure!(first == again, "same request id must reproduce the same owners");
    // 2. Bounded cross-id search: find ids whose owners actually differ.
    let mut differing = 0usize;
    let mut example_pair = None;
    for &id in &args.ids {
        let owners = owners_for(counts, topology.group_count(), args.layer, id)?;
        if owners != first {
            differing += 1;
            if example_pair.is_none() {
                example_pair = Some((args.ids[0], id, owners));
            }
        }
    }
    if topology.group_count() == 1 {
        ensure!(
            differing == 0,
            "EP1 must be tie-seed insensitive: {differing} ids changed owners"
        );
    } else {
        ensure!(
            differing > 0,
            "cross-id owner difference must be found by bounded search, not assumed"
        );
    }
    // 3. Ordinal rank/group/shard mapping is group-major: rank = group*TP + tp.
    for rank in 0..topology.world_size() {
        ensure!(
            topology.group(rank)? == (rank / topology.tp() as usize) as u8
                && topology.tp_rank(rank)? == (rank % topology.tp() as usize) as u8,
            "rank {rank} is not group-major for {topology:?}"
        );
    }
    // 4. Invalid owner/shape rejection through the real transport builder.
    let bad_len = vec![0u8; EXPERTS - 1];
    ensure!(
        request.clone().with_native_group_owners(&bad_len, topology).is_err(),
        "short owner array must be rejected"
    );
    let mut inactive = vec![0u8; EXPERTS];
    for &expert in &args.experts {
        inactive[expert as usize] = INACTIVE_REPLICATED_EXPERT_GROUP;
    }
    ensure!(
        request.clone().with_native_group_owners(&inactive, topology).is_err(),
        "inactive owner for a routed expert must be rejected"
    );
    let mut out_of_range = vec![0u8; EXPERTS];
    for &expert in &args.experts {
        out_of_range[expert as usize] = topology.group_count();
    }
    ensure!(
        request.clone().with_native_group_owners(&out_of_range, topology).is_err(),
        "out-of-range owner must be rejected"
    );
    let mut unflagged = request.clone();
    unflagged.header.flags &= !EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    ensure!(
        unflagged.with_native_group_owners(&first, topology).is_err(),
        "ownership without the compact BF16 contract must be rejected"
    );
    if let Some((_, id, _)) = example_pair {
        println!(
            "verify: topology={topology:?} differing_ids={differing}/{} example_pair=({}, {})",
            args.ids.len(),
            args.ids[0],
            id
        );
    } else {
        println!("verify: topology={topology:?} differing_ids=0/{}", args.ids.len());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let request = fixed_request(args.layer, &args.experts)?;
    let mut counts = [0u32; EXPERTS];
    for &expert in &args.experts {
        counts[expert as usize] += 1;
    }
    if args.verify {
        verify_focused(&args, &request, &counts)?;
    }
    let base = owners_for(&counts, args.topology.group_count(), args.layer, args.ids[0])?;
    // Prefer the first differing id for the cross-id pair; fall back to the
    // second id (EP1 negative control has identical owners by construction).
    let mut pair = (args.ids[0], args.ids[args.ids.len() - 1]);
    for &id in &args.ids {
        let owners = owners_for(&counts, args.topology.group_count(), args.layer, id)?;
        if owners != base {
            pair = (args.ids[0], id);
            break;
        }
    }
    let ids_scanned = args.ids.len();
    let case_same = case_json(&request, &args.experts, &base, args.topology, args.layer, pair.0)?;
    let owners_a = owners_for(&counts, args.topology.group_count(), args.layer, pair.1)?;
    let case_cross = case_json(&request, &args.experts, &owners_a, args.topology, args.layer, pair.1)?;
    let payload = json!({
        "schema": "v41-tie-seed-owner-export-v1",
        "kind": "diagnostic",
        "topology": format!("{:?}", args.topology),
        "tp": args.topology.tp(),
        "ep": args.topology.ep(),
        "layer": args.layer,
        "fixed_experts": args.experts,
        "ids_scanned": ids_scanned,
        "same_id": pair.0,
        "cross_id": pair.1,
        "owners_differ": owners_a != base,
        "case_same_id": case_same,
        "case_cross_id": case_cross,
    });
    let text = serde_json::to_string_pretty(&payload)?;
    if let Some(path) = args.out {
        std::fs::write(&path, text)?;
        println!(
            "wrote {} (topology={:?} same_id={} cross_id={} owners_differ={})",
            path.display(),
            args.topology,
            pair.0,
            pair.1,
            owners_a != base
        );
    } else {
        println!("{text}");
    }
    Ok(())
}
