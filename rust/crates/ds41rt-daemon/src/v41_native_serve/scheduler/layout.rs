//! Layout selection stays outside lane execution and uses static dispatch.
use super::*;
use crate::v41_experts::dspark::{DsparkChain, DistributedDsparkChain};
use crate::v41_target_pass::DistributedTargetPass;
use super::super::prefill_target::PrefillTarget;

pub(in crate::v41_native_serve) trait ServingTarget<'w, 'a>: PrefillTarget<'a> {
    type Chain: DraftChain<'a>;
    fn begin_request(transport: &mut Self::Transport) -> Result<()>;
    fn reset_connections(transport: &mut Self::Transport) -> Result<()>;
    fn decode_round(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
        first: &mut Self, second: &mut Self, requests: &mut Requests<'a>,
        first_transport: &mut Self::Transport, second_transport: &mut Self::Transport,
        active: &mut [Option<Active<'a>>], members: &[Vec<usize>; 2],
        draft: Option<&mut DraftRuntime<'w, 'a, Self::Chain>>,
        prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>) -> Result<()>;
}

impl<'w, 'a: 'w> ServingTarget<'w, 'a> for TargetPass<'_, 'a> {
    type Chain = DsparkChain<'w, 'a>;
    fn begin_request(transport: &mut Self::Transport) -> Result<()> {
        transport.begin_request(); Ok(())
    }
    fn reset_connections(transport: &mut Self::Transport) -> Result<()> {
        transport.reset_connections(); Ok(())
    }
    fn decode_round(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
        first: &mut Self, second: &mut Self, requests: &mut Requests<'a>,
        first_transport: &mut Self::Transport, second_transport: &mut Self::Transport,
        active: &mut [Option<Active<'a>>], members: &[Vec<usize>; 2],
        draft: Option<&mut DraftRuntime<'w, 'a, Self::Chain>>,
        prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>) -> Result<()> {
        if members.iter().all(|lane| !lane.is_empty()) {
            independent::run(lib, runtime, first, second, requests, first_transport,
                second_transport, active, draft, prefixes, receive, wake)
        } else {
            // Preserve the single-RTX C1 path without shared-bank/async delivery.
            let lane = usize::from(members[0].is_empty());
            let (pass, transport) = if lane == 0 { (first, first_transport) }
                else { (second, second_transport) };
            single_lane_round(lib, runtime, lane, pass, requests, transport,
                active, &members[lane], draft, prefixes.turn_bank_enabled())
        }
    }
}

impl<'w, 'a: 'w> ServingTarget<'w, 'a> for DistributedTargetPass<'_, 'a> {
    type Chain = DistributedDsparkChain<'w, 'a>;
    fn begin_request(transport: &mut Self::Transport) -> Result<()> {
        transport.begin_request(); Ok(())
    }
    fn reset_connections(transport: &mut Self::Transport) -> Result<()> {
        let device = transport.device;
        device.run(|| { transport.reset_connections(); Ok(()) })
    }
    fn decode_round(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
        first: &mut Self, second: &mut Self, requests: &mut Requests<'a>,
        first_transport: &mut Self::Transport, second_transport: &mut Self::Transport,
        active: &mut [Option<Active<'a>>], _members: &[Vec<usize>; 2],
        draft: Option<&mut DraftRuntime<'w, 'a, Self::Chain>>,
        prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>) -> Result<()> {
        // A single active lane still drives its distributed draft via polling.
        // The empty peer returns immediately without joining per-token work.
        independent::run(lib, runtime, first, second, requests, first_transport,
            second_transport, active, draft, prefixes, receive, wake)
    }
}
