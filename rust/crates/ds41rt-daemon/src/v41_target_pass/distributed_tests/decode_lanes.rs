//! Full-stack lane overlap, retirement and boundary migration with real weights.
use super::*;

pub(super) fn qualify<'w, 'a>(
    lib: &'a NativeLibrary,
    runtime: &tokio::runtime::Runtime,
    requests: &mut Requests<'a>,
    first: &mut TargetPass<'w, 'a>,
    second: &mut TargetPass<'w, 'a>,
    first_transport: &mut NativeTp4Wave<'a>,
    second_transport: &mut NativeTp4Wave<'a>,
    draft: &mut crate::v41_native_serve::speculative::DraftRuntime<'_, 'a>,
) -> Result<()> {
    for per_lane in [1usize, 3, 8] {
        let mut expected = Vec::new();
        let mut expected_proposals = Vec::new();
        for overlap in [false, true] {
            let leases = (0..2 * per_lane)
                .map(|slot| requests.admit(slot, 9000 + slot as u64))
                .collect::<Result<Vec<_>>>()?;
            for slot in 0..2 * per_lane { draft.admit(9000 + slot as u64)?; }
            assert!(draft.admit(9000).is_err(), "duplicate draft identity accepted");
            if per_lane == 8 { assert!(draft.admit(9999).is_err(), "draft capacity exceeded"); }
            let mut members = [
                (0..per_lane).collect::<Vec<_>>(),
                (per_lane..2 * per_lane).collect::<Vec<_>>(),
            ];
            let mut tokens: Vec<Vec<u32>> = (0..2 * per_lane)
                .map(|slot| (0..8).map(|i| ((slot * 97 + i * 7919 + 17) % 129280) as u32).collect())
                .collect();
            let mut retired = Vec::new();
            for step in 0..4 {
                // Both previous full stacks have committed. Retire requests from
                // lane zero, then migrate survivors without moving cache storage.
                if step == 2 && per_lane > 1 {
                    for _ in 0..2 {
                        let slot = members[0].pop().unwrap();
                        requests.release(leases[slot])?;
                        draft.release(9000 + slot as u64)?;
                        retired.push(slot);
                    }
                    let moved = members[1].pop().unwrap();
                    members[0].push(moved);
                }
                // Swap owners after another complete step to exercise migration
                // in both directions, including the two-single-request case.
                if step == 3 { members.swap(0, 1); }
                let work = members.each_ref().map(|slots| slots.iter().map(|&slot| RequestTokens {
                    lease: leases[slot], tokens: &tokens[slot], image_mask: None,
                    kind: if step == 0 { ExpertV2SourceKind::Prefill } else { ExpertV2SourceKind::Decode },
                }).collect::<Vec<_>>());
                let mut a = requests.prepare(&work[0])?;
                let mut b = requests.prepare(&work[1])?;
                let width = if step == 0 { 8 } else { 1 };
                let selected = members.each_ref().map(|slots|
                    (0..slots.len()).map(|i| (i + 1) * width - 1).collect::<Vec<_>>());
                let started = Instant::now();
                let (a_logits, b_logits) = runtime.block_on(async {
                    if overlap {
                        tokio::try_join!(
                            unsafe { first.execute(requests, &mut a, first_transport, 0, &selected[0]) },
                            unsafe { second.execute(requests, &mut b, second_transport, 0, &selected[1]) },
                        )
                    } else {
                        let a_logits = unsafe { first.execute(requests, &mut a, first_transport, 0, &selected[0]).await? };
                        let b_logits = unsafe { second.execute(requests, &mut b, second_transport, 0, &selected[1]).await? };
                        Ok::<_, anyhow::Error>((a_logits, b_logits))
                    }
                })?;
                let elapsed = started.elapsed();
                let mut outputs = Vec::new();
                for logits in [a_logits, b_logits] {
                    let mut bytes = vec![0; logits.logits.bytes];
                    lib.copy_d2h(&mut bytes, logits.logits)?;
                    ensure!(bytes.chunks_exact(4).all(|v| f32::from_ne_bytes(v.try_into().unwrap()).is_finite()),
                        "non-finite lane logit");
                    outputs.push(bytes);
                }
                if overlap {
                    assert_eq!(outputs, expected[step], "overlap differs: per_lane={per_lane}, step={step}");
                } else { expected.push(outputs.clone()); }
                draft.commit_batch(first, requests, &mut a, &vec![width as u32; members[0].len()])?;
                draft.commit_batch(second, requests, &mut b, &vec![width as u32; members[1].len()])?;
                for (slots, bytes) in members.iter().zip(&outputs) {
                    for (&slot, row) in slots.iter().zip(bytes.chunks_exact(129280 * 4)) {
                        let next = row.chunks_exact(4).enumerate().map(|(i, b)|
                            (i as u32, f32::from_ne_bytes(b.try_into().unwrap())))
                            .max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
                        tokens[slot] = vec![next];
                        assert_eq!(requests.cache().committed_end(leases[slot])?, 8 + step as u64);
                        draft.validate_position(9000 + slot as u64, 8 + step as u64)?;
                    }
                }
                let proposal_inputs: Vec<_> = members.iter().flatten().map(|&slot|
                    (9000 + slot as u64, tokens[slot][0], 8 + step as u64,
                     if step == 2 && slot % 2 == 0 { 1 } else { 6 })).collect();
                let proposals = draft.propose(lib, 0, &proposal_inputs)?;
                for (proposal, &(_, anchor, _, remaining)) in proposals.iter().zip(&proposal_inputs) {
                    assert_eq!(proposal[0], anchor);
                    assert_eq!(proposal.len(), remaining);
                }
                if overlap { assert_eq!(proposals, expected_proposals[step], "draft batch changed after overlap/migration"); }
                else { expected_proposals.push(proposals); }
                eprintln!("decode lanes per_lane={per_lane} overlap={overlap} step={step} members={:?} execute_us={}",
                    members.each_ref().map(|m| m.len()), elapsed.as_micros());
            }
            for (slot, lease) in leases.into_iter().enumerate() {
                if !retired.contains(&slot) { requests.release(lease)?; draft.release(9000 + slot as u64)?; }
            }
        }
    }
    Ok(())
}
