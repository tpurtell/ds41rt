//! CLI-resolved replicated Spark `TP×EP` topology shared by both daemons.
//!
//! The only source of rank identity is
//! [`ds41rt_transport::v41_expert::V41SparkTopology`]: this module never
//! re-derives executor ids, group indices or physical rank maps, so transport,
//! worker shard selection and coordinator assembly cannot drift apart.
use anyhow::{ensure, Result};
use ds41rt_loader::OfficialV41Catalog;
use ds41rt_transport::v41_expert::V41SparkTopology;

/// Resolve the opt-in `--spark-tp N --spark-ep M` pair for a command that knows
/// how many physical Spark ranks were launched.
///
/// Both keys are all-or-none: absent means the legacy topology for the rank
/// count, present means the explicit replicated-group contract. `world` is the
/// physical rank count (coordinator peers / worker `--world`), and it must equal
/// `TP * EP` exactly.
pub(crate) fn resolve(
    tp: Option<u8>,
    ep: Option<u8>,
    world: usize,
    component: &str,
) -> Result<Option<V41SparkTopology>> {
    let topology = match (tp, ep) {
        (None, None) => return Ok(None),
        (Some(tp), Some(ep)) => V41SparkTopology::new(tp, ep)?,
        _ => anyhow::bail!(
            "{component} --spark-tp and --spark-ep must be given together or not at all"
        ),
    };
    ensure!(
        topology.world_size() == world,
        "{component} Spark topology {}x{} needs {} physical ranks but {world} were given",
        topology.tp(),
        topology.ep(),
        topology.world_size()
    );
    Ok(Some(topology))
}

/// Explicit replicated groups are defined only for the official native
/// checkpoint. Reject EXL3 or NVFP4 publications before any weight allocation,
/// transport connection or readiness publication.
pub(crate) fn require_native(
    topology: Option<V41SparkTopology>,
    catalog: &OfficialV41Catalog,
) -> Result<()> {
    let Some(topology) = topology else {
        return Ok(());
    };
    ensure!(
        catalog.exl3().is_none() && catalog.nvfp4().is_none(),
        "explicit SPARK_TP/SPARK_EP {}x{} requires the official native checkpoint; \
         EXL3 and NVFP4 publications keep their separate non-topology paths",
        topology.tp(),
        topology.ep()
    );
    Ok(())
}

/// The group this physical rank belongs to, or `None` for the legacy topology.
pub(crate) fn group_of(topology: Option<V41SparkTopology>, rank: usize) -> Result<Option<u8>> {
    topology.map(|topology| topology.group(rank)).transpose()
}

/// The worker's tensor-parallel shard index inside its group, or `None` for the
/// legacy topology.
pub(crate) fn tp_rank_of(topology: Option<V41SparkTopology>, rank: usize) -> Result<Option<u8>> {
    topology.map(|topology| topology.tp_rank(rank)).transpose()
}

/// Roles published by the two new native Spark shard families, for callers that
/// need the expected role without loading the library. Three- and six-rank
/// layouts reduce through the native generic N-plane entry point; admission
/// uses `V41CompactReducer::require_rank_count` on the loaded library before any
/// allocation or readiness publication, so a missing artifact fails startup
/// rather than the first request.
pub(crate) const SPARK_TP2_ROLE: u32 = 5;
pub(crate) const SPARK_TP3_ROLE: u32 = 6;

#[cfg(test)]
mod tests {
    use super::*;

    fn topology(tp: u8, ep: u8) -> V41SparkTopology {
        V41SparkTopology::new(tp, ep).unwrap()
    }

    #[test]
    fn absent_keys_keep_the_legacy_topology() {
        assert_eq!(resolve(None, None, 4, "serve-native").unwrap(), None);
        // The world count is irrelevant to the legacy path.
        assert_eq!(resolve(None, None, 2, "expertd-native").unwrap(), None);
    }

    #[test]
    fn keys_are_all_or_none() {
        assert!(resolve(Some(2), None, 4, "serve-native").is_err());
        assert!(resolve(None, Some(2), 4, "serve-native").is_err());
    }

    #[test]
    fn approved_layouts_require_their_exact_rank_count() {
        for (tp, ep) in [(2u8, 1u8), (3, 1), (4, 1), (2, 2), (3, 2), (2, 3)] {
            let expected = topology(tp, ep);
            let resolved = resolve(Some(tp), Some(ep), expected.world_size(), "serve-native")
                .unwrap()
                .expect("explicit topology");
            assert_eq!(resolved, expected);
            assert_eq!(resolved.world_size(), tp as usize * ep as usize);
            assert!(resolve(Some(tp), Some(ep), expected.world_size() + 1, "serve-native").is_err());
        }
    }

    #[test]
    fn unapproved_layouts_and_ranges_are_rejected() {
        for (tp, ep) in [(1u8, 1u8), (2, 4), (4, 2), (3, 3), (0, 1)] {
            let world = tp as usize * ep as usize;
            assert!(resolve(Some(tp), Some(ep), world, "serve-native").is_err(), "{tp}x{ep}");
        }
    }

    #[test]
    fn rank_helpers_use_the_shared_topology_mapping() {
        let topology = topology(2, 3);
        for rank in 0..6 {
            assert_eq!(group_of(Some(topology), rank).unwrap(), Some((rank / 2) as u8));
            assert_eq!(tp_rank_of(Some(topology), rank).unwrap(), Some((rank % 2) as u8));
        }
        assert!(group_of(Some(topology), 6).is_err());
        assert_eq!(group_of(None, 99).unwrap(), None);
        assert_eq!(tp_rank_of(None, 99).unwrap(), None);
    }
}
