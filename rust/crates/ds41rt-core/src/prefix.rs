//! Token-prefix radix retention shared by the in-memory snapshot cache
//! (`ds41rt-daemon::v41_native_serve::prefix`) and the disk persistence index (`ds41rt-persist`).
//!
//! Moved here verbatim from the daemon so both tiers apply one reuse rule: the most computation
//! saved wins, where a partial match replays from an even-aligned position minus a 128-token
//! window and a populated exact ancestor can beat a slightly longer partial match
//! (`Reusable::skipped`). Prompt repeats and completed turns keep separate bounded banks
//! (`Retention`); a tie prefers the completed turn.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

struct Node<T> {
    edge: Vec<u32>,
    value: Option<(u64, T)>,
    children: BTreeMap<u32, Node<T>>,
}
#[derive(Clone, Copy)]
struct Reusable {
    common: usize,
    frontier: usize,
    clock: u64,
}
impl Reusable {
    fn skipped(self) -> usize {
        if self.common == self.frontier {
            self.common
        } else {
            (self.common / 2 * 2).saturating_sub(128)
        }
    }
}
impl<T> Node<T> {
    fn empty(edge: Vec<u32>) -> Self {
        Self {
            edge,
            value: None,
            children: BTreeMap::new(),
        }
    }
    fn insert(&mut self, tokens: &[u32], value: T, clock: u64) -> bool {
        if tokens.is_empty() {
            return self.value.replace((clock, value)).is_none();
        }
        let child = self
            .children
            .entry(tokens[0])
            .or_insert_with(|| Self::empty(tokens.to_vec()));
        let common = child
            .edge
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common < child.edge.len() {
            let mut old = std::mem::replace(child, Self::empty(tokens[..common].to_vec()));
            old.edge.drain(..common);
            child.children.insert(old.edge[0], old);
        }
        child.insert(&tokens[common..], value, clock)
    }
    fn lookup<'a>(
        &'a mut self,
        tokens: &[u32],
        position: usize,
        clock: u64,
    ) -> Option<(usize, &'a T)> {
        if let Some(first) = tokens.first() {
            if let Some(child) = self.children.get_mut(first) {
                if tokens.starts_with(&child.edge) {
                    let length = child.edge.len();
                    if let Some(found) = child.lookup(&tokens[length..], position + length, clock) {
                        return Some(found);
                    }
                }
            }
        }
        self.value.as_mut().map(|(used, value)| {
            *used = clock;
            (position, &*value)
        })
    }
    fn oldest(&self) -> Option<u64> {
        self.value
            .as_ref()
            .map(|v| v.0)
            .into_iter()
            .chain(self.children.values().filter_map(Self::oldest))
            .min()
    }
    fn any_frontier(&self, position: usize) -> Option<Reusable> {
        self.value
            .as_ref()
            .map(|&(clock, _)| Reusable {
                common: position,
                frontier: position,
                clock,
            })
            .or_else(|| {
                self.children
                    .values()
                    .find_map(|child| child.any_frontier(position + child.edge.len()))
            })
    }
    fn find_reusable(&self, tokens: &[u32], position: usize) -> Option<Reusable> {
        let mut best = self.value.as_ref().map(|&(clock, _)| Reusable {
            common: position,
            frontier: position,
            clock,
        });
        if let Some(child) = tokens.first().and_then(|first| self.children.get(first)) {
            let common = child
                .edge
                .iter()
                .zip(tokens)
                .take_while(|(a, b)| a == b)
                .count();
            let candidate = if common == child.edge.len() {
                child.find_reusable(&tokens[common..], position + common)
            } else {
                child
                    .any_frontier(position + child.edge.len())
                    .map(|found| Reusable {
                        common: position + common,
                        ..found
                    })
            };
            if let Some(candidate) = candidate {
                if best.is_none_or(|old| candidate.skipped() > old.skipped()) {
                    best = Some(candidate);
                }
            }
        }
        if best.is_none() && position > 0 {
            best = self.any_frontier(position).map(|found| Reusable {
                common: position,
                ..found
            });
        }
        best.filter(|found| found.skipped() > 0)
    }
    fn refresh(&mut self, old: u64, new: u64) -> Option<&T> {
        if let Some((clock, value)) = self.value.as_mut() {
            if *clock == old {
                *clock = new;
                return Some(value);
            }
        }
        self.children
            .values_mut()
            .find_map(|child| child.refresh(old, new))
    }
    fn exact_clock(&self, tokens: &[u32]) -> Option<u64> {
        if tokens.is_empty() {
            return self.value.as_ref().map(|v| v.0);
        }
        let child = self.children.get(&tokens[0])?;
        tokens
            .strip_prefix(child.edge.as_slice())
            .and_then(|rest| child.exact_clock(rest))
    }
    fn oldest_where(&self, keep: &dyn Fn(&T) -> bool) -> Option<u64> {
        self.value
            .as_ref()
            .filter(|(_, value)| !keep(value))
            .map(|v| v.0)
            .into_iter()
            .chain(self.children.values().filter_map(|child| child.oldest_where(keep)))
            .min()
    }
    fn evict(&mut self, clock: u64) -> Option<T> {
        let mut evicted = match self.value.as_ref() {
            Some((c, _)) if *c == clock => self.value.take().map(|(_, value)| value),
            _ => None,
        };
        for child in self.children.values_mut() {
            if let Some(value) = child.evict(clock) {
                evicted = Some(value);
            }
        }
        self.children
            .retain(|_, child| child.value.is_some() || !child.children.is_empty());
        for child in self.children.values_mut() {
            while child.value.is_none() && child.children.len() == 1 {
                let (_, next) = child.children.pop_first().unwrap();
                child.edge.extend(next.edge);
                child.value = next.value;
                child.children = next.children;
            }
        }
        evicted
    }
}
pub struct Radix<T> {
    root: Node<T>,
    clock: u64,
    entries: usize,
    limit: usize,
}
impl<T> Radix<T> {
    pub fn new(limit: usize) -> Self {
        Self {
            root: Node::empty(Vec::new()),
            clock: 0,
            entries: 0,
            limit,
        }
    }
    /// Insert, evicting the least recently used entry if the bank is over its limit; returns
    /// that evicted entry so its owner can act before it is dropped.
    pub fn insert(&mut self, tokens: &[u32], value: T) -> Option<T> {
        if self.limit == 0 || tokens.is_empty() {
            return None;
        }
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        self.entries += usize::from(self.root.insert(tokens, value, self.clock));
        let mut evicted = None;
        while self.entries > self.limit {
            evicted = self.evict_oldest();
        }
        evicted
    }
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(usize, &T)> {
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        self.root.lookup(tokens, 0, self.clock)
    }
    /// Prefer the most computation saved: a populated exact ancestor can beat
    /// a slightly longer partial match which needs a complete replay window.
    pub fn lookup_reusable(&mut self, tokens: &[u32]) -> Option<(usize, usize, &T)> {
        let found = self.root.find_reusable(tokens, 0)?;
        self.clock = self
            .clock
            .checked_add(1)
            .expect("prefix access clock exhausted");
        let value = self
            .root
            .refresh(found.clock, self.clock)
            .expect("selected retained frontier");
        Some((found.common, found.frontier, value))
    }
    /// Retained entries in this bank.
    pub fn entries(&self) -> usize {
        self.entries
    }
    /// The bound this bank was created with; zero disables retention entirely.
    pub fn limit(&self) -> usize {
        self.limit
    }
    /// True when no entry and no edge remain: eviction has fully collapsed the tree.
    pub fn is_empty(&self) -> bool {
        self.root.value.is_none() && self.root.children.is_empty()
    }
    pub fn evict_one(&mut self) -> bool {
        self.evict_oldest().is_some()
    }
    /// Evict the least recently used entry and return it; `None` when empty.
    pub fn evict_oldest(&mut self) -> Option<T> {
        let oldest = self.root.oldest()?;
        let evicted = self.root.evict(oldest);
        self.entries -= 1;
        evicted
    }
    /// Evict the least recently used entry for which `keep` is false and return it.
    pub fn evict_oldest_where(&mut self, keep: &dyn Fn(&T) -> bool) -> Option<T> {
        let oldest = self.root.oldest_where(keep)?;
        let evicted = self.root.evict(oldest);
        self.entries -= 1;
        evicted
    }
    /// Remove and return the entry whose token sequence is exactly `tokens`.
    pub fn remove_exact(&mut self, tokens: &[u32]) -> Option<T> {
        let clock = self.root.exact_clock(tokens)?;
        let removed = self.root.evict(clock);
        self.entries -= 1;
        removed
    }
}
/// Prompt repeats and completed agentic turns have separate bounded banks.
/// A turn slot cannot be consumed by its own prompt snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotKind {
    Prompt,
    Turn,
}
pub struct Retention<T> {
    prompts: Radix<T>,
    turns: Radix<T>,
}
impl<T> Retention<T> {
    pub fn new(limit: usize) -> Self {
        Self { prompts: Radix::new(limit), turns: Radix::new(limit) }
    }
    pub fn bank(&self, kind: SnapshotKind) -> &Radix<T> {
        match kind {
            SnapshotKind::Prompt => &self.prompts,
            SnapshotKind::Turn => &self.turns,
        }
    }
    pub fn bank_mut(&mut self, kind: SnapshotKind) -> &mut Radix<T> {
        match kind { SnapshotKind::Prompt => &mut self.prompts, SnapshotKind::Turn => &mut self.turns }
    }
    pub fn lookup_reusable(&mut self, tokens: &[u32]) -> Option<(usize, usize, &T)> {
        let prompt = self.prompts.root.find_reusable(tokens, 0);
        let turn = self.turns.root.find_reusable(tokens, 0);
        // Prefer completed turns on a tie. Refresh only the chosen bank's LRU.
        let kind = match (prompt, turn) {
            (Some(p), Some(t)) if p.skipped() > t.skipped() => SnapshotKind::Prompt,
            (_, Some(_)) => SnapshotKind::Turn,
            (Some(_), None) => SnapshotKind::Prompt,
            (None, None) => return None,
        };
        self.bank_mut(kind).lookup_reusable(tokens)
    }
    /// `evict_one`, returning the evicted entry and its bank.
    pub fn evict_oldest(&mut self) -> Option<(SnapshotKind, T)> {
        self.prompts
            .evict_oldest()
            .map(|value| (SnapshotKind::Prompt, value))
            .or_else(|| self.turns.evict_oldest().map(|value| (SnapshotKind::Turn, value)))
    }
    /// `evict_oldest` restricted to entries for which `keep` is false, same bank order.
    pub fn evict_one_where(&mut self, keep: &dyn Fn(&T) -> bool) -> Option<(SnapshotKind, T)> {
        self.prompts
            .evict_oldest_where(keep)
            .map(|value| (SnapshotKind::Prompt, value))
            .or_else(|| self.turns.evict_oldest_where(keep).map(|value| (SnapshotKind::Turn, value)))
    }
    pub fn evict_one(&mut self) -> bool {
        // Under global-page pressure, reclaim prompt snapshots before completed
        // turns; their pages may remain shared until the turn also expires.
        self.evict_oldest().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};
    #[test]
    fn twenty_four_completed_turns_do_not_compete_with_prompt_snapshots() {
        let mut retained = Retention::new(24);
        for i in 100..124 {
            retained.bank_mut(SnapshotKind::Prompt).insert(&[i, 1], (false, i));
            retained.bank_mut(SnapshotKind::Turn).insert(&[i, 1, 2], (true, i));
        }
        assert_eq!((retained.prompts.entries, retained.turns.entries), (24, 24));
        for i in 100..124 {
            assert_eq!(retained.lookup_reusable(&[i, 1]), Some((2, 2, &(false, i))));
            assert_eq!(retained.lookup_reusable(&[i, 1, 2, 3]), Some((3, 3, &(true, i))));
        }
        for i in 200..224 {
            retained.bank_mut(SnapshotKind::Prompt).insert(&[i, 1], (false, i));
        }
        for i in 100..124 {
            assert_eq!(retained.lookup_reusable(&[i, 1, 2, 3]), Some((3, 3, &(true, i))));
        }
        retained.bank_mut(SnapshotKind::Turn).insert(&[999, 1, 2], (true, 999));
        assert!(retained.lookup_reusable(&[100, 1, 2, 3]).is_none());
        assert_eq!(retained.turns.entries, 24);
        for _ in 0..24 { assert!(retained.evict_one()); }
        assert_eq!((retained.prompts.entries, retained.turns.entries), (0, 24));
        for _ in 0..24 { assert!(retained.evict_one()); }
        assert!(!retained.evict_one());
        let mut disabled = Retention::new(0);
        disabled.bank_mut(SnapshotKind::Turn).insert(&[1], (true, 1));
        assert!(disabled.lookup_reusable(&[1]).is_none());
    }

    /// Contract pin: generated-turn retention is opportunistic, prompt-prefix
    /// reuse is the guarantee. A child that re-renders a completed turn from text
    /// may re-tokenize it differently; the completed turn then matches only
    /// partially and, because a partial match replays a 128-token encoder window,
    /// it saves less computation than the exact prompt ancestor. The reuse rule
    /// therefore selects the prompt snapshot and the reported frontier is the
    /// prompt length, not the completed turn. This is the v10 65536/262144 shape
    /// (`hit == context`) and the reason it is not a runtime publication fault.
    #[test]
    fn short_partial_completed_turn_defers_to_the_exact_prompt_snapshot() {
        let context = 4096usize;
        let prompt: Vec<u32> = (1..=context as u32).collect();
        let mut turn = prompt.clone();
        turn.extend(10_000..10_079); // 79-token completed turn (v10 65536 shape)
        let mut retained = Retention::new(8);
        retained.bank_mut(SnapshotKind::Prompt).insert(&prompt, "prompt");
        retained.bank_mut(SnapshotKind::Turn).insert(&turn, "turn");
        // The child re-sends the prompt, a turn that re-encodes with one token
        // different before the end, and then its own instruction.
        let mut divergent = turn.clone();
        divergent[context + 78] = 77_777;
        divergent.extend(20_000..20_044);
        assert_eq!(
            retained.lookup_reusable(&divergent),
            Some((context, context, &"prompt"))
        );
        // The same child whose turn round-trips exactly reuses the whole turn.
        let mut exact = turn.clone();
        exact.extend(20_000..20_044);
        assert_eq!(
            retained.lookup_reusable(&exact),
            Some((context + 79, context + 79, &"turn"))
        );
    }

    /// A partial completed turn longer than the 128-token replay window can still
    /// save more than the prompt ancestor; the reported frontier is then the
    /// aligned LCP minus the replay window. That is the v9 262144 shape
    /// (`hit = 262158 = 262144 + 14` for a 147-token turn matching 142 tokens).
    #[test]
    fn partial_completed_turn_that_saves_more_than_the_prompt_is_reused_from_its_window() {
        let context = 4096usize;
        let prompt: Vec<u32> = (1..=context as u32).collect();
        let mut turn = prompt.clone();
        turn.extend(10_000..10_147); // 147-token completed turn (v9 262144 shape)
        let mut retained = Retention::new(8);
        retained.bank_mut(SnapshotKind::Prompt).insert(&prompt, "prompt");
        retained.bank_mut(SnapshotKind::Turn).insert(&turn, "turn");
        let mut divergent = turn.clone();
        divergent[context + 142] = 77_777; // matches 142 of 147 generated tokens
        divergent.extend(20_000..20_044);
        let (common, frontier, value) = retained.lookup_reusable(&divergent).unwrap();
        assert_eq!((common, frontier, value), (context + 142, context + 147, &"turn"));
        // `v41_native_serve::prefix::restore` reports the replay start for a
        // partial match: `(common / 2 * 2) - 128`.
        assert_eq!((common / 2 * 2) - 128, context + 14);
    }

    #[test]
    fn partial_radix_match_accounts_for_alignment_replay_and_exact_ancestors() {
        let tokens: Vec<u32> = (1..=512).collect();
        let mut radix = Radix::new(16);
        radix.insert(&tokens, 512);
        let mut partial = tokens[..451].to_vec();
        partial.push(9999);
        assert_eq!(radix.lookup_reusable(&partial), Some((451, 512, &512)));
        radix.insert(&tokens[..384], 384);
        // Replaying at 450 only skips 322 tokens; the populated 384-token
        // ancestor skips more work and preserves its complete window state.
        assert_eq!(radix.lookup_reusable(&partial), Some((384, 384, &384)));
        assert_eq!(radix.lookup_reusable(&tokens), Some((512, 512, &512)));
        assert!(radix.lookup_reusable(&tokens[..128]).is_none());
        assert!(radix.lookup_reusable(&[9999]).is_none());
        assert_eq!(
            radix.lookup_reusable(&tokens[..258]),
            Some((258, 384, &384))
        );
        assert!(radix.evict_one());
        assert_eq!(radix.entries, 1);
        assert_eq!(
            radix.lookup_reusable(&tokens[..258]),
            Some((258, 384, &384))
        );
        assert!(radix.evict_one());
        assert_eq!(radix.entries, 0);
        assert!(radix.root.children.is_empty());
    }

    #[test]
    fn token_radix_splits_edges_and_returns_longest_retained_frontier() {
        let mut radix = Radix::new(16);
        radix.insert(&[10, 20, 30, 40], 4);
        radix.insert(&[10, 20], 2);
        radix.insert(&[10, 20, 50], 3);
        assert_eq!(radix.lookup(&[10, 20, 30, 40, 60]), Some((4, &4)));
        assert_eq!(radix.lookup(&[10, 20, 30]), Some((2, &2)));
        assert_eq!(radix.lookup(&[10, 20, 50]), Some((3, &3)));
        assert!(radix.lookup(&[10]).is_none());
        assert!(radix.lookup(&[99, 20]).is_none());
        radix.insert(&[10, 20], 99);
        assert_eq!(radix.entries, 3);
        assert_eq!(radix.lookup(&[10, 20]), Some((2, &99)));
    }
    #[test]
    fn evict_oldest_where_skips_kept_entries_in_bank_order() {
        let mut retained = Retention::new(16);
        retained.bank_mut(SnapshotKind::Prompt).insert(&[1, 2], "p-old");
        retained.bank_mut(SnapshotKind::Prompt).insert(&[1, 3], "p-new");
        retained.bank_mut(SnapshotKind::Turn).insert(&[9, 9], "t-old");
        let keep = |v: &&str| *v == "p-old";
        assert_eq!(retained.evict_one_where(&keep), Some((SnapshotKind::Prompt, "p-new")));
        assert_eq!(retained.lookup_reusable(&[1, 3]), None);
        assert_eq!(retained.lookup_reusable(&[1, 2]), Some((2, 2, &"p-old")));
        assert_eq!(retained.evict_one_where(&keep), Some((SnapshotKind::Turn, "t-old")));
        assert_eq!(retained.lookup_reusable(&[9, 9]), None);
        assert_eq!(retained.evict_one_where(&keep), None);
        assert_eq!(retained.bank(SnapshotKind::Prompt).entries(), 1);
        assert_eq!(retained.evict_oldest(), Some((SnapshotKind::Prompt, "p-old")));
        assert!(retained.bank(SnapshotKind::Prompt).is_empty());
    }
    #[test]
    fn radix_eviction_drops_saved_owners_and_preserves_recent_branch() {
        struct Owner(Rc<Cell<usize>>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let dropped = Rc::new(Cell::new(0));
        let mut radix = Radix::new(2);
        radix.insert(&[1, 2], Owner(dropped.clone()));
        radix.insert(&[1, 2, 3], Owner(dropped.clone()));
        radix.lookup(&[1, 2]);
        radix.insert(&[1, 4], Owner(dropped.clone()));
        assert_eq!(dropped.get(), 1);
        assert_eq!(radix.lookup(&[1, 2, 3]).unwrap().0, 2);
        assert!(radix.evict_one());
        assert!(radix.evict_one());
        assert!(!radix.evict_one());
        assert_eq!(dropped.get(), 3);
        assert!(radix.root.children.is_empty());
    }
}
