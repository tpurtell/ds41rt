//! Shared model for the HC-2 suites: mirrors `Snapshots` with a separately maintained
//! `Retention<Key>` and exact page-reference and byte accounting, so a random or interleaved
//! sequence can be checked after every step. Plans are modelled separately from commits so the
//! concurrency suite can interleave plan, commit and abort steps.
#![allow(dead_code)]
use ds41rt_core::prefix::{Retention, SnapshotKind};
use ds41rt_hostcache::cache::HostCache;
use ds41rt_hostcache::copy::StubCopyEngine;
use ds41rt_hostcache::pool::testing::layout;
use ds41rt_hostcache::pool::Layout;
use ds41rt_hostcache::snapshot::{DevicePageId, Key, SnapshotMeta, Snapshots, StorePlan};
use ds41rt_hostcache::COMPRESSORS;
use std::collections::HashMap;

/// xorshift64*: deterministic, so a failing schedule reproduces from its seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The expected state after a sequence of operations.
pub struct Model {
    pub retention: Retention<Key>,
    pub snapshots: HashMap<Key, ModelSnapshot>,
    /// Model page id to reference count.
    pub page_refs: HashMap<u32, u32>,
    pub page_device: HashMap<u32, DevicePageId>,
    /// Device identity to the model page that currently holds its bytes.
    pub mapped: HashMap<DevicePageId, u32>,
    /// Plans that have been planned but not committed or aborted.
    pub plans: HashMap<Key, ModelPlan>,
    pub bytes: u64,
    pub next_key: Key,
    pub next_page: u32,
    pub layout: Layout,
}

pub struct ModelSnapshot {
    pub kind: SnapshotKind,
    pub tokens: Vec<u32>,
    pub pages: Vec<u32>,
    pub has_draft: bool,
    pub pins: u32,
}

/// A planned store: the pages it holds, the copies it still owes and the non-page bytes it
/// holds (page bytes are released with the page references).
pub struct ModelPlan {
    pub meta: SnapshotMeta,
    /// `(compressor, index, device identity, model page)` for pages to copy.
    pub copies: Vec<(u8, u32, DevicePageId, u32)>,
    /// Per compressor, in logical order, the page references (shared or newly allocated).
    pub pages: [Vec<u32>; COMPRESSORS],
    pub slab_bytes: u64,
}

impl Model {
    pub fn new() -> Self {
        Self {
            retention: Retention::new(usize::MAX),
            snapshots: HashMap::new(),
            page_refs: HashMap::new(),
            page_device: HashMap::new(),
            mapped: HashMap::new(),
            plans: HashMap::new(),
            bytes: 0,
            next_key: 0,
            next_page: 0,
            layout: layout(),
        }
    }

    /// Mirror a successful `plan_store`: allocate the parts and every unshared page, but do not
    /// fill the identity map until commit.
    pub fn plan(
        &mut self,
        meta: &SnapshotMeta,
        device_pages: &[Vec<DevicePageId>; COMPRESSORS],
    ) -> Key {
        let mut pages: [Vec<u32>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        let mut copies = Vec::new();
        let mut slab_bytes = self.layout.tail as u64;
        if self.layout.scores > 0 {
            slab_bytes += self.layout.scores as u64;
        }
        if meta.has_draft && self.layout.draft > 0 {
            slab_bytes += self.layout.draft as u64;
        }
        let mut page_bytes = 0u64;
        for (compressor, list) in device_pages.iter().enumerate() {
            for (index, &id) in list.iter().enumerate() {
                let page = match self.mapped.get(&id).copied() {
                    Some(page) => page,
                    None => {
                        let page = self.next_page;
                        self.next_page += 1;
                        self.page_device.insert(page, id);
                        self.page_refs.insert(page, 0);
                        page_bytes += self.layout.page as u64;
                        copies.push((compressor as u8, index as u32, id, page));
                        page
                    }
                };
                *self.page_refs.get_mut(&page).expect("model page") += 1;
                pages[compressor].push(page);
            }
        }
        let key = self.next_key;
        self.next_key += 1;
        self.bytes += slab_bytes + page_bytes;
        self.plans.insert(
            key,
            ModelPlan {
                meta: meta.clone(),
                copies,
                pages,
                slab_bytes,
            },
        );
        key
    }

    /// Mirror a successful `commit_store`: fill the identity map, release any duplicate page and
    /// make the snapshot resident.
    pub fn commit(&mut self, key: Key) -> Key {
        let plan = self.plans.remove(&key).expect("model plan");
        let mut pages = plan.pages;
        for &(compressor, index, id, page) in &plan.copies {
            match self.mapped.get(&id).copied() {
                Some(existing) if existing != page => {
                    self.release_page(page);
                    *self.page_refs.get_mut(&existing).expect("model page") += 1;
                    pages[compressor as usize][index as usize] = existing;
                }
                Some(_) => {}
                None => {
                    self.mapped.insert(id, page);
                }
            }
        }
        let meta = plan.meta;
        let replaced = self
            .retention
            .bank_mut(meta.kind)
            .lookup(&meta.tokens)
            .and_then(|(position, &old)| (position == meta.tokens.len()).then_some(old));
        if let Some(old) = replaced {
            self.release(old);
        }
        self.retention.bank_mut(meta.kind).insert(&meta.tokens, key);
        self.snapshots.insert(
            key,
            ModelSnapshot {
                kind: meta.kind,
                tokens: meta.tokens,
                pages: pages.into_iter().flatten().collect(),
                has_draft: meta.has_draft,
                pins: 0,
            },
        );
        key
    }

    /// Mirror a successful `plan_store` + `commit_store`.
    pub fn store(
        &mut self,
        meta: &SnapshotMeta,
        device_pages: &[Vec<DevicePageId>; COMPRESSORS],
    ) -> Key {
        let key = self.plan(meta, device_pages);
        self.commit(key)
    }

    /// Mirror `abort_store`: release the plan's parts and page references.
    pub fn abort(&mut self, key: Key) {
        let plan = self.plans.remove(&key).expect("model plan");
        for list in &plan.pages {
            for &page in list {
                self.release_page(page);
            }
        }
        self.bytes -= plan.slab_bytes;
    }

    /// Drop a snapshot and every page reference it holds.
    pub fn release(&mut self, key: Key) {
        let Some(snapshot) = self.snapshots.remove(&key) else {
            return;
        };
        self.bytes -= self.layout.tail as u64;
        if self.layout.scores > 0 {
            self.bytes -= self.layout.scores as u64;
        }
        if snapshot.has_draft && self.layout.draft > 0 {
            self.bytes -= self.layout.draft as u64;
        }
        for page in snapshot.pages {
            self.release_page(page);
        }
    }

    /// Drop one reference to a model page; free it at zero.
    fn release_page(&mut self, page: u32) {
        let count = self.page_refs.get_mut(&page).expect("model page");
        *count -= 1;
        if *count == 0 {
            self.page_refs.remove(&page);
            let device = self.page_device.remove(&page).expect("model device");
            if self.mapped.get(&device) == Some(&page) {
                self.mapped.remove(&device);
            }
            self.bytes -= self.layout.page as u64;
        }
    }

    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(usize, usize, Key)> {
        self.retention
            .lookup_reusable(tokens)
            .map(|(common, frontier, &key)| (common, frontier, key))
    }

    pub fn evict_to(&mut self, quota: u64) -> (Vec<Key>, u64) {
        let mut evicted = Vec::new();
        let mut freed = 0;
        while self.bytes > quota {
            let Some((key, bytes)) = self.evict_one() else {
                break;
            };
            evicted.push(key);
            freed += bytes;
        }
        (evicted, freed)
    }

    /// Evict the least recently used unpinned snapshot; `None` when only pinned snapshots
    /// remain. Mirrors `Snapshots::evict_one`.
    pub fn evict_one(&mut self) -> Option<(Key, u64)> {
        let next = {
            let snapshots = &self.snapshots;
            self.retention.evict_one_where(&|key: &Key| {
                snapshots.get(key).is_some_and(|snapshot| snapshot.pins > 0)
            })
        };
        let (_kind, key) = next?;
        let before = self.bytes;
        self.release(key);
        Some((key, before - self.bytes))
    }

    pub fn remove(&mut self, key: Key) -> bool {
        let Some(snapshot) = self.snapshots.get(&key) else {
            return false;
        };
        let kind = snapshot.kind;
        let tokens = snapshot.tokens.clone();
        self.retention.bank_mut(kind).remove_exact(&tokens);
        self.release(key);
        true
    }

    pub fn pin(&mut self, key: Key) {
        if let Some(snapshot) = self.snapshots.get_mut(&key) {
            snapshot.pins += 1;
        }
    }

    pub fn unpin(&mut self, key: Key) {
        if let Some(snapshot) = self.snapshots.get_mut(&key) {
            snapshot.pins = snapshot.pins.saturating_sub(1);
        }
    }

    pub fn device_page_freed(&mut self, id: DevicePageId) {
        self.mapped.remove(&id);
    }

    /// Assert the real store matches the model exactly.
    pub fn check(&self, store: &Snapshots) {
        assert_eq!(store.len(), self.snapshots.len(), "snapshot count");
        assert_eq!(store.bytes_used(), self.bytes, "bytes used");
        assert_eq!(store.live_pages(), self.page_refs.len(), "live pages");
        assert_eq!(
            store.page_ref_total(),
            self.page_refs
                .values()
                .map(|&refs| u64::from(refs))
                .sum::<u64>(),
            "page reference total"
        );
        let mut real = store.page_ref_counts();
        let mut model: Vec<u32> = self.page_refs.values().copied().collect();
        real.sort_unstable();
        model.sort_unstable();
        assert_eq!(real, model, "page reference counts");
        for kind in [SnapshotKind::Prompt, SnapshotKind::Turn] {
            assert_eq!(
                store.retention().bank(kind).entries(),
                self.retention.bank(kind).entries(),
                "retention entries for {kind:?}"
            );
        }
        for (&key, expected) in &self.snapshots {
            let actual = store.get(key).expect("snapshot present");
            assert_eq!(actual.meta.kind, expected.kind);
            assert_eq!(actual.meta.tokens, expected.tokens);
            assert_eq!(actual.pins, expected.pins);
        }
    }
}

/// One step of a suite's schedule. `Commit` and `Abort` act on the caller's pending plan.
#[derive(Clone, Debug)]
pub enum Op {
    Store {
        kind: SnapshotKind,
        tokens: Vec<u32>,
        has_draft: bool,
        pages: Vec<DevicePageId>,
    },
    Plan {
        kind: SnapshotKind,
        tokens: Vec<u32>,
        has_draft: bool,
        pages: Vec<DevicePageId>,
    },
    Commit,
    Abort,
    Lookup {
        tokens: Vec<u32>,
    },
    Evict {
        quota: u64,
    },
    Free {
        id: DevicePageId,
    },
    Pin {
        key: Key,
    },
    Unpin {
        key: Key,
    },
    Remove {
        key: Key,
    },
}

/// Spread a flat page list across the compressor banks, as the engine's page lists are.
fn distribute(pages: &[DevicePageId]) -> [Vec<DevicePageId>; COMPRESSORS] {
    let mut device_pages: [Vec<DevicePageId>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
    for (index, &id) in pages.iter().enumerate() {
        device_pages[index % COMPRESSORS].push(id);
    }
    device_pages
}

fn snapshot_meta(kind: SnapshotKind, tokens: &[u32], has_draft: bool) -> SnapshotMeta {
    SnapshotMeta {
        kind,
        tokens: tokens.to_vec(),
        end: tokens.len() as u32,
        has_draft,
    }
}

/// Apply `op` to the real store and the model, then assert they agree. `pending` holds the
/// caller's in-flight plan for `Commit` and `Abort`; a commit or abort with no pending plan is a
/// no-op. Returns the key a store or plan produced.
pub fn apply(
    store: &mut Snapshots,
    model: &mut Model,
    op: &Op,
    now_ns: u64,
    pending: &mut Option<StorePlan>,
) -> Option<Key> {
    match op {
        Op::Store {
            kind,
            tokens,
            has_draft,
            pages,
        } => {
            let device_pages = distribute(pages);
            let meta = snapshot_meta(*kind, tokens, *has_draft);
            let plan = store
                .plan_store(meta.clone(), &device_pages)
                .expect("plan succeeds");
            let key = store.commit_store(plan, now_ns);
            let model_key = model.store(&meta, &device_pages);
            assert_eq!(key, model_key, "key");
            model.check(store);
            Some(key)
        }
        Op::Plan {
            kind,
            tokens,
            has_draft,
            pages,
        } => {
            let device_pages = distribute(pages);
            let meta = snapshot_meta(*kind, tokens, *has_draft);
            let plan = store
                .plan_store(meta.clone(), &device_pages)
                .expect("plan succeeds");
            let key = plan.key;
            let model_key = model.plan(&meta, &device_pages);
            assert_eq!(key, model_key, "plan key");
            *pending = Some(plan);
            model.check(store);
            Some(key)
        }
        Op::Commit => {
            let plan = pending.take()?;
            let key = plan.key;
            let real = store.commit_store(plan, now_ns);
            let model_key = model.commit(key);
            assert_eq!(real, model_key, "commit key");
            model.check(store);
            Some(real)
        }
        Op::Abort => {
            let plan = pending.take()?;
            let key = plan.key;
            store.abort_store(plan);
            model.abort(key);
            model.check(store);
            None
        }
        Op::Lookup { tokens } => {
            let hit = store.lookup(tokens, now_ns);
            let expected = model.lookup(tokens);
            match (hit, expected) {
                (Some(hit), Some((common, frontier, key))) => {
                    assert_eq!(hit.key, key, "lookup key");
                    assert_eq!(hit.common, common, "lookup common");
                    assert_eq!(hit.frontier, frontier, "lookup frontier");
                    assert_eq!(hit.kind, model.snapshots[&key].kind, "lookup kind");
                }
                (None, None) => {}
                (hit, expected) => panic!("lookup mismatch: {hit:?} vs {expected:?}"),
            }
            model.check(store);
            None
        }
        Op::Evict { quota } => {
            let (keys, freed) = store.evict_to(*quota);
            let (model_keys, model_freed) = model.evict_to(*quota);
            assert_eq!(keys, model_keys, "evicted keys");
            assert_eq!(freed, model_freed, "freed bytes");
            model.check(store);
            None
        }
        Op::Free { id } => {
            store.device_page_freed(*id);
            model.device_page_freed(*id);
            model.check(store);
            None
        }
        Op::Pin { key } => {
            store.pin(*key);
            model.pin(*key);
            model.check(store);
            None
        }
        Op::Unpin { key } => {
            // `unpin` of an unpinned snapshot is a logic error in the store, so only unpin a
            // snapshot the model still holds a pin on.
            if model
                .snapshots
                .get(key)
                .is_some_and(|snapshot| snapshot.pins > 0)
            {
                store.unpin(*key);
                model.unpin(*key);
            }
            model.check(store);
            None
        }
        Op::Remove { key } => {
            assert_eq!(store.remove(*key), model.remove(*key), "remove");
            model.check(store);
            None
        }
    }
}

/// Mirror the cache's eviction-on-exhaustion: while planning a store the cache may evict
/// unpinned snapshots, so bring the model's resident set down to the cache's.
pub fn reconcile<P>(cache: &HostCache<StubCopyEngine, P>, model: &mut Model) {
    while model.snapshots.len() > cache.metrics().resident_snapshots as usize {
        if model.evict_one().is_none() {
            break;
        }
    }
}
