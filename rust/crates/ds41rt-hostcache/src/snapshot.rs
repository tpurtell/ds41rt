//! Host snapshots (packet HC-2): what the cache holds and how it finds it. A host snapshot is
//! the four parts of a retained snapshot in pinned slabs plus its key-space tokens, frontier and
//! kind. Pages are shared exactly as the device shares them: while a device page is allocated,
//! its identity maps to at most one host page, so a page is copied once however many snapshots
//! reference it. The identity map is filled at commit, so a page whose copy is still in flight
//! is never shared: a concurrent plan copies it into its own slab and the later commit releases
//! the duplicate in favour of a reference. Lookups use the engine's own `Retention` radix
//! (`ds41rt-core::prefix`), so a host hit is exactly a snapshot the device tier would have
//! chosen. Eviction follows the same bank order (prompts before turns, oldest access first) and
//! never touches a pinned snapshot.
use crate::pool::{Class, HostRange, PoolExhausted, Slab, SlabPool};
use crate::SnapshotKind;
use crate::COMPRESSORS;
use ds41rt_core::prefix::Retention;
use serde::Serialize;
use std::collections::HashMap;

/// Identity of a device page while it is allocated: the engine's page index in its compressor
/// pool and the generation that increments on every free, so a reused index is a new identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct DevicePageId {
    /// Compressor bank the page belongs to.
    pub compressor: u8,
    /// Page index within the compressor's pool.
    pub page: u32,
    /// Free generation; a reused index with a new generation is a new identity.
    pub generation: u32,
}

/// Cache-local snapshot id, unique for the life of the cache.
pub type Key = u64;

/// Index into the shared page table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct PageRef(
    /// Index into the shared page table; live exactly while the entry is present.
    pub u32,
);

/// What the engine tells the cache about a snapshot at store time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotMeta {
    /// Which retention bank the snapshot belongs to.
    pub kind: SnapshotKind,
    /// Key-space tokens (image spans already folded in by the engine).
    pub tokens: Vec<u32>,
    /// Tokens the snapshot covers; `tokens.len()` for a whole-prefix snapshot.
    pub end: u32,
    /// Whether the snapshot carries dSpark draft rings.
    pub has_draft: bool,
}

/// A resident host snapshot: its key, metadata, shared page references and its own slabs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostSnapshot {
    /// Cache-local id, unique for the life of the cache.
    pub key: Key,
    /// What the engine told the cache at store time.
    pub meta: SnapshotMeta,
    /// Per compressor, in logical order.
    pub pages: [Vec<PageRef>; COMPRESSORS],
    /// The backbone tail arena slot.
    pub tail: Slab,
    /// The dSpark draft rings, when the snapshot has them and the layout enables the class.
    pub draft: Option<Slab>,
    /// The scores row, when the layout enables the class.
    pub scores: Option<Slab>,
    /// Virtual-clock time of the last lookup hit, for eviction order.
    pub last_access_ns: u64,
    /// Restores in flight; a pinned snapshot is never evicted.
    pub pins: u32,
}

/// The result of planning a store: slabs allocated for every part and for each page that is not
/// already shared, with the device identity each unshared page must be copied from. The device
/// identity map is filled by [`Snapshots::commit_store`], not here.
#[derive(Debug)]
pub struct StorePlan {
    /// The key the committed snapshot will have.
    pub key: Key,
    /// What the engine told the cache at store time.
    pub meta: SnapshotMeta,
    /// `(compressor, logical index, device identity, destination slab)` for pages to copy.
    pub copies: Vec<(u8, u32, DevicePageId, Slab)>,
    /// Per compressor, in logical order, the final page references (shared or newly allocated).
    pub pages: [Vec<PageRef>; COMPRESSORS],
    /// The backbone tail arena slot.
    pub tail: Slab,
    /// The dSpark draft rings, when planned.
    pub draft: Option<Slab>,
    /// The scores row, when planned.
    pub scores: Option<Slab>,
}

/// A lookup hit: the engine's `(common, frontier)` semantics for the matched snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// The matched snapshot.
    pub key: Key,
    /// The matched snapshot's kind.
    pub kind: SnapshotKind,
    /// Tokens shared with the query.
    pub common: usize,
    /// The matched snapshot's frontier.
    pub frontier: usize,
}

/// One shared host page: its slab, how many snapshots reference it, and the device identity it
/// was copied from while that device page is alive.
struct PageEntry {
    slab: Slab,
    refs: u32,
    device: Option<DevicePageId>,
}

/// The store of host snapshots. Invariants: a page slab is held exactly while its reference
/// count is positive; the device map holds an entry only for a page that is both alive on the
/// device and present in a host slab, and is filled at commit so an in-flight copy is never
/// shared; `bytes_used()` equals the pool's; eviction order equals `Retention::evict_one` order
/// over unpinned snapshots.
pub struct Snapshots {
    pool: SlabPool,
    retention: Retention<Key>,
    snapshots: HashMap<Key, HostSnapshot>,
    /// Shared page table; a slot is `None` while its index is on `free_pages`.
    pages: Vec<Option<PageEntry>>,
    free_pages: Vec<u32>,
    /// Device identity to the one host page that holds its bytes, while it is shareable.
    device_map: HashMap<DevicePageId, PageRef>,
    next_key: Key,
}

impl Snapshots {
    /// A store over `pool` with no snapshots. Invariant: every slab this store hands out comes
    /// from `pool`, so `bytes_used()` never exceeds the pool's quota.
    pub fn new(pool: SlabPool) -> Self {
        Self {
            pool,
            retention: Retention::new(usize::MAX),
            snapshots: HashMap::new(),
            pages: Vec::new(),
            free_pages: Vec::new(),
            device_map: HashMap::new(),
            next_key: 0,
        }
    }

    /// Allocate slabs for `meta` and every page of `device_pages` not already shared. A class
    /// the layout disables (size zero) is never planned, so `scores` and `draft` come back
    /// `None`. On `PoolExhausted` nothing is held. A page whose identity is not yet mapped is
    /// copied into a fresh slab even if another in-flight plan is copying the same identity;
    /// [`commit_store`](Self::commit_store) releases the later duplicate in favour of a
    /// reference. Duplicate tokens for the same kind replace the older snapshot on commit (the
    /// newer frontier wins, as in the engine's radix).
    pub fn plan_store(
        &mut self,
        meta: SnapshotMeta,
        device_pages: &[Vec<DevicePageId>; COMPRESSORS],
    ) -> Result<StorePlan, PoolExhausted> {
        let tail = self.pool.take(Class::Tail)?;
        let scores = match self.take_optional(Class::Scores) {
            Ok(slab) => slab,
            Err(exhausted) => {
                self.pool.give_back(tail);
                return Err(exhausted);
            }
        };
        let draft = if meta.has_draft {
            match self.take_optional(Class::Draft) {
                Ok(slab) => slab,
                Err(exhausted) => {
                    self.release_slabs(tail, None, scores);
                    return Err(exhausted);
                }
            }
        } else {
            None
        };
        let mut pages: [Vec<PageRef>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        let mut copies = Vec::new();
        for (compressor, list) in device_pages.iter().enumerate() {
            for (index, &id) in list.iter().enumerate() {
                match self.device_map.get(&id).copied() {
                    Some(page) => {
                        self.add_page_ref(page);
                        pages[compressor].push(page);
                    }
                    None => match self.pool.take(Class::Page) {
                        Ok(slab) => {
                            let page = self.insert_page(slab);
                            pages[compressor].push(page);
                            copies.push((compressor as u8, index as u32, id, slab));
                        }
                        Err(exhausted) => {
                            self.release_pages(&pages);
                            self.release_slabs(tail, draft, scores);
                            return Err(exhausted);
                        }
                    },
                }
            }
        }
        let key = self.next_key;
        self.next_key += 1;
        Ok(StorePlan {
            key,
            meta,
            copies,
            pages,
            tail,
            draft,
            scores,
        })
    }

    /// After every copy of `plan` completed: fill the device map from the plan's copies, release
    /// any page whose identity another plan already mapped, and make the snapshot
    /// lookup-visible. Invariant: the returned key is the plan's key and the snapshot is
    /// resident until removed or evicted.
    pub fn commit_store(&mut self, plan: StorePlan, now_ns: u64) -> Key {
        let StorePlan {
            key,
            meta,
            copies,
            mut pages,
            tail,
            draft,
            scores,
        } = plan;
        self.map_copied_pages(&copies, &mut pages);
        let kind = meta.kind;
        let replaced = self
            .retention
            .bank_mut(kind)
            .lookup(&meta.tokens)
            .and_then(|(position, &old)| (position == meta.tokens.len()).then_some(old));
        if let Some(old) = replaced {
            if let Some(snapshot) = self.snapshots.remove(&old) {
                self.release_snapshot(snapshot);
            }
        }
        self.retention.bank_mut(kind).insert(&meta.tokens, key);
        self.snapshots.insert(
            key,
            HostSnapshot {
                key,
                meta,
                pages,
                tail,
                draft,
                scores,
                last_access_ns: now_ns,
                pins: 0,
            },
        );
        key
    }

    /// A store that failed: release its slabs and drop its page references. Invariant: the plan
    /// is consumed and none of its slabs or references remain held.
    pub fn abort_store(&mut self, plan: StorePlan) {
        self.release_pages(&plan.pages);
        self.release_slabs(plan.tail, plan.draft, plan.scores);
    }

    /// The engine's reuse rule over host snapshots; refreshes the hit's access clock. Invariant:
    /// a hit is exactly a `Retention::lookup_reusable` hit over the resident snapshots.
    pub fn lookup(&mut self, tokens: &[u32], now_ns: u64) -> Option<Hit> {
        let (common, frontier, key) = {
            let (common, frontier, &key) = self.retention.lookup_reusable(tokens)?;
            (common, frontier, key)
        };
        let snapshot = self.snapshots.get_mut(&key)?;
        snapshot.last_access_ns = now_ns;
        Some(Hit {
            key,
            kind: snapshot.meta.kind,
            common,
            frontier,
        })
    }

    /// The resident snapshot `key`, if any. Invariant: the returned snapshot's slabs and page
    /// references are held by this store.
    pub fn get(&self, key: Key) -> Option<&HostSnapshot> {
        self.snapshots.get(&key)
    }

    /// Add one pin to `key`. Invariant: a pinned snapshot is never evicted; an unknown key is a
    /// no-op (the engine may pin a snapshot that was already evicted).
    pub fn pin(&mut self, key: Key) {
        if let Some(snapshot) = self.snapshots.get_mut(&key) {
            snapshot.pins += 1;
        }
    }

    /// Drop one pin from `key`. Invariant: `key` is resident and pinned; an unknown key is a
    /// no-op and an unpinned snapshot is a logic error (debug-asserted).
    pub fn unpin(&mut self, key: Key) {
        let Some(snapshot) = self.snapshots.get_mut(&key) else {
            return;
        };
        debug_assert!(snapshot.pins > 0, "unpin of an unpinned snapshot");
        snapshot.pins -= 1;
    }

    /// A device page was freed: its identity can no longer be shared from the device. Invariant:
    /// the host page that held its bytes stays resident; only the shareable identity is dropped.
    /// The engine frees a device page only after its copy has completed or been aborted, so a
    /// pending plan's identity is never freed.
    pub fn device_page_freed(&mut self, id: DevicePageId) {
        if let Some(page) = self.device_map.remove(&id) {
            if let Some(entry) = self.live_page_mut(page) {
                entry.device = None;
            }
        }
    }

    /// Evict the least recently used unpinned snapshot; returns its key and the bytes freed, or
    /// `None` when only pinned snapshots remain. Invariant: a pinned snapshot is never evicted.
    pub fn evict_one(&mut self) -> Option<(Key, u64)> {
        loop {
            let next = {
                let snapshots = &self.snapshots;
                self.retention.evict_one_where(&|key: &Key| {
                    snapshots.get(key).is_some_and(|snapshot| snapshot.pins > 0)
                })
            };
            let (_kind, key) = next?;
            let before = self.pool.bytes_used();
            if let Some(snapshot) = self.snapshots.remove(&key) {
                self.release_snapshot(snapshot);
                return Some((key, before - self.pool.bytes_used()));
            }
        }
    }

    /// Evict in the engine's order until `bytes_used() <= quota`, skipping pinned snapshots;
    /// returns the evicted keys and the bytes freed. Invariant: a pinned snapshot is never
    /// evicted; stops early if only pinned snapshots remain.
    pub fn evict_to(&mut self, quota: u64) -> (Vec<Key>, u64) {
        let mut evicted = Vec::new();
        let mut freed = 0;
        while self.pool.bytes_used() > quota {
            let Some((key, bytes)) = self.evict_one() else {
                break;
            };
            evicted.push(key);
            freed += bytes;
        }
        (evicted, freed)
    }

    /// Remove `key` and release everything it holds; `false` when it was not resident.
    /// Invariant: a removed key is absent from the radix and its slabs are back in the pool.
    pub fn remove(&mut self, key: Key) -> bool {
        let Some(snapshot) = self.snapshots.remove(&key) else {
            return false;
        };
        self.retention
            .bank_mut(snapshot.meta.kind)
            .remove_exact(&snapshot.meta.tokens);
        self.release_snapshot(snapshot);
        true
    }

    /// The number of resident snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether no snapshot is resident.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes currently held, exactly the pool's `bytes_used()`.
    pub fn bytes_used(&self) -> u64 {
        self.pool.bytes_used()
    }

    /// The pinned host range of a held slab; its length is the class size. Invariant: `slab` is
    /// held by this store (the pool debug-asserts it), so the range names live pinned memory.
    pub fn location(&self, slab: Slab) -> HostRange {
        self.pool.location(slab)
    }

    /// The pinned host range of a live shared page, or `None` when its index is free. Invariant:
    /// the range names the slab that holds the page's bytes.
    pub fn page_location(&self, page: PageRef) -> Option<HostRange> {
        self.pages
            .get(page.0 as usize)
            .and_then(Option::as_ref)
            .map(|entry| self.pool.location(entry.slab))
    }

    /// Record that device page `id` now holds the bytes of host page `page` (after a restore), so
    /// a later store of `id` shares `page` instead of copying it. Invariant: `page` is live; the
    /// map holds at most one identity per live page, so the page's previous identity is dropped.
    pub fn register_device_page(&mut self, id: DevicePageId, page: PageRef) {
        let Some(previous) = self.live_page_mut(page).map(|entry| entry.device) else {
            debug_assert!(false, "registered a page that is not live");
            return;
        };
        if let Some(previous) = previous {
            if previous != id && self.device_map.get(&previous) == Some(&page) {
                self.device_map.remove(&previous);
            }
        }
        if let Some(existing) = self.device_map.insert(id, page) {
            if existing != page {
                if let Some(entry) = self.live_page_mut(existing) {
                    entry.device = None;
                }
            }
        }
        if let Some(entry) = self.live_page_mut(page) {
            entry.device = Some(id);
        }
    }

    /// The radix, for the suites that assert a host hit equals a `Retention` hit.
    pub fn retention(&self) -> &Retention<Key> {
        &self.retention
    }

    /// The reference count of a shared page; zero when the index is free. Test support.
    #[doc(hidden)]
    pub fn page_ref_count(&self, page: PageRef) -> u32 {
        self.pages
            .get(page.0 as usize)
            .and_then(|entry| entry.as_ref())
            .map_or(0, |entry| entry.refs)
    }

    /// The device identity a shared page was copied from, if it is still shareable. Test support.
    #[doc(hidden)]
    pub fn page_device(&self, page: PageRef) -> Option<DevicePageId> {
        self.pages
            .get(page.0 as usize)
            .and_then(|entry| entry.as_ref())
            .and_then(|entry| entry.device)
    }

    /// The reference counts of every live shared page. Test support.
    #[doc(hidden)]
    pub fn page_ref_counts(&self) -> Vec<u32> {
        self.pages
            .iter()
            .flatten()
            .map(|entry| entry.refs)
            .collect()
    }

    /// The number of live shared pages. Test support.
    #[doc(hidden)]
    pub fn live_pages(&self) -> usize {
        self.pages.iter().filter(|entry| entry.is_some()).count()
    }

    /// The sum of every live page's reference count. Test support.
    #[doc(hidden)]
    pub fn page_ref_total(&self) -> u64 {
        self.pages
            .iter()
            .flatten()
            .map(|entry| u64::from(entry.refs))
            .sum()
    }

    /// Take a slab of `class` unless the layout disables it (size zero), in which case `None`.
    fn take_optional(&mut self, class: Class) -> Result<Option<Slab>, PoolExhausted> {
        if self.pool.layout().size(class) == 0 {
            return Ok(None);
        }
        self.pool.take(class).map(Some)
    }

    /// The live entry for `page`, if its index is live.
    fn live_page_mut(&mut self, page: PageRef) -> Option<&mut PageEntry> {
        self.pages.get_mut(page.0 as usize).and_then(Option::as_mut)
    }

    /// Add one reference to a live page; a missing entry is a broken internal invariant.
    fn add_page_ref(&mut self, page: PageRef) {
        match self.live_page_mut(page) {
            Some(entry) => entry.refs += 1,
            None => debug_assert!(false, "reference to a page that is not live"),
        }
    }

    /// Fill the device map from a plan's copies at commit. A page whose identity another plan
    /// already mapped is released and its reference replaced by the mapped page.
    fn map_copied_pages(
        &mut self,
        copies: &[(u8, u32, DevicePageId, Slab)],
        pages: &mut [Vec<PageRef>; COMPRESSORS],
    ) {
        for &(compressor, index, id, _slab) in copies {
            let page = pages[compressor as usize][index as usize];
            match self.device_map.get(&id).copied() {
                Some(existing) if existing != page => {
                    self.release_page(page);
                    self.add_page_ref(existing);
                    pages[compressor as usize][index as usize] = existing;
                }
                Some(_) => {}
                None => {
                    self.device_map.insert(id, page);
                    match self.live_page_mut(page) {
                        Some(entry) => entry.device = Some(id),
                        None => debug_assert!(false, "copied page is not live"),
                    }
                }
            }
        }
    }

    /// Claim a free page index (or grow the table) and return it with one reference. The device
    /// identity is recorded by `commit_store`, not here.
    fn insert_page(&mut self, slab: Slab) -> PageRef {
        let index = match self.free_pages.pop() {
            Some(index) => index,
            None => {
                self.pages.push(None);
                (self.pages.len() - 1) as u32
            }
        };
        self.pages[index as usize] = Some(PageEntry {
            slab,
            refs: 1,
            device: None,
        });
        PageRef(index)
    }

    /// Drop one reference to `page`; release its slab and device-map entry at zero.
    fn release_page(&mut self, page: PageRef) {
        let index = page.0 as usize;
        let Some(entry) = self.pages[index].as_mut() else {
            debug_assert!(false, "released a page that is not live");
            return;
        };
        entry.refs -= 1;
        if entry.refs > 0 {
            return;
        }
        let slab = entry.slab;
        if let Some(id) = entry.device {
            if self.device_map.get(&id) == Some(&page) {
                self.device_map.remove(&id);
            }
        }
        self.pages[index] = None;
        self.free_pages.push(page.0);
        self.pool.give_back(slab);
    }

    /// Drop one reference to every page in `pages`.
    fn release_pages(&mut self, pages: &[Vec<PageRef>; COMPRESSORS]) {
        for list in pages {
            for &page in list {
                self.release_page(page);
            }
        }
    }

    /// Release a snapshot's parts and every page reference it holds.
    fn release_snapshot(&mut self, snapshot: HostSnapshot) {
        self.release_pages(&snapshot.pages);
        self.release_slabs(snapshot.tail, snapshot.draft, snapshot.scores);
    }

    /// Return the non-page slabs of a plan or snapshot to the pool.
    fn release_slabs(&mut self, tail: Slab, draft: Option<Slab>, scores: Option<Slab>) {
        if let Some(scores) = scores {
            self.pool.give_back(scores);
        }
        if let Some(draft) = draft {
            self.pool.give_back(draft);
        }
        self.pool.give_back(tail);
    }
}

/// Test support for the snapshot suites: builders for device pages, metadata and stores. Hidden
/// from the public docs; the unit tests and both integration suites share it.
#[doc(hidden)]
pub mod testing {
    use super::*;
    use crate::pool::testing::{layout, pool, FakePinned, CHUNK};
    use crate::pool::{Layout, SlabPool};

    /// A `Snapshots` over a fresh pool of `quota` bytes with the pool suites' layout.
    pub fn snapshots(quota: u64) -> Snapshots {
        Snapshots::new(pool(quota).0)
    }

    /// A `Snapshots` over a fresh pool of `quota` bytes with `layout`, using a chunk large
    /// enough for the layout's largest class.
    pub fn snapshots_with_layout(quota: u64, layout: Layout) -> Snapshots {
        let chunk = layout
            .page
            .max(layout.tail)
            .max(layout.draft)
            .max(layout.scores)
            .max(1);
        let mut mem = FakePinned::new(usize::MAX);
        Snapshots::new(SlabPool::new(quota, chunk, layout, &mut mem).expect("snapshot test pool"))
    }

    /// A device page identity in compressor 0, generation 0.
    pub fn id(page: u32) -> DevicePageId {
        DevicePageId {
            compressor: 0,
            page,
            generation: 0,
        }
    }

    /// A snapshot meta of `kind` over `tokens`.
    pub fn meta(kind: SnapshotKind, tokens: &[u32], has_draft: bool) -> SnapshotMeta {
        SnapshotMeta {
            kind,
            tokens: tokens.to_vec(),
            end: tokens.len() as u32,
            has_draft,
        }
    }

    /// A page list with `ids` in compressor 0.
    pub fn pages(ids: &[DevicePageId]) -> [Vec<DevicePageId>; COMPRESSORS] {
        let mut pages: [Vec<DevicePageId>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        pages[0].extend_from_slice(ids);
        pages
    }

    /// Build `count` resident turn snapshots of `tokens` tokens each sharing a `prefix`-token
    /// prefix. Pages are modelled at one per 100 tokens so the shared prefix is shared exactly.
    pub fn resident_snapshots(count: usize, tokens: usize, prefix: usize) -> (Snapshots, Vec<Key>) {
        let layout = layout();
        let unique_pages = (prefix / 100) + count * ((tokens - prefix) / 100);
        let bytes = unique_pages as u64 * layout.page as u64
            + count as u64 * (layout.tail + layout.scores) as u64;
        let mut store = snapshots(bytes + bytes / 2 + CHUNK as u64);
        let prefix_tokens: Vec<u32> = (0..prefix as u32).collect();
        let prefix_pages: Vec<DevicePageId> = (0..(prefix / 100) as u32)
            .map(|page| DevicePageId {
                compressor: 0,
                page,
                generation: 0,
            })
            .collect();
        let mut keys = Vec::with_capacity(count);
        for index in 0..count {
            let mut sequence = prefix_tokens.clone();
            sequence
                .extend((0..(tokens - prefix) as u32).map(|token| (index as u32) * 8192 + token));
            let mut pages: [Vec<DevicePageId>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
            pages[0].extend_from_slice(&prefix_pages);
            pages[0].extend(
                (0..((tokens - prefix) / 100) as u32).map(|page| DevicePageId {
                    compressor: 0,
                    page: 1_000_000 + (index as u32) * 1000 + page,
                    generation: 0,
                }),
            );
            let meta = SnapshotMeta {
                kind: SnapshotKind::Turn,
                tokens: sequence,
                end: tokens as u32,
                has_draft: false,
            };
            let plan = store.plan_store(meta, &pages).expect("resident plan");
            keys.push(store.commit_store(plan, 0));
        }
        (store, keys)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{id, meta, pages, snapshots, snapshots_with_layout};
    use super::*;
    use crate::pool::testing::CHUNK;
    use crate::pool::Layout;

    #[test]
    fn plan_allocates_one_slab_per_part_and_page() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(
                meta(SnapshotKind::Turn, &[1, 2, 3], true),
                &pages(&[id(1), id(2)]),
            )
            .expect("plan");
        assert_eq!(plan.copies.len(), 2);
        assert_eq!(plan.pages[0].len(), 2);
        assert!(plan.draft.is_some());
        assert!(plan.scores.is_some());
        let layout = crate::pool::testing::layout();
        let expected = (layout.tail + layout.scores + layout.draft + 2 * layout.page) as u64;
        assert_eq!(store.bytes_used(), expected);
        store.abort_store(plan);
        assert_eq!(store.bytes_used(), 0);
        assert_eq!(store.live_pages(), 0);
    }

    #[test]
    fn plan_exhaustion_releases_everything_it_allocated() {
        // Two chunks: the tail and scores classes take one each, leaving none for a page.
        let mut store = snapshots(2 * CHUNK as u64);
        let error = store
            .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
            .expect_err("page class exhausted");
        assert_eq!(error.class, Class::Page);
        assert_eq!(store.bytes_used(), 0);
        assert_eq!(store.live_pages(), 0);
    }

    #[test]
    fn shared_pages_are_referenced_not_recopied() {
        let mut store = snapshots(1 << 30);
        let first = store
            .plan_store(
                meta(SnapshotKind::Turn, &[1, 2], false),
                &pages(&[id(1), id(2)]),
            )
            .expect("plan");
        assert_eq!(first.copies.len(), 2);
        let first_key = store.commit_store(first, 0);
        let second = store
            .plan_store(
                meta(SnapshotKind::Turn, &[1, 2, 3], false),
                &pages(&[id(1), id(2), id(3)]),
            )
            .expect("plan");
        assert_eq!(second.copies.len(), 1);
        assert_eq!(second.copies[0].2, id(3));
        let second_key = store.commit_store(second, 1);
        assert_eq!(store.live_pages(), 3);
        assert_eq!(store.page_ref_total(), 5);
        assert_eq!(
            store.page_ref_count(store.get(first_key).expect("first").pages[0][0]),
            2
        );
        assert!(store.remove(first_key));
        assert_eq!(store.live_pages(), 3);
        assert!(store.remove(second_key));
        assert_eq!(store.live_pages(), 0);
        assert_eq!(store.bytes_used(), 0);
    }

    #[test]
    fn freeing_a_device_page_forgets_the_share_but_keeps_the_host_page() {
        let mut store = snapshots(1 << 30);
        let first = store
            .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
            .expect("plan");
        let first_key = store.commit_store(first, 0);
        let page = store.get(first_key).expect("first").pages[0][0];
        store.device_page_freed(id(1));
        assert_eq!(store.page_device(page), None);
        assert_eq!(store.page_ref_count(page), 1);
        let second = store
            .plan_store(
                meta(SnapshotKind::Turn, &[1, 2], false),
                &pages(&[id(1), id(2)]),
            )
            .expect("plan");
        assert_eq!(second.copies.len(), 2);
        // The old host page persists; the freed identity is copied into a new one.
        assert_eq!(store.live_pages(), 3);
    }

    #[test]
    fn commit_replaces_a_snapshot_with_the_same_tokens_and_kind() {
        let mut store = snapshots(1 << 30);
        let first = store
            .plan_store(meta(SnapshotKind::Turn, &[7, 8], false), &pages(&[id(1)]))
            .expect("plan");
        let first_key = store.commit_store(first, 0);
        let second = store
            .plan_store(meta(SnapshotKind::Turn, &[7, 8], false), &pages(&[id(2)]))
            .expect("plan");
        let second_key = store.commit_store(second, 1);
        assert_eq!(store.len(), 1);
        assert!(store.get(first_key).is_none());
        assert!(store.get(second_key).is_some());
        assert_eq!(store.live_pages(), 1);
        let layout = crate::pool::testing::layout();
        assert_eq!(
            store.bytes_used(),
            (layout.tail + layout.scores + layout.page) as u64
        );
    }

    #[test]
    fn eviction_skips_pinned_and_follows_bank_order() {
        let mut store = snapshots(1 << 30);
        let prompt = store
            .plan_store(
                SnapshotMeta {
                    kind: SnapshotKind::Prompt,
                    ..meta(SnapshotKind::Turn, &[1], false)
                },
                &pages(&[id(1)]),
            )
            .expect("plan");
        let prompt_key = store.commit_store(prompt, 0);
        let turn = store
            .plan_store(meta(SnapshotKind::Turn, &[2], false), &pages(&[id(2)]))
            .expect("plan");
        let turn_key = store.commit_store(turn, 1);
        store.pin(turn_key);
        let (evicted, freed) = store.evict_to(0);
        assert_eq!(evicted, vec![prompt_key]);
        assert!(freed > 0);
        assert_eq!(store.len(), 1);
        assert!(store.get(turn_key).is_some());
        store.unpin(turn_key);
        let (evicted, _) = store.evict_to(0);
        assert_eq!(evicted, vec![turn_key]);
        assert!(store.is_empty());
        assert_eq!(store.bytes_used(), 0);
    }

    #[test]
    fn lookup_matches_the_retention_rule_and_refreshes_access() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(
                meta(SnapshotKind::Turn, &[1, 2, 3, 4], false),
                &pages(&[id(1)]),
            )
            .expect("plan");
        let key = store.commit_store(plan, 0);
        let hit = store.lookup(&[1, 2, 3, 4, 5], 42).expect("hit");
        assert_eq!(hit.key, key);
        assert_eq!(hit.kind, SnapshotKind::Turn);
        assert_eq!((hit.common, hit.frontier), (4, 4));
        assert_eq!(store.get(key).expect("snapshot").last_access_ns, 42);
        assert!(store.lookup(&[9], 43).is_none());
    }

    #[test]
    fn page_indices_are_reused_after_release() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
            .expect("plan");
        let key = store.commit_store(plan, 0);
        let page = store.get(key).expect("snapshot").pages[0][0];
        assert!(store.remove(key));
        let plan = store
            .plan_store(meta(SnapshotKind::Turn, &[2], false), &pages(&[id(2)]))
            .expect("plan");
        let key = store.commit_store(plan, 1);
        assert_eq!(store.get(key).expect("snapshot").pages[0][0], page);
    }

    #[test]
    fn commit_dedups_a_page_two_plans_copied() {
        let mut store = snapshots(1 << 30);
        let first = store
            .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
            .expect("plan");
        let second = store
            .plan_store(meta(SnapshotKind::Turn, &[2], false), &pages(&[id(1)]))
            .expect("plan");
        // Neither plan shares: the identity is not mapped until commit.
        assert_eq!(first.copies.len(), 1);
        assert_eq!(second.copies.len(), 1);
        assert_eq!(store.live_pages(), 2);
        let first_key = store.commit_store(first, 0);
        let second_key = store.commit_store(second, 1);
        // The second commit released its duplicate and references the first's page.
        assert_eq!(store.live_pages(), 1);
        assert_eq!(store.page_ref_total(), 2);
        let page = store.get(first_key).expect("first").pages[0][0];
        assert_eq!(store.get(second_key).expect("second").pages[0][0], page);
        assert_eq!(store.page_ref_count(page), 2);
    }

    #[test]
    fn register_device_page_aliases_an_identity_to_the_newer_page() {
        let mut store = snapshots(1 << 30);
        let first = store
            .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
            .expect("plan");
        let first_key = store.commit_store(first, 0);
        let second = store
            .plan_store(meta(SnapshotKind::Turn, &[2], false), &pages(&[id(2)]))
            .expect("plan");
        let second_key = store.commit_store(second, 1);
        let page_a = store.get(first_key).expect("first").pages[0][0];
        let page_b = store.get(second_key).expect("second").pages[0][0];
        assert_eq!(store.page_device(page_a), Some(id(1)));
        assert_eq!(store.page_device(page_b), Some(id(2)));

        // Restoring page B's bytes onto identity id(1): the newer mapping wins and the older
        // page loses its identity, while both reference counts stay exact.
        store.register_device_page(id(1), page_b);
        assert_eq!(store.page_device(page_b), Some(id(1)));
        assert_eq!(store.page_device(page_a), None);
        assert_eq!(store.page_ref_count(page_a), 1);
        assert_eq!(store.page_ref_count(page_b), 1);

        // A store of id(1) now shares page B, not page A.
        let third = store
            .plan_store(meta(SnapshotKind::Turn, &[3], false), &pages(&[id(1)]))
            .expect("plan");
        assert!(third.copies.is_empty());
        assert_eq!(third.pages[0][0], page_b);
    }

    #[test]
    fn register_device_page_maps_an_unmapped_identity() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
            .expect("plan");
        let key = store.commit_store(plan, 0);
        let page = store.get(key).expect("snapshot").pages[0][0];
        store.device_page_freed(id(1));
        assert_eq!(store.page_device(page), None);

        store.register_device_page(id(7), page);
        assert_eq!(store.page_device(page), Some(id(7)));
        assert_eq!(store.page_ref_count(page), 1);

        let second = store
            .plan_store(meta(SnapshotKind::Turn, &[2], false), &pages(&[id(7)]))
            .expect("plan");
        assert!(second.copies.is_empty());
        assert_eq!(second.pages[0][0], page);
    }

    #[test]
    fn zero_scores_layout_stores_without_a_scores_slab() {
        let layout = Layout::engine(0);
        let mut store = snapshots_with_layout(8 * layout.tail as u64, layout);
        let plan = store
            .plan_store(meta(SnapshotKind::Turn, &[1, 2], false), &pages(&[id(1)]))
            .expect("plan");
        assert!(plan.scores.is_none());
        let key = store.commit_store(plan, 0);
        assert!(store.get(key).expect("snapshot").scores.is_none());
        assert_eq!(store.bytes_used(), (layout.tail + layout.page) as u64);
    }

    #[test]
    fn pin_and_unpin_of_an_unknown_key_are_no_ops() {
        let mut store = snapshots(1 << 30);
        store.pin(7);
        store.unpin(7);
        assert!(store.is_empty());
        assert_eq!(store.bytes_used(), 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "unpin of an unpinned snapshot")]
    fn unpin_of_unpinned_snapshot_panics_in_debug() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
            .expect("plan");
        let key = store.commit_store(plan, 0);
        store.unpin(key);
    }
}
