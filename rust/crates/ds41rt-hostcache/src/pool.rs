//! Pinned slab pool (packet HC-1): all pinned host memory is allocated at construction to the
//! configured quota in fixed chunks and never grows. Each chunk is carved for one size class
//! (page, tail, draft, scores); each class keeps a free list; a class claims a free chunk when
//! its list is empty and releases a chunk when every slab in it is free. Allocation and release
//! are O(1); bytes in use is exact and equals slabs held × class size.
use serde::Serialize;
use thiserror::Error;

/// The four slab sizes a snapshot needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Class {
    /// One source page.
    Page,
    /// One backbone tail arena slot.
    Tail,
    /// The dSpark draft rings.
    Draft,
    /// One scores row.
    Scores,
}

impl Class {
    /// Every class, in the pool's per-class array order.
    pub const ALL: [Class; 4] = [Class::Page, Class::Tail, Class::Draft, Class::Scores];

    /// Index into the pool's per-class arrays.
    const fn index(self) -> usize {
        match self {
            Class::Page => 0,
            Class::Tail => 1,
            Class::Draft => 2,
            Class::Scores => 3,
        }
    }
}

/// Slab sizes in bytes. The engine layout uses the crate constants for pages, tails and drafts;
/// the scores row size comes from the engine at boot (verified in the daemon binding).
///
/// A size of zero disables that class: the pool allocates nothing for it, [`SlabPool::take`]
/// always reports [`PoolExhausted`] with `free_bytes` 0, [`SlabPool::free_bytes`] reports 0, and
/// its occupancy stays zero. The engine uses `Layout::engine(0)` because scores are host memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Layout {
    /// Bytes of a page slab.
    pub page: usize,
    /// Bytes of a tail slab.
    pub tail: usize,
    /// Bytes of a draft slab.
    pub draft: usize,
    /// Bytes of a scores slab.
    pub scores: usize,
}

impl Layout {
    /// The engine's layout: crate constants for page, tail and draft; `scores` from the engine.
    pub fn engine(scores: usize) -> Self {
        Self {
            page: crate::PAGE_BYTES,
            tail: crate::TAIL_BYTES,
            draft: crate::DRAFT_BYTES,
            scores,
        }
    }
    /// Bytes of one slab of `class`.
    pub fn size(&self, class: Class) -> usize {
        match class {
            Class::Page => self.page,
            Class::Tail => self.tail,
            Class::Draft => self.draft,
            Class::Scores => self.scores,
        }
    }
}

/// One pinned chunk as the memory provider knows it: an id the copy engine maps to an address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct HostChunk {
    /// Provider-assigned chunk id.
    pub id: u32,
    /// Bytes the provider allocated for this chunk.
    pub bytes: usize,
}

/// Where pinned memory comes from: the CUDA engine registers it for DMA; the stub fakes it.
pub trait PinnedMemory {
    /// Allocate one pinned chunk of `bytes`; the provider owns it until `release_chunk`.
    fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<HostChunk>;
    /// Release a chunk previously returned by `allocate_chunk`.
    fn release_chunk(&mut self, chunk: HostChunk) -> anyhow::Result<()>;
}

/// A slab handle: which chunk, which index within it. Copy engines address it through
/// [`SlabPool::location`]. Handles are plain data; the pool is the authority on validity.
///
/// `chunk` is the pool's internal chunk index; [`SlabPool::location`] maps it to the provider's
/// chunk id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Slab {
    /// Size class this slab was carved for.
    pub class: Class,
    /// Pool chunk index.
    pub chunk: u32,
    /// Slab index within the chunk.
    pub index: u32,
}

/// A byte range inside a pinned chunk, the unit every copy is expressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct HostRange {
    /// Provider chunk id.
    pub chunk: u32,
    /// Byte offset within the chunk.
    pub offset: usize,
    /// Byte length of the range.
    pub bytes: usize,
}

/// No free slab or free chunk can satisfy a `take` of `class`. A disabled (zero-size) class
/// always reports exhaustion with `needed_bytes` and `free_bytes` both 0.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("pinned pool exhausted for {class:?}: {free_bytes} bytes free, {needed_bytes} needed")]
pub struct PoolExhausted {
    /// The class that could not be satisfied.
    pub class: Class,
    /// Bytes the failed `take` needed.
    pub needed_bytes: usize,
    /// Bytes the pool could still have handed to `class`.
    pub free_bytes: u64,
}

/// Per-class occupancy for the metrics export.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ClassOccupancy {
    /// Slabs currently held.
    pub slabs_in_use: u64,
    /// Slabs free in chunks carved for the class.
    pub slabs_free: u64,
    /// Chunks currently carved for the class.
    pub chunks: u32,
}

/// Sentinel for "no chunk" in the intrusive chunk lists.
const NONE: u32 = u32::MAX;

/// One pinned chunk and its carve state. A chunk is either free (listed in
/// [`SlabPool::free_chunks`]) or carved for exactly one class. A carved chunk with free slabs
/// sits in that class's chunk list and owns a free-slab stack over its `total` slabs; the
/// remainder of `chunk_bytes` is unused.
struct Chunk {
    host: HostChunk,
    class: Option<Class>,
    total: u32,
    free: u32,
    free_head: u32,
    slab_next: Vec<u32>,
    slab_free: Vec<bool>,
    next: u32,
    prev: u32,
}

impl Chunk {
    fn free(host: HostChunk) -> Self {
        Self {
            host,
            class: None,
            total: 0,
            free: 0,
            free_head: NONE,
            slab_next: Vec::new(),
            slab_free: Vec::new(),
            next: NONE,
            prev: NONE,
        }
    }
}

/// The pool. Invariants: `bytes_used() <= quota()`; a `Slab` handed out by `take` is valid until
/// its `give_back`; `give_back` of a slab not held is a logic error (debug-asserted); chunks
/// carved for a class hold `chunk_bytes / size` slabs and the remainder is unused.
pub struct SlabPool {
    quota_bytes: u64,
    chunk_bytes: usize,
    layout: Layout,
    chunks: Vec<Chunk>,
    /// Indices of chunks not carved for any class.
    free_chunks: Vec<u32>,
    /// Head of each class's list of carved chunks that still have free slabs.
    class_head: [u32; Class::ALL.len()],
    slabs_in_use: [u64; Class::ALL.len()],
    slabs_free: [u64; Class::ALL.len()],
    chunks_carved: [u32; Class::ALL.len()],
    bytes_used: u64,
}

impl SlabPool {
    /// Allocate `quota_bytes / chunk_bytes` chunks up front through `memory`. Fails if the
    /// provider cannot supply them; a partial allocation is released before returning.
    pub fn new(
        quota_bytes: u64,
        chunk_bytes: usize,
        layout: Layout,
        memory: &mut dyn PinnedMemory,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(chunk_bytes > 0, "chunk_bytes must be positive");
        for class in Class::ALL {
            let size = layout.size(class);
            anyhow::ensure!(
                size <= chunk_bytes,
                "{class:?} slab size {size} must be <= {chunk_bytes}"
            );
        }
        let count = (quota_bytes / chunk_bytes as u64) as usize;
        let mut chunks = Vec::with_capacity(count);
        for _ in 0..count {
            match memory.allocate_chunk(chunk_bytes) {
                Ok(host) => chunks.push(Chunk::free(host)),
                Err(err) => {
                    for chunk in &chunks {
                        let _ = memory.release_chunk(chunk.host);
                    }
                    return Err(err);
                }
            }
        }
        let free_chunks = (0..chunks.len() as u32).collect();
        Ok(Self {
            quota_bytes,
            chunk_bytes,
            layout,
            chunks,
            free_chunks,
            class_head: [NONE; Class::ALL.len()],
            slabs_in_use: [0; Class::ALL.len()],
            slabs_free: [0; Class::ALL.len()],
            chunks_carved: [0; Class::ALL.len()],
            bytes_used: 0,
        })
    }

    /// Hand out one slab of `class`, carving a free chunk if the class has no free slab. O(1).
    /// A disabled (zero-size) class always reports exhaustion.
    pub fn take(&mut self, class: Class) -> Result<Slab, PoolExhausted> {
        let ci = class.index();
        let size = self.layout.size(class);
        if size == 0 {
            return Err(PoolExhausted {
                class,
                needed_bytes: 0,
                free_bytes: 0,
            });
        }
        let slab = if self.class_head[ci] != NONE {
            self.pop_free(self.class_head[ci], class)
        } else if let Some(chunk) = self.free_chunks.pop() {
            self.carve(chunk, class)
        } else {
            return Err(PoolExhausted {
                class,
                needed_bytes: size,
                free_bytes: self.free_bytes(class),
            });
        };
        self.slabs_in_use[ci] += 1;
        self.bytes_used += size as u64;
        Ok(slab)
    }

    /// Return a held slab; the chunk goes back to the free list once every slab in it is free.
    /// O(1). `slab` must have come from `take` and not already been given back.
    pub fn give_back(&mut self, slab: Slab) {
        debug_assert!(self.is_held(slab), "give_back of a slab not held: {slab:?}");
        let ci = slab.class.index();
        let size = self.layout.size(slab.class);
        let chunk_idx = slab.chunk as usize;
        let index = slab.index as usize;
        let free_before = self.chunks[chunk_idx].free;
        {
            let chunk = &mut self.chunks[chunk_idx];
            chunk.slab_free[index] = true;
            chunk.slab_next[index] = chunk.free_head;
            chunk.free_head = slab.index;
            chunk.free += 1;
        }
        self.slabs_free[ci] += 1;
        self.slabs_in_use[ci] -= 1;
        self.bytes_used -= size as u64;
        let free = self.chunks[chunk_idx].free;
        let total = self.chunks[chunk_idx].total;
        if free == total {
            if free_before > 0 {
                self.unlink_class(slab.chunk, slab.class);
            }
            self.release_chunk(slab.chunk, slab.class);
        } else if free_before == 0 {
            self.link_class(slab.chunk, slab.class);
        }
    }

    /// The chunk and byte offset of a held slab; its length is the class size.
    pub fn location(&self, slab: Slab) -> HostRange {
        debug_assert!(self.is_held(slab), "location of a slab not held: {slab:?}");
        let size = self.layout.size(slab.class);
        HostRange {
            chunk: self.chunks[slab.chunk as usize].host.id,
            offset: slab.index as usize * size,
            bytes: size,
        }
    }

    /// The slab sizes the pool was built with.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Bytes currently held: exactly the sum of held slabs' class sizes.
    pub fn bytes_used(&self) -> u64 {
        self.bytes_used
    }

    /// The configured quota; `bytes_used() <= quota()` always holds.
    pub fn quota(&self) -> u64 {
        self.quota_bytes
    }

    /// Bytes a `take` of `class` could still satisfy from free slabs and free chunks. A disabled
    /// (zero-size) class reports 0.
    pub fn free_bytes(&self, class: Class) -> u64 {
        let ci = class.index();
        let size = self.layout.size(class);
        if size == 0 {
            return 0;
        }
        let size = size as u64;
        let per_chunk = (self.chunk_bytes / self.layout.size(class)) as u64;
        self.slabs_free[ci] * size + self.free_chunks.len() as u64 * per_chunk * size
    }

    /// Per-class held, free and carved-chunk counts, in [`Class::ALL`] order.
    pub fn occupancy(&self) -> [(Class, ClassOccupancy); Class::ALL.len()] {
        Class::ALL.map(|class| {
            let ci = class.index();
            (
                class,
                ClassOccupancy {
                    slabs_in_use: self.slabs_in_use[ci],
                    slabs_free: self.slabs_free[ci],
                    chunks: self.chunks_carved[ci],
                },
            )
        })
    }

    /// Release every chunk back to the provider (drop order: the engine outlives the pool).
    pub fn release_all(self, memory: &mut dyn PinnedMemory) -> anyhow::Result<()> {
        let mut first_err = None;
        for chunk in self.chunks {
            if let Err(err) = memory.release_chunk(chunk.host) {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Is `slab` currently held? O(1); the authority behind the `debug_assert!`s.
    fn is_held(&self, slab: Slab) -> bool {
        let Some(chunk) = self.chunks.get(slab.chunk as usize) else {
            return false;
        };
        chunk.class == Some(slab.class)
            && (slab.index as usize) < chunk.slab_free.len()
            && !chunk.slab_free[slab.index as usize]
    }

    /// Pop the head slab of a carved chunk's free stack. The chunk must have a free slab.
    fn pop_free(&mut self, chunk_idx: u32, class: Class) -> Slab {
        let ci = class.index();
        let index = {
            let chunk = &mut self.chunks[chunk_idx as usize];
            let index = chunk.free_head;
            chunk.free_head = chunk.slab_next[index as usize];
            chunk.slab_free[index as usize] = false;
            chunk.free -= 1;
            index
        };
        self.slabs_free[ci] -= 1;
        if self.chunks[chunk_idx as usize].free == 0 {
            self.unlink_class(chunk_idx, class);
        }
        Slab {
            class,
            chunk: chunk_idx,
            index,
        }
    }

    /// Carve a free chunk for `class` and return its first slab.
    fn carve(&mut self, chunk_idx: u32, class: Class) -> Slab {
        let ci = class.index();
        let total = (self.chunk_bytes / self.layout.size(class)) as u32;
        debug_assert!(total > 0, "chunk too small for {class:?}");
        {
            let chunk = &mut self.chunks[chunk_idx as usize];
            chunk.class = Some(class);
            chunk.total = total;
            chunk.free = total;
            chunk.free_head = 0;
            chunk.slab_next = (1..total).chain(std::iter::once(NONE)).collect();
            chunk.slab_free = vec![true; total as usize];
            chunk.next = NONE;
            chunk.prev = NONE;
        }
        self.link_class(chunk_idx, class);
        self.chunks_carved[ci] += 1;
        self.slabs_free[ci] += total as u64;
        self.pop_free(chunk_idx, class)
    }

    /// Return a fully free chunk to the free list. The caller has already unlinked it.
    fn release_chunk(&mut self, chunk_idx: u32, class: Class) {
        let ci = class.index();
        let free = self.chunks[chunk_idx as usize].free as u64;
        self.slabs_free[ci] -= free;
        self.chunks_carved[ci] -= 1;
        let chunk = &mut self.chunks[chunk_idx as usize];
        chunk.class = None;
        chunk.total = 0;
        chunk.free = 0;
        chunk.free_head = NONE;
        chunk.slab_next = Vec::new();
        chunk.slab_free = Vec::new();
        chunk.next = NONE;
        chunk.prev = NONE;
        self.free_chunks.push(chunk_idx);
    }

    /// Push a carved chunk onto its class's list of chunks with free slabs.
    fn link_class(&mut self, chunk_idx: u32, class: Class) {
        let ci = class.index();
        let head = self.class_head[ci];
        {
            let chunk = &mut self.chunks[chunk_idx as usize];
            chunk.next = head;
            chunk.prev = NONE;
        }
        if head != NONE {
            self.chunks[head as usize].prev = chunk_idx;
        }
        self.class_head[ci] = chunk_idx;
    }

    /// Remove a carved chunk from its class's list.
    fn unlink_class(&mut self, chunk_idx: u32, class: Class) {
        let ci = class.index();
        let (prev, next) = {
            let chunk = &self.chunks[chunk_idx as usize];
            (chunk.prev, chunk.next)
        };
        if prev != NONE {
            self.chunks[prev as usize].next = next;
        } else {
            self.class_head[ci] = next;
        }
        if next != NONE {
            self.chunks[next as usize].prev = prev;
        }
        let chunk = &mut self.chunks[chunk_idx as usize];
        chunk.next = NONE;
        chunk.prev = NONE;
    }
}

/// Test doubles for the pool's memory provider. Hidden from the public docs; HC-3 and HC-5
/// reuse it.
#[doc(hidden)]
pub mod testing {
    use super::{HostChunk, Layout, PinnedMemory, SlabPool};
    use std::collections::HashMap;

    /// The chunk size the pool suites use: small enough to exhaust in a few takes.
    pub const CHUNK: usize = 1 << 16;

    /// The layout the pool suites use: four distinct sizes, none a divisor of [`CHUNK`].
    pub fn layout() -> Layout {
        Layout {
            page: 4096,
            tail: 8192,
            draft: 2048,
            scores: 1024,
        }
    }

    /// A pool of `quota` bytes over [`CHUNK`] and [`layout`], with its provider. The provider
    /// never refuses a chunk, so the pool's own quota is the only limit.
    pub fn pool(quota: u64) -> (SlabPool, FakePinned) {
        let mut mem = FakePinned::new(usize::MAX);
        let pool = SlabPool::new(quota, CHUNK, layout(), &mut mem).expect("pool");
        (pool, mem)
    }

    /// A [`PinnedMemory`] that hands out sequential chunk ids, counts allocations and releases,
    /// and refuses to allocate more than `limit` chunks at once.
    pub struct FakePinned {
        limit: usize,
        next_id: u32,
        live: HashMap<u32, usize>,
        allocations: usize,
        releases: usize,
        live_bytes: usize,
    }

    impl FakePinned {
        /// A provider that refuses the `limit + 1`-th live chunk.
        pub fn new(limit: usize) -> Self {
            Self {
                limit,
                next_id: 0,
                live: HashMap::new(),
                allocations: 0,
                releases: 0,
                live_bytes: 0,
            }
        }
        /// Chunks handed out over the provider's life.
        pub fn allocations(&self) -> usize {
            self.allocations
        }
        /// Chunks released over the provider's life.
        pub fn releases(&self) -> usize {
            self.releases
        }
        /// Chunks currently held by the pool.
        pub fn live_chunks(&self) -> usize {
            self.live.len()
        }
        /// Bytes currently held by the pool.
        pub fn live_bytes(&self) -> usize {
            self.live_bytes
        }
    }

    impl PinnedMemory for FakePinned {
        fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<HostChunk> {
            anyhow::ensure!(
                self.live.len() < self.limit,
                "FakePinned: refusing chunk {} beyond limit {}",
                self.live.len(),
                self.limit
            );
            let id = self.next_id;
            self.next_id += 1;
            self.live.insert(id, bytes);
            self.allocations += 1;
            self.live_bytes += bytes;
            Ok(HostChunk { id, bytes })
        }

        fn release_chunk(&mut self, chunk: HostChunk) -> anyhow::Result<()> {
            let bytes = self
                .live
                .remove(&chunk.id)
                .ok_or_else(|| anyhow::anyhow!("FakePinned: chunk {} is not live", chunk.id))?;
            self.releases += 1;
            self.live_bytes -= bytes;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{layout, pool, FakePinned, CHUNK};
    use super::*;

    #[test]
    fn is_held_tracks_take_and_give_back() {
        let (mut pool, _mem) = pool(CHUNK as u64);
        let slab = pool.take(Class::Page).expect("take");
        assert!(pool.is_held(slab));
        pool.give_back(slab);
        assert!(!pool.is_held(slab));
    }

    #[test]
    fn carve_links_and_release_unlinks() {
        let (mut pool, _mem) = pool(CHUNK as u64);
        let slab = pool.take(Class::Tail).expect("take");
        let occ = pool.occupancy();
        assert_eq!(occ[1].1.chunks, 1);
        assert_eq!(occ[1].1.slabs_in_use, 1);
        assert_eq!(occ[1].1.slabs_free, (CHUNK / layout().tail - 1) as u64);
        pool.give_back(slab);
        let occ = pool.occupancy();
        assert_eq!(occ[1].1.chunks, 0);
        assert_eq!(occ[1].1.slabs_free, 0);
    }

    #[test]
    fn free_bytes_uses_usable_slabs_not_raw_chunk_bytes() {
        // 10000 / 4096 = 2 usable slabs; the 1808-byte remainder is never handed out.
        let mut mem = FakePinned::new(usize::MAX);
        let layout = Layout {
            page: 4096,
            tail: 4096,
            draft: 4096,
            scores: 4096,
        };
        let pool = SlabPool::new(10_000, 10_000, layout, &mut mem).expect("pool");
        assert_eq!(pool.free_bytes(Class::Page), 8192);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "give_back of a slab not held")]
    fn give_back_of_unheld_slab_panics_in_debug() {
        let (mut pool, _mem) = pool(CHUNK as u64);
        let slab = pool.take(Class::Page).expect("take");
        pool.give_back(slab);
        pool.give_back(slab);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "location of a slab not held")]
    fn location_of_unheld_slab_panics_in_debug() {
        let (mut pool, _mem) = pool(CHUNK as u64);
        let slab = pool.take(Class::Page).expect("take");
        pool.give_back(slab);
        let _ = pool.location(slab);
    }
}
