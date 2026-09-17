//! Paired index/FP4 KV pages owned by compressor request leases.
use crate::v41_memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Kv};
use std::ffi::c_void;
use std::{cell::RefCell, rc::Rc};
mod ownership;
use ownership::PagePool;
mod reservation;
use reservation::PageReservation;
pub(crate) use ownership::SourcePrefix;

/// Capacity failure for one entry in the caller's ordered append transaction.
/// Binding/ownership failures deliberately use different error types.
#[derive(Debug)]
pub(crate) struct SourcePoolExhausted {
    pub work_index: usize,
    pub needed: usize,
    pub available: usize,
}
impl std::fmt::Display for SourcePoolExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compressed KV pool exhausted at work item {}: need {} pages, {} available",
            self.work_index, self.needed, self.available)
    }
}
impl std::error::Error for SourcePoolExhausted {}

pub(super) const PAGE_ROWS: usize = 256;
const KV_VALUES: usize = V41Kv::COMPRESSED_VALUE_BYTES;
const KV_SCALES: usize = V41Kv::COMPRESSED_SCALE_BYTES;
const SOURCE_ROW_BYTES: usize = 68 + V41Kv::COMPRESSED_ROW_BYTES;
pub(crate) struct SourceCache<'a> {
    pub packed: DeviceAllocation<'a>,
    pub scales: DeviceAllocation<'a>,
    pub kv_values: DeviceAllocation<'a>,
    pub kv_scales: DeviceAllocation<'a>,
    pub capacity: usize,
    page_table: DeviceAllocation<'a>,
    lengths: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    stride: usize,
    pages: [Vec<u32>; 16],
    rows: [usize; 16],
    pool: Rc<RefCell<PagePool>>,
    writing: Rc<std::cell::Cell<u16>>,
}
pub(super) struct IndexPlan {
    additions: Vec<(usize, Vec<u32>)>,
    used: usize,
    lengths: Vec<(usize, u64)>,
    // Shared partial pages are copied before accepted rows are appended.
    replacements: Vec<(usize, usize, u32, u32)>,
    reservation: Option<PageReservation>,
}
/// Only rows below `rows` are initialized. Logical row r uses physical page
/// pages[r / 256], offset r % 256. Drain device consumers before mutating owner.
/// The borrow prevents host-side release/commit while this view is in use.
pub(crate) struct IndexCacheView<'a> {
    pub packed: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub pages: &'a [u32],
    pub rows: usize,
    /// U32 physical page IDs, capacity entries; only ceil(rows/256) are valid.
    pub device_pages: Ds41rtDeviceBuffer,
    /// U64 committed row count, published after accepted value/scale writes.
    pub device_rows: Ds41rtDeviceBuffer,
}
/// FP4 K16 serving KV shares physical pages and publication with index keys.
pub(crate) struct KvCacheView<'a> {
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub pages: &'a [u32],
    pub rows: usize,
    pub device_pages: Ds41rtDeviceBuffer,
    pub device_rows: Ds41rtDeviceBuffer,
}
impl<'a> SourceCache<'a> {
    pub fn device_bytes(pages: usize, slots: usize) -> Result<usize> {
        ensure!(
            (1..=262144).contains(&pages),
            "invalid index pool page count"
        );
        ensure!((1..=16).contains(&slots), "invalid index slot count");
        Ok(pages * PAGE_ROWS * SOURCE_ROW_BYTES + slots * (pages.min(4096) * 4 + 8))
    }
    pub fn new(library: &'a NativeLibrary, pages: usize, slots: usize) -> Result<Self> {
        Self::device_bytes(pages, slots)?;
        let lengths = DeviceAllocation::new(library, slots * 8)?;
        library.copy_h2d(lengths.buffer, &vec![0; slots * 8])?;
        Ok(Self {
            page_table: DeviceAllocation::new(library, slots * pages.min(4096) * 4)?,
            lengths,
            staging: HostAllocation::new(library, pages * 4 + slots * 8)?,
            stride: pages.min(4096),
            packed: DeviceAllocation::new(library, pages * PAGE_ROWS * 64)?,
            scales: DeviceAllocation::new(library, pages * PAGE_ROWS * 4)?,
            kv_values: DeviceAllocation::new(library, pages * PAGE_ROWS * KV_VALUES)?,
            kv_scales: DeviceAllocation::new(library, pages * PAGE_ROWS * KV_SCALES)?,
            capacity: pages * PAGE_ROWS,
            pages: std::array::from_fn(|_| vec![]),
            rows: [0; 16],
            pool: Rc::new(RefCell::new(PagePool::new(pages))),
            writing: Default::default(),
        })
    }
    pub fn view(&self, slot: usize, rows: usize) -> IndexCacheView<'_> {
        IndexCacheView {
            packed: self.packed.buffer,
            scales: self.scales.buffer,
            pages: &self.pages[slot],
            rows,
            device_pages: slice(
                self.page_table.buffer,
                slot * self.stride * 4,
                self.stride * 4,
            ),
            device_rows: slice(self.lengths.buffer, slot * 8, 8),
        }
    }
    pub fn kv_view(&self, slot: usize, rows: usize) -> KvCacheView<'_> {
        let index = self.view(slot, rows);
        KvCacheView {
            values: self.kv_values.buffer,
            scales: self.kv_scales.buffer,
            pages: index.pages,
            rows: index.rows,
            device_pages: index.device_pages,
            device_rows: index.device_rows,
        }
    }
    pub fn ensure_idle(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.lengths.buffer.bytes / 8 && self.writing.get() & (1 << slot) == 0,
            "compressed cache slot has a pending append");
        Ok(())
    }
    pub fn release(&mut self, slot: usize) -> Result<()> {
        self.ensure_idle(slot)?;
        // Caller first revokes the host lease, and has drained all consumers.
        self.pool.borrow_mut().release(&self.pages[slot]);
        self.pages[slot].clear();
        self.rows[slot] = 0;
        // The lease is revoked and consumers drained. reset/restore installs
        // replacement device metadata before a new owner becomes usable.
        Ok(())
    }
    pub fn reset(&self, slot: usize) -> Result<()> {
        self.ensure_idle(slot)?;
        self.lengths
            .library
            .copy_h2d(slice(self.lengths.buffer, slot * 8, 8), &[0; 8])
    }
    /// The four device segments holding `page`'s rows (packed index, index scales, KV values,
    /// KV scales), in the order the host cache stores them.
    pub fn page_segments(&self, page: u32) -> [Ds41rtDeviceBuffer; 4] {
        let rows = |buffer: Ds41rtDeviceBuffer, bytes: usize| {
            slice(buffer, page as usize * PAGE_ROWS * bytes, PAGE_ROWS * bytes)
        };
        [
            rows(self.packed.buffer, 64),
            rows(self.scales.buffer, 4),
            rows(self.kv_values.buffer, KV_VALUES),
            rows(self.kv_scales.buffer, KV_SCALES),
        ]
    }
    /// Identity generation of `page`: changes whenever the page is freed and reused.
    pub fn page_generation(&self, page: u32) -> u32 {
        self.pool.borrow().generation(page)
    }
    /// A prefix over `count` freshly allocated pages holding `rows` rows, for the host cache to
    /// fill; the prefix owns the pages. `SourcePoolExhausted` when fewer pages are free.
    pub fn allocate_prefix(&self, count: usize, rows: usize) -> Result<SourcePrefix> {
        ensure!(rows <= count * PAGE_ROWS, "allocated prefix rows exceed its pages");
        let mut pool = self.pool.borrow_mut();
        let available = pool.free.len();
        let pages = pool.allocate(count).ok_or(SourcePoolExhausted {
            work_index: 0,
            needed: count,
            available,
        })?;
        Ok(SourcePrefix {
            pool: Rc::clone(&self.pool),
            pages,
            rows,
        })
    }
    /// Retain initialized source rows without copying GPU data. Callers drain
    /// consumers and supply the authoritative committed row count.
    pub fn retain_prefix(&self, slot: usize, rows: usize) -> Result<SourcePrefix> {
        self.ensure_idle(slot)?;
        ensure!(
            slot < self.lengths.buffer.bytes / 8 && rows <= self.rows[slot],
            "source prefix exceeds initialized page table"
        );
        let pages = self.pages[slot][..rows.div_ceil(PAGE_ROWS)].to_vec();
        self.pool.borrow_mut().retain(&pages);
        Ok(SourcePrefix {
            pool: Rc::clone(&self.pool),
            pages,
            rows,
        })
    }
    /// Attach a retained source to a fresh request. Device metadata is installed
    /// before host publication; a later append privately copies a shared tail.
    pub fn restore_prefix(&mut self, slot: usize, prefix: &SourcePrefix) -> Result<()> {
        self.ensure_idle(slot)?;
        ensure!(
            slot < self.lengths.buffer.bytes / 8
                && self.pages[slot].is_empty()
                && Rc::ptr_eq(&self.pool, &prefix.pool)
                && prefix.pages.len() <= self.stride,
            "foreign source prefix or occupied destination"
        );
        let library = self.lengths.library;
        let bytes: Vec<u8> = prefix.pages.iter().flat_map(|p| p.to_ne_bytes()).collect();
        if !bytes.is_empty() {
            library.copy_h2d(
                slice(self.page_table.buffer, slot * self.stride * 4, bytes.len()),
                &bytes,
            )?;
        }
        library.copy_h2d(
            slice(self.lengths.buffer, slot * 8, 8),
            &(prefix.rows as u64).to_ne_bytes(),
        )?;
        self.pool.borrow_mut().retain(&prefix.pages);
        self.pages[slot] = prefix.pages.clone();
        self.rows[slot] = prefix.rows;
        Ok(())
    }
    /// # Safety
    /// Enqueue before accepted row writes. The plan owns destinations and append
    /// slots; old shared tails retain their request references through completion.
    /// Disjoint reservations may coexist. Drain before applying/discarding a plan.
    pub unsafe fn copy_shared_tails(&self, plan: &IndexPlan, stream: *mut c_void) -> Result<()> {
        self.validate_plan(plan)?;
        for &(_, _, source, destination) in &plan.replacements {
            for (buffer, row_bytes) in [
                (self.packed.buffer, 64),
                (self.scales.buffer, 4),
                (self.kv_values.buffer, KV_VALUES),
                (self.kv_scales.buffer, KV_SCALES),
            ] {
                let bytes = PAGE_ROWS * row_bytes;
                unsafe {
                    self.lengths.library.copy_d2d_async(
                        slice(buffer, destination as usize * bytes, bytes),
                        slice(buffer, source as usize * bytes, bytes),
                        bytes,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// # Safety
    /// Value/scales writes precede this call on stream. Drain the stream before
    /// reusing staging, releasing a slot or publishing the plan on the host.
    pub unsafe fn upload(&mut self, plan: &IndexPlan, stream: *mut c_void) -> Result<()> {
        self.validate_plan(plan)?;
        let starts = std::array::from_fn(|slot| self.pages[slot].len());
        unsafe { upload_metadata(self.lengths.library, self.page_table.buffer, self.lengths.buffer,
            self.stride, starts, plan, self.staging.bytes_mut(), stream) }
    }
    /// # Safety
    /// Same publication ordering as upload. Staging belongs to the producer and
    /// remains pinned and untouched until this stream drains.
    pub unsafe fn upload_staged(&self, plan: &IndexPlan, staging: &mut [u8], stream: *mut c_void) -> Result<()> {
        self.validate_plan(plan)?;
        let starts = std::array::from_fn(|slot| self.pages[slot].len());
        unsafe { upload_metadata(self.lengths.library, self.page_table.buffer, self.lengths.buffer,
            self.stride, starts, plan, staging, stream) }
    }
    pub fn validate_plan(&self, plan: &IndexPlan) -> Result<()> {
        let reservation = plan.reservation.as_ref().ok_or_else(|| anyhow::anyhow!("source plan not reserved"))?;
        ensure!(Rc::ptr_eq(&reservation.pool, &self.pool)
            && self.writing.get() & reservation.mask == reservation.mask, "foreign or lost source reservation");
        Ok(())
    }
    /// Atomically claim the append slots and free pages after validating every
    /// participant. Disjoint plans may coexist and apply in either order. After
    /// queueing GPU writes, drain before applying or dropping the plan.
    pub fn reserve(&self, appends: &[(usize, usize, usize)]) -> Result<IndexPlan> {
        let mut plan = IndexPlan {
            additions: vec![],
            used: 0,
            lengths: vec![],
            replacements: vec![],
            reservation: None,
        };
        let mut pool = self.pool.borrow_mut();
        let mut seen = [false; 16];
        for &(slot, old, new) in appends {
            ensure!(
                slot < self.lengths.buffer.bytes / 8 && !seen[slot],
                "duplicate or invalid index slot"
            );
            self.ensure_idle(slot)?;
            seen[slot] = true;
            ensure!(
                old == self.rows[slot]
                    && old <= new
                    && new <= 1048576
                    && self.pages[slot].len() == old.div_ceil(PAGE_ROWS),
                "index history binding differs"
            );
        }
        for (position, &(slot, old, new)) in appends.iter().enumerate() {
            plan.lengths.push((slot, new as u64));
            if new > old && old % PAGE_ROWS != 0 {
                let logical = old / PAGE_ROWS;
                let source = self.pages[slot][logical];
                if pool.shared(source) {
                    // If every owner appends in this transaction, one can keep
                    // the original. All tail copies precede every accepted write,
                    // including writes by that owner. A snapshot or non-appending
                    // owner prevents this optimization. Exclusive appends avoid
                    // this bounded (at most sixteen owners) scan entirely.
                    let mut writers = 0;
                    let mut last = position;
                    for (i, &(other, begin, end)) in appends.iter().enumerate() {
                        if end > begin
                            && begin % PAGE_ROWS != 0
                            && self.pages[other][begin / PAGE_ROWS] == source
                        {
                            writers += 1;
                            last = i;
                        }
                    }
                    if writers != pool.references(source) || position != last {
                        ensure!(
                            plan.used < pool.free.len(),
                            SourcePoolExhausted { work_index: position, needed: 1, available: pool.free.len() - plan.used }
                        );
                        let destination = pool.free[pool.free.len() - plan.used - 1];
                        plan.used += 1;
                        plan.replacements.push((slot, logical, source, destination));
                    }
                }
            }
            let extra = new.div_ceil(PAGE_ROWS) - self.pages[slot].len();
            ensure!(
                extra <= pool.free.len() - plan.used,
                SourcePoolExhausted { work_index: position, needed: extra, available: pool.free.len() - plan.used }
            );
            let end = pool.free.len() - plan.used;
            plan.additions.push((
                slot,
                pool.free[end - extra..end].iter().rev().copied().collect(),
            ));
            plan.used += extra;
        }
        let remaining = pool.free.len() - plan.used;
        let pages = pool.free.split_off(remaining);
        let mask = seen.iter().enumerate().fold(0u16, |mask, (slot, &used)|
            mask | if used { 1 << slot } else { 0 });
        self.writing.set(self.writing.get() | mask);
        plan.reservation = Some(PageReservation { pool: self.pool.clone(), pages,
            flags: self.writing.clone(), mask });
        Ok(plan)
    }
    pub fn destination(&self, plan: &IndexPlan, slot: usize, row: usize) -> Result<u64> {
        let logical_page = row / PAGE_ROWS;
        let old = &self.pages[slot];
        let page = if let Some(&(_, _, _, destination)) = plan
            .replacements
            .iter()
            .find(|&&(s, logical, _, _)| s == slot && logical == logical_page)
        {
            destination
        } else if logical_page < old.len() {
            old[logical_page]
        } else {
            let new = plan
                .additions
                .iter()
                .find(|(s, _)| *s == slot)
                .ok_or_else(|| anyhow::anyhow!("index append missing"))?;
            *new.1
                .get(logical_page - old.len())
                .ok_or_else(|| anyhow::anyhow!("index append outside reservation"))?
        };
        Ok(u64::from(page) * PAGE_ROWS as u64 + (row % PAGE_ROWS) as u64)
    }
    pub fn apply(&mut self, mut plan: IndexPlan) {
        let mut reservation = plan.reservation.take().expect("source plan is not reserved");
        assert!(Rc::ptr_eq(&self.pool, &reservation.pool), "foreign source plan");
        let mut pool = self.pool.borrow_mut();
        for (slot, logical, old, new) in plan.replacements {
            pool.retain(&[new]);
            pool.release(&[old]);
            self.pages[slot][logical] = new;
        }
        for (slot, pages) in plan.additions {
            pool.retain(&pages);
            self.pages[slot].extend(pages);
        }
        for (slot, rows) in plan.lengths {
            self.rows[slot] = rows as usize;
        }
        reservation.pages.clear(); // Page references now belong to request tables.
        drop(pool); // Reservation drop must not reborrow an active pool borrow.
        drop(reservation);
    }
}

unsafe fn upload_metadata(library: &NativeLibrary, page_table: Ds41rtDeviceBuffer,
    lengths: Ds41rtDeviceBuffer, stride: usize, starts: [usize; 16], plan: &IndexPlan,
    staging: &mut [u8], stream: *mut c_void) -> Result<()> {
    let bytes = plan.replacements.len()*4 + plan.additions.iter().map(|(_, p)| p.len()*4).sum::<usize>()
        + plan.lengths.len()*8;
    ensure!(staging.len() >= bytes, "source metadata staging too small");
        let mut offset = 0;
        for &(slot, logical, _, destination) in &plan.replacements {
            let staging = &mut staging[offset..offset + 4];
            staging.copy_from_slice(&destination.to_ne_bytes());
            unsafe {
                library.copy_h2d_async(
                    slice(
                        page_table,
                        (slot * stride + logical) * 4,
                        4,
                    ),
                    staging,
                    stream,
                )?;
            }
            offset += 4;
        }
        for (slot, pages) in &plan.additions {
            let bytes = pages.len() * 4;
            if bytes == 0 {
                continue;
            }
            let staging = &mut staging[offset..offset + bytes];
            for (out, page) in staging.chunks_exact_mut(4).zip(pages) {
                out.copy_from_slice(&page.to_ne_bytes());
            }
            let dst = (slot * stride + starts[*slot]) * 4;
            unsafe {
                library.copy_h2d_async(
                    slice(page_table, dst, bytes),
                    staging,
                    stream,
                )?;
            }
            offset += bytes;
        }
        for &(slot, rows) in &plan.lengths {
            let staging = &mut staging[offset..offset + 8];
            staging.copy_from_slice(&rows.to_ne_bytes());
            unsafe {
                library.copy_h2d_async(slice(lengths, slot * 8, 8), staging, stream)?;
            }
            offset += 8;
        }
        Ok(())
}

fn slice(buffer: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    Ds41rtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buffer
    }
}

#[cfg(test)]
mod high_pages;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_memory::LoadStream;

    pub(super) fn append(
        cache: &mut SourceCache<'_>,
        slot: usize,
        old: usize,
        new: usize,
        value: u8,
        stream: *mut c_void,
    ) -> Result<()> {
        append_many(cache, &[(slot, old, new, value)], stream)
    }

    fn append_many(
        cache: &mut SourceCache<'_>,
        work: &[(usize, usize, usize, u8)],
        stream: *mut c_void,
    ) -> Result<()> {
        let appends: Vec<_> = work
            .iter()
            .map(|&(slot, old, new, _)| (slot, old, new))
            .collect();
        let plan = cache.reserve(&appends)?;
        let library = cache.lengths.library;
        unsafe {
            cache.copy_shared_tails(&plan, stream)?;
            library.cuda_stream_synchronize(stream)?;
        }
        for &(slot, old, new, value) in work {
            for row in old..new {
                let destination = cache.destination(&plan, slot, row)? as usize;
                for (buffer, width) in [
                    (cache.packed.buffer, 64),
                    (cache.scales.buffer, 4),
                    (cache.kv_values.buffer, KV_VALUES),
                    (cache.kv_scales.buffer, KV_SCALES),
                ] {
                    library.copy_h2d(
                        slice(buffer, destination * width, width),
                        &vec![value; width],
                    )?;
                }
            }
        }
        unsafe {
            cache.upload(&plan, stream)?;
            library.cuda_stream_synchronize(stream)?;
        }
        cache.apply(plan);
        Ok(())
    }

    pub(super) fn read(cache: &SourceCache<'_>, slot: usize, rows: usize) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for row in 0..rows {
            let physical =
                cache.pages[slot][row / PAGE_ROWS] as usize * PAGE_ROWS + row % PAGE_ROWS;
            for (buffer, width) in [
                (cache.packed.buffer, 64),
                (cache.scales.buffer, 4),
                (cache.kv_values.buffer, KV_VALUES),
                (cache.kv_scales.buffer, KV_SCALES),
            ] {
                let start = bytes.len();
                bytes.resize(start + width, 0);
                cache
                    .lengths
                    .library
                    .copy_d2h(&mut bytes[start..], slice(buffer, physical * width, width))?;
            }
        }
        Ok(bytes)
    }

    #[test]
    #[ignore = "requires a CUDA native library in DS41RT_NATIVE_LIB"]
    fn retained_source_truncation_preserves_future_rows_after_original_release() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        let mut cache = SourceCache::new(&library, 3, 2)?;
        append(&mut cache, 0, 0, 300, 0x11, stream.raw)?;
        let saved = cache.retain_prefix(0, 300)?;
        cache.release(0)?;
        assert!(saved.truncate(301).is_err());
        let empty = saved.truncate(0)?;
        assert!(empty.pages.is_empty());
        let short = saved.truncate(100)?;
        assert_eq!(short.pages.len(), 1);
        cache.restore_prefix(1, &short)?;
        assert!(cache.retain_prefix(1, 101).is_err());
        append(&mut cache, 1, 100, 130, 0x22, stream.raw)?;
        cache.restore_prefix(0, &saved)?;
        assert!(read(&cache, 0, 300)?.iter().all(|&v| v == 0x11));
        let branch = read(&cache, 1, 130)?;
        assert!(branch[..100 * SOURCE_ROW_BYTES].iter().all(|&v| v == 0x11));
        assert!(branch[100 * SOURCE_ROW_BYTES..].iter().all(|&v| v == 0x22));
        drop(short);
        drop(saved);
        drop(empty);
        cache.release(0)?;
        cache.release(1)?;
        assert_eq!(cache.pool.borrow().free.len(), 3);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA native library in DS41RT_NATIVE_LIB"]
    fn native_source_shared_writers_fit_exact_pool_and_preserve_nonwriters() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        for nonwriter in [false, true] {
            let mut cache = SourceCache::new(&library, 3, 3)?;
            append(&mut cache, 0, 0, 6, 0x11, stream.raw)?;
            let short = cache.retain_prefix(0, 3)?;
            cache.restore_prefix(1, &short)?;
            cache.restore_prefix(2, &short)?;
            let original = cache.pages[0][0];
            // The retained snapshot needs the original page, so three writers
            // cannot fit in two spare pages. Planning must leave owners intact.
            assert!(cache.reserve(&[(0, 6, 7), (1, 3, 4), (2, 3, 4)]).is_err());
            assert_eq!(cache.pool.borrow().free.len(), 2);
            drop(short);
            assert!(cache.reserve(&[(0, 6, 7), (0, 6, 7)]).is_err());
            assert!(cache.reserve(&[(16, 6, 7)]).is_err());
            let end = if nonwriter { 3 } else { 4 };
            append_many(
                &mut cache,
                &[(0, 6, 7, 0x22), (1, 3, 4, 0x33), (2, 3, end, 0x44)],
                stream.raw,
            )?;
            assert_eq!(cache.pool.borrow().free.len(), 0);
            assert_eq!(cache.pages[2][0], original);
            assert_ne!(cache.pages[0][0], original);
            assert_ne!(cache.pages[1][0], original);
            assert_ne!(cache.pages[0][0], cache.pages[1][0]);
            // Slot two may overwrite rows that belonged to slot zero's longer
            // prefix; its copy must have completed before any branch wrote.
            for (slot, old, new, value) in [(0, 6, 7, 0x22), (1, 3, 4, 0x33), (2, 3, end, 0x44)] {
                let bytes = read(&cache, slot, new)?;
                assert!(bytes[..old * SOURCE_ROW_BYTES].iter().all(|&v| v == 0x11));
                assert!(bytes[old * SOURCE_ROW_BYTES..].iter().all(|&v| v == value));
                cache.release(slot)?;
            }
            assert_eq!(cache.pool.borrow().free.len(), 3);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA native library in DS41RT_NATIVE_LIB"]
    fn native_source_prefix_copy_on_write_and_eviction() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        let mut cache = SourceCache::new(&library, 5, 2)?;
        append(&mut cache, 0, 0, 270, 0x31, stream.raw)?;
        let original = read(&cache, 0, 270)?;
        assert!(cache.retain_prefix(0, 271).is_err());
        assert!(original.iter().all(|&v| v == 0x31));
        let prefix = cache.retain_prefix(0, 270)?;
        cache.restore_prefix(1, &prefix)?;
        let original_pages = cache.pages[0].clone();
        // Only the partial last page is copied. The first page stays shared.
        append(&mut cache, 1, 270, 300, 0x72, stream.raw)?;
        assert_eq!(cache.pages[1][0], original_pages[0]);
        assert_ne!(cache.pages[1][1], original_pages[1]);
        assert_eq!(read(&cache, 0, 270)?, original);
        let branch = read(&cache, 1, 300)?;
        assert_eq!(&branch[..original.len()], original.as_slice());
        assert!(branch[original.len()..].iter().all(|&v| v == 0x72));
        // A divergent shorter prefix can share a physical page containing
        // future rows; appending must not overwrite those retained rows.
        let shorter = cache.retain_prefix(0, 259)?;
        cache.release(1)?;
        cache.restore_prefix(1, &shorter)?;
        append(&mut cache, 1, 259, 280, 0x53, stream.raw)?;
        assert_eq!(read(&cache, 0, 270)?, original);
        let branch = read(&cache, 1, 280)?;
        assert!(branch[..259 * SOURCE_ROW_BYTES].iter().all(|&v| v == 0x31));
        assert!(branch[259 * SOURCE_ROW_BYTES..].iter().all(|&v| v == 0x53));
        cache.release(0)?;
        cache.release(1)?;
        assert_eq!(cache.pool.borrow().free.len(), 3);
        drop(shorter);
        assert_eq!(cache.pool.borrow().free.len(), 3);
        // Restore after every original request lease was released.
        cache.restore_prefix(1, &prefix)?;
        assert_eq!(read(&cache, 1, 270)?, original);
        drop(prefix);
        assert_eq!(cache.pool.borrow().free.len(), 3);
        cache.release(1)?;
        assert_eq!(cache.pool.borrow().free.len(), 5);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA native library in DS41RT_NATIVE_LIB"]
    fn native_source_prefix_exhaustion_and_page_boundary() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        let mut cache = SourceCache::new(&library, 2, 2)?;
        append(&mut cache, 0, 0, 256, 0x11, stream.raw)?;
        let prefix = cache.retain_prefix(0, 256)?;
        cache.restore_prefix(1, &prefix)?;
        let full = cache.pages[0][0];
        append(&mut cache, 1, 256, 257, 0x22, stream.raw)?;
        assert_eq!(cache.pages[1][0], full);
        let partial = cache.retain_prefix(1, 257)?;
        let error = cache.reserve(&[(1, 257, 258)]).err().expect("shared tail must need a page");
        let pressure = error.downcast_ref::<SourcePoolExhausted>().expect("typed pool pressure");
        assert_eq!((pressure.work_index, pressure.needed, pressure.available), (0, 1, 0));
        let error = cache.reserve(&[(0, 256, 256), (1, 257, 513)]).err().expect("joint append cannot fit");
        assert_eq!(error.downcast_ref::<SourcePoolExhausted>().unwrap().work_index, 1);
        // Invalid binding must never be classified as recoverable pool pressure.
        let error = cache.reserve(&[(1, 256, 258)]).err().expect("wrong committed end");
        assert!(error.downcast_ref::<SourcePoolExhausted>().is_none());
        let before = read(&cache, 1, 257)?;
        drop(partial);
        // The tail is now exclusive: append succeeds with no free pages.
        append(&mut cache, 1, 257, 258, 0x33, stream.raw)?;
        assert_eq!(read(&cache, 1, 257)?, before);
        cache.release(0)?;
        cache.release(1)?;
        drop(prefix);
        assert_eq!(cache.pool.borrow().free.len(), 2);
        Ok(())
    }
}
