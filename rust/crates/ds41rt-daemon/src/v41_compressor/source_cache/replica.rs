//! Peer FP4 payload storage. Page ownership remains with the authoritative cache.
use super::*;
use crate::v41_memory::device::{Allocation, Device};

pub(crate) struct SourceCacheReplica<'a> {
    pub storage: Rc<SourceReplica<'a>>,
    restore: crate::v41_memory::device::Stream<'a>,
}
impl<'a> SourceCache<'a> {
    pub fn enable_replica(&mut self,peer:Device<'a>)->Result<Rc<SourceReplica<'a>>> {
        ensure!(self.replica.is_none(),"source replica already configured");
        let storage=Rc::new(SourceReplica::new(self,peer)?);
        let restore=crate::v41_memory::device::Stream::new(peer)?;
        self.replica=Some(SourceCacheReplica { storage:storage.clone(),restore });
        Ok(storage)
    }
    /// Complete RAM-restored payload publication before exposing the rebuilt
    /// prefix. This admission path already waits for the host-cache upload.
    /// # Safety
    /// Host uploads are complete and the prefix's pages remain exclusively owned.
    pub unsafe fn publish_restored_prefix(&self,prefix:&SourcePrefix)->Result<()> {
        let Some(replica)=&self.replica else { return Ok(()); };
        let copied=unsafe { replica.storage.copy_restored_pages(self,prefix,replica.restore.raw) };
        // A partial copy must finish before the caller can release its pages.
        let drained=replica.restore.drain();
        copied.and(drained)
    }
}

pub(crate) struct SourceReplica<'a> {
    copy: ds41rt_ffi::V41PeerCopy<'a>,
    values: Allocation<'a>,
    scales: Allocation<'a>,
    pages: Allocation<'a>,
    lengths: Allocation<'a>,
    pool: Rc<RefCell<PagePool>>,
    capacity: usize,
    stride: usize,
}
impl<'a> SourceReplica<'a> {
    pub(super) fn install_metadata(&self,slot:usize,pages:&[u8],rows:usize)->Result<()> {
        ensure!(slot<self.lengths.buffer.bytes/8 && pages.len()<=self.stride*4
            && pages.len()%4==0 && rows.div_ceil(PAGE_ROWS)<=pages.len()/4,
            "source replica metadata extent differs");
        self.values.device.run(|| {
            if !pages.is_empty() { self.values.device.library.copy_h2d(
                slice(self.pages.buffer,slot*self.stride*4,pages.len()),pages)?; }
            self.values.device.library.copy_h2d(slice(self.lengths.buffer,slot*8,8),&(rows as u64).to_ne_bytes())
        })
    }
    pub fn device(&self) -> Device<'a> { self.values.device }
    pub fn validate_owner(&self, source: &SourceCache<'_>) -> Result<()> { self.check(source) }
    pub fn device_bytes(pages: usize, slots: usize) -> Result<usize> {
        SourceCache::device_bytes(pages, slots)?;
        Ok(pages * PAGE_ROWS * (KV_VALUES + KV_SCALES) + slots * (pages.min(4096)*4+8))
    }
    /// Construct before admitting requests; enabling a replica on a populated
    /// pool would require copying every retained snapshot, not just live slots.
    pub fn new(source: &SourceCache<'a>, device: Device<'a>) -> Result<Self> {
        ensure!(std::ptr::eq(source.lengths.library, device.library)
            && source.kv_values.buffer.device_id != device.id,
            "source replica requires a peer device from the same library");
        ensure!(source.writing.get() == 0 && source.pool.borrow().free.len()*PAGE_ROWS == source.capacity,
            "source replica requires an empty source pool");
        device.run(|| device.library.cuda_enable_peer(source.kv_values.buffer.device_id))?;
        let lengths = Allocation::new(device, source.lengths.buffer.bytes)?;
        device.run(|| device.library.copy_h2d(lengths.buffer, &vec![0; lengths.buffer.bytes]))?;
        Ok(Self {
            copy: device.run(|| device.library.v41_peer_copy())?,
            values: Allocation::new(device, source.capacity*KV_VALUES)?,
            scales: Allocation::new(device, source.capacity*KV_SCALES)?,
            pages: Allocation::new(device, source.page_table.buffer.bytes)?,
            lengths, pool: source.pool.clone(), capacity: source.capacity, stride: source.stride,
        })
    }
    fn check(&self, source: &SourceCache<'_>) -> Result<()> {
        ensure!(Rc::ptr_eq(&self.pool, &source.pool), "foreign source replica");
        Ok(())
    }
    /// # Safety
    /// The destination stream waits for authoritative payload and metadata writes.
    /// Keep source, replica and plan alive until that stream drains, including on
    /// failure/cancellation. Consumers may read only after the copy completes.
    /// Disjoint source reservations may use separate lane-owned peer streams.
    pub unsafe fn copy_append(&self, source: &SourceCache<'_>, plan: &IndexPlan,
        stream: *mut c_void) -> Result<()> {
        self.check(source)?;
        source.validate_plan(plan)?;
        self.values.device.run(|| {
            // A shared tail moved to a new physical page needs its old prefix too.
            for &(_, _, _, destination) in &plan.replacements {
                unsafe { self.copy_rows(source, destination as usize*PAGE_ROWS, PAGE_ROWS, stream)?; }
            }
            for &(slot, new) in &plan.lengths {
                let mut row = source.rows[slot];
                while row < new as usize {
                    let count = (PAGE_ROWS-row%PAGE_ROWS).min(new as usize-row);
                    let physical = source.destination(plan, slot, row)? as usize;
                    unsafe { self.copy_rows(source, physical, count, stream)?; }
                    row += count;
                }
            }
            // Publish changed page-table entries after payload copies. Lengths
            // follow every table entry, so a ready length never exposes stale KV.
            for &(slot, logical, _, _) in &plan.replacements {
                unsafe { self.copy_metadata(source, slot*self.stride*4+logical*4, 4, stream)?; }
            }
            for (slot, pages) in &plan.additions {
                if !pages.is_empty() {
                    unsafe { self.copy_metadata(source,
                        (slot*self.stride+source.pages[*slot].len())*4, pages.len()*4, stream)?; }
                }
            }
            for &(slot, _) in &plan.lengths {
                unsafe { self.copy_length(source, slot, stream)?; }
            }
            Ok(())
        })
    }
    /// Copy freshly restored host-cache pages before attaching/publishing a slot.
    /// Existing retained GPU snapshots already share the replica's physical pages.
    /// # Safety
    /// Source writes are ordered before this destination stream. Retain both
    /// allocations and the prefix until completion, draining on all error paths.
    pub unsafe fn copy_restored_pages(&self, source: &SourceCache<'_>, prefix: &SourcePrefix,
        stream: *mut c_void) -> Result<()> {
        self.check(source)?;
        ensure!(Rc::ptr_eq(&self.pool, &prefix.pool), "foreign restored source prefix");
        self.values.device.run(|| {
            let mut remaining = prefix.rows;
            for &page in &prefix.pages {
                let count = remaining.min(PAGE_ROWS);
                if count > 0 { unsafe { self.copy_rows(source, page as usize*PAGE_ROWS, count, stream)?; } }
                remaining -= count;
            }
            ensure!(remaining == 0, "restored source prefix is truncated");
            Ok(())
        })
    }
    /// # Safety
    /// All referenced replica pages are ready or ordered on stream, and source
    /// restore/reset metadata is ready. Drain before publishing the host lease.
    pub unsafe fn copy_slot_metadata(&self, source: &SourceCache<'_>, slot: usize,
        stream: *mut c_void) -> Result<()> {
        self.check(source)?;
        source.ensure_idle(slot)?;
        self.values.device.run(|| {
            let bytes = source.pages[slot].len()*4;
            if bytes > 0 { unsafe { self.copy_metadata(source, slot*self.stride*4, bytes, stream)?; } }
            unsafe { self.copy_length(source, slot, stream) }
        })
    }
    /// # Safety
    /// The caller has completed replica writes for this slot and holds the
    /// authoritative source lease through every consumer. This storage owns no
    /// additional page references; it never increases logical token capacity.
    pub unsafe fn view<'s>(&'s self, source: &'s SourceCache<'_>, slot: usize,
        rows: usize) -> Result<KvCacheView<'s>> {
        source.ensure_idle(slot)?;
        unsafe { self.committed_view(source,slot,rows) }
    }
    /// # Safety
    /// As for view, but a follower may append beyond these committed rows.
    /// Its publication must copy replacement-page payload before page IDs and
    /// lengths; consumers must retain their causal bounds and authoritative lease.
    pub unsafe fn committed_view<'s>(&'s self, source:&'s SourceCache<'_>,slot:usize,
        rows:usize)->Result<KvCacheView<'s>> {
        self.check(source)?;
        ensure!(slot<source.lengths.buffer.bytes/8,"replica slot out of range");
        ensure!(rows <= source.rows[slot], "replica view exceeds committed rows");
        Ok(KvCacheView { values: self.values.buffer, scales: self.scales.buffer,
            pages: &source.pages[slot], rows,
            device_pages: slice(self.pages.buffer, slot*self.stride*4, self.stride*4),
            device_rows: slice(self.lengths.buffer, slot*8, 8) })
    }
    unsafe fn copy_rows(&self, source: &SourceCache<'_>, row: usize, count: usize,
        stream: *mut c_void) -> Result<()> {
        ensure!(row <= self.capacity && count <= self.capacity-row, "replica copy outside pool");
        for (dst, src, width) in [(self.values.buffer, source.kv_values.buffer, KV_VALUES),
            (self.scales.buffer, source.kv_scales.buffer, KV_SCALES)] {
            unsafe { self.copy.launch(slice(dst,row*width,count*width),
                slice(src,row*width,count*width),count*width,stream)?; }
        }
        Ok(())
    }
    unsafe fn copy_metadata(&self, source: &SourceCache<'_>, offset: usize, bytes: usize,
        stream: *mut c_void) -> Result<()> {
        unsafe { self.copy.launch(slice(self.pages.buffer,offset,bytes),
            slice(source.page_table.buffer,offset,bytes),bytes,stream) }
    }
    unsafe fn copy_length(&self, source: &SourceCache<'_>, slot: usize, stream: *mut c_void) -> Result<()> {
        unsafe { self.copy.launch(slice(self.lengths.buffer,slot*8,8),
            slice(source.lengths.buffer,slot*8,8),8,stream) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_memory::LoadStream;

    #[test]
    fn replica_budget_counts_only_fp4_payload_and_metadata() -> Result<()> {
        assert_eq!(SourceReplica::device_bytes(4096,16)?, 4096*256*288+16*(4096*4+8));
        assert!(SourceReplica::device_bytes(0,16).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and two CUDA GPUs"]
    fn peer_source_replica_append_cow_and_host_restore() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        for gpu in 0..2 {
            let owner = Device { library: &lib, id: gpu };
            let peer = Device { library: &lib, id: 1-gpu };
            owner.run(|| {
                let mut source = SourceCache::new(&lib,8,3)?;
                let replica = source.enable_replica(peer)?;
                let producer = LoadStream { library: &lib, raw: lib.cuda_stream_create()? };
                let mut publication = crate::v41_memory::peer_publication::PeerPublication::new(owner,peer)?;
                let mut append = |source: &mut SourceCache<'_>, old, new, value| -> Result<()> {
                    let plan = source.reserve(&[(0,old,new)])?;
                    unsafe { source.copy_shared_tails(&plan,producer.raw)?;
                        lib.cuda_stream_synchronize(producer.raw)?; }
                    for row in old..new {
                        let physical = source.destination(&plan,0,row)? as usize;
                        for (b,width) in [(source.kv_values.buffer,KV_VALUES),(source.kv_scales.buffer,KV_SCALES)] {
                            lib.copy_h2d(slice(b,physical*width,width),&vec![value;width])?;
                        }
                    }
                    unsafe { source.upload(&plan,producer.raw)?;
                        publication.enqueue(producer.raw,|stream| replica.copy_append(source,&plan,stream))?;
                        lib.cuda_stream_synchronize(producer.raw)?; }
                    source.apply(plan);
                    Ok(())
                };
                let check = |source: &SourceCache<'_>, slot, expected: &[u8]| -> Result<()> {
                    let view = unsafe { replica.view(source,slot,expected.len())? };
                    peer.run(|| {
                        let mut length=[0u8;8];lib.copy_d2h(&mut length,view.device_rows)?;
                        assert_eq!(u64::from_ne_bytes(length),expected.len() as u64);
                        let mut pages=vec![0u8;view.pages.len()*4];
                        lib.copy_d2h(&mut pages,slice(view.device_pages,0,view.pages.len()*4))?;
                        assert_eq!(pages,view.pages.iter().flat_map(|p|p.to_ne_bytes()).collect::<Vec<_>>());
                        for (row,&value) in expected.iter().enumerate() {
                            let physical=view.pages[row/PAGE_ROWS] as usize*PAGE_ROWS+row%PAGE_ROWS;
                            for (b,width) in [(view.values,KV_VALUES),(view.scales,KV_SCALES)] {
                                let mut bytes=vec![0;width];lib.copy_d2h(&mut bytes,slice(b,physical*width,width))?;
                                assert!(bytes.iter().all(|&b|b==value),"peer KV row {row} differs");
                            }
                        }
                        Ok(())
                    })
                };
                append(&mut source,0,255,0x22)?;
                check(&source,0,&vec![0x22;255])?;
                let retained=source.retain_prefix(0,255)?;
                source.restore_prefix(1,&retained)?;
                append(&mut source,255,258,0x44)?;
                let mut expected=vec![0x22;255];expected.extend([0x44;3]);
                check(&source,0,&expected)?;
                check(&source,1,&vec![0x22;255])?;
                let restored=source.allocate_prefix(2,300)?;
                for &page in &restored.pages {
                    for b in &source.page_segments(page)[2..] { lib.copy_h2d(*b,&vec![0x66;b.bytes])?; }
                }
                unsafe { source.publish_restored_prefix(&restored)?; }
                source.restore_prefix(2,&restored)?;
                check(&source,2,&vec![0x66;300])?;
                source.release(2)?;
                source.reset(2)?;
                check(&source,2,&[])?;
                assert!(SourceReplica::new(&source,peer).is_err());
                Ok(())
            })?;
        }
        Ok(())
    }
}
