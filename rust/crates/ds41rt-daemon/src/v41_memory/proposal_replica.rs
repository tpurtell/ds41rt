//! Lane-owned peer storage for private attention KV proposals.
use super::device::{Allocation, Device};
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer,V41PeerCopy};
use std::ffi::c_void;

#[derive(Clone,Copy)]
pub(crate) enum ProposalFormat { WindowFp8, CompressedFp4 }
impl ProposalFormat {
    fn widths(self)->(usize,usize) {
        match self { Self::WindowFp8=>(512,16),Self::CompressedFp4=>(256,32) }
    }
}
/// One owner per producer wave, retained across the layers consuming that wave.
/// Copy once after production; never duplicate the committed cache here.
pub(crate) struct ProposalReplica<'a> {
    values:Allocation<'a>,
    scales:Allocation<'a>,
    copy:V41PeerCopy<'a>,
    source:Device<'a>,
    format:ProposalFormat,
    capacity:usize,
}
impl<'a> ProposalReplica<'a> {
    pub fn device_bytes(capacity:usize,format:ProposalFormat)->Result<usize> {
        ensure!((1..=4096).contains(&capacity),"invalid proposal replica capacity");
        let (values,scales)=format.widths();Ok(capacity*(values+scales))
    }
    pub fn new(source:Device<'a>,destination:Device<'a>,capacity:usize,format:ProposalFormat)->Result<Self> {
        Self::device_bytes(capacity,format)?;
        ensure!(source.id!=destination.id && std::ptr::eq(source.library,destination.library),
            "proposal replica requires peer devices from one library");
        destination.run(||destination.library.cuda_enable_peer(source.id))?;
        let (values,scales)=format.widths();
        Ok(Self { values:Allocation::new(destination,capacity*values)?,
            scales:Allocation::new(destination,capacity*scales)?,
            copy:destination.run(||destination.library.v41_peer_copy())?,source,format,capacity })
    }
    /// # Safety
    /// Inputs are the matching producer's immutable proposal planes. The peer
    /// stream follows producer completion and belongs to this destination device.
    /// Preserve source/replica buffers through publication and every consumer;
    /// drain on cancellation/error before releasing them or reusing this wave.
    /// Offset/stride are kept unchanged so request metadata remains authoritative.
    pub unsafe fn copy_rows(&self,values:Ds41rtDeviceBuffer,scales:Ds41rtDeviceBuffer,
        offset:usize,count:usize,step:usize,stream:*mut c_void)->Result<()> {
        ensure!(values.device_id==self.source.id && scales.device_id==self.source.id,
            "proposal source device differs");
        let (value_width,scale_width)=self.format.widths();
        let rows=span(self.capacity,offset,count,step)?;
        if rows==0 { return Ok(()); }
        // Producer views expose only this wave's used rows, while the replica
        // remains provisioned for the maximum batch. Validate the referenced
        // extent, not the replica's reserved capacity.
        ensure!(values.bytes>=(offset+rows)*value_width && scales.bytes>=(offset+rows)*scale_width,
            "proposal source extent differs");
        self.values.device.run(|| {
            // Preserve physical offsets while skipping unused stride gaps.
            for (dst,src,width) in [(self.values.buffer,values,value_width),
                (self.scales.buffer,scales,scale_width)] {
                unsafe { self.copy.launch_rows(slice(dst,offset*width,rows*width),
                    slice(src,offset*width,rows*width),width,count,step*width,step*width,stream)?; }
            }
            Ok(())
        })
    }
    /// Raw storage is usable only under the publication/lifetime contract above.
    pub fn buffers(&self)->(Ds41rtDeviceBuffer,Ds41rtDeviceBuffer) {
        (self.values.buffer,self.scales.buffer)
    }
}
fn span(capacity:usize,offset:usize,count:usize,step:usize)->Result<usize> {
    ensure!(matches!(step,1|2) && offset<=capacity,"invalid proposal span");
    if count==0 { return Ok(0); }
    ensure!(offset<capacity && count-1<=(capacity-1-offset)/step,"proposal span exceeds capacity");
    Ok((count-1)*step+1)
}
fn slice(mut buffer:Ds41rtDeviceBuffer,offset:usize,bytes:usize)->Ds41rtDeviceBuffer {
    debug_assert!(offset+bytes<=buffer.bytes);
    buffer.ptr=unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };buffer.bytes=bytes;buffer
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proposal_span_preserves_offsets_without_overflow() -> Result<()> {
        assert_eq!(span(4096,3,4,2)?,7);
        assert_eq!(span(4096,4096,0,1)?,0);
        assert_eq!(span(4096,4095,1,2)?,1);
        assert!(span(4096,4095,2,2).is_err());
        assert!(span(4096,0,usize::MAX,2).is_err());
        assert!(span(4096,usize::MAX,1,1).is_err());
        assert!(span(4096,0,0,0).is_err());
        assert_eq!(ProposalReplica::device_bytes(4096,ProposalFormat::WindowFp8)?,4096*528);
        assert_eq!(ProposalReplica::device_bytes(4096,ProposalFormat::CompressedFp4)?,4096*288);
        Ok(())
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB with SM peer copy and two CUDA GPUs"]
    fn proposal_replica_preserves_strides_and_untouched_rows() -> Result<()> {
        use super::super::{device::Stream,peer_publication::PeerPublication};
        let lib=unsafe { ds41rt_ffi::NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        for gpu in 0..2 { for format in [ProposalFormat::WindowFp8,ProposalFormat::CompressedFp4] {
            let owner=Device { library:&lib,id:gpu };let peer=Device { library:&lib,id:1-gpu };
            let replica=ProposalReplica::new(owner,peer,16,format)?;
            let (vw,sw)=format.widths();
            let source_values=Allocation::new(owner,16*vw)?;
            let source_scales=Allocation::new(owner,16*sw)?;
            let producer=Stream::new(owner)?;
            let mut publication=PeerPublication::new(owner,peer)?;
            let (destination_values,destination_scales)=replica.buffers();
            peer.run(|| {
                lib.copy_h2d(destination_values,&vec![0xcd;16*vw])?;
                lib.copy_h2d(destination_scales,&vec![0xcd;16*sw])
            })?;
            let mut expected_values=vec![0xcd;16*vw];let mut expected_scales=vec![0xcd;16*sw];
            for (offset,count,step,value) in [(2,3,2,0x22),(3,1,1,0x66),(16,0,1,0x77)] {
                owner.run(|| {
                    lib.copy_h2d(source_values.buffer,&vec![value;16*vw])?;
                    lib.copy_h2d(source_scales.buffer,&vec![value+1;16*sw])
                })?;
                let used=if count==0 { 0 } else { offset+(count-1)*step+1 };
                let values=slice(source_values.buffer,0,used*vw);
                let scales=slice(source_scales.buffer,0,used*sw);
                if used>0 {
                    assert!(unsafe { replica.copy_rows(slice(values,0,used*vw-1),scales,
                        offset,count,step,producer.raw) }.is_err());
                }
                unsafe { publication.enqueue(producer.raw,|stream|replica.copy_rows(
                    values,scales,offset,count,step,stream))?; }
                producer.drain()?;
                for row in (0..count).map(|i|offset+i*step) {
                    expected_values[row*vw..(row+1)*vw].fill(value);
                    expected_scales[row*sw..(row+1)*sw].fill(value+1);
                }
                peer.run(|| {
                    let mut actual=vec![0;16*vw];lib.copy_d2h(&mut actual,destination_values)?;
                    assert_eq!(actual,expected_values);
                    let mut actual=vec![0;16*sw];lib.copy_d2h(&mut actual,destination_scales)?;
                    assert_eq!(actual,expected_scales);Ok(())
                })?;
                assert_eq!(replica.buffers().0.ptr,destination_values.ptr);
            }
            assert!(unsafe { replica.copy_rows(source_values.buffer,source_scales.buffer,
                15,2,2,producer.raw) }.is_err());
        }}
        Ok(())
    }

}
