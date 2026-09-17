//! Native TP2 draft expert ownership and graph integration.
mod rank;
mod pair;
#[cfg(test)]
mod checkpoint_tests;

pub(super) use rank::Weights;
use super::{DsparkRouter, DsparkSharedFfn, DsparkWeights};
use crate::v41_memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use std::{ffi::c_void, rc::Rc};

/// One stage in one lane. The containing chain owns its external capture stream.
pub(super) struct RoutedWave<'w, 'a> {
    stream: Stream<'a>,
    pair: pair::Pair<'a>,
    inputs: [Allocation<'a>;3],
    shared: Allocation<'a>,
    owner: &'w DsparkWeights<'a>,
    stage: usize,
    capacity: u32,
}
impl<'w,'a> RoutedWave<'w,'a> {
    pub fn device_bytes(library:&NativeLibrary,capacity:u32)->Result<[usize;2]> {
        let mut bytes=pair::Pair::device_bytes(library,capacity)?;
        bytes[1]+=capacity as usize*(10240*2+12*2);
        Ok(bytes)
    }
    pub fn new(owner:&'w DsparkWeights<'a>,weights:[Rc<Weights<'a>>;2],stage:usize,capacity:u32)->Result<Self> {
        ensure!(stage<3 && owner.library.cuda_get_device()?==1,"draft TP2 transformer must reside on RTX1");
        let device=Device {library:owner.library,id:1};
        Ok(Self {stream:Stream::new(device)?,pair:pair::Pair::new(weights,capacity)?,
            inputs:[Allocation::new(device,capacity as usize*10240)?,Allocation::new(device,capacity as usize*12)?,Allocation::new(device,capacity as usize*12)?],
            shared:Allocation::new(device,capacity as usize*10240)?,owner,stage,capacity})
    }
    pub fn stream(&self)->*mut c_void {self.stream.raw}
    pub fn synchronize(&self)->Result<()> {let root=self.stream.drain();let peer=self.pair.drain_peer();root.and(peer)}
    pub fn inputs(&self)->[Ds41rtDeviceBuffer;3] {self.inputs.each_ref().map(|a|a.buffer)}
    pub fn output(&self)->Ds41rtDeviceBuffer {self.pair.output()}
    /// Caller retains/drains external stream and destroys captured graphs first.
    pub unsafe fn enqueue(&mut self,router:&mut DsparkRouter<'_, '_>,shared:&mut DsparkSharedFfn<'_, '_>,rows:u32,stream:*mut c_void)->Result<()> {
        ensure!(rows>0 && rows<=self.capacity && router.matches_stage(self.owner,self.stage)
            && shared.matches_stage(self.owner,self.stage),"draft TP2 FFN stage or extent differs");
        let inputs=self.inputs();
        unsafe {
            router.enqueue(inputs,rows as usize,stream)?;
            let shared_output=self.shared.buffer;
            self.pair.enqueue_with_shared(self.stage,rows,inputs,Some(shared_output),stream,
                ||shared.enqueue(inputs[0],shared_output,rows,stream))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::v41_memory::device::{Allocation,Device,Stream};
    use anyhow::{ensure,Result};
    use ds41rt_ffi::NativeLibrary;

    fn bf16(value:f32)->u16 {
        let bits=value.to_bits();
        ((bits.wrapping_add(0x7fff+((bits>>16)&1)))>>16) as u16
    }
    fn f32_bf16(value:f32)->f32 {f32::from_bits(u32::from(bf16(value))<<16)}

    #[test]
    #[ignore = "requires native draft TP2 AOT and two CUDA GPUs"]
    fn cuda_dspark_tp2_interfaces_and_ordered_reduction()->Result<()> {
        let library=unsafe {NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)?};
        let mut comparisons=0;
        let mut rounding_distinguished=false;
        for id in [0,1] {
            let device=Device {library:&library,id};
            for capacity in [1,16,80,256,1024,4096] {
                let kernel=device.run(||library.v41_dspark_tp2_expert_kernel(capacity))?;
                let info=kernel.info();
                ensure!((info.role,info.experts,info.logical_intermediate,info.topk,info.input_dtype)
                    ==(4,128,1152,3,7),"draft TP2 native geometry differs");
                assert!(!kernel.accumulates_tokens());
                let backbone=device.run(||library.v41_tp2_expert_kernel(capacity))?;
                assert_eq!(backbone.info().role,3);
            }
            let reducer=library.v41_route_reducer()?;
            let stream=Stream::new(device)?;
            for rows in [1usize,7,16] {
                for (ranks,topk) in [(1usize,3usize),(2,3),(4,6)] {
                    let count=rows*5120;let plane_count=count*topk;
                    let planes=(0..ranks).map(|_|Allocation::new(device,plane_count*4)).collect::<Result<Vec<_>>>()?;
                    let shared=Allocation::new(device,count*2)?;
                    let output=Allocation::new(device,count*2)?;
                    let pointers=std::array::from_fn(|rank|planes.get(rank).map_or(std::ptr::null(),|p|p.buffer.ptr.cast::<f32>()));
                    for with_shared in [false,true] {
                        let shared_ptr=if with_shared {shared.buffer.ptr.cast::<u16>()} else {std::ptr::null_mut()};
                        let graph=device.run(||unsafe {
                            library.cuda_graph_begin_capture(stream.raw)?;
                            let queued=reducer.launch(pointers,shared_ptr,output.buffer.ptr.cast(),rows as u32,
                                ranks as u32,topk as u32,stream.raw);
                            let captured=library.cuda_graph_end_capture(stream.raw);
                            match (queued,captured) {
                                (Ok(()),Ok(graph))=>Ok(graph),
                                (Err(error),Ok(graph))=>{library.cuda_graph_exec_destroy(graph)?;Err(error)},
                                (Err(error),Err(_)) | (Ok(()),Err(error))=>Err(error),
                            }
                        })?;
                        let checked=(||->Result<()> {
                            for seed in [0usize,3,11] {
                                let values:Vec<Vec<f32>>=(0..ranks).map(|rank|(0..plane_count).map(|i|
                                    (((i*17+rank*13+seed)%43) as f32-21.0)*0.047+rank as f32*0.003).collect()).collect();
                                for (plane,values) in planes.iter().zip(&values) {
                                    let bytes:Vec<u8>=values.iter().flat_map(|v|v.to_ne_bytes()).collect();
                                    device.run(||library.copy_h2d(plane.buffer,&bytes))?;
                                }
                                let shared_values:Vec<f32>=(0..count).map(|i|f32_bf16(((i+seed)%13) as f32*0.031)).collect();
                                let bytes:Vec<u8>=shared_values.iter().flat_map(|&v|bf16(v).to_ne_bytes()).collect();
                                device.run(||library.copy_h2d(shared.buffer,&bytes))?;
                                device.run(||unsafe {library.cuda_graph_launch(graph,stream.raw)})?;
                                stream.drain()?;
                                let mut actual=vec![0;count*2];device.run(||library.copy_d2h(&mut actual,output.buffer))?;
                                for index in 0..count {
                                    let (row,col)=(index/5120,index%5120);let mut total=0f32;let mut wrong=0f32;
                                    for route in 0..topk {
                                        let offset=(row*topk+route)*5120+col;
                                        let mut sum=values[0][offset];let mut rounded=f32_bf16(sum);
                                        for rank in 1..ranks {sum+=values[rank][offset];rounded+=f32_bf16(values[rank][offset]);}
                                        total+=f32_bf16(sum);wrong+=rounded;
                                    }
                                    if with_shared {total+=shared_values[index];wrong+=shared_values[index];}
                                    if ranks==2 && bf16(total)!=bf16(wrong) {rounding_distinguished=true;}
                                    let got=u16::from_ne_bytes([actual[index*2],actual[index*2+1]]);
                                    ensure!(got==bf16(total),"route reduction differs: gpu={id} rows={rows} ranks={ranks} shared={with_shared} seed={seed} index={index}");
                                }
                                comparisons+=1;
                            }
                            Ok(())
                        })();
                        device.run(||unsafe {library.cuda_graph_exec_destroy(graph)})?;
                        checked?;
                    }
                }
            }
        }
        ensure!(rounding_distinguished,"fixture did not distinguish premature rank rounding");
        eprintln!("draft/backbone metadata coexist; {comparisons} changed-input graph reductions exact");
        Ok(())
    }
}
