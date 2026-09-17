#include "ds41rt_v41_peer_copy.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>
static void Check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
  int devices=0;Check(cudaGetDeviceCount(&devices));
  if(devices<2) { std::cout<<"Requires two CUDA devices"<<std::endl;return 77; }
  for(int device=0;device<2;++device) { Check(cudaSetDevice(device));Check(cudaDeviceEnablePeerAccess(1-device,0)); }
  int checks=0;
  for(int source_device=0;source_device<2;++source_device) {
    const int destination_device=1-source_device;
    Check(cudaSetDevice(source_device));
    for(size_t bytes: {size_t(1),3ul,4ul,8ul,15ul,16ul,257ul,65536ul})for(size_t offset: {0ul,1ul,4ul}) {
      const size_t capacity=bytes+32;
      uint8_t *source,*destination;
      Check(cudaMalloc(&source,capacity));
      std::vector<uint8_t> host(capacity),actual(capacity);
      for(size_t i=0;i<capacity;++i)host[i]=uint8_t(i*17+19);
      Check(cudaMemcpy(source,host.data(),capacity,cudaMemcpyHostToDevice));
      Check(cudaSetDevice(destination_device));
      Check(cudaMalloc(&destination,capacity));Check(cudaMemset(destination,0xcd,capacity));
      Check(static_cast<cudaError_t>(ds41rt_v41_peer_copy_initialize()));
      cudaStream_t stream;Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
      if(ds41rt_v41_peer_copy_async(destination,source,0,stream)!=cudaErrorInvalidValue ||
         ds41rt_v41_peer_copy_async(destination,destination,bytes,stream)!=cudaErrorInvalidValue)
        throw std::runtime_error("invalid peer copy accepted");
      cudaGraph_t graph;cudaGraphExec_t executable;
      Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
      Check(static_cast<cudaError_t>(ds41rt_v41_peer_copy_async(destination+offset,source+offset,bytes,stream)));
      Check(cudaStreamEndCapture(stream,&graph));Check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
      for(int replay=0;replay<2;++replay) {
        Check(cudaGraphLaunch(executable,stream));Check(cudaStreamSynchronize(stream));
        Check(cudaMemcpy(actual.data(),destination,capacity,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<capacity;++i)
          if(actual[i]!=(i>=offset && i<offset+bytes?host[i]:uint8_t(0xcd)))
            throw std::runtime_error("peer copy payload or guard mismatch");
        Check(cudaSetDevice(source_device));
        for(auto& value:host)value^=0x55;
        Check(cudaMemcpy(source,host.data(),capacity,cudaMemcpyHostToDevice));
        Check(cudaSetDevice(destination_device));
      }
      Check(cudaGraphExecDestroy(executable));Check(cudaGraphDestroy(graph));
      Check(cudaStreamDestroy(stream));Check(cudaFree(destination));
      Check(cudaSetDevice(source_device));Check(cudaFree(source));++checks;
    }
  }

  for(int owner=0;owner<2;++owner)for(size_t width:{7ul,12ul,32ul,32768ul})for(size_t rows:{1ul,7ul,64ul}) {
    const int peer=1-owner;const size_t bytes=rows*width*2;
    uint8_t *source,*joined,*half[2];cudaStream_t stream[2];
    Check(cudaSetDevice(owner));Check(cudaMalloc(&source,bytes));Check(cudaMalloc(&joined,bytes+32));
    Check(cudaStreamCreateWithFlags(&stream[owner],cudaStreamNonBlocking));
    Check(static_cast<cudaError_t>(ds41rt_v41_peer_copy_initialize()));
    std::vector<uint8_t> input(bytes),output(bytes+32);
    for(size_t i=0;i<bytes;++i)input[i]=uint8_t((i%width)*17+(i/width)*31+9);
    Check(cudaMemcpy(source,input.data(),bytes,cudaMemcpyHostToDevice));Check(cudaMemset(joined,0xcd,bytes+32));
    Check(cudaSetDevice(peer));Check(cudaMalloc(&half[0],rows*width));Check(cudaMalloc(&half[1],rows*width));
    Check(cudaStreamCreateWithFlags(&stream[peer],cudaStreamNonBlocking));
    Check(static_cast<cudaError_t>(ds41rt_v41_peer_copy_initialize()));
    if(ds41rt_v41_peer_copy_rows_async(half[0],source,width,rows,width-1,2*width,stream[peer])!=cudaErrorInvalidValue ||
       ds41rt_v41_peer_copy_rows_async(half[0],source,width,4097,width,2*width,stream[peer])!=cudaErrorInvalidValue)
      throw std::runtime_error("invalid pitched span accepted");
    cudaGraph_t graph;cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(stream[peer],cudaStreamCaptureModeThreadLocal));
    for(int rank=0;rank<2;++rank)Check(static_cast<cudaError_t>(ds41rt_v41_peer_copy_rows_async(
        half[rank],source+rank*width,width,rows,width,2*width,stream[peer])));
    Check(cudaStreamEndCapture(stream[peer],&graph));Check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int replay=0;replay<2;++replay) {
      Check(cudaGraphLaunch(executable,stream[peer]));Check(cudaStreamSynchronize(stream[peer]));
      Check(cudaSetDevice(owner));
      for(int rank=0;rank<2;++rank)Check(static_cast<cudaError_t>(ds41rt_v41_peer_copy_rows_async(
          joined+rank*width,half[rank],width,rows,2*width,width,stream[owner])));
      Check(cudaStreamSynchronize(stream[owner]));Check(cudaMemcpy(output.data(),joined,bytes+32,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<bytes+32;++i)if(output[i]!=(i<bytes?input[i]:uint8_t(0xcd)))
        throw std::runtime_error("pitched split/gather payload or guard mismatch");
      for(auto& value:input)value^=0x5a;
      Check(cudaMemcpy(source,input.data(),bytes,cudaMemcpyHostToDevice));Check(cudaSetDevice(peer));
    }
    Check(cudaGraphExecDestroy(executable));Check(cudaGraphDestroy(graph));
    Check(cudaFree(half[0]));Check(cudaFree(half[1]));Check(cudaStreamDestroy(stream[peer]));
    Check(cudaSetDevice(owner));
    uint8_t* local;Check(cudaMalloc(&local,rows*width));
    Check(static_cast<cudaError_t>(ds41rt_v41_peer_copy_rows_async(local,source+width,width,rows,width,2*width,stream[owner])));
    Check(cudaStreamSynchronize(stream[owner]));
    std::vector<uint8_t> local_output(rows*width);Check(cudaMemcpy(local_output.data(),local,rows*width,cudaMemcpyDeviceToHost));
    for(size_t row=0;row<rows;++row)for(size_t col=0;col<width;++col)
      if(local_output[row*width+col]!=input[row*2*width+width+col])throw std::runtime_error("local head split differs");
    Check(cudaFree(local));Check(cudaFree(source));Check(cudaFree(joined));Check(cudaStreamDestroy(stream[owner]));
    ++checks;
  }
  std::cout<<"Peer copy alignment, guards and replay checks passed: "<<checks<<std::endl;
}
