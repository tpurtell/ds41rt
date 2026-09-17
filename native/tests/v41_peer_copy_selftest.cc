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
  std::cout<<"Peer copy alignment, guards and replay checks passed: "<<checks<<std::endl;
}
