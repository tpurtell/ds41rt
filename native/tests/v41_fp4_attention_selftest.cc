#include "ds41rt_v41_sparse_attention.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  void* allocation = nullptr;
  uint8_t* data = nullptr;
  size_t bytes;
  explicit Buffer(size_t size, bool unaligned = false) : bytes(size) {
    Check(cudaMalloc(&allocation, size + size_t(unaligned)));
    data = static_cast<uint8_t*>(allocation) + size_t(unaligned);
  }
  ~Buffer() { cudaFree(allocation); }
  Buffer(const Buffer&) = delete;
  void Fill(int value) { Check(cudaMemset(data, value, bytes)); }
  template<class T> void Copy(const std::vector<T>& values) {
    if (values.size()*sizeof(T) != bytes) throw std::runtime_error("test buffer size mismatch");
    Check(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice));
  }
};
static uint16_t BFloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

int main() {
  Check(static_cast<cudaError_t>(ds41rt_v41_sparse_attention_initialize()));
  Check(static_cast<cudaError_t>(ds41rt_v41_sparse_attention_heads32_initialize()));
  cudaStream_t test_stream;
  Check(cudaStreamCreateWithFlags(&test_stream,cudaStreamNonBlocking));
  int checks = 0;
  for(int format : {0,1,2}) for (int rows : {1, 6, 128, 256}) {
    Buffer query(size_t(rows)*64*512*2), output(query.bytes), sink(64*4);
    Buffer window(128*512, !(rows==6 && format==2)), window_scales(128*16, !(rows==6 && format==2));
    Buffer proposal(size_t(rows)*512, !(rows==6 && format==2)), proposal_scales(size_t(rows)*16, !(rows==6 && format==2));
    Buffer source(768*(format==2?256:512), !(rows==6 && format==2)), source_scales(768*(format==2?32:16), !(rows==6 && format==2));
    Buffer private_source(3*(format==2?256:512), !(rows==6 && format==2)), private_scales(3*(format==2?32:16), !(rows==6 && format==2));
    Buffer window_end(8), source_end(8), pages(8), metadata(size_t(rows)*10*8);
    Buffer selected(size_t(rows)*512*4), bounds(size_t(rows)*8);
    Buffer scratch(size_t(rows)*10*64*514*4);
    query.Fill(0); sink.Fill(0); window.Fill(0); proposal.Fill(0);
    window_scales.Fill(127); proposal_scales.Fill(127);
    source.Fill(format==2?0x22:0x38); private_source.Fill(format==2?0x22:0x38); // Quantized ones.
    source_scales.Fill(format==2?0x38:127); private_scales.Fill(format==2?0x38:127); // Scale one.
    window_end.Copy<uint64_t>({2048}); source_end.Copy<uint64_t>({512});
    pages.Copy<uint32_t>({1, 0});
    ds41rt_v41_sparse_kv_t view{};
    view.values[0]=window.data; view.values[1]=proposal.data;
    view.values[2]=source.data; view.values[3]=private_source.data;
    view.scales[0]=window_scales.data; view.scales[1]=proposal_scales.data;
    view.scales[2]=source_scales.data; view.scales[3]=private_scales.data;
    view.window_end=reinterpret_cast<const uint64_t*>(window_end.data);
    view.source_end=reinterpret_cast<const uint64_t*>(source_end.data);
    view.pages=reinterpret_cast<const uint32_t*>(pages.data);
    view.window_proposal_capacity=rows; view.source_capacity=768;
    view.source_proposal_capacity=3; view.page_stride=2; view.compressed=format;
    for (int parts : {0, 3, 10}) for (int begin : {-1, 0, 2048, 2049}) for (int mode : {0, 1, 2}) {
      std::vector<uint64_t> meta;
      std::vector<int32_t> ids(size_t(rows)*512, -1);
      for (int row=0; row<rows; ++row) {
        const uint64_t fields[]={2048,0,uint64_t(rows),uint64_t(2048+row),0,
            uint64_t(mode?514:512),512,uint64_t(mode?2:0),uint64_t(mode==1?1:0),uint64_t(mode==2?2:1)};
        meta.insert(meta.end(), fields, fields+10);
        for (int key=0; key<(mode?2:512); ++key) ids[size_t(row)*512+key]=(mode?512:0)+key;
      }
      metadata.Copy(meta); selected.Copy(ids);
      bounds.Copy(std::vector<uint64_t>(rows, begin<0?0:uint64_t(begin)));
      output.Fill(0xcd);
      const auto* q=reinterpret_cast<const uint16_t*>(query.data);
      auto* out=reinterpret_cast<uint16_t*>(output.data);
      const auto* m=reinterpret_cast<const uint64_t*>(metadata.data);
      const auto* s=reinterpret_cast<const int32_t*>(selected.data);
      const auto* sk=reinterpret_cast<const float*>(sink.data);
      int status;
      if (begin>=0) status=ds41rt_v41_sparse_attention_bounded(q,sk,m,s,out,rows,0,&view,nullptr,
          reinterpret_cast<const uint64_t*>(bounds.data),parts?reinterpret_cast<float*>(scratch.data):nullptr,
          parts?scratch.bytes:0,parts?parts:1);
      else if (parts) status=ds41rt_v41_sparse_attention_split(q,sk,m,s,out,rows,0,&view,nullptr,
          reinterpret_cast<float*>(scratch.data),scratch.bytes,parts);
      else status=ds41rt_v41_sparse_attention(q,sk,m,s,out,rows,0,&view,nullptr);
      Check(static_cast<cudaError_t>(status));
      Check(cudaDeviceSynchronize());
      std::vector<uint16_t> actual(output.bytes/2);
      Check(cudaMemcpy(actual.data(), output.data, output.bytes, cudaMemcpyDeviceToHost));
      for (int row=0; row<rows; ++row) {
        // Zero queries/sink give equal attention weights. Window values are
        // zero, source values are one, and the sink contributes one to the denominator.
        int windows=begin==2048?std::min(row+1,128):128;
        int sources=format?(mode?2:512):0;
        uint16_t expected=begin==2049?0:BFloat16(float(sources)/float(sources+windows+1));
        for (size_t col=0; col<64*512; ++col)
          if (actual[size_t(row)*64*512+col]!=expected)
            throw std::runtime_error("attention disagrees with closed-form reference");
      }
      // Compare compact local heads with the full-head kernel using nonuniform
      // queries and sinks. This catches row/head strides and rank-one sink offsets.
      std::vector<uint16_t> varied_query(query.bytes/2);
      std::vector<float> varied_sink(64);
      for(size_t i=0;i<varied_query.size();++i)
        varied_query[i]=BFloat16(float(int((i*17+i/512)%31)-15)/64.0f);
      for(int head=0;head<64;++head)varied_sink[head]=float(head-32)/16.0f;
      query.Copy(varied_query);sink.Copy(varied_sink);
      const auto* wb=reinterpret_cast<const uint64_t*>(bounds.data);
      Check(static_cast<cudaError_t>(ds41rt_v41_sparse_attention_bounded(q,sk,m,s,out,rows,0,
          &view,nullptr,wb,parts?reinterpret_cast<float*>(scratch.data):nullptr,
          parts?scratch.bytes:0,parts?parts:1)));
      Check(cudaMemcpy(actual.data(),output.data,output.bytes,cudaMemcpyDeviceToHost));
      Buffer local_query(size_t(rows)*32*512*2),local_output(local_query.bytes),local_sink(32*4);
      Buffer local_scratch(size_t(rows)*(parts?parts:1)*32*514*4);
      for(int rank=0;rank<2;++rank) {
        std::vector<uint16_t> compact(local_query.bytes/2);
        for(int row=0;row<rows;++row)
          std::copy_n(varied_query.data()+(size_t(row)*64+rank*32)*512,32*512,
              compact.data()+size_t(row)*32*512);
        local_query.Copy(compact);
        local_sink.Copy(std::vector<float>(varied_sink.begin()+rank*32,varied_sink.begin()+(rank+1)*32));
        local_output.Fill(0xcd);local_scratch.Fill(0xcd);
        auto launch=[&](uint64_t bytes) {
          return ds41rt_v41_sparse_attention_heads32_bounded(
              reinterpret_cast<const uint16_t*>(local_query.data),reinterpret_cast<const float*>(local_sink.data),
              m,s,reinterpret_cast<uint16_t*>(local_output.data),rows,0,&view,test_stream,wb,
              parts?reinterpret_cast<float*>(local_scratch.data):nullptr,bytes,parts?parts:1);
        };
        if(parts && launch(local_scratch.bytes-1)!=cudaErrorInvalidValue)
          throw std::runtime_error("compact attention accepted undersized scratch");
        Check(static_cast<cudaError_t>(launch(parts?local_scratch.bytes:0)));
        if(rows==6 && begin==0 && mode==0) {
          Check(cudaStreamSynchronize(test_stream));
          cudaGraph_t graph;cudaGraphExec_t executable;
          Check(cudaStreamBeginCapture(test_stream,cudaStreamCaptureModeThreadLocal));
          Check(static_cast<cudaError_t>(launch(parts?local_scratch.bytes:0)));
          Check(cudaStreamEndCapture(test_stream,&graph));
          Check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
          bounds.Copy(std::vector<uint64_t>(rows,2049));
          Check(cudaGraphLaunch(executable,test_stream));Check(cudaStreamSynchronize(test_stream));
          Check(cudaMemcpy(compact.data(),local_output.data,local_output.bytes,cudaMemcpyDeviceToHost));
          if(std::any_of(compact.begin(),compact.end(),[](uint16_t v){return v!=0;}))
            throw std::runtime_error("compact replay ignored updated invalid bounds");
          bounds.Copy(std::vector<uint64_t>(rows,0));
          Check(cudaGraphLaunch(executable,test_stream));Check(cudaStreamSynchronize(test_stream));
          Check(cudaGraphExecDestroy(executable));Check(cudaGraphDestroy(graph));
        }
        Check(cudaStreamSynchronize(test_stream));
        Check(cudaMemcpy(compact.data(),local_output.data,local_output.bytes,cudaMemcpyDeviceToHost));
        for(int row=0;row<rows;++row)for(size_t col=0;col<32*512;++col)
          if(compact[size_t(row)*32*512+col]!=actual[(size_t(row)*64+rank*32)*512+col])
            throw std::runtime_error("compact TP2 attention disagrees with full-head output");
      }
      if(rows==6 && begin==0 && mode==0 && parts==3) {
        const int batch_parts=format?10:2;
        Buffer device_views(rows*sizeof(view)),bad_end(8);
        bad_end.Copy<uint64_t>({0});
        std::vector<ds41rt_v41_sparse_kv_t> views(rows,view);
        for(int row=1;row<rows;row+=2)
          views[row].window_end=reinterpret_cast<const uint64_t*>(bad_end.data);
        device_views.Copy(views);
        auto* dv=reinterpret_cast<const ds41rt_v41_sparse_kv_t*>(device_views.data);
        Buffer batch_scratch(size_t(rows)*batch_parts*64*514*4);
        auto* bs=reinterpret_cast<float*>(batch_scratch.data);
        Check(static_cast<cudaError_t>(ds41rt_v41_sparse_attention_batch_validate(
            q,sk,m,s,out,rows,views.data(),dv,wb,bs,batch_scratch.bytes,batch_parts,format)));
        auto batch_launch=ds41rt_v41_sparse_attention_batch;
        auto half_batch_launch=ds41rt_v41_sparse_attention_heads32_batch;
#ifdef DS41RT_TEST_ATTENTION_AOT
        if(format==2) {
          batch_launch=ds41rt_v41_sparse_attention_batch_aot;
          half_batch_launch=ds41rt_v41_sparse_attention_heads32_batch_aot;
        }
#endif
        Check(static_cast<cudaError_t>(batch_launch(
            q,sk,m,s,out,rows,dv,test_stream,wb,bs,batch_parts,format)));
        Check(cudaStreamSynchronize(test_stream));
        Check(cudaMemcpy(actual.data(),output.data,output.bytes,cudaMemcpyDeviceToHost));
        Buffer half_query(size_t(rows)*32*512*2),half_output(half_query.bytes),half_sink(128);
        Buffer half_scratch(batch_scratch.bytes/2);
        for(int rank=0;rank<2;++rank) {
          std::vector<uint16_t> compact(half_query.bytes/2);
          for(int row=0;row<rows;++row)
            std::copy_n(varied_query.data()+(size_t(row)*64+rank*32)*512,32*512,
                compact.data()+size_t(row)*32*512);
          half_query.Copy(compact);
          half_sink.Copy(std::vector<float>(varied_sink.begin()+rank*32,varied_sink.begin()+(rank+1)*32));
          auto* hq=reinterpret_cast<const uint16_t*>(half_query.data);
          auto* hs=reinterpret_cast<const float*>(half_sink.data);
          auto* ho=reinterpret_cast<uint16_t*>(half_output.data);
          auto* hp=reinterpret_cast<float*>(half_scratch.data);
          auto validate=[&](uint64_t bytes,const ds41rt_v41_sparse_kv_t* descriptors) {
            return ds41rt_v41_sparse_attention_heads32_batch_validate(hq,hs,m,s,ho,rows,
                views.data(),descriptors,wb,hp,bytes,batch_parts,format);
          };
          Check(static_cast<cudaError_t>(validate(half_scratch.bytes,dv)));
          if(validate(half_scratch.bytes-1,dv)!=cudaErrorInvalidValue ||
              validate(half_scratch.bytes,reinterpret_cast<const ds41rt_v41_sparse_kv_t*>(half_output.data))!=cudaErrorInvalidValue)
            throw std::runtime_error("compact batch validation missed scratch/descriptor overlap");
          cudaGraph_t graph;cudaGraphExec_t executable;
          Check(cudaStreamBeginCapture(test_stream,cudaStreamCaptureModeThreadLocal));
          Check(static_cast<cudaError_t>(half_batch_launch(
              hq,hs,m,s,ho,rows,dv,test_stream,wb,hp,batch_parts,format)));
          Check(cudaStreamEndCapture(test_stream,&graph));
          Check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
          for(int replay=0;replay<2;++replay) {
            Check(cudaGraphLaunch(executable,test_stream));Check(cudaStreamSynchronize(test_stream));
            Check(cudaMemcpy(compact.data(),half_output.data,half_output.bytes,cudaMemcpyDeviceToHost));
            for(int row=0;row<rows;++row)for(size_t col=0;col<32*512;++col) {
              const auto expected=replay?0:actual[(size_t(row)*64+rank*32)*512+col];
              if(compact[size_t(row)*32*512+col]!=expected)
                throw std::runtime_error("compact batch descriptor replay mismatch");
            }
            if(replay==0) {
              auto invalid=views;
              for(auto& v:invalid)v.window_end=reinterpret_cast<const uint64_t*>(bad_end.data);
              Check(static_cast<cudaError_t>(ds41rt_v41_sparse_attention_heads32_batch_validate(
                  hq,hs,m,s,ho,rows,invalid.data(),dv,wb,hp,half_scratch.bytes,batch_parts,format)));
              device_views.Copy(invalid);
            }
          }
          Check(cudaGraphExecDestroy(executable));Check(cudaGraphDestroy(graph));
          device_views.Copy(views);
        }
      }
      query.Fill(0);sink.Fill(0);
      ++checks;
    }
    std::cout << "PASS format=" << format << " rows=" << rows << std::endl;
  }
  Check(cudaStreamDestroy(test_stream));
  std::cout << "Attention closed-form and compact-head checks passed: " << checks << std::endl;
}
