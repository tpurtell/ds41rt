#include "ds41rt_native.h"
#include "ds41rt_v41_attention_ops.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
static void check(bool value) { if (!value) std::abort(); }
static void ok(ds41rt_status_t value) { check(value == DS41RT_STATUS_OK); }
int main() {
  for (int rows : {1, 40, 2048}) for (int heads : {1, 64}) for (int inverse : {0, 1}) {
    const size_t count = size_t(rows)*heads*512, bytes = count*2;
    std::vector<uint16_t> input(count + 1);
    for (size_t i=0; i<count; ++i) input[i] = uint16_t(0x3d00 + (i*17)%1024) | ((i%3 == 0) ? 0x8000 : 0);
    input[count] = 0x7bcd;
    std::vector<float> frequency(rows*64);
    for (size_t i=0; i<frequency.size(); i+=2) {
      const float angle = float(i%71)*0.071f;
      frequency[i] = std::cos(angle); frequency[i+1] = std::sin(angle);
    }
    ds41rt_device_buffer_t source{}, separate{}, inplace{}, freq{};
    ok(ds41rt_alloc_device_buffer(bytes+2, &source));
    ok(ds41rt_alloc_device_buffer(bytes+2, &separate));
    ok(ds41rt_alloc_device_buffer(bytes+2, &inplace));
    ok(ds41rt_alloc_device_buffer(frequency.size()*4, &freq));
    ok(ds41rt_copy_h2d(source, input.data(), bytes+2));
    ok(ds41rt_copy_h2d(separate, input.data(), bytes+2));
    ok(ds41rt_copy_h2d(inplace, input.data(), bytes+2));
    ok(ds41rt_copy_h2d(freq, frequency.data(), frequency.size()*4));
    void* stream=nullptr; ok(ds41rt_cuda_stream_create(&stream));
    check(ds41rt_v41_attention_rope(static_cast<uint16_t*>(source.ptr), static_cast<float*>(freq.ptr),
        static_cast<uint16_t*>(separate.ptr), rows, heads, inverse, stream) == 0);
    check(ds41rt_v41_attention_rope(static_cast<uint16_t*>(inplace.ptr), static_cast<float*>(freq.ptr),
        static_cast<uint16_t*>(inplace.ptr), rows, heads, inverse, stream) == 0);
    check(ds41rt_v41_attention_rope(static_cast<uint16_t*>(source.ptr), static_cast<float*>(freq.ptr),
        static_cast<uint16_t*>(source.ptr)+1, rows, heads, inverse, stream) != 0);
    ok(ds41rt_cuda_stream_synchronize(stream));
    std::vector<uint16_t> expected(count+1), actual(count+1);
    ok(ds41rt_copy_d2h(expected.data(), separate, bytes+2));
    ok(ds41rt_copy_d2h(actual.data(), inplace, bytes+2));
    check(actual == expected && actual[count] == input[count]);
    for(size_t vector=0; vector<size_t(rows)*heads; ++vector)
      for(size_t col=0; col<448; ++col) check(actual[vector*512+col] == input[vector*512+col]);
    ok(ds41rt_cuda_stream_destroy(stream));
    ok(ds41rt_free_device_buffer(&freq)); ok(ds41rt_free_device_buffer(&inplace));
    ok(ds41rt_free_device_buffer(&separate)); ok(ds41rt_free_device_buffer(&source));
  }
  std::cout << "in-place RoPE matches separate output exactly at 1/40/2048 rows; partial overlap rejected\n";
}
