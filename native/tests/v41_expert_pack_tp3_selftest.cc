// Exact host-reference check for the native expert packer at the Spark TP3
// local intermediate (768). This is a plain CUDA representation transform, so
// it runs on any supported device (SM120 RTX included); it does not need the
// SM121 AOT expert export.
//
// The host reference re-implements the N256/K128 lane-major transform from
// native/cuda/kernels/v41_expert_pack.cu and every output byte is compared, so
// a wrong tile/row/column mapping or a wrong packed extent fails loudly. Also
// checks the accepted-extent list and byte sizes (32-scale alignment, 128
// storage padding).
//
// Requires a CUDA device; exits 77 (ctest SKIP) when none is present.
#include "ds41rt_v41_experts.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kIntermediate = 768;

void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "v41 expert pack tp3 selftest: %s\n", message);
    std::exit(1);
  }
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "v41 expert pack tp3 selftest: %s failed: %s\n",
                 what, cudaGetErrorString(status));
    std::exit(1);
  }
}

// Deterministic non-uniform payload; any byte-level mapping error changes it.
uint8_t pattern(uint64_t index, uint32_t salt) {
  uint64_t value = index * 0x9e3779b97f4a7c15ull + uint64_t(salt) * 0x100000001b3ull;
  value ^= value >> 29;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 32;
  return static_cast<uint8_t>(value & 0xffu);
}

// Direct port of the device `pack<Gated,Scales>` index math.
std::vector<uint8_t> reference_pack(bool gated, bool scales, const uint8_t* first,
    const uint8_t* second, uint32_t n, uint32_t k, uint32_t n_pad, uint64_t count) {
  const uint32_t k_tiles = (k + 127) / 128;
  const uint32_t tile_elements = scales ? 1024 : 4096;
  const uint32_t divisor = scales ? 32 : 8;
  std::vector<uint8_t> out(size_t(count) * (scales ? 1 : 4), 0);
  for (uint64_t index = 0; index < count; ++index) {
    const uint64_t tile = index / tile_elements;
    const uint32_t local = static_cast<uint32_t>(index % tile_elements);
    const uint32_t nt = static_cast<uint32_t>(tile / k_tiles);
    const uint32_t kt = static_cast<uint32_t>(tile % k_tiles);
    uint32_t row = 0, col = 0;
    if (scales) {
      row = nt * 256 + local / 4;
      col = kt * 4 + local % 4;
    } else {
      const uint32_t n8i = local & 3;
      const uint32_t combined = (local >> 2) & 31;
      const uint32_t n8c = (local >> 7) & 7;
      const uint32_t k32 = local >> 10;
      row = nt * 256 + n8c * 32 + n8i * 8 + (combined >> 2);
      col = kt * 16 + k32 * 4 + (combined & 3);
    }
    const uint8_t* source = first;
    if (gated && row >= n_pad) {
      row -= n_pad;
      source = second;
    }
    const bool valid = row < n && col < k / divisor;
    if (scales) {
      out[index] = valid ? source[uint64_t(row) * (k / divisor) + col] : 0;
    } else {
      uint32_t value = 0;
      if (valid)
        std::memcpy(&value, source + (uint64_t(row) * (k / divisor) + col) * 4, 4);
      out[size_t(index) * 4 + 0] = static_cast<uint8_t>(value & 0xffu);
      out[size_t(index) * 4 + 1] = static_cast<uint8_t>((value >> 8) & 0xffu);
      out[size_t(index) * 4 + 2] = static_cast<uint8_t>((value >> 16) & 0xffu);
      out[size_t(index) * 4 + 3] = static_cast<uint8_t>((value >> 24) & 0xffu);
    }
  }
  return out;
}

std::vector<uint8_t> make_source(size_t bytes, uint32_t salt) {
  std::vector<uint8_t> source(bytes);
  for (size_t i = 0; i < bytes; ++i) source[i] = pattern(i, salt);
  return source;
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    std::printf("v41 expert pack tp3 selftest: no CUDA device, skipping\n");
    return 77;
  }
  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  // Accepted extents and exact byte model for TP3 (padded == 768, no padding).
  uint64_t sizes[4] = {};
  require(ds41rt_v41_expert_packed_sizes(kIntermediate, sizes) == cudaSuccess,
          "packed sizes rejected 768");
  require(sizes[0] == uint64_t(kIntermediate) * kHidden, "W13 extent");
  require(sizes[1] == uint64_t(kIntermediate) * kHidden / 16, "S13 extent");
  require(sizes[2] == uint64_t(kHidden) * kIntermediate / 2, "W2 extent");
  require(sizes[3] == uint64_t(kHidden) * kIntermediate / 32, "S2 extent");
  for (uint32_t rejected : {0u, 1u, 640u, 577u, 800u, 2303u, 4096u})
    require(ds41rt_v41_expert_packed_sizes(rejected, sizes) != cudaSuccess,
            "packed sizes accepted an unsupported extent");

  const uint64_t weight_bytes = uint64_t(kIntermediate) * kHidden / 2;
  const uint64_t scale_bytes = weight_bytes / 16;
  const std::vector<uint8_t> w1 = make_source(weight_bytes, 1);
  const std::vector<uint8_t> w3 = make_source(weight_bytes, 2);
  const std::vector<uint8_t> w2 = make_source(weight_bytes, 3);
  const std::vector<uint8_t> s1 = make_source(scale_bytes, 4);
  const std::vector<uint8_t> s3 = make_source(scale_bytes, 5);
  const std::vector<uint8_t> s2 = make_source(scale_bytes, 6);

  const uint8_t* sources[6] = {w1.data(), w3.data(), w2.data(),
                               s1.data(), s3.data(), s2.data()};
  uint8_t* destinations[4] = {};
  for (int i = 0; i < 4; ++i)
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&destinations[i]), sizes[i]),
               "cudaMalloc destination");

  const std::vector<uint8_t> expected_w13 = reference_pack(
      /*gated=*/true, /*scales=*/false, w3.data(), w1.data(), kIntermediate, kHidden,
      kIntermediate, sizes[0] / 4);
  const std::vector<uint8_t> expected_s13 = reference_pack(
      /*gated=*/true, /*scales=*/true, s3.data(), s1.data(), kIntermediate, kHidden,
      kIntermediate, sizes[1]);
  const std::vector<uint8_t> expected_w2 = reference_pack(
      /*gated=*/false, /*scales=*/false, w2.data(), nullptr, kHidden, kIntermediate, 0,
      sizes[2] / 4);
  const std::vector<uint8_t> expected_s2 = reference_pack(
      /*gated=*/false, /*scales=*/true, s2.data(), nullptr, kHidden, kIntermediate, 0,
      sizes[3]);

  require(ds41rt_v41_pack_expert_async(sources, destinations, kIntermediate, nullptr) ==
              cudaSuccess,
          "pack launch failed");
  check_cuda(cudaStreamSynchronize(nullptr), "synchronize");

  const std::vector<uint8_t>* expected[4] = {&expected_w13, &expected_s13, &expected_w2,
                                             &expected_s2};
  for (int i = 0; i < 4; ++i) {
    std::vector<uint8_t> actual(sizes[i]);
    check_cuda(cudaMemcpy(actual.data(), destinations[i], sizes[i], cudaMemcpyDeviceToHost),
               "copy destination");
    require(actual.size() == expected[i]->size(), "destination extent mismatch");
    for (size_t byte = 0; byte < actual.size(); ++byte)
      if (actual[byte] != (*expected[i])[byte]) {
        std::fprintf(stderr,
                     "destination %d byte %zu actual=%02x expected=%02x\n", i, byte,
                     actual[byte], (*expected[i])[byte]);
        std::exit(1);
      }
    check_cuda(cudaFree(destinations[i]), "cudaFree destination");
  }

  std::printf("v41 expert pack tp3 selftest: ok (intermediate=%u, %llu+%llu+%llu+%llu bytes)\n",
              kIntermediate, static_cast<unsigned long long>(sizes[0]),
              static_cast<unsigned long long>(sizes[1]),
              static_cast<unsigned long long>(sizes[2]),
              static_cast<unsigned long long>(sizes[3]));
  return 0;
}
