// Exact host-reference check for the native expert packer across EVERY native
// geometry, including the padded TP4 shard and the TP4 four-rank band partition.
//
// The packer is a plain CUDA representation transform, so this runs on any
// supported device (SM120 RTX included) and does not need the SM121 AOT export.
//
// Two independent properties are proven per geometry:
//
// 1. Value fidelity. The device output is compared BYTE FOR BYTE against a host
//    reference that is a direct port of the device `pack<Gated,Scales>` index
//    math, over that geometry's own sliced inputs.
// 2. Storage padding and partition. The packed layout is swizzled, so the
//    padded region is decoded back to (source row, source K word) with the same
//    index math instead of being treated as dense rows:
//      * N-axis planes (W13 payload slot 0, S13 scales slot 1) are gated: the
//        logical row is `half_row = plane_row - kernel_intermediate` for the
//        gate half, and padded words are exactly `half_row >= intermediate`.
//      * K-axis planes (W2 payload slot 2, S2 scales slot 3) are non-gated: the
//        K word is `col`, and padded words are exactly `col >= intermediate/d`.
//    Padded words must be zero AND the padded word count must equal the exact
//    arithmetic total, so a silently truncated region fails too.
// 3. The four 576-wide band cases (`band4/0..3`, offsets 0/576/1152/1728) are
//    the TP4 four-rank logical partition. Each packed word is re-decoded and
//    checked against the band's own source word, marked in the un-padded global
//    layout, and the union of all four bands must cover the full logical extent
//    exactly once with no word claimed twice: `2 * 2304 * 640` W13 words (the up
//    half then the gate half) and `5120 * 288` W2 words. No per-rank slab is
//    concatenated to a full pack: the partition is asserted in logical
//    coordinates, because the swizzle tile origin is rank-local.
//
// Cases (name: per-rank logical intermediate, source offset of this rank's band):
//   tp6      384   offset 0
//   tp3      768   offset 0
//   tp2      1152  offset 0
//   tp4      576   offset 0      (pads to 640; the production Spark TP4 shard)
//   full     2304  offset 0
//   band4/0  576   offset 0
//   band4/1  576   offset 576
//   band4/2  576   offset 1152
//   band4/3  576   offset 1728
//
// Requires a CUDA device; exits 77 (ctest SKIP) when none is present, or 1 when
// DS41RT_REQUIRE_CUDA is set (so a scheduled gate cannot pass by skipping).
#include "ds41rt_v41_experts.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kFullIntermediate = 2304;

struct Case {
  const char* name;
  uint32_t intermediate;  // per-rank LOGICAL width passed to the packer
  uint32_t offset;        // source row/column offset of this rank's band
};

const Case kCases[] = {
    {"tp6", 384, 0},
    {"tp3", 768, 0},
    {"tp2", 1152, 0},
    {"tp4", 576, 0},
    {"full", 2304, 0},
    {"band4/0", 576, 0},
    {"band4/1", 576, 576},
    {"band4/2", 576, 1152},
    {"band4/3", 576, 1728},
};

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "v41 expert pack geometry selftest: %s\n",
                 message.c_str());
    std::exit(1);
  }
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "v41 expert pack geometry selftest: %s failed: %s\n",
                 what, cudaGetErrorString(status));
    std::exit(1);
  }
}

uint8_t pattern(uint64_t index, uint32_t salt) {
  uint64_t value = index * 0x9e3779b97f4a7c15ull + uint64_t(salt) * 0x100000001b3ull;
  value ^= value >> 29;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 32;
  return static_cast<uint8_t>(value & 0xffu);
}

std::vector<uint8_t> make_source(size_t bytes, uint32_t salt) {
  std::vector<uint8_t> source(bytes);
  for (size_t i = 0; i < bytes; ++i) source[i] = pattern(i, salt);
  return source;
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

// w1/w3 are [intermediate, hidden] FP4; s1/s3 are [intermediate, hidden/32].
std::vector<uint8_t> band_rows(const std::vector<uint8_t>& full, uint32_t offset,
    uint32_t rows, uint32_t row_bytes) {
  const uint64_t start = uint64_t(offset) * row_bytes;
  return std::vector<uint8_t>(full.begin() + start,
                              full.begin() + start + uint64_t(rows) * row_bytes);
}

// w2 is [hidden, intermediate] FP4; s2 is [hidden, intermediate/32]. Both are
// sliced on their LAST axis, which is why a band offset is divided by the
// element rate (2 for FP4 bytes, 32 for K32 scales).
std::vector<uint8_t> band_cols(const std::vector<uint8_t>& full, uint32_t rows,
    uint32_t row_bytes_full, uint32_t col_start, uint32_t cols) {
  std::vector<uint8_t> out(uint64_t(rows) * cols);
  for (uint32_t row = 0; row < rows; ++row)
    std::memcpy(out.data() + uint64_t(row) * cols,
                full.data() + uint64_t(row) * row_bytes_full + col_start, cols);
  return out;
}

// Flat payload word index -> (local row in the 256-row tile, K word) exactly as
// the device derives them for the 4-byte payload path.
struct PayloadCoord {
  uint32_t row;
  uint32_t word;
};

// K-relative word inside a tile: `k32*4 + (combined & 3)`; the caller adds
// `kt*16` for the full packed column.
uint32_t tile_word(uint32_t idx) {
  const uint32_t combined = (idx >> 2) & 31;
  const uint32_t k32 = idx >> 10;
  return k32 * 4 + (combined & 3);
}

PayloadCoord payload_coord(uint32_t idx) {
  const uint32_t n8i = idx & 3;
  const uint32_t combined = (idx >> 2) & 31;
  const uint32_t n8c = (idx >> 7) & 7;
  return PayloadCoord{n8c * 32 + n8i * 8 + (combined >> 2), tile_word(idx)};
}

// Mark the un-padded logical W13 words this rank's packed output proves it owns
// and verify each against the band's own source word. `covered` is indexed by
// the global logical word id; a second write or a missing word is a partition
// failure.
void claim_w13(const uint32_t* packed, const uint32_t* w3, const uint32_t* w1,
               uint32_t intermediate, uint32_t offset,
               std::vector<uint8_t>& covered) {
  const uint32_t kernel_intermediate = (intermediate + 127) / 128 * 128;
  const uint32_t n_tiles = (2 * kernel_intermediate + 255) / 256;
  const uint32_t k_tiles = (kHidden + 127) / 128;
  const uint32_t words_per_row = kHidden / 8;
  for (uint32_t nt = 0; nt < n_tiles; ++nt)
    for (uint32_t kt = 0; kt < k_tiles; ++kt)
      for (uint32_t idx = 0; idx < 4096; ++idx) {
        const PayloadCoord coord = payload_coord(idx);
        const uint32_t plane_row = nt * 256 + coord.row;
        const uint32_t col = kt * 16 + coord.word;
        const bool gate = plane_row >= kernel_intermediate;
        const uint32_t half_row = gate ? plane_row - kernel_intermediate : plane_row;
        if (half_row >= intermediate) continue;  // padded N row
        if (col >= words_per_row) continue;      // K tail (never occurs at 384/576/etc.)
        // The logical W13 matrix is [2*intermediate, hidden]: the up half (W3)
        // occupies rows [0, intermediate) and the gate half (W1) rows
        // [intermediate, 2*intermediate). The destination's plane rows [0,640)
        // are W3 and [640,1280) are W1, so the halves must stay separate here or
        // the two logical halves collide in the coverage map.
        const uint32_t global_row =
            (gate ? kFullIntermediate : 0) + offset + half_row;
        require(global_row < 2 * kFullIntermediate,
                "W13 band maps past the official intermediate");
        const uint32_t global_word = global_row * words_per_row + col;
        require(global_word < covered.size(), "W13 coverage index out of range");
        require(covered[global_word] == 0, "W13 word claimed twice");
        const uint32_t expected = (gate ? w1 : w3)[uint64_t(half_row) * words_per_row + col];
        require(packed[(nt * k_tiles + kt) * 4096 + idx] == expected,
                "W13 packed word does not match its band source");
        covered[global_word] = 1;
      }
}

// Same for the W2 down-projection: the K axis is the local intermediate, so the
// padded words are `col >= intermediate/8`, and `row` is the HIDDEN row.
void claim_w2(const uint32_t* packed, const uint32_t* w2, uint32_t intermediate,
              uint32_t offset, std::vector<uint8_t>& covered) {
  const uint32_t kernel_intermediate = (intermediate + 127) / 128 * 128;
  const uint32_t n_tiles = (kHidden + 255) / 256;
  const uint32_t k_tiles = (kernel_intermediate + 127) / 128;
  const uint32_t local_k_words = intermediate / 8;
  const uint32_t full_k_words = kFullIntermediate / 8;
  for (uint32_t nt = 0; nt < n_tiles; ++nt)
    for (uint32_t kt = 0; kt < k_tiles; ++kt)
      for (uint32_t idx = 0; idx < 4096; ++idx) {
        const PayloadCoord coord = payload_coord(idx);
        const uint32_t col = kt * 16 + coord.word;
        if (col >= local_k_words) continue;  // padded K word
        const uint32_t hidden_row = nt * 256 + coord.row;
        require(hidden_row < kHidden, "W2 band maps past the hidden size");
        const uint32_t global_k = offset / 8 + col;
        require(global_k < full_k_words, "W2 band maps past the intermediate");
        const uint32_t global_word = hidden_row * full_k_words + global_k;
        require(global_word < covered.size(), "W2 coverage index out of range");
        require(covered[global_word] == 0, "W2 word claimed twice");
        const uint32_t expected = w2[uint64_t(hidden_row) * local_k_words + col];
        require(packed[(nt * k_tiles + kt) * 4096 + idx] == expected,
                "W2 packed word does not match its band source");
        covered[global_word] = 1;
      }
}

// Exact padded-word totals for a 576 -> 640 extent, in packed coordinates.
struct PadCounts {
  long expected[4];
};

PadCounts expected_pad_counts(uint32_t intermediate, uint32_t kernel_intermediate) {
  const uint32_t pad_rows = kernel_intermediate - intermediate;
  PadCounts counts{};
  counts.expected[0] = 2L * pad_rows * (kHidden / 8);    // W13 payload N pad
  counts.expected[1] = 2L * pad_rows * (kHidden / 32);   // S13 scales N pad
  counts.expected[2] = long(kHidden) * (pad_rows / 8);   // W2 payload K pad
  counts.expected[3] = long(kHidden) * (pad_rows / 32);  // S2 scales K pad
  return counts;
}

// Decode every destination word, count the padded words and how many are not
// zero, and require both the exact count and all-zero contents. Slots 0/1 pad on
// the N axis (half_row); slots 2/3 pad on the K axis (col).
void check_padding(const std::array<std::vector<uint8_t>, 4>& actual,
                   uint32_t intermediate, uint32_t kernel_intermediate,
                   const std::string& name) {
  const PadCounts expected = expected_pad_counts(intermediate, kernel_intermediate);
  const uint32_t k_tiles = (kHidden + 127) / 128;
  const uint32_t gated_n_tiles = (2 * kernel_intermediate + 255) / 256;
  const uint32_t down_n_tiles = (kHidden + 255) / 256;
  const uint32_t down_k_tiles = (kernel_intermediate + 127) / 128;
  const uint32_t local_k_words = intermediate / 8;
  const uint32_t local_k_scale_words = intermediate / 32;
  long pad_words[4] = {0, 0, 0, 0};
  long pad_nonzero[4] = {0, 0, 0, 0};

  {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(actual[0].data());
    for (uint32_t nt = 0; nt < gated_n_tiles; ++nt)
      for (uint32_t kt = 0; kt < k_tiles; ++kt)
        for (uint32_t idx = 0; idx < 4096; ++idx) {
          const PayloadCoord coord = payload_coord(idx);
          const uint32_t plane_row = nt * 256 + coord.row;
          const uint32_t half_row =
              plane_row >= kernel_intermediate ? plane_row - kernel_intermediate : plane_row;
          if (half_row < intermediate) continue;
          ++pad_words[0];
          if (w[(nt * k_tiles + kt) * 4096 + idx] != 0) ++pad_nonzero[0];
        }
  }
  {
    const uint8_t* w = actual[1].data();
    for (uint32_t nt = 0; nt < gated_n_tiles; ++nt)
      for (uint32_t kt = 0; kt < k_tiles; ++kt)
        for (uint32_t idx = 0; idx < 1024; ++idx) {
          const uint32_t plane_row = nt * 256 + idx / 4;
          const uint32_t half_row =
              plane_row >= kernel_intermediate ? plane_row - kernel_intermediate : plane_row;
          if (half_row < intermediate) continue;
          ++pad_words[1];
          if (w[(nt * k_tiles + kt) * 1024 + idx] != 0) ++pad_nonzero[1];
        }
  }
  {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(actual[2].data());
    for (uint32_t nt = 0; nt < down_n_tiles; ++nt)
      for (uint32_t kt = 0; kt < down_k_tiles; ++kt)
        for (uint32_t idx = 0; idx < 4096; ++idx) {
          const uint32_t combined = (idx >> 2) & 31;
          const uint32_t k32 = idx >> 10;
          const uint32_t col = kt * 16 + k32 * 4 + (combined & 3);
          if (col < local_k_words) continue;
          ++pad_words[2];
          if (w[(nt * down_k_tiles + kt) * 4096 + idx] != 0) ++pad_nonzero[2];
        }
  }
  {
    const uint8_t* w = actual[3].data();
    for (uint32_t nt = 0; nt < down_n_tiles; ++nt)
      for (uint32_t kt = 0; kt < down_k_tiles; ++kt)
        for (uint32_t idx = 0; idx < 1024; ++idx) {
          const uint32_t col = kt * 4 + idx % 4;
          if (col < local_k_scale_words) continue;
          ++pad_words[3];
          if (w[(nt * down_k_tiles + kt) * 1024 + idx] != 0) ++pad_nonzero[3];
        }
  }

  require(pad_nonzero[0] == 0 && pad_nonzero[1] == 0 && pad_nonzero[2] == 0 &&
              pad_nonzero[3] == 0,
          name + ": padded storage is not zero (payload " +
              std::to_string(pad_nonzero[0]) + ", scale " + std::to_string(pad_nonzero[1]) +
              ", W2 " + std::to_string(pad_nonzero[2]) + ", S2 " +
              std::to_string(pad_nonzero[3]) + ")");
  require(pad_words[0] == expected.expected[0] && pad_words[1] == expected.expected[1] &&
              pad_words[2] == expected.expected[2] && pad_words[3] == expected.expected[3],
          name + ": padded word counts differ from the exact extent arithmetic");
  std::printf("%-8s intermediate=%-4u kernel=%-4u pad words W13=%ld S13=%ld W2=%ld S2=%ld "
              "(all zero)\n",
              name.c_str(), intermediate, kernel_intermediate, pad_words[0],
              pad_words[1], pad_words[2], pad_words[3]);
}
}  // namespace

int main() {
  int devices = 0;
  const bool required = std::getenv("DS41RT_REQUIRE_CUDA") != nullptr;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    if (required) {
      std::fprintf(stderr,
                   "v41 expert pack geometry selftest: DS41RT_REQUIRE_CUDA is set "
                   "but no CUDA device is present\n");
      return 1;
    }
    std::printf("v41 expert pack geometry selftest: no CUDA device, skipping\n");
    return 77;
  }
  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  const uint64_t full_weight_bytes = uint64_t(kFullIntermediate) * kHidden / 2;
  const uint64_t full_scale_bytes = full_weight_bytes / 16;
  const std::vector<uint8_t> full_w1 = make_source(full_weight_bytes, 11);
  const std::vector<uint8_t> full_w3 = make_source(full_weight_bytes, 12);
  const std::vector<uint8_t> full_w2 = make_source(full_weight_bytes, 13);
  const std::vector<uint8_t> full_s1 = make_source(full_scale_bytes, 14);
  const std::vector<uint8_t> full_s3 = make_source(full_scale_bytes, 15);
  const std::vector<uint8_t> full_s2 = make_source(full_scale_bytes, 16);

  // Union coverage over the four band cases in the un-padded global layouts.
  // W13 has two logical halves (up then gate): 2*2304 rows x 640 words.
  std::vector<uint8_t> w13_covered(size_t(2) * kFullIntermediate * (kHidden / 8), 0);
  std::vector<uint8_t> w2_covered(size_t(kHidden) * (kFullIntermediate / 8), 0);

  for (const Case& item : kCases) {
    const uint32_t intermediate = item.intermediate;
    const std::string name = item.name;
    uint64_t sizes[4] = {};
    require(ds41rt_v41_expert_packed_sizes(intermediate, sizes) == cudaSuccess,
            name + ": packer rejected the extent");

    const uint64_t weight_bytes = uint64_t(intermediate) * kHidden / 2;
    const uint64_t scale_bytes = weight_bytes / 16;
    const std::vector<uint8_t> w1 = band_rows(full_w1, item.offset, intermediate, kHidden / 2);
    const std::vector<uint8_t> w3 = band_rows(full_w3, item.offset, intermediate, kHidden / 2);
    const std::vector<uint8_t> s1 = band_rows(full_s1, item.offset, intermediate, kHidden / 32);
    const std::vector<uint8_t> s3 = band_rows(full_s3, item.offset, intermediate, kHidden / 32);
    const std::vector<uint8_t> w2 = band_cols(full_w2, kHidden, kFullIntermediate / 2,
                                             item.offset / 2, intermediate / 2);
    const std::vector<uint8_t> s2 = band_cols(full_s2, kHidden, kFullIntermediate / 32,
                                             item.offset / 32, intermediate / 32);
    require(w1.size() == weight_bytes && s2.size() == scale_bytes,
            name + ": band slice extents are wrong");

    const uint8_t* sources[6] = {w1.data(), w3.data(), w2.data(),
                                 s1.data(), s3.data(), s2.data()};
    uint8_t* destinations[4] = {};
    for (int i = 0; i < 4; ++i)
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&destinations[i]), sizes[i]),
                 "cudaMalloc destination");
    require(ds41rt_v41_pack_expert_async(sources, destinations, intermediate,
                                         nullptr) == cudaSuccess,
            name + ": pack launch failed");
    check_cuda(cudaStreamSynchronize(nullptr), "synchronize");

    const uint32_t kernel_intermediate = (intermediate + 127) / 128 * 128;
    std::vector<uint8_t> expected_w13 = reference_pack(
        /*gated=*/true, /*scales=*/false, w3.data(), w1.data(), intermediate,
        kHidden, kernel_intermediate, sizes[0] / 4);
    std::vector<uint8_t> expected_s13 = reference_pack(
        /*gated=*/true, /*scales=*/true, s3.data(), s1.data(), intermediate,
        kHidden, kernel_intermediate, sizes[1]);
    std::vector<uint8_t> expected_w2 = reference_pack(
        /*gated=*/false, /*scales=*/false, w2.data(), nullptr, kHidden,
        intermediate, 0, sizes[2] / 4);
    std::vector<uint8_t> expected_s2 = reference_pack(
        /*gated=*/false, /*scales=*/true, s2.data(), nullptr, kHidden,
        intermediate, 0, sizes[3]);
    const std::vector<uint8_t>* expected[4] = {&expected_w13, &expected_s13,
                                               &expected_w2, &expected_s2};

    std::array<std::vector<uint8_t>, 4> actual;
    for (int i = 0; i < 4; ++i) {
      actual[i].resize(sizes[i]);
      check_cuda(cudaMemcpy(actual[i].data(), destinations[i], sizes[i],
                            cudaMemcpyDeviceToHost),
                 "copy destination");
      for (size_t byte = 0; byte < actual[i].size(); ++byte)
        if (actual[i][byte] != (*expected[i])[byte]) {
          std::fprintf(stderr,
                       "%s destination %d byte %zu actual=%02x expected=%02x\n",
                       name.c_str(), i, byte, actual[i][byte], (*expected[i])[byte]);
          std::exit(1);
        }
      check_cuda(cudaFree(destinations[i]), "cudaFree destination");
    }

    if (kernel_intermediate != intermediate) {
      require(intermediate == 576 && kernel_intermediate == 640,
              name + ": unexpected padded extent");
      check_padding(actual, intermediate, kernel_intermediate, name);
    } else {
      std::printf("%-8s intermediate=%-4u kernel=%-4u w13=%llu s13=%llu w2=%llu s2=%llu\n",
                  name.c_str(), intermediate, kernel_intermediate,
                  static_cast<unsigned long long>(sizes[0]),
                  static_cast<unsigned long long>(sizes[1]),
                  static_cast<unsigned long long>(sizes[2]),
                  static_cast<unsigned long long>(sizes[3]));
    }

    // TP4 four-rank logical partition: the four 576-wide bands must claim every
    // un-padded global word exactly once, and each packed word must equal the
    // band's own source word.
    if (name.rfind("band4/", 0) == 0) {
      claim_w13(reinterpret_cast<const uint32_t*>(actual[0].data()),
                reinterpret_cast<const uint32_t*>(w3.data()),
                reinterpret_cast<const uint32_t*>(w1.data()), intermediate, item.offset,
                w13_covered);
      claim_w2(reinterpret_cast<const uint32_t*>(actual[2].data()),
               reinterpret_cast<const uint32_t*>(w2.data()), intermediate, item.offset,
               w2_covered);
    }
  }

  size_t missing_w13 = 0;
  for (uint8_t value : w13_covered) missing_w13 += (value == 0);
  size_t missing_w2 = 0;
  for (uint8_t value : w2_covered) missing_w2 += (value == 0);
  require(missing_w13 == 0 && missing_w2 == 0,
          "band4 partition left " + std::to_string(missing_w13) + " W13 and " +
              std::to_string(missing_w2) + " W2 words uncovered");
  std::printf("band4 partition: %zu W13 + %zu W2 words covered exactly once\n",
              w13_covered.size(), w2_covered.size());

  std::printf("v41 expert pack geometry selftest: ok (%zu geometries, incl. padded 576->640)\n",
              sizeof(kCases) / sizeof(kCases[0]));
  return 0;
}
