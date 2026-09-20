// Exact-reference check for the replicated-group N-plane compact reducer.
//
// Independent host scalar reference: each physical rank contributes one BF16
// [rows,5120] plane, the 2/3/4/6 planes are summed in rank order in FP32, the
// optional BF16 shared expert is added exactly once, and the result is rounded
// once to BF16. Also checks:
//   * the historical TP2/TP4 entry points agree bit-for-bit with the generic
//     entry point for the same ordered planes;
//   * the destination is written exactly over [0, rows*5120) and every trailing
//     byte of the capacity-sized buffer stays at its poison value;
//   * the rows=4096 upper boundary;
//   * a captured graph re-reads changed plane/shared contents on every replay
//     (ranks 3 and 6) instead of baking them;
//   * all-zero and single-zero active planes contribute exact zeros;
//   * every malformed case is rejected with exactly one violation in an
//     otherwise-valid six-slot argument set.
//
// Requires a CUDA device; exits 77 (ctest SKIP) when none is present.
#include "ds41rt_v41_experts.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t kHidden = 5120;
constexpr uint8_t kPoison = 0xa5;

void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "v41 route reduce planes selftest: %s\n", message);
    std::exit(1);
  }
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "v41 route reduce planes selftest: %s failed: %s\n",
                 what, cudaGetErrorString(status));
    std::exit(1);
  }
}

// Round-to-nearest-even BF16 encoding, independent of CUDA intrinsics.
uint16_t bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

float from_bf16(uint16_t value) {
  const uint32_t bits = uint32_t(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// BF16-representable, order-sensitive payloads: the cancellation pairs expose
// an ordered FP32 sum, and the small offsets expose an intermediate rounding.
float payload(uint32_t rank, uint32_t index) {
  static const float first[6] = {16777216.0f, 1.0f, -16777216.0f, 1.0f, 0.5f, -0.5f};
  static const float second[6] = {256.0f, 1.0f, 2.0f, -1.0f, 0.25f, -0.25f};
  static const float third[6] = {-256.0f, -1.0f, -2.0f, 1.0f, -0.25f, 0.25f};
  switch (index % 8) {
    case 0: return first[rank];
    case 1: return second[rank];
    case 2: return third[rank];
    case 3: return rank % 2 ? 0.00390625f : 1.0f;  // 2^-8 cancellation-ish
    default: return float(int(index / kHidden % 7) + int(rank) * 3 - int(index % 11)) * 0.25f;
  }
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    std::printf("v41 route reduce planes selftest: no CUDA device, skipping\n");
    return 77;
  }
  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  void* raw[6] = {};
  void* shared_raw = nullptr;
  void* output_raw = nullptr;
  // Second independent destination for the generic-vs-fixed equivalence check.
  void* equivalence_raw = nullptr;
  const size_t plane_bytes = size_t(4096) * kHidden * 2;
  for (auto& pointer : raw) check_cuda(cudaMalloc(&pointer, plane_bytes), "cudaMalloc plane");
  check_cuda(cudaMalloc(&shared_raw, plane_bytes), "cudaMalloc shared");
  check_cuda(cudaMalloc(&output_raw, plane_bytes), "cudaMalloc output");
  check_cuda(cudaMalloc(&equivalence_raw, plane_bytes), "cudaMalloc equivalence");

  // Host-side mirror of the BF16 values currently resident in raw[rank]; the
  // scalar reference always reads this, never the device contents.
  std::vector<uint16_t> host_plane[6];
  std::vector<uint16_t> host_shared;

  auto fill_host = [&](uint32_t rows, float scale, float shared_value) {
    for (uint32_t rank = 0; rank < 6; ++rank) {
      host_plane[rank].resize(size_t(rows) * kHidden);
      for (size_t i = 0; i < host_plane[rank].size(); ++i)
        host_plane[rank][i] = bf16(payload(rank, static_cast<uint32_t>(i)) * scale);
    }
    host_shared.assign(size_t(rows) * kHidden, bf16(shared_value));
  };
  auto upload_planes = [&]() {
    for (uint32_t rank = 0; rank < 6; ++rank)
      check_cuda(cudaMemcpy(raw[rank], host_plane[rank].data(),
                            host_plane[rank].size() * 2, cudaMemcpyHostToDevice),
                 "upload plane");
  };
  auto upload_shared = [&](uint32_t rows) {
    check_cuda(cudaMemcpy(shared_raw, host_shared.data(), size_t(rows) * kHidden * 2,
                          cudaMemcpyHostToDevice), "upload shared");
  };
  // Six-slot view with exactly `ranks` active planes; inactive slots are null.
  auto slots = [&](uint32_t ranks) {
    std::array<const uint16_t*, 6> selected{};
    for (uint32_t rank = 0; rank < 6; ++rank)
      selected[rank] = rank < ranks ? reinterpret_cast<const uint16_t*>(raw[rank]) : nullptr;
    return selected;
  };
  auto reference = [&](uint32_t rows, uint32_t ranks, bool use_shared) {
    std::vector<uint16_t> expected(size_t(rows) * kHidden);
    for (size_t i = 0; i < expected.size(); ++i) {
      float sum = from_bf16(host_plane[0][i]);
      for (uint32_t rank = 1; rank < ranks; ++rank)
        sum += from_bf16(host_plane[rank][i]);
      if (use_shared) sum += from_bf16(host_shared[i]);
      expected[i] = bf16(sum);
    }
    return expected;
  };
  // Copy the full destination capacity back and require the live region to match
  // the scalar reference and every trailing byte to remain poison.
  auto verify = [&](uint16_t* destination, uint32_t rows, uint32_t ranks,
                    bool use_shared, const char* label) {
    std::vector<uint16_t> actual(plane_bytes / 2);
    check_cuda(cudaMemcpy(actual.data(), destination, plane_bytes, cudaMemcpyDeviceToHost),
               "copy destination");
    const std::vector<uint16_t> expected = reference(rows, ranks, use_shared);
    for (size_t i = 0; i < expected.size(); ++i)
      if (actual[i] != expected[i]) {
        std::fprintf(stderr,
                     "%s: mismatch rows=%u ranks=%u element=%zu actual=%u expected=%u\n",
                     label, rows, ranks, i, actual[i], expected[i]);
        std::exit(1);
      }
    for (size_t i = expected.size(); i < actual.size(); ++i)
      if (actual[i] != uint16_t(0xa5a5u)) {
        std::fprintf(stderr, "%s: destination written past rows=%u at element=%zu\n",
                     label, rows, i);
        std::exit(1);
      }
  };
  // Poison the actual destination, restore shared if used, launch, and verify.
  auto launch_and_verify = [&](uint32_t rows, uint32_t ranks, bool use_shared,
                               bool alias_shared, const char* label) {
    const auto selected = slots(ranks);
    uint16_t* destination =
        alias_shared ? reinterpret_cast<uint16_t*>(shared_raw)
                     : reinterpret_cast<uint16_t*>(output_raw);
    check_cuda(cudaMemset(destination, kPoison, plane_bytes), "poison destination");
    if (use_shared) upload_shared(rows);
    require(ds41rt_v41_reduce_compact_bf16_planes_async(
                selected.data(), use_shared ? reinterpret_cast<const uint16_t*>(shared_raw)
                                            : nullptr,
                destination, rows, ranks, nullptr) == cudaSuccess,
            label);
    check_cuda(cudaStreamSynchronize(nullptr), "synchronize");
    verify(destination, rows, ranks, use_shared, label);
  };

  // Positive matrix: every capacity boundary and every rank count, with no
  // shared, with shared, and with output aliasing shared exactly.
  for (uint32_t rows : {1u, 16u, 80u, 3u, 4096u}) {
    fill_host(rows, 1.0f, 1.0f);
    upload_planes();
    for (uint32_t ranks : {2u, 3u, 4u, 6u}) {
      launch_and_verify(rows, ranks, false, false, "no shared");
      launch_and_verify(rows, ranks, true, false, "with shared");
      launch_and_verify(rows, ranks, true, true, "shared exact alias");

      // The historical 2- and 4-plane entry points must be bit-identical to the
      // generic entry point for the same ordered planes.
      if (ranks == 2 || ranks == 4) {
        const auto selected = slots(ranks);
        uint16_t* generic = reinterpret_cast<uint16_t*>(output_raw);
        uint16_t* fixed_output = reinterpret_cast<uint16_t*>(equivalence_raw);
        check_cuda(cudaMemset(generic, 0, plane_bytes), "memset generic");
        check_cuda(cudaMemset(fixed_output, 0, plane_bytes), "memset fixed");
        require(ds41rt_v41_reduce_compact_bf16_planes_async(
                    selected.data(), nullptr, generic, rows, ranks, nullptr) == cudaSuccess,
                "generic equivalence launch failed");
        if (ranks == 2) {
          const uint16_t* fixed[2] = {
              reinterpret_cast<const uint16_t*>(raw[0]),
              reinterpret_cast<const uint16_t*>(raw[1])};
          require(ds41rt_v41_reduce_tp2_compact_bf16_async(fixed, nullptr, fixed_output,
                      rows, nullptr) == cudaSuccess, "fixed tp2 launch failed");
        } else {
          const uint16_t* fixed[4] = {
              reinterpret_cast<const uint16_t*>(raw[0]),
              reinterpret_cast<const uint16_t*>(raw[1]),
              reinterpret_cast<const uint16_t*>(raw[2]),
              reinterpret_cast<const uint16_t*>(raw[3])};
          require(ds41rt_v41_reduce_compact_bf16_async(fixed, nullptr, fixed_output,
                      rows, nullptr) == cudaSuccess, "fixed tp4 launch failed");
        }
        check_cuda(cudaStreamSynchronize(nullptr), "synchronize equivalence");
        std::vector<uint16_t> a(size_t(rows) * kHidden), b(size_t(rows) * kHidden);
        check_cuda(cudaMemcpy(a.data(), generic, a.size() * 2, cudaMemcpyDeviceToHost),
                   "copy generic");
        check_cuda(cudaMemcpy(b.data(), fixed_output, b.size() * 2, cudaMemcpyDeviceToHost),
                   "copy fixed");
        require(a == b, "generic and fixed reducers disagree");
      }
    }
  }

  // Zero active planes: an empty/unassigned rank returns exact zeros, and an
  // all-zero plan must neither corrupt nor depend on the shared expert.
  {
    const uint32_t rows = 16;
    fill_host(rows, 1.0f, 1.0f);
    upload_planes();
    for (uint32_t ranks : {2u, 3u, 4u, 6u}) {
      for (uint32_t zero = 0; zero < ranks; ++zero) {
        const std::vector<uint16_t> saved = host_plane[zero];
        std::fill(host_plane[zero].begin(), host_plane[zero].end(), uint16_t(0));
        check_cuda(cudaMemcpy(raw[zero], host_plane[zero].data(),
                              size_t(rows) * kHidden * 2, cudaMemcpyHostToDevice),
                   "zero active plane");
        launch_and_verify(rows, ranks, true, false, "single zero active plane");
        host_plane[zero] = saved;
        check_cuda(cudaMemcpy(raw[zero], host_plane[zero].data(),
                              size_t(rows) * kHidden * 2, cudaMemcpyHostToDevice),
                   "restore active plane");
      }
      for (uint32_t rank = 0; rank < 6; ++rank) {
        std::fill(host_plane[rank].begin(), host_plane[rank].end(), uint16_t(0));
        check_cuda(cudaMemcpy(raw[rank], host_plane[rank].data(),
                              size_t(rows) * kHidden * 2, cudaMemcpyHostToDevice),
                   "zero all planes");
      }
      launch_and_verify(rows, ranks, true, false, "all-zero planes with shared");
      launch_and_verify(rows, ranks, false, false, "all-zero planes without shared");
      fill_host(rows, 1.0f, 1.0f);
      upload_planes();
    }
  }

  // Graph capture/replay: capture once at ranks 3 and 6, then replay with
  // changed plane and shared contents and require the replay to consume the new
  // contents. No allocation happens between instantiate and replay.
  {
    const uint32_t rows = 16;
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create capture stream");
    for (uint32_t ranks : {3u, 6u}) {
      fill_host(rows, 1.0f, 1.0f);
      upload_planes();
      upload_shared(rows);
      check_cuda(cudaMemset(output_raw, kPoison, plane_bytes), "poison output A");
      const auto selected = slots(ranks);
      check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                 "begin capture");
      require(ds41rt_v41_reduce_compact_bf16_planes_async(
                  selected.data(), reinterpret_cast<const uint16_t*>(shared_raw),
                  reinterpret_cast<uint16_t*>(output_raw), rows, ranks, stream) == cudaSuccess,
              "captured launch failed");
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t executable = nullptr;
      check_cuda(cudaStreamEndCapture(stream, &graph), "end capture");
      check_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");

      fill_host(rows, 0.5f, -1.0f);
      upload_planes();
      upload_shared(rows);
      check_cuda(cudaMemset(output_raw, kPoison, plane_bytes), "poison output B");
      check_cuda(cudaGraphLaunch(executable, stream), "graph launch B");
      check_cuda(cudaStreamSynchronize(stream), "graph sync B");
      verify(reinterpret_cast<uint16_t*>(output_raw), rows, ranks, true, "graph replay B");

      fill_host(rows, -1.0f, 3.0f);
      upload_planes();
      upload_shared(rows);
      check_cuda(cudaMemset(output_raw, kPoison, plane_bytes), "poison output C");
      check_cuda(cudaGraphLaunch(executable, stream), "graph launch C");
      check_cuda(cudaStreamSynchronize(stream), "graph sync C");
      verify(reinterpret_cast<uint16_t*>(output_raw), rows, ranks, true, "graph replay C");

      check_cuda(cudaGraphExecDestroy(executable), "destroy executable");
      check_cuda(cudaGraphDestroy(graph), "destroy graph");
    }
    check_cuda(cudaStreamDestroy(stream), "destroy capture stream");
  }

  // Validation: every negative case is otherwise valid with exactly one
  // violation, so a rejection cannot pass for the wrong reason.
  {
    const uint32_t rows = 16;
    fill_host(rows, 1.0f, 1.0f);
    upload_planes();
    upload_shared(rows);
    auto* output = reinterpret_cast<uint16_t*>(output_raw);
    auto call = [&](const uint16_t* const* planes, const uint16_t* shared,
                    uint16_t* destination, uint32_t call_rows, uint32_t ranks) {
      return ds41rt_v41_reduce_compact_bf16_planes_async(
          planes, shared, destination, call_rows, ranks, nullptr);
    };

    const auto all_six = slots(6);
    require(call(all_six.data(), nullptr, output, rows, 5) != cudaSuccess,
            "ranks=5 accepted");
    require(call(all_six.data(), nullptr, output, rows, 0) != cudaSuccess,
            "ranks=0 accepted");
    require(call(all_six.data(), nullptr, output, 0, 6) != cudaSuccess,
            "rows=0 accepted");
    require(call(all_six.data(), nullptr, output, 4097, 6) != cudaSuccess,
            "rows=4097 accepted");

    // Exactly one null active slot; the other active slots and all inactive
    // slots are valid.
    const auto null_active = [&] {
      auto selected = slots(3);
      selected[2] = nullptr;
      return selected;
    }();
    require(call(null_active.data(), nullptr, output, rows, 3) != cudaSuccess,
            "null active slot accepted");

    // Exactly one non-null inactive slot; the three active slots are valid.
    const auto extra_slot = [&] {
      auto selected = slots(3);
      selected[3] = reinterpret_cast<const uint16_t*>(raw[3]);
      return selected;
    }();
    require(call(extra_slot.data(), nullptr, output, rows, 3) != cudaSuccess,
            "non-null inactive slot accepted");

    // Exactly one violation: plane 0 is the output buffer.
    const auto alias = [&] {
      auto selected = slots(2);
      selected[0] = output;
      return selected;
    }();
    require(call(alias.data(), nullptr, output, rows, 2) != cudaSuccess,
            "output/plane0 aliasing accepted");

    // Exactly one violation: shared partially overlaps output while differing
    // from it (both are 16-byte aligned).
    const auto* partial_shared =
        reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(output_raw) + 16);
    require(call(all_six.data(), partial_shared, output, rows, 6) != cudaSuccess,
            "partial shared/output overlap accepted");

    // Exactly one violation: byte-misaligned output, otherwise valid.
    auto* misaligned_output = reinterpret_cast<uint16_t*>(
        reinterpret_cast<uint8_t*>(output_raw) + 1);
    require(call(all_six.data(), nullptr, misaligned_output, rows, 6) != cudaSuccess,
            "misaligned output accepted");

    // Exactly one violation: byte-misaligned plane 1, otherwise valid six-slot.
    const auto misaligned_plane = [&] {
      auto selected = slots(6);
      selected[1] = reinterpret_cast<const uint16_t*>(
          reinterpret_cast<const uint8_t*>(raw[1]) + 1);
      return selected;
    }();
    require(call(misaligned_plane.data(), nullptr, output, rows, 6) != cudaSuccess,
            "misaligned plane accepted");

    // Positive boundary: output == shared exactly is accepted and correct.
    check_cuda(cudaMemset(output_raw, kPoison, plane_bytes), "poison shared/output");
    check_cuda(cudaMemcpy(output_raw, host_shared.data(), size_t(rows) * kHidden * 2,
                          cudaMemcpyHostToDevice), "seed shared/output");
    require(call(all_six.data(), output, output, rows, 6) == cudaSuccess,
            "exact shared/output alias rejected");
    check_cuda(cudaStreamSynchronize(nullptr), "synchronize exact alias");
    verify(output, rows, 6, true, "exact shared/output alias");
  }

  for (auto& pointer : raw) check_cuda(cudaFree(pointer), "cudaFree plane");
  check_cuda(cudaFree(shared_raw), "cudaFree shared");
  check_cuda(cudaFree(output_raw), "cudaFree output");
  check_cuda(cudaFree(equivalence_raw), "cudaFree equivalence");
  std::printf("v41 route reduce planes selftest: ok\n");
  return 0;
}
