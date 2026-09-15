// CPU-only FP8/E8M0 -> NVFP4. No full-size decoded weight allocation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <omp.h>

static float fp8(unsigned x) {
    unsigned e = x >> 3, m = x & 7;
    return e ? std::ldexp(float(8 + m), int(e) - 10) : std::ldexp(float(m), -9);
}
static unsigned round8(float x) {
    x = std::clamp(x, std::ldexp(1.0f, -9), 448.0f);
    unsigned lo = 0, hi = 126;
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (fp8(mid) < x) lo = mid + 1; else hi = mid;
    }
    if (!lo) return 0;
    float a = x - fp8(lo - 1), b = fp8(lo) - x;
    return a < b || (a == b && ((lo - 1) % 2 == 0)) ? lo - 1 : lo;
}
static unsigned round4(float x) {
    const float bounds[] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
    unsigned i = 0;
    while (i < 7 && (x > bounds[i] || (x == bounds[i] && i % 2))) ++i;
    return i;
}
extern "C" {
void ple_stats(const uint8_t* w, const uint8_t* s, size_t rows, uint8_t* maxima) {
    #pragma omp parallel num_threads(8)
    {
        uint8_t local[256] = {};
        #pragma omp for
        for (size_t b = 0; b < rows * 8; ++b) {
            unsigned m = 0;
            for (unsigned j = 0; j < 32; ++j) m = std::max(m, unsigned(w[b * 32 + j] & 127));
            local[s[b]] = std::max(local[s[b]], uint8_t(m));
            // A separate sentinel catches NaN weights even with finite scales.
            if (m == 127 || s[b] == 255) local[255] = 127;
        }
        #pragma omp critical
        for (unsigned e = 0; e < 256; ++e) maxima[e] = std::max(maxima[e], local[e]);
    }
}
float ple_amax(const uint8_t* maxima) {
    if (maxima[255]) return NAN;
    float a = 0;
    for (unsigned e = 0; e < 255; ++e)
        if (maxima[e]) a = std::max(a, std::ldexp(fp8(maxima[e]), int(e) - 127));
    return a;
}
void ple_lut(float global, uint8_t* codes, uint8_t* scales) {
    for (unsigned e = 0; e < 255; ++e) for (unsigned m = 0; m < 127; ++m) {
        float a = std::ldexp(fp8(m), int(e) - 127);
        float scale = m ? a / (6.f * global) : 1.f;
        unsigned sc = round8(scale);
        scales[e * 128 + m] = sc;
        float divisor = fp8(sc) * global;
        for (unsigned v = 0; v < 256; ++v) {
            float val = std::ldexp(fp8(v & 127), int(e) - 127);
            unsigned code = round4(val / divisor);
            // Negative zero follows ModelOpt's weight < 0 sign test.
            if ((v & 128) && (v & 127)) code |= 8;
            codes[(e * 128 + m) * 256 + v] = code;
        }
    }
}
void ple_convert(const uint8_t* w, const uint8_t* s, size_t rows,
                 const uint8_t* codes, const uint8_t* scales, uint8_t* out, uint8_t* os) {
    #pragma omp parallel for num_threads(8)
    for (size_t b = 0; b < rows * 16; ++b) {
        const uint8_t* src = w + b * 16;
        unsigned m = 0;
        for (unsigned j = 0; j < 16; ++j) m = std::max(m, unsigned(src[j] & 127));
        unsigned key = unsigned(s[b / 2]) * 128 + m;
        os[b] = scales[key];
        const uint8_t* lut = codes + key * 256;
        for (unsigned j = 0; j < 8; ++j)
            out[b * 8 + j] = lut[src[j * 2]] | (lut[src[j * 2 + 1]] << 4);
    }
}
}
