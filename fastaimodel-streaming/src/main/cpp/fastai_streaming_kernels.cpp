#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

#if defined(_MSC_VER)
#define EXPORT_API extern "C" __declspec(dllexport)
#else
#define EXPORT_API extern "C" __attribute__((visibility("default")))
#endif

// Block dimensions matching GGML / GGUF specifications
#define Q4_0_BLOCK 32
#define Q8_0_BLOCK 32

// Structure matching GGML block_q4_0
#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;           // delta (fp16)
    uint8_t qs[16];       // 32 nibbles packed into 16 bytes
};

struct block_q8_0 {
    uint16_t d;           // delta (fp16)
    int8_t  qs[32];       // 32 signed 8-bit integers
};
#pragma pack(pop)

// Helper: convert fp16 bits to fp32 float using F16C intrinsic if available or bitcast
static inline float fp16_to_fp32(uint16_t h) {
#if defined(__F16C__) || defined(__AVX2__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000;
    uint32_t two_w = w + w;
    uint32_t exp = two_w >> 24;
    if (exp == 0) {
        return 0.0f;
    }
    if (exp == 0xFF) {
        return (sign ? -1.0f : 1.0f) * INFINITY;
    }
    uint32_t f_exp = (exp + (127 - 15)) << 23;
    uint32_t f_sig = (two_w >> 1) & 0x007FFFFF;
    uint32_t result = sign | f_exp | f_sig;
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
#endif
}

// Single row Q4_0 dot-product with AVX2 & FMA
static inline float dot_q4_0_row_avx2(const block_q4_0* __restrict row_blocks,
                                      const float* __restrict vec_in,
                                      int in_cols) {
    const int num_blocks = in_cols / Q4_0_BLOCK;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    const __m128i mask_low = _mm_set1_epi8(0x0F);
    const __m256 c_sub8 = _mm256_set1_ps(8.0f);

    for (int b = 0; b < num_blocks; b++) {
        const block_q4_0& blk = row_blocks[b];
        const float d = fp16_to_fp32(blk.d);
        const __m256 v_scale = _mm256_set1_ps(d);

        // Load 16 bytes containing 32 nibbles
        __m128i raw16 = _mm_loadu_si128((const __m128i*)blk.qs);

        // Extract low 4 bits (qs0..qs15)
        __m128i low_nibbles = _mm_and_si128(raw16, mask_low);
        // Extract high 4 bits (qs16..qs31)
        __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask_low);

        // Interleave so we get [q0, q1, q2, q3, ...]
        __m128i unpacked_low = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
        __m128i unpacked_high = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

        // Convert first 16 int8 values to 32-bit floats
        __m128i q0_7 = _mm_cvtepu8_epi32(unpacked_low);
        __m128i q8_15 = _mm_cvtepu8_epi32(_mm_srli_si128(unpacked_low, 4));
        __m128i q16_23 = _mm_cvtepu8_epi32(_mm_srli_si128(unpacked_low, 8));
        __m128i q24_31 = _mm_cvtepu8_epi32(_mm_srli_si128(unpacked_low, 12));

        __m256 fq0_7 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(q8_15, q0_7)), c_sub8);
        __m256 fq8_15 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(q24_31, q16_23)), c_sub8);

        // For remaining 16 values
        __m128i q32_39 = _mm_cvtepu8_epi32(unpacked_high);
        __m128i q40_47 = _mm_cvtepu8_epi32(_mm_srli_si128(unpacked_high, 4));
        __m128i q48_55 = _mm_cvtepu8_epi32(_mm_srli_si128(unpacked_high, 8));
        __m128i q56_63 = _mm_cvtepu8_epi32(_mm_srli_si128(unpacked_high, 12));

        __m256 fq16_23 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(q40_47, q32_39)), c_sub8);
        __m256 fq24_31 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_set_m128i(q56_63, q48_55)), c_sub8);

        // Multiply by scale
        fq0_7 = _mm256_mul_ps(fq0_7, v_scale);
        fq8_15 = _mm256_mul_ps(fq8_15, v_scale);
        fq16_23 = _mm256_mul_ps(fq16_23, v_scale);
        fq24_31 = _mm256_mul_ps(fq24_31, v_scale);

        // Load 32 floats of vec_in
        const float* vx = vec_in + b * Q4_0_BLOCK;
        __m256 vx0 = _mm256_loadu_ps(vx);
        __m256 vx1 = _mm256_loadu_ps(vx + 8);
        __m256 vx2 = _mm256_loadu_ps(vx + 16);
        __m256 vx3 = _mm256_loadu_ps(vx + 24);

        acc0 = _mm256_fmadd_ps(fq0_7, vx0, acc0);
        acc1 = _mm256_fmadd_ps(fq8_15, vx1, acc1);
        acc0 = _mm256_fmadd_ps(fq16_23, vx2, acc0);
        acc1 = _mm256_fmadd_ps(fq24_31, vx3, acc1);
    }

    __m256 sum256 = _mm256_add_ps(acc0, acc1);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    return _mm_cvtss_f32(sum128);
}

// Single row Q8_0 dot-product with AVX2 & FMA
static inline float dot_q8_0_row_avx2(const block_q8_0* __restrict row_blocks,
                                      const float* __restrict vec_in,
                                      int in_cols) {
    const int num_blocks = in_cols / Q8_0_BLOCK;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (int b = 0; b < num_blocks; b++) {
        const block_q8_0& blk = row_blocks[b];
        const float d = fp16_to_fp32(blk.d);
        const __m256 v_scale = _mm256_set1_ps(d);

        // Load 32 signed int8 values
        __m128i qs0 = _mm_loadu_si128((const __m128i*)blk.qs);
        __m128i qs1 = _mm_loadu_si128((const __m128i*)(blk.qs + 16));

        // Convert to 32-bit floats
        __m256 f0 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi8_epi32(_mm_srli_si128(qs0, 4)), _mm_cvtepi8_epi32(qs0)));
        __m256 f1 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi8_epi32(_mm_srli_si128(qs0, 12)), _mm_cvtepi8_epi32(_mm_srli_si128(qs0, 8))));
        __m256 f2 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi8_epi32(_mm_srli_si128(qs1, 4)), _mm_cvtepi8_epi32(qs1)));
        __m256 f3 = _mm256_cvtepi32_ps(_mm256_set_m128i(_mm_cvtepi8_epi32(_mm_srli_si128(qs1, 12)), _mm_cvtepi8_epi32(_mm_srli_si128(qs1, 8))));

        f0 = _mm256_mul_ps(f0, v_scale);
        f1 = _mm256_mul_ps(f1, v_scale);
        f2 = _mm256_mul_ps(f2, v_scale);
        f3 = _mm256_mul_ps(f3, v_scale);

        const float* vx = vec_in + b * Q8_0_BLOCK;
        __m256 vx0 = _mm256_loadu_ps(vx);
        __m256 vx1 = _mm256_loadu_ps(vx + 8);
        __m256 vx2 = _mm256_loadu_ps(vx + 16);
        __m256 vx3 = _mm256_loadu_ps(vx + 24);

        acc0 = _mm256_fmadd_ps(f0, vx0, acc0);
        acc1 = _mm256_fmadd_ps(f1, vx1, acc1);
        acc0 = _mm256_fmadd_ps(f2, vx2, acc0);
        acc1 = _mm256_fmadd_ps(f3, vx3, acc1);
    }

    __m256 sum256 = _mm256_add_ps(acc0, acc1);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
    return _mm_cvtss_f32(sum128);
}

// Multithreaded GEMV runner
template <typename F>
static inline void parallel_for(int start, int end, F func) {
    const int total = end - start;
    if (total <= 0) return;
    const unsigned int n_threads = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    if (total <= 32 || n_threads <= 1) {
        func(start, end);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    const int chunk = (total + n_threads - 1) / n_threads;

    for (unsigned int t = 0; t < n_threads; t++) {
        int s = start + t * chunk;
        int e = std::min(end, s + chunk);
        if (s < e) {
            pool.emplace_back([func, s, e]() {
                func(s, e);
            });
        }
    }

    for (auto& th : pool) {
        if (th.joinable()) th.join();
    }
}

// ====================================================================
// Exported C APIs for Java Foreign Function & Memory (FFM)
// ====================================================================

EXPORT_API int fastai_get_kernel_version() {
    return 100; // 1.0.0
}

EXPORT_API void gemv_q4_0_avx2(int out_rows, int in_cols,
                              const uint8_t* weights,
                              const float* vec_in,
                              float* vec_out,
                              int row_bytes) {
    parallel_for(0, out_rows, [=](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r++) {
            const block_q4_0* row_blocks = (const block_q4_0*)(weights + (size_t)r * row_bytes);
            vec_out[r] = dot_q4_0_row_avx2(row_blocks, vec_in, in_cols);
        }
    });
}

EXPORT_API void gemv_q8_0_avx2(int out_rows, int in_cols,
                              const uint8_t* weights,
                              const float* vec_in,
                              float* vec_out,
                              int row_bytes) {
    parallel_for(0, out_rows, [=](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r++) {
            const block_q8_0* row_blocks = (const block_q8_0*)(weights + (size_t)r * row_bytes);
            vec_out[r] = dot_q8_0_row_avx2(row_blocks, vec_in, in_cols);
        }
    });
}

// Batched GEMM: decodes weight block once and accumulates over all prompt tokens T
EXPORT_API void gemm_q4_0_avx2(int out_rows, int in_cols,
                              const uint8_t* weights,
                              const float* in_batch_flat, // [batchSize * in_cols]
                              float* out_batch_flat,      // [batchSize * out_rows]
                              int batch_size,
                              int row_bytes) {
    const int num_blocks = in_cols / Q4_0_BLOCK;

    parallel_for(0, out_rows, [=](int r_start, int r_end) {
        float q_unpacked[32];
        const __m128i mask_low = _mm_set1_epi8(0x0F);

        for (int r = r_start; r < r_end; r++) {
            const block_q4_0* row_blocks = (const block_q4_0*)(weights + (size_t)r * row_bytes);

            // Initialize outputs for this row across all batch items
            for (int b = 0; b < batch_size; b++) {
                out_batch_flat[b * out_rows + r] = 0.0f;
            }

            for (int blk_idx = 0; blk_idx < num_blocks; blk_idx++) {
                const block_q4_0& blk = row_blocks[blk_idx];
                const float scale = fp16_to_fp32(blk.d);

                // Unpack 32 nibbles into q_unpacked
                for (int i = 0; i < 16; i++) {
                    uint8_t byte_val = blk.qs[i];
                    q_unpacked[i * 2]     = (float)((int)(byte_val & 0x0F) - 8);
                    q_unpacked[i * 2 + 1] = (float)((int)(byte_val >> 4) - 8);
                }

                const int x_base = blk_idx * Q4_0_BLOCK;

                for (int t = 0; t < batch_size; t++) {
                    const float* x = in_batch_flat + t * in_cols + x_base;
                    __m256 vq0 = _mm256_loadu_ps(q_unpacked);
                    __m256 vq1 = _mm256_loadu_ps(q_unpacked + 8);
                    __m256 vq2 = _mm256_loadu_ps(q_unpacked + 16);
                    __m256 vq3 = _mm256_loadu_ps(q_unpacked + 24);

                    __m256 vx0 = _mm256_loadu_ps(x);
                    __m256 vx1 = _mm256_loadu_ps(x + 8);
                    __m256 vx2 = _mm256_loadu_ps(x + 16);
                    __m256 vx3 = _mm256_loadu_ps(x + 24);

                    __m256 sum0 = _mm256_mul_ps(vq0, vx0);
                    __m256 sum1 = _mm256_fmadd_ps(vq1, vx1, sum0);
                    __m256 sum2 = _mm256_fmadd_ps(vq2, vx2, sum1);
                    __m256 sum3 = _mm256_fmadd_ps(vq3, vx3, sum2);

                    __m128 hi128 = _mm256_extractf128_ps(sum3, 1);
                    __m128 lo128 = _mm256_castps256_ps128(sum3);
                    __m128 sum128 = _mm_add_ps(lo128, hi128);
                    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
                    float blk_sum = _mm_cvtss_f32(sum128);

                    out_batch_flat[t * out_rows + r] += scale * blk_sum;
                }
            }
        }
    });
}

EXPORT_API void gemm_q8_0_avx2(int out_rows, int in_cols,
                              const uint8_t* weights,
                              const float* in_batch_flat,
                              float* out_batch_flat,
                              int batch_size,
                              int row_bytes) {
    const int num_blocks = in_cols / Q8_0_BLOCK;

    parallel_for(0, out_rows, [=](int r_start, int r_end) {
        for (int r = r_start; r < r_end; r++) {
            const block_q8_0* row_blocks = (const block_q8_0*)(weights + (size_t)r * row_bytes);

            for (int b = 0; b < batch_size; b++) {
                out_batch_flat[b * out_rows + r] = 0.0f;
            }

            for (int blk_idx = 0; blk_idx < num_blocks; blk_idx++) {
                const block_q8_0& blk = row_blocks[blk_idx];
                const float scale = fp16_to_fp32(blk.d);

                float q_unpacked[32];
                for (int i = 0; i < 32; i++) {
                    q_unpacked[i] = (float)blk.qs[i];
                }

                const int x_base = blk_idx * Q8_0_BLOCK;

                for (int t = 0; t < batch_size; t++) {
                    const float* x = in_batch_flat + t * in_cols + x_base;
                    __m256 vq0 = _mm256_loadu_ps(q_unpacked);
                    __m256 vq1 = _mm256_loadu_ps(q_unpacked + 8);
                    __m256 vq2 = _mm256_loadu_ps(q_unpacked + 16);
                    __m256 vq3 = _mm256_loadu_ps(q_unpacked + 24);

                    __m256 vx0 = _mm256_loadu_ps(x);
                    __m256 vx1 = _mm256_loadu_ps(x + 8);
                    __m256 vx2 = _mm256_loadu_ps(x + 16);
                    __m256 vx3 = _mm256_loadu_ps(x + 24);

                    __m256 sum0 = _mm256_mul_ps(vq0, vx0);
                    __m256 sum1 = _mm256_fmadd_ps(vq1, vx1, sum0);
                    __m256 sum2 = _mm256_fmadd_ps(vq2, vx2, sum1);
                    __m256 sum3 = _mm256_fmadd_ps(vq3, vx3, sum2);

                    __m128 hi128 = _mm256_extractf128_ps(sum3, 1);
                    __m128 lo128 = _mm256_castps256_ps128(sum3);
                    __m128 sum128 = _mm_add_ps(lo128, hi128);
                    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));
                    float blk_sum = _mm_cvtss_f32(sum128);

                    out_batch_flat[t * out_rows + r] += scale * blk_sum;
                }
            }
        }
    });
}
