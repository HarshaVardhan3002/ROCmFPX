#include "rocmfpx.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Finite unsigned E4M3 scale bytes decoded to FP32. Precomputed from the same
// exp/mant formula rocmfpx_ue4m3_to_fp32() used to evaluate with ldexpf():
//   exp == 0 -> mant * 2^-10 ; otherwise (8 + mant) * 2^(exp - 11).
// The scale search re-decodes candidate bytes for every block, and dequant
// decodes a scale for every element, so keeping this as a table (identical to
// the former per-call ldexpf result) removes the transcendental from both hot
// paths without changing any produced value.
#define ROCMFPX_SCALE_SUB(M) ((M) * 0x1p-10f)
#define ROCMFPX_SCALE_E(B, M) ((8 + (M)) * (B))

static const float rocmfpx_scale_ue4m3[127] = {
    ROCMFPX_SCALE_SUB(0),      ROCMFPX_SCALE_SUB(1),      ROCMFPX_SCALE_SUB(2),      ROCMFPX_SCALE_SUB(3),
    ROCMFPX_SCALE_SUB(4),      ROCMFPX_SCALE_SUB(5),      ROCMFPX_SCALE_SUB(6),      ROCMFPX_SCALE_SUB(7),
    ROCMFPX_SCALE_E(0x1p-10f,0), ROCMFPX_SCALE_E(0x1p-10f,1), ROCMFPX_SCALE_E(0x1p-10f,2), ROCMFPX_SCALE_E(0x1p-10f,3),
    ROCMFPX_SCALE_E(0x1p-10f,4), ROCMFPX_SCALE_E(0x1p-10f,5), ROCMFPX_SCALE_E(0x1p-10f,6), ROCMFPX_SCALE_E(0x1p-10f,7),
    ROCMFPX_SCALE_E(0x1p-9f,0),  ROCMFPX_SCALE_E(0x1p-9f,1),  ROCMFPX_SCALE_E(0x1p-9f,2),  ROCMFPX_SCALE_E(0x1p-9f,3),
    ROCMFPX_SCALE_E(0x1p-9f,4),  ROCMFPX_SCALE_E(0x1p-9f,5),  ROCMFPX_SCALE_E(0x1p-9f,6),  ROCMFPX_SCALE_E(0x1p-9f,7),
    ROCMFPX_SCALE_E(0x1p-8f,0),  ROCMFPX_SCALE_E(0x1p-8f,1),  ROCMFPX_SCALE_E(0x1p-8f,2),  ROCMFPX_SCALE_E(0x1p-8f,3),
    ROCMFPX_SCALE_E(0x1p-8f,4),  ROCMFPX_SCALE_E(0x1p-8f,5),  ROCMFPX_SCALE_E(0x1p-8f,6),  ROCMFPX_SCALE_E(0x1p-8f,7),
    ROCMFPX_SCALE_E(0x1p-7f,0),  ROCMFPX_SCALE_E(0x1p-7f,1),  ROCMFPX_SCALE_E(0x1p-7f,2),  ROCMFPX_SCALE_E(0x1p-7f,3),
    ROCMFPX_SCALE_E(0x1p-7f,4),  ROCMFPX_SCALE_E(0x1p-7f,5),  ROCMFPX_SCALE_E(0x1p-7f,6),  ROCMFPX_SCALE_E(0x1p-7f,7),
    ROCMFPX_SCALE_E(0x1p-6f,0),  ROCMFPX_SCALE_E(0x1p-6f,1),  ROCMFPX_SCALE_E(0x1p-6f,2),  ROCMFPX_SCALE_E(0x1p-6f,3),
    ROCMFPX_SCALE_E(0x1p-6f,4),  ROCMFPX_SCALE_E(0x1p-6f,5),  ROCMFPX_SCALE_E(0x1p-6f,6),  ROCMFPX_SCALE_E(0x1p-6f,7),
    ROCMFPX_SCALE_E(0x1p-5f,0),  ROCMFPX_SCALE_E(0x1p-5f,1),  ROCMFPX_SCALE_E(0x1p-5f,2),  ROCMFPX_SCALE_E(0x1p-5f,3),
    ROCMFPX_SCALE_E(0x1p-5f,4),  ROCMFPX_SCALE_E(0x1p-5f,5),  ROCMFPX_SCALE_E(0x1p-5f,6),  ROCMFPX_SCALE_E(0x1p-5f,7),
    ROCMFPX_SCALE_E(0x1p-4f,0),  ROCMFPX_SCALE_E(0x1p-4f,1),  ROCMFPX_SCALE_E(0x1p-4f,2),  ROCMFPX_SCALE_E(0x1p-4f,3),
    ROCMFPX_SCALE_E(0x1p-4f,4),  ROCMFPX_SCALE_E(0x1p-4f,5),  ROCMFPX_SCALE_E(0x1p-4f,6),  ROCMFPX_SCALE_E(0x1p-4f,7),
    ROCMFPX_SCALE_E(0x1p-3f,0),  ROCMFPX_SCALE_E(0x1p-3f,1),  ROCMFPX_SCALE_E(0x1p-3f,2),  ROCMFPX_SCALE_E(0x1p-3f,3),
    ROCMFPX_SCALE_E(0x1p-3f,4),  ROCMFPX_SCALE_E(0x1p-3f,5),  ROCMFPX_SCALE_E(0x1p-3f,6),  ROCMFPX_SCALE_E(0x1p-3f,7),
    ROCMFPX_SCALE_E(0x1p-2f,0),  ROCMFPX_SCALE_E(0x1p-2f,1),  ROCMFPX_SCALE_E(0x1p-2f,2),  ROCMFPX_SCALE_E(0x1p-2f,3),
    ROCMFPX_SCALE_E(0x1p-2f,4),  ROCMFPX_SCALE_E(0x1p-2f,5),  ROCMFPX_SCALE_E(0x1p-2f,6),  ROCMFPX_SCALE_E(0x1p-2f,7),
    ROCMFPX_SCALE_E(0x1p-1f,0),  ROCMFPX_SCALE_E(0x1p-1f,1),  ROCMFPX_SCALE_E(0x1p-1f,2),  ROCMFPX_SCALE_E(0x1p-1f,3),
    ROCMFPX_SCALE_E(0x1p-1f,4),  ROCMFPX_SCALE_E(0x1p-1f,5),  ROCMFPX_SCALE_E(0x1p-1f,6),  ROCMFPX_SCALE_E(0x1p-1f,7),
    ROCMFPX_SCALE_E(0x1p0f,0),   ROCMFPX_SCALE_E(0x1p0f,1),   ROCMFPX_SCALE_E(0x1p0f,2),   ROCMFPX_SCALE_E(0x1p0f,3),
    ROCMFPX_SCALE_E(0x1p0f,4),   ROCMFPX_SCALE_E(0x1p0f,5),   ROCMFPX_SCALE_E(0x1p0f,6),   ROCMFPX_SCALE_E(0x1p0f,7),
    ROCMFPX_SCALE_E(0x1p1f,0),   ROCMFPX_SCALE_E(0x1p1f,1),   ROCMFPX_SCALE_E(0x1p1f,2),   ROCMFPX_SCALE_E(0x1p1f,3),
    ROCMFPX_SCALE_E(0x1p1f,4),   ROCMFPX_SCALE_E(0x1p1f,5),   ROCMFPX_SCALE_E(0x1p1f,6),   ROCMFPX_SCALE_E(0x1p1f,7),
    ROCMFPX_SCALE_E(0x1p2f,0),   ROCMFPX_SCALE_E(0x1p2f,1),   ROCMFPX_SCALE_E(0x1p2f,2),   ROCMFPX_SCALE_E(0x1p2f,3),
    ROCMFPX_SCALE_E(0x1p2f,4),   ROCMFPX_SCALE_E(0x1p2f,5),   ROCMFPX_SCALE_E(0x1p2f,6),   ROCMFPX_SCALE_E(0x1p2f,7),
    ROCMFPX_SCALE_E(0x1p3f,0),   ROCMFPX_SCALE_E(0x1p3f,1),   ROCMFPX_SCALE_E(0x1p3f,2),   ROCMFPX_SCALE_E(0x1p3f,3),
    ROCMFPX_SCALE_E(0x1p3f,4),   ROCMFPX_SCALE_E(0x1p3f,5),   ROCMFPX_SCALE_E(0x1p3f,6),   ROCMFPX_SCALE_E(0x1p3f,7),
    ROCMFPX_SCALE_E(0x1p4f,0),   ROCMFPX_SCALE_E(0x1p4f,1),   ROCMFPX_SCALE_E(0x1p4f,2),   ROCMFPX_SCALE_E(0x1p4f,3),
    ROCMFPX_SCALE_E(0x1p4f,4),   ROCMFPX_SCALE_E(0x1p4f,5),   ROCMFPX_SCALE_E(0x1p4f,6),
};

#undef ROCMFPX_SCALE_SUB
#undef ROCMFPX_SCALE_E

float rocmfpx_ue4m3_to_fp32(uint8_t e) {
    return rocmfpx_scale_is_valid(e) ? rocmfpx_scale_ue4m3[e] : 0.0f;
}

bool rocmfpx_scale_is_valid(uint8_t e) {
    return e <= 0x7e;
}

// Precomputed table of all 127 valid UE4M3 scale values (e = 0x00..0x7e).
// Avoids repeated ldexpf calls in the hot MSE inner loops and in nearest-scale
// binary search. Matches the style of rocmfp4_scale_ue4m3_half in rocmfp4.c.
#define ROCMFPX_SCALE_SUB(M) ((M) * 0x1p-10f)
#define ROCMFPX_SCALE_E1(M)  ((8 + (M)) * 0x1p-10f)
#define ROCMFPX_SCALE_E2(M)  ((8 + (M)) * 0x1p-9f)
#define ROCMFPX_SCALE_E3(M)  ((8 + (M)) * 0x1p-8f)
#define ROCMFPX_SCALE_E4(M)  ((8 + (M)) * 0x1p-7f)
#define ROCMFPX_SCALE_E5(M)  ((8 + (M)) * 0x1p-6f)
#define ROCMFPX_SCALE_E6(M)  ((8 + (M)) * 0x1p-5f)
#define ROCMFPX_SCALE_E7(M)  ((8 + (M)) * 0x1p-4f)
#define ROCMFPX_SCALE_E8(M)  ((8 + (M)) * 0x1p-3f)
#define ROCMFPX_SCALE_E9(M)  ((8 + (M)) * 0x1p-2f)
#define ROCMFPX_SCALE_E10(M) ((8 + (M)) * 0x1p-1f)
#define ROCMFPX_SCALE_E11(M) ((8 + (M)) * 0x1p0f)
#define ROCMFPX_SCALE_E12(M) ((8 + (M)) * 0x1p1f)
#define ROCMFPX_SCALE_E13(M) ((8 + (M)) * 0x1p2f)
#define ROCMFPX_SCALE_E14(M) ((8 + (M)) * 0x1p3f)
#define ROCMFPX_SCALE_E15(M) ((8 + (M)) * 0x1p4f)

static const float rocmfpx_scale_table[127] = {
    ROCMFPX_SCALE_SUB(0), ROCMFPX_SCALE_SUB(1), ROCMFPX_SCALE_SUB(2), ROCMFPX_SCALE_SUB(3),
    ROCMFPX_SCALE_SUB(4), ROCMFPX_SCALE_SUB(5), ROCMFPX_SCALE_SUB(6), ROCMFPX_SCALE_SUB(7),
    ROCMFPX_SCALE_E1(0),  ROCMFPX_SCALE_E1(1),  ROCMFPX_SCALE_E1(2),  ROCMFPX_SCALE_E1(3),
    ROCMFPX_SCALE_E1(4),  ROCMFPX_SCALE_E1(5),  ROCMFPX_SCALE_E1(6),  ROCMFPX_SCALE_E1(7),
    ROCMFPX_SCALE_E2(0),  ROCMFPX_SCALE_E2(1),  ROCMFPX_SCALE_E2(2),  ROCMFPX_SCALE_E2(3),
    ROCMFPX_SCALE_E2(4),  ROCMFPX_SCALE_E2(5),  ROCMFPX_SCALE_E2(6),  ROCMFPX_SCALE_E2(7),
    ROCMFPX_SCALE_E3(0),  ROCMFPX_SCALE_E3(1),  ROCMFPX_SCALE_E3(2),  ROCMFPX_SCALE_E3(3),
    ROCMFPX_SCALE_E3(4),  ROCMFPX_SCALE_E3(5),  ROCMFPX_SCALE_E3(6),  ROCMFPX_SCALE_E3(7),
    ROCMFPX_SCALE_E4(0),  ROCMFPX_SCALE_E4(1),  ROCMFPX_SCALE_E4(2),  ROCMFPX_SCALE_E4(3),
    ROCMFPX_SCALE_E4(4),  ROCMFPX_SCALE_E4(5),  ROCMFPX_SCALE_E4(6),  ROCMFPX_SCALE_E4(7),
    ROCMFPX_SCALE_E5(0),  ROCMFPX_SCALE_E5(1),  ROCMFPX_SCALE_E5(2),  ROCMFPX_SCALE_E5(3),
    ROCMFPX_SCALE_E5(4),  ROCMFPX_SCALE_E5(5),  ROCMFPX_SCALE_E5(6),  ROCMFPX_SCALE_E5(7),
    ROCMFPX_SCALE_E6(0),  ROCMFPX_SCALE_E6(1),  ROCMFPX_SCALE_E6(2),  ROCMFPX_SCALE_E6(3),
    ROCMFPX_SCALE_E6(4),  ROCMFPX_SCALE_E6(5),  ROCMFPX_SCALE_E6(6),  ROCMFPX_SCALE_E6(7),
    ROCMFPX_SCALE_E7(0),  ROCMFPX_SCALE_E7(1),  ROCMFPX_SCALE_E7(2),  ROCMFPX_SCALE_E7(3),
    ROCMFPX_SCALE_E7(4),  ROCMFPX_SCALE_E7(5),  ROCMFPX_SCALE_E7(6),  ROCMFPX_SCALE_E7(7),
    ROCMFPX_SCALE_E8(0),  ROCMFPX_SCALE_E8(1),  ROCMFPX_SCALE_E8(2),  ROCMFPX_SCALE_E8(3),
    ROCMFPX_SCALE_E8(4),  ROCMFPX_SCALE_E8(5),  ROCMFPX_SCALE_E8(6),  ROCMFPX_SCALE_E8(7),
    ROCMFPX_SCALE_E9(0),  ROCMFPX_SCALE_E9(1),  ROCMFPX_SCALE_E9(2),  ROCMFPX_SCALE_E9(3),
    ROCMFPX_SCALE_E9(4),  ROCMFPX_SCALE_E9(5),  ROCMFPX_SCALE_E9(6),  ROCMFPX_SCALE_E9(7),
    ROCMFPX_SCALE_E10(0), ROCMFPX_SCALE_E10(1), ROCMFPX_SCALE_E10(2), ROCMFPX_SCALE_E10(3),
    ROCMFPX_SCALE_E10(4), ROCMFPX_SCALE_E10(5), ROCMFPX_SCALE_E10(6), ROCMFPX_SCALE_E10(7),
    ROCMFPX_SCALE_E11(0), ROCMFPX_SCALE_E11(1), ROCMFPX_SCALE_E11(2), ROCMFPX_SCALE_E11(3),
    ROCMFPX_SCALE_E11(4), ROCMFPX_SCALE_E11(5), ROCMFPX_SCALE_E11(6), ROCMFPX_SCALE_E11(7),
    ROCMFPX_SCALE_E12(0), ROCMFPX_SCALE_E12(1), ROCMFPX_SCALE_E12(2), ROCMFPX_SCALE_E12(3),
    ROCMFPX_SCALE_E12(4), ROCMFPX_SCALE_E12(5), ROCMFPX_SCALE_E12(6), ROCMFPX_SCALE_E12(7),
    ROCMFPX_SCALE_E13(0), ROCMFPX_SCALE_E13(1), ROCMFPX_SCALE_E13(2), ROCMFPX_SCALE_E13(3),
    ROCMFPX_SCALE_E13(4), ROCMFPX_SCALE_E13(5), ROCMFPX_SCALE_E13(6), ROCMFPX_SCALE_E13(7),
    ROCMFPX_SCALE_E14(0), ROCMFPX_SCALE_E14(1), ROCMFPX_SCALE_E14(2), ROCMFPX_SCALE_E14(3),
    ROCMFPX_SCALE_E14(4), ROCMFPX_SCALE_E14(5), ROCMFPX_SCALE_E14(6), ROCMFPX_SCALE_E14(7),
    ROCMFPX_SCALE_E15(0), ROCMFPX_SCALE_E15(1), ROCMFPX_SCALE_E15(2), ROCMFPX_SCALE_E15(3),
    ROCMFPX_SCALE_E15(4), ROCMFPX_SCALE_E15(5), ROCMFPX_SCALE_E15(6),
};

#undef ROCMFPX_SCALE_SUB
#undef ROCMFPX_SCALE_E1
#undef ROCMFPX_SCALE_E2
#undef ROCMFPX_SCALE_E3
#undef ROCMFPX_SCALE_E4
#undef ROCMFPX_SCALE_E5
#undef ROCMFPX_SCALE_E6
#undef ROCMFPX_SCALE_E7
#undef ROCMFPX_SCALE_E8
#undef ROCMFPX_SCALE_E9
#undef ROCMFPX_SCALE_E10
#undef ROCMFPX_SCALE_E11
#undef ROCMFPX_SCALE_E12
#undef ROCMFPX_SCALE_E13
#undef ROCMFPX_SCALE_E14
#undef ROCMFPX_SCALE_E15

// O(1) table lookup - use this everywhere inside this file instead of calling
// the public rocmfpx_ue4m3_to_fp32() which goes through the ldexpf path.
static inline float rocmfpx_scale_lookup(uint8_t e) {
    return e <= 0x7e ? rocmfpx_scale_table[e] : 0.0f;
}

// Binary search for the UE4M3 entry nearest to `target`.
// rocmfpx_scale_table is monotonically increasing in [0..0x7e], so we can
// binary-search to a two-element window then pick the closer neighbor.
// This replaces the original O(126) linear scan. Matches rocmfp4_nearest_scale_ue4m3.
static uint8_t rocmfpx_nearest_scale_ue4m3(float target) {
    if (!(target > 0.0f) || !isfinite(target)) {
        return 0;
    }

    int lo = 1;
    int hi = 126;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (rocmfpx_scale_table[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == 1) {
        return 1;
    }

    const float hi_scale = rocmfpx_scale_table[lo];
    const float lo_scale = rocmfpx_scale_table[lo - 1];

    // Ties keep the lower (smaller) scale byte, matching the old ascending scan.
    return (target - lo_scale <= hi_scale - target) ? (uint8_t)(lo - 1) : (uint8_t) lo;
}


size_t rocmfpx_row_size_fp3(int64_t k) {
    assert(k % QK_ROCMFP3 == 0);
    return (size_t) (k / QK_ROCMFP3) * sizeof(block_rocmfp3);
}

size_t rocmfpx_row_size_fp2(int64_t k) {
    assert(k % QK_ROCMFP2 == 0);
    return (size_t) (k / QK_ROCMFP2) * sizeof(block_rocmfp2);
}

size_t rocmfpx_row_size_fp6(int64_t k) {
    assert(k % QK_ROCMFP6 == 0);
    return (size_t) (k / QK_ROCMFP6) * sizeof(block_rocmfp6);
}

size_t rocmfpx_row_size_fp8(int64_t k) {
    assert(k % QK_ROCMFP8 == 0);
    return (size_t) (k / QK_ROCMFP8) * sizeof(block_rocmfp8);
}



static float rocmfpx_max_abs(const float * x, int n) {
    float max_abs = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float ax = fabsf(x[i]);
        if (ax > max_abs) {
            max_abs = ax;
        }
    }

    return max_abs;
}

static void rocmfpx_prepare_mse_weights(
        float * dst, const float * x, int n, const float * quant_weights, float sigma2,
        float * max_abs, float * max_abs_weight, bool * all_finite) {
    *max_abs = 0.0f;
    *max_abs_weight = 0.0f;
    *all_finite = true;

    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        const float qw = quant_weights[i];
        const float weight = isfinite(qw) && qw > 0.0f && isfinite(x[i]) ? qw * sqrtf(sigma2 + x[i]*x[i]) : 0.0f;
        *all_finite = *all_finite && isfinite(x[i]);

        if (isfinite(x[i])) {
            if (ax > *max_abs) {
                *max_abs = ax;
                *max_abs_weight = weight;
            } else if (ax == *max_abs && weight > *max_abs_weight) {
                *max_abs_weight = weight;
            }
        }

        // Match llama.cpp imatrix weighting style: calibration importance is
        // scaled by row energy so large activations stay protected.
        dst[i] = weight;
    }
}

// ROCmFP2 S40 uses the frozen MORD code order {-4, -1, +1, +4}.
static inline int rocmfpx_decode_fp2_code(uint8_t code) {
    static const int8_t values[4] = { -4, -1, 1, 4 };
    return values[code & 3u];
}

static inline uint8_t rocmfpx_quantize_fp2_code(float x, float inv_scale) {
    if (!isfinite(x) || !(inv_scale > 0.0f)) {
        return 0;
    }

    const float magnitude = fabsf(x * inv_scale);
    const bool outer = magnitude > 2.5f;
    if (signbit(x)) {
        return outer ? 0u : 1u;
    }
    return outer ? 3u : 2u;
}

static float rocmfpx_fp2_group_mse_for_scale(
        const float * x, const float * mse_weights, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float reconstructed = (float) rocmfpx_decode_fp2_code(rocmfpx_quantize_fp2_code(x[i], inv_scale)) * scale;
        const float delta = x[i] - reconstructed;
        err += (mse_weights ? mse_weights[i] : 1.0f) * delta * delta;
        if (err > best_err) {
            return err;
        }
    }
    return err;
}

static uint8_t rocmfpx_choose_scale_fp2_mse(
        const float * x, int n, const float * quant_weights, float sigma2) {
    float mse_weights[QK_ROCMFP2/2];
    float max_abs = 0.0f;
    float max_abs_weight = 0.0f;
    bool all_finite = true;

    if (quant_weights) {
        rocmfpx_prepare_mse_weights(
                mse_weights, x, n, quant_weights, sigma2,
                &max_abs, &max_abs_weight, &all_finite);
    } else {
        max_abs = rocmfpx_max_abs(x, n);
        max_abs_weight = 1.0f;
    }
    GGML_UNUSED(all_finite);

    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const float * weights = quant_weights ? mse_weights : NULL;
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 4.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            const float clip_delta = max_abs - 4.0f * scale;
            const float clip_err = max_abs_weight * clip_delta * clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = rocmfpx_fp2_group_mse_for_scale(x, weights, n, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfpx_fp2_group_mse_for_scale(x, weights, n, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }
    return best_e;
}

static void rocmfpx_quantize_row_fp2_impl(
        const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y,
        int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP2 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i] * x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP2;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib * QK_ROCMFP2;
        const float * qw = quant_weights ? quant_weights + ib * QK_ROCMFP2 : NULL;
        block_rocmfp2 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP2 / 2);
            const float * xh = xb + half_off;
            const float * qh = qw ? qw + half_off : NULL;
            yb->e[half] = rocmfpx_choose_scale_fp2_mse(xh, QK_ROCMFP2 / 2, qh, sigma2);

            const float scale = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (int packed = 0; packed < 4; ++packed) {
                uint8_t byte = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    const int j = 4 * packed + lane;
                    byte |= (uint8_t) (rocmfpx_quantize_fp2_code(xh[j], inv_scale) << (2 * lane));
                }
                yb->qs[half * 4 + packed] = byte;
            }
        }
    }
}

void rocmfpx_quantize_row_fp2_ref(
        const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp2_impl(x, y, k, NULL);
}

void rocmfpx_dequantize_row_fp2(
        const block_rocmfp2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP2 == 0);
    const int64_t nb = k / QK_ROCMFP2;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp2 * xb = x + ib;
        float * yb = y + ib * QK_ROCMFP2;
        for (int half = 0; half < 2; ++half) {
            const float scale = rocmfpx_scale_lookup(xb->e[half]);
            for (int j = 0; j < QK_ROCMFP2 / 2; ++j) {
                const uint8_t code = (xb->qs[half * 4 + j / 4] >> (2 * (j % 4))) & 3u;
                yb[half * (QK_ROCMFP2 / 2) + j] = (float) rocmfpx_decode_fp2_code(code) * scale;
            }
        }
    }
}

void rocmfpx_quantize_row_fp2(
        const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp2_ref(x, (block_rocmfp2 *) y, k);
}

size_t rocmfpx_quantize_fp2(
        const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp2(n_per_row);
    char * qrow = (char *) dst;
    for (int64_t row = 0; row < nrows; ++row) {
        rocmfpx_quantize_row_fp2_impl(
                src + row * n_per_row, (block_rocmfp2 *) qrow, n_per_row, imatrix);
        qrow += row_size;
    }
    return (size_t) nrows * row_size;
}

// ---------------------------------------------------------------------------
// FP3 group pack/unpack: 8 x 3-bit codes -> 3 bytes (24 bits)
//
// Bit layout across 3 bytes (LSB first within each byte):
//   byte 0: [v0 2:0][v1 2:0][v2 1:0]
//   byte 1: [v2.2  ][v3 2:0][v4 2:0][v5.0]
//   byte 2: [v5 2:1][v6 2:0][v7 2:0]
//
// A 16-element half-block is two consecutive pack8 groups.
// Half 0 -> qs[0..5], Half 1 -> qs[6..11].
// ---------------------------------------------------------------------------
static inline void rocmfpx_fp3_pack8(uint8_t * dst, const uint8_t * c) {
    dst[0] = (uint8_t)( (c[0] & 7u)        | ((c[1] & 7u) << 3) | ((c[2] & 3u) << 6) );
    dst[1] = (uint8_t)( ((c[2] >> 2) & 1u) | ((c[3] & 7u) << 1) | ((c[4] & 7u) << 4) | ((c[5] & 1u) << 7) );
    dst[2] = (uint8_t)( ((c[5] >> 1) & 3u) | ((c[6] & 7u) << 2) | ((c[7] & 7u) << 5) );
}

static inline void rocmfpx_fp3_unpack8(const uint8_t * src, uint8_t * c) {
    c[0] =  src[0]        & 7u;
    c[1] = (src[0] >> 3)  & 7u;
    c[2] = ((src[0] >> 6) & 3u) | ((src[1] & 1u) << 2);
    c[3] = (src[1] >> 1)  & 7u;
    c[4] = (src[1] >> 4)  & 7u;
    c[5] = ((src[1] >> 7) & 1u) | ((src[2] & 3u) << 1);
    c[6] = (src[2] >> 2)  & 7u;
    c[7] = (src[2] >> 5)  & 7u;
}

// ---------------------------------------------------------------------------
// FP6 group pack/unpack: 4 x 6-bit codes -> 3 bytes (24 bits)
//
// Bit layout:
//   byte 0: [v0 5:0][v1 1:0]
//   byte 1: [v1 5:2][v2 3:0]
//   byte 2: [v2 5:4][v3 5:0]
//
// A 16-element half-block is four consecutive pack4 groups.
// Half 0 -> qs[0..11], Half 1 -> qs[12..23].
// ---------------------------------------------------------------------------
static inline void rocmfpx_fp6_pack4(uint8_t * dst, const uint8_t * c) {
    dst[0] = (uint8_t)( (c[0] & 0x3fu)        | ((c[1] & 0x03u) << 6) );
    dst[1] = (uint8_t)( ((c[1] >> 2) & 0x0fu) | ((c[2] & 0x0fu) << 4) );
    dst[2] = (uint8_t)( ((c[2] >> 4) & 0x03u) | ((c[3] & 0x3fu) << 2) );
}

static inline void rocmfpx_fp6_unpack4(const uint8_t * src, uint8_t * c) {
    c[0] =  src[0]         & 0x3fu;
    c[1] = ((src[0] >> 6)  & 0x03u) | ((src[1] & 0x0fu) << 2);
    c[2] = ((src[1] >> 4)  & 0x0fu) | ((src[2] & 0x03u) << 4);
    c[3] =  (src[2] >> 2)  & 0x3fu;
}

static int rocmfpx_decode_fp3_code(uint8_t code) {
    static const int mag[4] = { 0, 1, 2, 4 };
    const int value = mag[code & 3u];
    return (code & 4u) ? -value : value;
}

static uint8_t rocmfpx_quantize_fp3_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    const float ax = fabsf(x * inv_scale);
    uint8_t mag;

    if (ax <= 0.5f) {
        mag = 0;
    } else if (ax <= 1.5f) {
        mag = 1;
    } else if (ax <= 3.0f) {
        mag = 2;
    } else {
        mag = 3;
    }

    return mag == 0 ? 0 : (uint8_t) ((x < 0.0f ? 4u : 0u) | mag);
}

// Fused threshold + decode for finite values in the exhaustive scale search.
// The result exactly matches quantize_fp3_code followed by decode_fp3_code.
static inline float rocmfpx_fp3_decoded_mag(float x, float inv_scale) {
    const float a = fabsf(x * inv_scale);
    float mag;
    if (a <= 0.5f) {
        return 0.0f;
    } else if (a <= 1.5f) {
        mag = 1.0f;
    } else if (a <= 3.0f) {
        mag = 2.0f;
    } else {
        mag = 4.0f;
    }
    return x < 0.0f ? -mag : mag;
}

static float rocmfpx_fp3_block_mse_for_scale(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite - skips isfinite() per element.
static float rocmfpx_fp3_block_mse_for_scale_finite(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfpx_fp3_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite.
static float rocmfpx_fp3_block_weighted_mse_for_scale_finite(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfpx_fp3_decoded_mag(x[i], inv_scale) * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp3_mse_impl(
        const float * x, int n, const float * mse_weights,
        float max_abs, float max_abs_weight, bool all_finite) {
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 4.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            const float clip_delta = max_abs - 4.0f*scale;
            const float clip_err = mse_weights ? max_abs_weight*clip_delta*clip_delta : clip_delta*clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = mse_weights ?
                    (all_finite ?
                        rocmfpx_fp3_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e0, best_err) :
                        rocmfpx_fp3_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err)) :
                    (all_finite ?
                        rocmfpx_fp3_block_mse_for_scale_finite(x, n, (uint8_t) e0, best_err) :
                        rocmfpx_fp3_block_mse_for_scale(x, n, (uint8_t) e0, best_err));
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = mse_weights ?
                (all_finite ?
                    rocmfpx_fp3_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e1, best_err) :
                    rocmfpx_fp3_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err)) :
                (all_finite ?
                    rocmfpx_fp3_block_mse_for_scale_finite(x, n, (uint8_t) e1, best_err) :
                    rocmfpx_fp3_block_mse_for_scale(x, n, (uint8_t) e1, best_err));
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

static uint8_t rocmfpx_choose_scale_fp3_mse(const float * x, int n) {
    const float max_abs = rocmfpx_max_abs(x, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }
    // rocmfpx_max_abs skips non-finite, but we need to know if all are finite
    // to select the fast path. Use a simple scan here since max_abs already ran.
    bool all_finite = true;
    for (int i = 0; i < n; ++i) {
        all_finite = all_finite && isfinite(x[i]);
    }

    return rocmfpx_choose_scale_fp3_mse_impl(x, n, NULL, max_abs, 0.0f, all_finite);
}

static uint8_t rocmfpx_choose_scale_fp3_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP3);
    float mse_weights[QK_ROCMFP3];
    float max_abs;
    float max_abs_weight;
    bool all_finite;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight, &all_finite);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp3_mse_impl(x, n, mse_weights, max_abs, max_abs_weight, all_finite);
}

static int rocmfpx_decode_fp6_code(uint8_t code) {
    const int mag = code & 31u;
    return (code & 32u) ? -(mag == 0 ? 32 : mag) : mag;
}

static uint8_t rocmfpx_quantize_fp6_code(float x, float inv_scale) {
    if (!isfinite(x) || !isfinite(inv_scale) || inv_scale <= 0.0f) {
        return 0;
    }

    int q = (int) lroundf(x * inv_scale);
    if (q > 31) {
        q = 31;
    } else if (q < -32) {
        q = -32;
    }

    return q == 0 ? 0 : (uint8_t) (q < 0 ? (32u | ((uint8_t) -q & 31u)) : (uint8_t) q);
}

// Fused round, clamp, and decode for finite values in the scale search. Keep
// current main's asymmetric signed range [-32, 31], including the encoded -32
// endpoint, rather than the older experimental branch's [-31, 31] behavior.
static inline float rocmfpx_fp6_decoded_value(float x, float inv_scale) {
    int q = (int) lroundf(x * inv_scale);
    if (q > 31) {
        q = 31;
    } else if (q < -32) {
        q = -32;
    }

    return (float) q;
}

static float rocmfpx_fp6_block_mse_for_scale(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float y = rocmfpx_fp6_decoded_value(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite.
static float rocmfpx_fp6_block_mse_for_scale_finite(const float * x, int n, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfpx_fp6_decoded_value(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static float rocmfpx_fp6_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float y = rocmfpx_fp6_decoded_value(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite.
static float rocmfpx_fp6_block_weighted_mse_for_scale_finite(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float y = rocmfpx_fp6_decoded_value(x[i], inv_scale) * scale;
        const float d = x[i] - y;
        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp6_mse_impl(
        const float * x, int n, const float * mse_weights,
        float max_abs, float max_abs_weight, bool all_finite) {
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 31.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            const float clip_delta = max_abs - 31.0f*scale;
            const float clip_err = mse_weights ? max_abs_weight*clip_delta*clip_delta : clip_delta*clip_delta;
            if (clip_delta > 0.0f && clip_err > best_err) {
                lower_done = true;
            } else {
                const float err = mse_weights ?
                    (all_finite ?
                        rocmfpx_fp6_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e0, best_err) :
                        rocmfpx_fp6_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err)) :
                    (all_finite ?
                        rocmfpx_fp6_block_mse_for_scale_finite(x, n, (uint8_t) e0, best_err) :
                        rocmfpx_fp6_block_mse_for_scale(x, n, (uint8_t) e0, best_err));
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = mse_weights ?
                (all_finite ?
                    rocmfpx_fp6_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e1, best_err) :
                    rocmfpx_fp6_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err)) :
                (all_finite ?
                    rocmfpx_fp6_block_mse_for_scale_finite(x, n, (uint8_t) e1, best_err) :
                    rocmfpx_fp6_block_mse_for_scale(x, n, (uint8_t) e1, best_err));
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

static uint8_t rocmfpx_choose_scale_fp6_mse(const float * x, int n) {
    const float max_abs = rocmfpx_max_abs(x, n);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }
    bool all_finite = true;
    for (int i = 0; i < n; ++i) {
        all_finite = all_finite && isfinite(x[i]);
    }

    return rocmfpx_choose_scale_fp6_mse_impl(x, n, NULL, max_abs, 0.0f, all_finite);
}

static uint8_t rocmfpx_choose_scale_fp6_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP6);
    float mse_weights[QK_ROCMFP6];
    float max_abs;
    float max_abs_weight;
    bool all_finite;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight, &all_finite);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    return rocmfpx_choose_scale_fp6_mse_impl(x, n, mse_weights, max_abs, max_abs_weight, all_finite);
}

static int8_t rocmfpx_quantize_fp8_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    int q = (int) lroundf(x * inv_scale);
    if (q > 127) {
        q = 127;
    } else if (q < -127) {
        q = -127;
    }

    return (int8_t) q;
}

static float rocmfpx_fp8_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }

        const int8_t code = rocmfpx_quantize_fp8_code(x[i], inv_scale);
        const float y = (float) code * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

// Fast path: caller guarantees all x[i] are finite.
static float rocmfpx_fp8_block_weighted_mse_for_scale_finite(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const int8_t code = rocmfpx_quantize_fp8_code(x[i], inv_scale);
        const float y = (float) code * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_fp8_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMFP8);
    float mse_weights[QK_ROCMFP8];
    float max_abs;
    float max_abs_weight;
    bool all_finite;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight, &all_finite);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 127.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            const float clip_delta = max_abs - 127.0f*scale;
            if (clip_delta > 0.0f && max_abs_weight*clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = all_finite ?
                    rocmfpx_fp8_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e0, best_err) :
                    rocmfpx_fp8_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = all_finite ?
                rocmfpx_fp8_block_weighted_mse_for_scale_finite(x, n, mse_weights, (uint8_t) e1, best_err) :
                rocmfpx_fp8_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

void rocmfpx_quantize_row_fp3_ref(const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP3 == 0);

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP3;
        block_rocmfp3 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP3/2);
            const float * xh   = xb + half_off;
            yb->e[half] = rocmfpx_choose_scale_fp3_mse(xh, QK_ROCMFP3/2);

            const float scale     = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            // Compute 16 codes then pack into 6 bytes (2 groups of 8).
            uint8_t codes[QK_ROCMFP3/2];
            for (int j = 0; j < QK_ROCMFP3/2; ++j) {
                codes[j] = rocmfpx_quantize_fp3_code(xh[j], inv_scale);
            }
            rocmfpx_fp3_pack8(yb->qs + half_off*3/8,     codes);
            rocmfpx_fp3_pack8(yb->qs + half_off*3/8 + 3, codes + 8);
        }
    }
}

static void rocmfpx_quantize_row_fp3_weighted(
        const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP3 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i]*x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP3;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP3 : NULL;
        block_rocmfp3 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP3/2);
            const float * xh   = xb + half_off;
            const float * qh   = qw ? qw + half_off : NULL;
            yb->e[half] = qh ?
                rocmfpx_choose_scale_fp3_weighted_mse(xh, QK_ROCMFP3/2, qh, sigma2) :
                rocmfpx_choose_scale_fp3_mse(xh, QK_ROCMFP3/2);

            const float scale     = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            uint8_t codes[QK_ROCMFP3/2];
            for (int j = 0; j < QK_ROCMFP3/2; ++j) {
                codes[j] = rocmfpx_quantize_fp3_code(xh[j], inv_scale);
            }
            rocmfpx_fp3_pack8(yb->qs + half_off*3/8,     codes);
            rocmfpx_fp3_pack8(yb->qs + half_off*3/8 + 3, codes + 8);
        }
    }
}

void rocmfpx_dequantize_row_fp3(const block_rocmfp3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP3 == 0);

    const int64_t nb = k / QK_ROCMFP3;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp3 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP3;

        // Unpack all 32 codes in 4 groups of 8 (4 x 3 bytes).
        uint8_t codes[QK_ROCMFP3];
        rocmfpx_fp3_unpack8(xb->qs,      codes);
        rocmfpx_fp3_unpack8(xb->qs + 3,  codes + 8);
        rocmfpx_fp3_unpack8(xb->qs + 6,  codes + 16);
        rocmfpx_fp3_unpack8(xb->qs + 9,  codes + 24);

        for (int half = 0; half < 2; ++half) {
            const float scale = rocmfpx_scale_lookup(xb->e[half]);
            for (int j = 0; j < QK_ROCMFP3/2; ++j) {
                const int i = half*(QK_ROCMFP3/2) + j;
                yb[i] = (float) rocmfpx_decode_fp3_code(codes[i]) * scale;
            }
        }
    }
}

void rocmfpx_quantize_row_fp3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp3_ref(x, (block_rocmfp3 *) y, k);
}

size_t rocmfpx_quantize_fp3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp3(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp3_weighted(src + row*n_per_row, (block_rocmfp3 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp3_ref(src + row*n_per_row, (block_rocmfp3 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

static void rocmfpx_quantize_row_fp6_weighted(
        const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP6 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i]*x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP6;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP6 : NULL;
        block_rocmfp6 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP6/2);
            const float * xh   = xb + half_off;
            const float * qh   = qw ? qw + half_off : NULL;
            yb->e[half] = qh ?
                rocmfpx_choose_scale_fp6_weighted_mse(xh, QK_ROCMFP6/2, qh, sigma2) :
                rocmfpx_choose_scale_fp6_mse(xh, QK_ROCMFP6/2);

            const float scale     = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            // 16 codes packed into 4 groups of 4 (4 x 3 bytes = 12 bytes per half).
            uint8_t codes[QK_ROCMFP6/2];
            for (int j = 0; j < QK_ROCMFP6/2; ++j) {
                codes[j] = rocmfpx_quantize_fp6_code(xh[j], inv_scale);
            }
            uint8_t * qsdst = yb->qs + half_off * 6 / 8;
            rocmfpx_fp6_pack4(qsdst,      codes);
            rocmfpx_fp6_pack4(qsdst +  3, codes +  4);
            rocmfpx_fp6_pack4(qsdst +  6, codes +  8);
            rocmfpx_fp6_pack4(qsdst +  9, codes + 12);
        }
    }
}

void rocmfpx_quantize_row_fp6_ref(const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP6 == 0);

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP6;
        block_rocmfp6 * yb = y + ib;

        for (int half = 0; half < 2; ++half) {
            const int half_off = half * (QK_ROCMFP6/2);
            const float * xh   = xb + half_off;
            yb->e[half] = rocmfpx_choose_scale_fp6_mse(xh, QK_ROCMFP6/2);

            const float scale     = rocmfpx_scale_lookup(yb->e[half]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

            uint8_t codes[QK_ROCMFP6/2];
            for (int j = 0; j < QK_ROCMFP6/2; ++j) {
                codes[j] = rocmfpx_quantize_fp6_code(xh[j], inv_scale);
            }
            uint8_t * qsdst = yb->qs + half_off * 6 / 8;
            rocmfpx_fp6_pack4(qsdst,      codes);
            rocmfpx_fp6_pack4(qsdst +  3, codes +  4);
            rocmfpx_fp6_pack4(qsdst +  6, codes +  8);
            rocmfpx_fp6_pack4(qsdst +  9, codes + 12);
        }
    }
}

void rocmfpx_dequantize_row_fp6(const block_rocmfp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP6 == 0);

    const int64_t nb = k / QK_ROCMFP6;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp6 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP6;

        // Unpack all 32 codes in 8 groups of 4 (8 x 3 bytes = 24 bytes).
        uint8_t codes[QK_ROCMFP6];
        rocmfpx_fp6_unpack4(xb->qs,      codes);
        rocmfpx_fp6_unpack4(xb->qs +  3, codes +  4);
        rocmfpx_fp6_unpack4(xb->qs +  6, codes +  8);
        rocmfpx_fp6_unpack4(xb->qs +  9, codes + 12);
        rocmfpx_fp6_unpack4(xb->qs + 12, codes + 16);
        rocmfpx_fp6_unpack4(xb->qs + 15, codes + 20);
        rocmfpx_fp6_unpack4(xb->qs + 18, codes + 24);
        rocmfpx_fp6_unpack4(xb->qs + 21, codes + 28);

        for (int half = 0; half < 2; ++half) {
            const float scale = rocmfpx_scale_lookup(xb->e[half]);
            for (int j = 0; j < QK_ROCMFP6/2; ++j) {
                const int i = half*(QK_ROCMFP6/2) + j;
                yb[i] = (float) rocmfpx_decode_fp6_code(codes[i]) * scale;
            }
        }
    }
}

void rocmfpx_quantize_row_fp6(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp6_ref(x, (block_rocmfp6 *) y, k);
}

size_t rocmfpx_quantize_fp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp6(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp6_weighted(src + row*n_per_row, (block_rocmfp6 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp6_ref(src + row*n_per_row, (block_rocmfp6 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

static void rocmfpx_quantize_row_fp8_weighted(
        const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFP8 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i]*x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP8;
        const float * qw = quant_weights ? quant_weights + ib*QK_ROCMFP8 : NULL;
        block_rocmfp8 * yb = y + ib;

        yb->e = qw ? rocmfpx_choose_scale_fp8_weighted_mse(xb, QK_ROCMFP8, qw, sigma2) :
                     rocmfpx_nearest_scale_ue4m3(rocmfpx_max_abs(xb, QK_ROCMFP8) / 127.0f);

        const float scale = rocmfpx_scale_lookup(yb->e);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb->qs[i] = rocmfpx_quantize_fp8_code(xb[i], inv_scale);
        }
    }
}

void rocmfpx_quantize_row_fp8_ref(const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP8 == 0);

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib*QK_ROCMFP8;
        block_rocmfp8 * yb = y + ib;

        const float max_abs = rocmfpx_max_abs(xb, QK_ROCMFP8);
        yb->e = rocmfpx_nearest_scale_ue4m3(max_abs / 127.0f);

        const float scale = rocmfpx_scale_lookup(yb->e);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb->qs[i] = rocmfpx_quantize_fp8_code(xb[i], inv_scale);
        }
    }
}

void rocmfpx_dequantize_row_fp8(const block_rocmfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFP8 == 0);

    const int64_t nb = k / QK_ROCMFP8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmfp8 * xb = x + ib;
        float * yb = y + ib*QK_ROCMFP8;

        const float scale = rocmfpx_scale_lookup(xb->e);
        for (int i = 0; i < QK_ROCMFP8; ++i) {
            yb[i] = (float) xb->qs[i] * scale;
        }
    }
}

void rocmfpx_quantize_row_fp8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fp8_ref(x, (block_rocmfp8 *) y, k);
}

size_t rocmfpx_quantize_fp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_fp8(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_fp8_weighted(src + row*n_per_row, (block_rocmfp8 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_fp8_ref(src + row*n_per_row, (block_rocmfp8 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

bool rocmfpx_validate_row_data_fp2(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp2) != 0) {
        return false;
    }

    const block_rocmfp2 * blocks = (const block_rocmfp2 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp2);
    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }
    return true;
}

bool rocmfpx_validate_row_data_fp3(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp3) != 0) {
        return false;
    }

    const block_rocmfp3 * blocks = (const block_rocmfp3 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp3);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }

    return true;
}

bool rocmfpx_validate_row_data_fp6(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp6) != 0) {
        return false;
    }

    const block_rocmfp6 * blocks = (const block_rocmfp6 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp6);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e[0]) || !rocmfpx_scale_is_valid(blocks[i].e[1])) {
            return false;
        }
    }

    return true;
}

bool rocmfpx_validate_row_data_fp8(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmfp8) != 0) {
        return false;
    }

    const block_rocmfp8 * blocks = (const block_rocmfp8 *) data;
    const size_t nb = nbytes / sizeof(block_rocmfp8);

    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e)) {
            return false;
        }
    }

    return true;
}

static int8_t rocmi4_quantize_code(float x, float inv_scale) {
    if (!isfinite(x) || inv_scale <= 0.0f) {
        return 0;
    }

    int q = (int) lroundf(x * inv_scale);
    if (q > 7) {
        q = 7;
    } else if (q < -8) {
        q = -8;
    }
    return (int8_t) q;
}

static int8_t rocmi4_nibble_to_i8(uint8_t nibble) {
    return (int8_t) ((nibble & 0x8u) ? (int) (nibble | 0xF0u) : (int) (nibble & 0x7u));
}

size_t rocmfpx_row_size_i4(int64_t k) {
    assert(k % QK_ROCMI4 == 0);
    return (size_t) (k / QK_ROCMI4) * sizeof(block_rocmi4);
}

// ---------------------------------------------------------------------------
// Importance-matrix (imatrix) weighted quantisation for Q4_0_ROCMI4.
//
// ROCMI4 has exactly one free parameter per block: the UE4M3 scale. The codes
// then follow by rounding. So imatrix support means choosing the scale that
// minimises the importance-weighted error over the block, instead of simply
// taking amax/7. Same approach the FP2/FP3/FP6/FP8 siblings already use.
// ---------------------------------------------------------------------------

static float rocmfpx_i4_block_weighted_mse_for_scale(const float * x, int n, const float * mse_weights, uint8_t e, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const int8_t code = rocmi4_quantize_code(x[i], inv_scale);
        const float y = (float) code * scale;
        const float d = x[i] - y;

        err += mse_weights[i]*d*d;
        if (err > best_err) {
            return err;
        }
    }

    return err;
}

static uint8_t rocmfpx_choose_scale_i4_weighted_mse(const float * x, int n, const float * quant_weights, float sigma2) {
    assert(n <= QK_ROCMI4);
    float mse_weights[QK_ROCMI4];
    float max_abs;
    float max_abs_weight;
    bool all_finite;
    rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight, &all_finite);
    GGML_UNUSED(all_finite);
    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / 7.0f);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float scale = rocmfpx_scale_lookup((uint8_t) e0);
            // Below this scale the largest magnitude clips; once that clipping
            // error alone exceeds the best seen, no smaller scale can win.
            const float clip_delta = max_abs - 7.0f*scale;
            if (clip_delta > 0.0f && max_abs_weight*clip_delta*clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = rocmfpx_i4_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfpx_i4_block_weighted_mse_for_scale(x, n, mse_weights, (uint8_t) e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }

    return best_e;
}

static void rocmfpx_quantize_row_i4_weighted(
        const float * GGML_RESTRICT x, block_rocmi4 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMI4 == 0);

    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i]*x[i] : 0.0f;
    }
    const float sigma2 = sum_x2 / (float) k;

    const int64_t nb = k / QK_ROCMI4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib * QK_ROCMI4;
        const float * qw = quant_weights ? quant_weights + ib * QK_ROCMI4 : NULL;
        block_rocmi4 * yb = y + ib;

        yb->e = qw ? rocmfpx_choose_scale_i4_weighted_mse(xb, QK_ROCMI4, qw, sigma2) :
                     rocmfpx_nearest_scale_ue4m3(rocmfpx_max_abs(xb, QK_ROCMI4) / 7.0f);

        const float scale = rocmfpx_scale_lookup(yb->e);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        for (int j = 0; j < QS_ROCMI4; ++j) {
            const int8_t q0 = rocmi4_quantize_code(xb[j], inv_scale);
            const int8_t q1 = rocmi4_quantize_code(xb[j + QS_ROCMI4], inv_scale);
            yb->qs[j] = (uint8_t) ((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
    }
}

void rocmfpx_quantize_row_i4_ref(const float * GGML_RESTRICT x, block_rocmi4 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMI4 == 0);

    const int64_t nb = k / QK_ROCMI4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib * QK_ROCMI4;
        block_rocmi4 * yb = y + ib;

        const float max_abs = rocmfpx_max_abs(xb, QK_ROCMI4);
        yb->e = rocmfpx_nearest_scale_ue4m3(max_abs / 7.0f);

        const float scale = rocmfpx_scale_lookup(yb->e);
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        for (int j = 0; j < QS_ROCMI4; ++j) {
            const int8_t q0 = rocmi4_quantize_code(xb[j], inv_scale);
            const int8_t q1 = rocmi4_quantize_code(xb[j + QS_ROCMI4], inv_scale);
            yb->qs[j] = (uint8_t) ((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
    }
}

void rocmfpx_dequantize_row_i4(const block_rocmi4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMI4 == 0);

    const int64_t nb = k / QK_ROCMI4;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const block_rocmi4 * xb = x + ib;
        float * yb = y + ib * QK_ROCMI4;
        const float scale = rocmfpx_scale_lookup(xb->e);
        for (int j = 0; j < QS_ROCMI4; ++j) {
            yb[j]              = (float) rocmi4_nibble_to_i8(xb->qs[j] & 0x0Fu) * scale;
            yb[j + QS_ROCMI4]  = (float) rocmi4_nibble_to_i8(xb->qs[j] >> 4) * scale;
        }
    }
}

void rocmfpx_quantize_row_i4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_i4_ref(x, (block_rocmi4 *) y, k);
}

size_t rocmfpx_quantize_i4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = rocmfpx_row_size_i4(n_per_row);
    char * qrow = (char *) dst;

    for (int64_t row = 0; row < nrows; ++row) {
        if (imatrix) {
            rocmfpx_quantize_row_i4_weighted(src + row * n_per_row, (block_rocmi4 *) qrow, n_per_row, imatrix);
        } else {
            rocmfpx_quantize_row_i4_ref(src + row * n_per_row, (block_rocmi4 *) qrow, n_per_row);
        }
        qrow += row_size;
    }

    return (size_t) nrows * row_size;
}

bool rocmfpx_validate_row_data_i4(const void * data, size_t nbytes) {
    if (nbytes % sizeof(block_rocmi4) != 0) {
        return false;
    }

    const block_rocmi4 * blocks = (const block_rocmi4 *) data;
    const size_t nb = nbytes / sizeof(block_rocmi4);
    for (size_t i = 0; i < nb; ++i) {
        if (!rocmfpx_scale_is_valid(blocks[i].e)) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// ROCmFP2-L (S40 codes) and ROCmFP-T (ternary codes): 256-weight blocks, one UE4M3 scale per 64 weights.
// The scale search is the FP2 one with qmax = 4 (S40) or 1 (ternary) and groups of 64.

static inline int rocmfpx_quantize_trit(float x, float inv_scale) {
    if (!isfinite(x) || !(inv_scale > 0.0f)) {
        return 0;
    }
    const float v = x * inv_scale;
    return v > 0.5f ? 1 : (v < -0.5f ? -1 : 0);
}

static inline float rocmfpx_g64_reconstruct(float x, float scale, float inv_scale, int qmax) {
    return qmax == 1 ? (float) rocmfpx_quantize_trit(x, inv_scale) * scale
                     : (float) rocmfpx_decode_fp2_code(rocmfpx_quantize_fp2_code(x, inv_scale)) * scale;
}

static float rocmfpx_g64_mse_for_scale(const float * x, const float * w, int n, uint8_t e, int qmax, float best_err) {
    const float scale = rocmfpx_scale_lookup(e);
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    float err = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            continue;
        }
        const float delta = x[i] - rocmfpx_g64_reconstruct(x[i], scale, inv_scale, qmax);
        err += (w ? w[i] : 1.0f) * delta * delta;
        if (err > best_err) {
            return err;
        }
    }
    return err;
}

static uint8_t rocmfpx_choose_scale_g64(const float * x, int n, const float * quant_weights, float sigma2, int qmax) {
    float mse_weights[QG_ROCMFPX];
    float max_abs = 0.0f;
    float max_abs_weight = 0.0f;
    bool all_finite = true;

    assert(n <= QG_ROCMFPX);
    if (quant_weights) {
        rocmfpx_prepare_mse_weights(mse_weights, x, n, quant_weights, sigma2, &max_abs, &max_abs_weight, &all_finite);
    } else {
        max_abs = rocmfpx_max_abs(x, n);
        max_abs_weight = 1.0f;
    }
    GGML_UNUSED(all_finite);

    if (!(max_abs > 0.0f) || !isfinite(max_abs)) {
        return 0;
    }

    const float * weights = quant_weights ? mse_weights : NULL;
    const uint8_t start_e = rocmfpx_nearest_scale_ue4m3(max_abs / (float) qmax);
    uint8_t best_e = start_e;
    float best_err = INFINITY;
    bool lower_done = false;

    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = (int) start_e - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float clip_delta = max_abs - (float) qmax * rocmfpx_scale_lookup((uint8_t) e0);
            if (clip_delta > 0.0f && max_abs_weight * clip_delta * clip_delta > best_err) {
                lower_done = true;
            } else {
                const float err = rocmfpx_g64_mse_for_scale(x, weights, n, (uint8_t) e0, qmax, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) {
                    best_err = err;
                    best_e = (uint8_t) e0;
                }
            }
        }

        const int e1 = (int) start_e + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = rocmfpx_g64_mse_for_scale(x, weights, n, (uint8_t) e1, qmax, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) {
                best_err = err;
                best_e = (uint8_t) e1;
            }
        }

        if ((lower_done || e0 <= 1) && e1 >= 126) {
            break;
        }
    }
    return best_e;
}

static float rocmfpx_row_sigma2(const float * x, int64_t k) {
    float sum_x2 = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        sum_x2 += isfinite(x[i]) ? x[i] * x[i] : 0.0f;
    }
    return sum_x2 / (float) k;
}

static void rocmfpx_quantize_row_fpl_impl(const float * GGML_RESTRICT x, block_rocmfpl * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFPL == 0);
    const float sigma2 = rocmfpx_row_sigma2(x, k);
    const int64_t nb = k / QK_ROCMFPL;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib * QK_ROCMFPL;
        const float * qw = quant_weights ? quant_weights + ib * QK_ROCMFPL : NULL;
        block_rocmfpl * yb = y + ib;
        memset(yb->qs, 0, sizeof(yb->qs));
        for (int g = 0; g < QK_ROCMFPL / QG_ROCMFPX; ++g) {
            const float * xg = xb + g * QG_ROCMFPX;
            yb->e[g] = rocmfpx_choose_scale_g64(xg, QG_ROCMFPX, qw ? qw + g * QG_ROCMFPX : NULL, sigma2, 4);
            const float scale = rocmfpx_scale_lookup(yb->e[g]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (int j = 0; j < QG_ROCMFPX; ++j) {
                const int w = g * QG_ROCMFPX + j;
                yb->qs[w / 4] |= (uint8_t) (rocmfpx_quantize_fp2_code(xg[j], inv_scale) << (2 * (w % 4)));
            }
        }
    }
}

void rocmfpx_quantize_row_fpl_ref(const float * GGML_RESTRICT x, block_rocmfpl * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fpl_impl(x, y, k, NULL);
}

void rocmfpx_dequantize_row_fpl(const block_rocmfpl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMFPL == 0);
    const int64_t nb = k / QK_ROCMFPL;
    for (int64_t ib = 0; ib < nb; ++ib) {
        for (int w = 0; w < QK_ROCMFPL; ++w) {
            const uint8_t code = (x[ib].qs[w / 4] >> (2 * (w % 4))) & 3u;
            y[ib * QK_ROCMFPL + w] = (float) rocmfpx_decode_fp2_code(code) * rocmfpx_scale_lookup(x[ib].e[w / QG_ROCMFPX]);
        }
    }
}

void rocmfpx_quantize_row_fpl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fpl_ref(x, (block_rocmfpl *) y, k);
}

size_t rocmfpx_quantize_fpl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = (size_t) (n_per_row / QK_ROCMFPL) * sizeof(block_rocmfpl);
    for (int64_t row = 0; row < nrows; ++row) {
        rocmfpx_quantize_row_fpl_impl(src + row * n_per_row, (block_rocmfpl *) ((char *) dst + row * row_size), n_per_row, imatrix);
    }
    return (size_t) nrows * row_size;
}

// TQ1_0 trit order: qs[m] (m < 32) holds weights m + 32*n, qs[32 + m] holds 160 + m + 16*n, qh[j] holds 240 + j + 4*n,
// n = trit position, the first trit is the most significant one
static inline int rocmfpx_fpt_weight(int byte, int n) {
    return byte < 32 ? byte + 32 * n : (byte < 48 ? 160 + (byte - 32) + 16 * n : 240 + (byte - 48) + 4 * n);
}

static void rocmfpx_quantize_row_fpt_impl(const float * GGML_RESTRICT x, block_rocmfpt * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMFPT == 0);
    const float sigma2 = rocmfpx_row_sigma2(x, k);
    const int64_t nb = k / QK_ROCMFPT;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib * QK_ROCMFPT;
        const float * qw = quant_weights ? quant_weights + ib * QK_ROCMFPT : NULL;
        block_rocmfpt * yb = y + ib;
        uint8_t t[QK_ROCMFPT];
        for (int g = 0; g < QK_ROCMFPT / QG_ROCMFPX; ++g) {
            const float * xg = xb + g * QG_ROCMFPX;
            yb->e[g] = rocmfpx_choose_scale_g64(xg, QG_ROCMFPX, qw ? qw + g * QG_ROCMFPX : NULL, sigma2, 1);
            const float scale = rocmfpx_scale_lookup(yb->e[g]);
            const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (int j = 0; j < QG_ROCMFPX; ++j) {
                t[g * QG_ROCMFPX + j] = (uint8_t) (rocmfpx_quantize_trit(xg[j], inv_scale) + 1);
            }
        }
        for (int byte = 0; byte < 52; ++byte) {
            const int nt = byte < 48 ? 5 : 4;
            uint16_t q = 0;
            for (int n = 0; n < nt; ++n) {
                q = (uint16_t) (q * 3 + t[rocmfpx_fpt_weight(byte, n)]);
            }
            if (nt == 4) {
                q *= 3;
            }
            // ceiling division (243 == pow(3, 5)), as in TQ1_0
            const uint8_t v = (uint8_t) ((q * 256 + (243 - 1)) / 243);
            if (byte < 48) {
                yb->qs[byte] = v;
            } else {
                yb->qh[byte - 48] = v;
            }
        }
    }
}

void rocmfpx_quantize_row_fpt_ref(const float * GGML_RESTRICT x, block_rocmfpt * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fpt_impl(x, y, k, NULL);
}

void rocmfpx_dequantize_row_fpt(const block_rocmfpt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    assert(k % QK_ROCMFPT == 0);
    const int64_t nb = k / QK_ROCMFPT;
    for (int64_t ib = 0; ib < nb; ++ib) {
        float * yb = y + ib * QK_ROCMFPT;
        for (int byte = 0; byte < 52; ++byte) {
            const uint8_t v = byte < 48 ? x[ib].qs[byte] : x[ib].qh[byte - 48];
            for (int n = 0; n < (byte < 48 ? 5 : 4); ++n) {
                const uint8_t q = (uint8_t) (v * pow3[n]);
                const int w = rocmfpx_fpt_weight(byte, n);
                yb[w] = (float) ((((uint16_t) q * 3) >> 8) - 1) * rocmfpx_scale_lookup(x[ib].e[w / QG_ROCMFPX]);
            }
        }
    }
}

void rocmfpx_quantize_row_fpt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_fpt_ref(x, (block_rocmfpt *) y, k);
}

size_t rocmfpx_quantize_fpt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = (size_t) (n_per_row / QK_ROCMFPT) * sizeof(block_rocmfpt);
    for (int64_t row = 0; row < nrows; ++row) {
        rocmfpx_quantize_row_fpt_impl(src + row * n_per_row, (block_rocmfpt *) ((char *) dst + row * row_size), n_per_row, imatrix);
    }
    return (size_t) nrows * row_size;
}

static bool rocmfpx_validate_g64_scales(const uint8_t * blocks, size_t block_size, size_t scale_offset, size_t nbytes) {
    if (nbytes % block_size != 0) {
        return false;
    }
    for (size_t i = 0; i < nbytes / block_size; ++i) {
        for (int g = 0; g < 4; ++g) {
            if (!rocmfpx_scale_is_valid(blocks[i * block_size + scale_offset + g])) {
                return false;
            }
        }
    }
    return true;
}

bool rocmfpx_validate_row_data_fpl(const void * data, size_t nbytes) {
    return rocmfpx_validate_g64_scales((const uint8_t *) data, sizeof(block_rocmfpl), offsetof(block_rocmfpl, e), nbytes);
}

bool rocmfpx_validate_row_data_fpt(const void * data, size_t nbytes) {
    return rocmfpx_validate_g64_scales((const uint8_t *) data, sizeof(block_rocmfpt), offsetof(block_rocmfpt, e), nbytes);
}

// ---------------------------------------------------------------- ROCmIQ2_XXS / ROCmIQ1_S
// 256-weight blocks with a 16-bit scale word (UE4M3 d + 2-bit sub exponent per 64).  The encoder mirrors the research
// quantizer (rq3.py, diag path): imatrix * sqrt(sigma2 + x^2) weights; per 64 a free scale on a 2^(1/8) grid around the
// rms fit; d near the largest of those, every 64 picks its k; then the weighted nearest code per vector.

static const uint64_t rocmfpx_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

// fixed ROCmIQ1_S codebook: Lloyd on N(0,1)^4 (seed 0), int8 = round(c * 127 / max|c|), value = int8 / 32
static const int8_t rocmfpx_rocmiq1_codebook[64][4] = {
    {-125,  -36,  -11,   -5},  {-106,   51,   14,  -36},  { -93,   24,   28,   54},  { -88,   51,  -57,   45},
    { -77,  -50,   76,   56},  { -75,  -34,  -35,   78},  { -74,  -45,   74,  -35},  { -70,  -41,    1,  -82},
    { -70,   -5,  -95,  -58},  { -67,   -6,  -31,   -5},  { -65,  -99,    5,    5},  { -60,   43,   83,    2},
    { -51,   81,  -69,  -27},  { -47,  -65,  -83,   21},  { -43,   44,  -19,  -87},  { -42,   32,   74,  -85},
    { -41,   96,   31,   59},  { -39,   -1,   24,  -32},  { -37,   50,   -4,    7},  { -36,  -23,   33,   28},
    { -25,   -4,   26,  109},  { -22,  112,   19,  -33},  { -21,   17, -103,   45},  { -19,  -82,  -57,  -60},
    { -15,  -86,    1,   81},  { -15,    5,  -58,  -27},  { -13,  -51,  -14,   -6},  { -13,    1,  -28,   43},
    { -12,   18,   95,   73},  { -10,   48,  -33,  102},  {  -5,  -99,   40,  -53},  {  -1,  -17,  114,  -16},
    {   5,  -21,  -16, -127},  {  11,  -84,   75,   40},  {  10,   38,   40,  -28},  {  11,   27,   30,   38},
    {  11,  105,  -48,   35},  {  15,   -5,   -8,  -55},  {  17,  -33,   41,  -15},  {  21,   82,   92,    3},
    {  23,  -42,  -65,   89},  {  23,   45,  -26,   -3},  {  25,  -32,   63,  -89},  {  29, -114,  -21,   17},
    {  31,   46, -111,   -7},  {  33,  -37,   11,   44},  {  32,   65,   33,  -95},  {  32,   73,  -51,  -71},
    {  35,  -11,  -85,  -77},  {  36,  -23,  -45,    5},  {  41,  -56, -107,    3},  {  52,   74,   30,   73},
    {  61,  -20,   25,  107},  {  61,   -4,   86,   41},  {  62,    5,   11,   -4},  {  64,  -71,  -21,  -55},
    {  67,   31,  -51,   63},  {  74,   91,    7,  -12},  {  77,   25,   82,  -41},  {  87,  -69,   55,  -15},
    {  94,    7,    0,  -75},  { 100,  -57,  -26,   37},  { 101,   19,  -61,  -17},  { 121,   21,   24,   31},
};

static const float rocmfpx_rocmiq_kf[4] = { 1.0f, 0.70710678118654752f, 0.5f, 0.35355339059327376f };

static inline float rocmfpx_rocmiq_sub_scale(uint16_t sc, int g) {
    return rocmfpx_scale_lookup((uint8_t) (sc & 0xFF)) * rocmfpx_rocmiq_kf[(sc >> (8 + 2*g)) & 3];
}

static inline float rocmfpx_iq2_grid_f(int c, int j) {
    return (float) ((rocmfpx_iq2xxs_grid[c] >> (8*j)) & 0xFF) * 0.125f;
}

// codebook geometry: ROCMIQ2 = 256 x 8 magnitudes (signs from the data), ROCMIQ1 = 64 x 4 signed codes
enum { ROCMIQ_K2 = 256, ROCMIQ_D2 = 8, ROCMIQ_K1 = 64, ROCMIQ_D1 = 4 };

static inline float rocmfpx_rocmiq_cb(int iq2, int c, int j) {
    return iq2 ? rocmfpx_iq2_grid_f(c, j) : (float) rocmfpx_rocmiq1_codebook[c][j] * 0.03125f;
}

static float rocmfpx_rocmiq_rms_cb(int iq2) {
    const int K = iq2 ? ROCMIQ_K2 : ROCMIQ_K1;
    const int D = iq2 ? ROCMIQ_D2 : ROCMIQ_D1;
    double s = 0.0;
    for (int c = 0; c < K; ++c) {
        for (int j = 0; j < D; ++j) {
            const double v = rocmfpx_rocmiq_cb(iq2, c, j);
            s += v * v;
        }
    }
    return (float) sqrt(s / (K * D));
}

// per-vector terms of one 64-weight group: err(s, c) = A - (2 s B[c] - s^2 Q[c]); iq2 works on |x|
typedef struct {
    float A[16];
    float B[16][ROCMIQ_K2];
    float Q[16][ROCMIQ_K2];
} rocmiq_terms;

static void rocmfpx_rocmiq_terms(int iq2, const float * x, const float * w, rocmiq_terms * t) {
    const int K = iq2 ? ROCMIQ_K2 : ROCMIQ_K1;
    const int D = iq2 ? ROCMIQ_D2 : ROCMIQ_D1;
    const int nv = 64 / D;
    for (int v = 0; v < nv; ++v) {
        const float * xv = x + v*D;
        const float * wv = w + v*D;
        float a[8];
        float A = 0.0f;
        for (int j = 0; j < D; ++j) {
            a[j] = iq2 ? fabsf(xv[j]) : xv[j];
            A += wv[j] * a[j] * a[j];
        }
        t->A[v] = A;
        for (int c = 0; c < K; ++c) {
            float B = 0.0f;
            float Q = 0.0f;
            for (int j = 0; j < D; ++j) {
                const float cv = rocmfpx_rocmiq_cb(iq2, c, j);
                B += wv[j] * a[j] * cv;
                Q += wv[j] * cv * cv;
            }
            t->B[v][c] = B;
            t->Q[v][c] = Q;
        }
    }
}

static float rocmfpx_rocmiq_err(int iq2, const rocmiq_terms * t, float s) {
    const int K = iq2 ? ROCMIQ_K2 : ROCMIQ_K1;
    const int nv = iq2 ? 8 : 16;
    float err = 0.0f;
    for (int v = 0; v < nv; ++v) {
        float best = -INFINITY;
        for (int c = 0; c < K; ++c) {
            const float g = 2.0f * s * t->B[v][c] - s * s * t->Q[v][c];
            best = g > best ? g : best;
        }
        err += t->A[v] - best;
    }
    return err;
}

static void rocmfpx_rocmiq_weights(const float * x, const float * qw, float sigma2, float * w) {
    for (int j = 0; j < QK_ROCMIQ; ++j) {
        w[j] = qw ? (isfinite(qw[j]) && qw[j] > 0.0f ? qw[j] * sqrtf(sigma2 + x[j]*x[j]) : 0.0f) : 1.0f;
    }
}

static inline int rocmfpx_rocmiq_clamp_e(int e) {
    return e < 1 ? 1 : (e > 126 ? 126 : e);
}

// scale word of one block; t is scratch
static uint16_t rocmfpx_rocmiq_scale_word(int iq2, const float * x, const float * w, rocmiq_terms * t) {
    const float rms_c = rocmfpx_rocmiq_rms_cb(iq2);
    float amax = 0.0f;
    for (int j = 0; j < QK_ROCMIQ; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }
    if (!(amax > 0.0f) || !isfinite(amax)) {
        return 0;
    }
    // stage 1: free scale per 64 on a 2^(1/8) grid around the rms fit
    float mx = 0.0f;
    for (int g = 0; g < 4; ++g) {
        const float * xg = x + 64*g;
        float m2 = 0.0f;
        for (int j = 0; j < 64; ++j) {
            m2 += xg[j] * xg[j];
        }
        const float s0 = sqrtf(m2 / 64.0f) / rms_c;
        rocmfpx_rocmiq_terms(iq2, xg, w + 64*g, t);
        float be = INFINITY;
        float sb = s0;
        for (int m = -12; m <= 12; ++m) {
            const float s = s0 * exp2f((float) m / 8.0f);
            const float e = rocmfpx_rocmiq_err(iq2, t, s);
            if (e < be) {
                be = e;
                sb = s;
            }
        }
        mx = fmaxf(mx, sb);
    }
    if (!(mx > 0.0f)) {
        return 0;
    }
    // stage 2: d near the largest sub scale, every 64 picks its k
    const int e0 = rocmfpx_rocmiq_clamp_e(rocmfpx_nearest_scale_ue4m3(mx));
    float err[6][4][4];
    for (int g = 0; g < 4; ++g) {
        rocmfpx_rocmiq_terms(iq2, x + 64*g, w + 64*g, t);
        for (int de = 0; de < 6; ++de) {
            const int e = rocmfpx_rocmiq_clamp_e(e0 - 2 + de);
            for (int k = 0; k < 4; ++k) {
                err[de][g][k] = rocmfpx_rocmiq_err(iq2, t, rocmfpx_scale_lookup((uint8_t) e) * rocmfpx_rocmiq_kf[k]);
            }
        }
    }
    float btot = INFINITY;
    uint16_t word = 0;
    for (int de = 0; de < 6; ++de) {
        const int e = rocmfpx_rocmiq_clamp_e(e0 - 2 + de);
        float tot = 0.0f;
        uint16_t ks = 0;
        for (int g = 0; g < 4; ++g) {
            int kb = 0;
            for (int k = 1; k < 4; ++k) {
                if (err[de][g][k] < err[de][g][kb]) {
                    kb = k;
                }
            }
            tot += err[de][g][kb];
            ks |= (uint16_t) (kb << (8 + 2*g));
        }
        if (tot < btot) {
            btot = tot;
            word = (uint16_t) (e | ks);
        }
    }
    return word;
}

// weighted nearest IQ2 code of one 8-vector: signs from the data, even parity (flip the cheapest), +/-0.5 shift
static uint16_t rocmfpx_rocmiq2_code(const float * x, const float * w, float s) {
    uint16_t best_code = 0;
    float best_err = INFINITY;
    for (int sh = 0; sh < 2; ++sh) {
        const float o = s * 0.5f * (sh ? -1.0f : 1.0f);
        float a[8];
        int neg[8];
        int nneg = 0;
        for (int j = 0; j < 8; ++j) {
            const float y = x[j] - o;
            a[j] = fabsf(y);
            neg[j] = signbit(y) ? 1 : 0;
            nneg += neg[j];
        }
        int ci = 0;
        float cg = -INFINITY;
        for (int c = 0; c < 256; ++c) {
            float B = 0.0f;
            float Q = 0.0f;
            for (int j = 0; j < 8; ++j) {
                const float cv = rocmfpx_iq2_grid_f(c, j);
                B += w[j] * a[j] * cv;
                Q += w[j] * cv * cv;
            }
            const float g = 2.0f * s * B - s * s * Q;
            if (g > cg) {
                cg = g;
                ci = c;
            }
        }
        if (nneg & 1) {
            int jf = 0;
            float cf = INFINITY;
            for (int j = 0; j < 8; ++j) {
                const float cost = w[j] * a[j] * (rocmfpx_iq2_grid_f(ci, j) * s);
                if (cost < cf) {
                    cf = cost;
                    jf = j;
                }
            }
            neg[jf] ^= 1;
        }
        float err = 0.0f;
        uint16_t signs = 0;
        for (int j = 0; j < 8; ++j) {
            const float m = rocmfpx_iq2_grid_f(ci, j) * s;
            const float q = (neg[j] ? -m : m) + o;
            err += w[j] * (x[j] - q) * (x[j] - q);
            if (j < 7 && neg[j]) {
                signs |= (uint16_t) (1u << j);
            }
        }
        if (sh == 0 || err < best_err) {
            best_err = err;
            best_code = (uint16_t) (ci | (signs << 8) | (sh << 15));
        }
    }
    return best_code;
}

static void rocmfpx_quantize_row_rocmiq2_impl(const float * GGML_RESTRICT x, block_rocmiq2 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMIQ == 0);
    const float sigma2 = rocmfpx_row_sigma2(x, k);
    rocmiq_terms * t = (rocmiq_terms *) malloc(sizeof(rocmiq_terms));
    float w[QK_ROCMIQ];
    for (int64_t ib = 0; ib < k / QK_ROCMIQ; ++ib) {
        const float * xb = x + ib * QK_ROCMIQ;
        rocmfpx_rocmiq_weights(xb, quant_weights ? quant_weights + ib * QK_ROCMIQ : NULL, sigma2, w);
        const uint16_t sc = rocmfpx_rocmiq_scale_word(1, xb, w, t);
        y[ib].sc = sc;
        for (int v = 0; v < QK_ROCMIQ/8; ++v) {
            const float s = rocmfpx_rocmiq_sub_scale(sc, v / 8);
            y[ib].qs[v] = s > 0.0f ? rocmfpx_rocmiq2_code(xb + 8*v, w + 8*v, s) : 0;
        }
    }
    free(t);
}

static void rocmfpx_quantize_row_rocmiq1_impl(const float * GGML_RESTRICT x, block_rocmiq1 * GGML_RESTRICT y, int64_t k, const float * GGML_RESTRICT quant_weights) {
    assert(k % QK_ROCMIQ == 0);
    const float sigma2 = rocmfpx_row_sigma2(x, k);
    rocmiq_terms * t = (rocmiq_terms *) malloc(sizeof(rocmiq_terms));
    float w[QK_ROCMIQ];
    for (int64_t ib = 0; ib < k / QK_ROCMIQ; ++ib) {
        const float * xb = x + ib * QK_ROCMIQ;
        rocmfpx_rocmiq_weights(xb, quant_weights ? quant_weights + ib * QK_ROCMIQ : NULL, sigma2, w);
        const uint16_t sc = rocmfpx_rocmiq_scale_word(0, xb, w, t);
        y[ib].sc = sc;
        memset(y[ib].ql, 0, sizeof(y[ib].ql));
        memset(y[ib].qh, 0, sizeof(y[ib].qh));
        for (int g = 0; g < 4; ++g) {
            const float s = rocmfpx_rocmiq_sub_scale(sc, g);
            rocmfpx_rocmiq_terms(0, xb + 64*g, w + 64*g, t);
            for (int v = 0; v < 16; ++v) {
                int ci = 0;
                float cg = -INFINITY;
                for (int c = 0; c < 64; ++c) {
                    const float gv = 2.0f * s * t->B[v][c] - s * s * t->Q[v][c];
                    if (gv > cg) {
                        cg = gv;
                        ci = c;
                    }
                }
                const int i = 16*g + v;
                y[ib].ql[i/2] |= (uint8_t) ((ci & 15) << (4*(i % 2)));
                y[ib].qh[i/4] |= (uint8_t) ((ci >> 4) << (2*(i % 4)));
            }
        }
    }
    free(t);
}

void rocmfpx_rocmiq2_unpack(const block_rocmiq2 * GGML_RESTRICT x, int8_t * GGML_RESTRICT q, float * GGML_RESTRICT sc) {
    for (int g = 0; g < 4; ++g) {
        sc[g] = rocmfpx_rocmiq_sub_scale(x->sc, g) * 0.125f;
    }
    for (int v = 0; v < QK_ROCMIQ/8; ++v) {
        const uint16_t code = x->qs[v];
        const uint64_t grid = rocmfpx_iq2xxs_grid[code & 0xFF];
        unsigned signs = (code >> 8) & 0x7Fu;
        unsigned par = signs;
        par ^= par >> 4;
        par ^= par >> 2;
        par ^= par >> 1;
        signs |= (par & 1u) << 7;
        const int sh = (code >> 15) ? -4 : 4;
        for (int j = 0; j < 8; ++j) {
            const int g = (int) ((grid >> (8*j)) & 0xFF);
            q[8*v + j] = (int8_t) ((((signs >> j) & 1u) ? -g : g) + sh);
        }
    }
}

void rocmfpx_rocmiq1_unpack(const block_rocmiq1 * GGML_RESTRICT x, int8_t * GGML_RESTRICT q, float * GGML_RESTRICT sc) {
    for (int g = 0; g < 4; ++g) {
        sc[g] = rocmfpx_rocmiq_sub_scale(x->sc, g) * 0.03125f;
    }
    for (int i = 0; i < QK_ROCMIQ/4; ++i) {
        const int ci = ((x->ql[i/2] >> (4*(i % 2))) & 15) | (((x->qh[i/4] >> (2*(i % 4))) & 3) << 4);
        memcpy(q + 4*i, rocmfpx_rocmiq1_codebook[ci], 4);
    }
}

void rocmfpx_quantize_row_rocmiq2_ref(const float * GGML_RESTRICT x, block_rocmiq2 * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_rocmiq2_impl(x, y, k, NULL);
}

void rocmfpx_quantize_row_rocmiq1_ref(const float * GGML_RESTRICT x, block_rocmiq1 * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_rocmiq1_impl(x, y, k, NULL);
}

void rocmfpx_dequantize_row_rocmiq2(const block_rocmiq2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMIQ == 0);
    int8_t q[QK_ROCMIQ];
    float sc[4];
    for (int64_t ib = 0; ib < k / QK_ROCMIQ; ++ib) {
        rocmfpx_rocmiq2_unpack(&x[ib], q, sc);
        for (int j = 0; j < QK_ROCMIQ; ++j) {
            y[ib * QK_ROCMIQ + j] = (float) q[j] * sc[j / 64];
        }
    }
}

void rocmfpx_dequantize_row_rocmiq1(const block_rocmiq1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ROCMIQ == 0);
    int8_t q[QK_ROCMIQ];
    float sc[4];
    for (int64_t ib = 0; ib < k / QK_ROCMIQ; ++ib) {
        rocmfpx_rocmiq1_unpack(&x[ib], q, sc);
        for (int j = 0; j < QK_ROCMIQ; ++j) {
            y[ib * QK_ROCMIQ + j] = (float) q[j] * sc[j / 64];
        }
    }
}

void rocmfpx_quantize_row_rocmiq2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_rocmiq2_ref(x, (block_rocmiq2 *) y, k);
}

void rocmfpx_quantize_row_rocmiq1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    rocmfpx_quantize_row_rocmiq1_ref(x, (block_rocmiq1 *) y, k);
}

size_t rocmfpx_quantize_rocmiq2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = (size_t) (n_per_row / QK_ROCMIQ) * sizeof(block_rocmiq2);
    for (int64_t row = 0; row < nrows; ++row) {
        rocmfpx_quantize_row_rocmiq2_impl(src + row * n_per_row, (block_rocmiq2 *) ((char *) dst + row * row_size), n_per_row, imatrix);
    }
    return (size_t) nrows * row_size;
}

size_t rocmfpx_quantize_rocmiq1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    const size_t row_size = (size_t) (n_per_row / QK_ROCMIQ) * sizeof(block_rocmiq1);
    for (int64_t row = 0; row < nrows; ++row) {
        rocmfpx_quantize_row_rocmiq1_impl(src + row * n_per_row, (block_rocmiq1 *) ((char *) dst + row * row_size), n_per_row, imatrix);
    }
    return (size_t) nrows * row_size;
}

static bool rocmfpx_validate_rocmiq_words(const uint8_t * blocks, size_t block_size, size_t nbytes) {
    if (nbytes % block_size != 0) {
        return false;
    }
    for (size_t i = 0; i < nbytes / block_size; ++i) {
        uint16_t sc;
        memcpy(&sc, blocks + i * block_size, sizeof(sc));
        if (!rocmfpx_scale_is_valid((uint8_t) (sc & 0xFF))) {
            return false;
        }
    }
    return true;
}

bool rocmfpx_validate_row_data_rocmiq2(const void * data, size_t nbytes) {
    return rocmfpx_validate_rocmiq_words((const uint8_t *) data, sizeof(block_rocmiq2), nbytes);
}

bool rocmfpx_validate_row_data_rocmiq1(const void * data, size_t nbytes) {
    return rocmfpx_validate_rocmiq_words((const uint8_t *) data, sizeof(block_rocmiq1), nbytes);
}
