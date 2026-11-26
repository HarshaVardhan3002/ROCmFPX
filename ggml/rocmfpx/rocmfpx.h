#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QK_ROCMFPX 32

#define QK_ROCMFP2 QK_ROCMFPX
#define QK_ROCMFP3 QK_ROCMFPX
#define QK_ROCMFP6 QK_ROCMFPX
#define QK_ROCMFP8 QK_ROCMFPX

#define QS_ROCMFP2 ((QK_ROCMFP2 * 2) / 8)
#define QR_ROCMFP2 1
#define QI_ROCMFP2 (QK_ROCMFP2 / (4 * QR_ROCMFP2))
#define QS_ROCMFP3 ((QK_ROCMFP3 * 3) / 8)
#define QS_ROCMFP6 ((QK_ROCMFP6 * 6) / 8)
#define QS_ROCMFP8 QK_ROCMFP8

#define QR_ROCMFP3 1
#define QI_ROCMFP3 (QK_ROCMFP3 / (4 * QR_ROCMFP3))

#define QR_ROCMFP6 1
#define QI_ROCMFP6 (QK_ROCMFP6 / (4 * QR_ROCMFP6))

#define QR_ROCMFP8 1
#define QI_ROCMFP8 (QK_ROCMFP8 / (4 * QR_ROCMFP8))

#define QK_ROCMI4 QK_ROCMFPX
#define QS_ROCMI4 (QK_ROCMI4 / 2)
#define QR_ROCMI4 2
#define QI_ROCMI4 (QK_ROCMI4 / (4 * QR_ROCMI4))

// AMD-native experimental family layouts. The GGUF types are registered, but
// the layouts stay isolated from the promoted ROCmFP4 formats while evaluated.
typedef struct {
    uint8_t qs[QS_ROCMFP2];
    uint8_t e[2];
} block_rocmfp2;

typedef struct {
    uint8_t qs[QS_ROCMFP3];
    uint8_t e[2];
} block_rocmfp3;

typedef struct {
    uint8_t qs[QS_ROCMFP6];
    uint8_t e[2];
} block_rocmfp6;

typedef struct {
    int8_t  qs[QS_ROCMFP8];
    uint8_t e;
} block_rocmfp8;

// Native 4-bit integer block: 32 two's-complement nibbles + one UE4M3 scale.
// Layout matches Q4_0_ROCMFP4_FAST (17 bytes) so loaders stay dword-friendly,
// but codes are the integer values themselves - no Codebook10 expansion.
typedef struct {
    uint8_t qs[QS_ROCMI4];
    uint8_t e;
} block_rocmi4;

// Sub-2.5 bpw layouts: 256 weights per block and one UE4M3 scale per 64 weights (4-byte aligned blocks).
#define QK_ROCMFPL 256
#define QK_ROCMFPT 256
#define QG_ROCMFPX 64

// ROCmFP2-L, 2.125 bpw: S40 codes, weight j in bits 2*(j%4) of qs[j/4]
typedef struct {
    uint8_t qs[QK_ROCMFPL/4];
    uint8_t e[QK_ROCMFPL/QG_ROCMFPX];
} block_rocmfpl;

// ROCmFP-T, 1.75 bpw: ternary codes in the TQ1_0 order (qs: 5 per byte, qh: 4 per byte)
typedef struct {
    uint8_t qs[48];
    uint8_t qh[4];
    uint8_t e[QK_ROCMFPT/QG_ROCMFPX];
} block_rocmfpt;

// ROCmIQ family: 256 weights per block and one 16-bit scale word: bits 0-7 = UE4M3 index d, bits 8+2g..9+2g =
// sub exponent k_g of the 64 weights g (sub scale = d * 2^(-k_g/2)).
#define QK_ROCMIQ 256

// ROCmIQ2_XXS, 2.0625 bpw: one 16-bit code per 8 weights: bits 0-7 = iq2xxs_grid index (magnitudes 8, 25, 43),
// bits 8-14 = signs of weights 0..6 (weight 7: even parity), bit 15 = shift (0: +4, 1: -4).
// value = sub_scale / 8 * (+/-grid + shift)
typedef struct {
    uint16_t sc;
    uint16_t qs[QK_ROCMIQ/8];
} block_rocmiq2;

// ROCmIQ1_S, 1.5625 bpw: one 6-bit index per 4 weights into a fixed 64 x 4 int8 codebook (rocmfpx_rocmiq1_codebook),
// index i: low nibble in ql[i/2] >> 4*(i%2), high 2 bits in qh[i/4] >> 2*(i%4).  value = sub_scale / 32 * code
typedef struct {
    uint16_t sc;
    uint8_t  ql[QK_ROCMIQ/8];
    uint8_t  qh[QK_ROCMIQ/16];
} block_rocmiq1;

#if defined(__cplusplus)
static_assert(sizeof(block_rocmiq2) == 66, "wrong rocmiq2 block size/padding");
static_assert(sizeof(block_rocmiq1) == 50, "wrong rocmiq1 block size/padding");
#else
_Static_assert(sizeof(block_rocmiq2) == 66, "wrong rocmiq2 block size/padding");
_Static_assert(sizeof(block_rocmiq1) == 50, "wrong rocmiq1 block size/padding");
#endif

#if defined(__cplusplus)
static_assert(sizeof(block_rocmfp2) == QS_ROCMFP2 + 2*sizeof(uint8_t), "wrong rocmfp2 block size/padding");
static_assert(sizeof(block_rocmfp3) == QS_ROCMFP3 + 2*sizeof(uint8_t), "wrong rocmfp3 block size/padding");
static_assert(sizeof(block_rocmfp6) == QS_ROCMFP6 + 2*sizeof(uint8_t), "wrong rocmfp6 block size/padding");
static_assert(sizeof(block_rocmfp8) == QS_ROCMFP8 + sizeof(uint8_t), "wrong rocmfp8 block size/padding");
static_assert(sizeof(block_rocmi4) == QS_ROCMI4 + sizeof(uint8_t), "wrong rocmi4 block size/padding");
static_assert(sizeof(block_rocmfpl) == 68, "wrong rocmfpl block size/padding");
static_assert(sizeof(block_rocmfpt) == 56, "wrong rocmfpt block size/padding");
#else
_Static_assert(sizeof(block_rocmfp2) == QS_ROCMFP2 + 2*sizeof(uint8_t), "wrong rocmfp2 block size/padding");
_Static_assert(sizeof(block_rocmfp3) == QS_ROCMFP3 + 2*sizeof(uint8_t), "wrong rocmfp3 block size/padding");
_Static_assert(sizeof(block_rocmfp6) == QS_ROCMFP6 + 2*sizeof(uint8_t), "wrong rocmfp6 block size/padding");
_Static_assert(sizeof(block_rocmfp8) == QS_ROCMFP8 + sizeof(uint8_t), "wrong rocmfp8 block size/padding");
_Static_assert(sizeof(block_rocmi4) == QS_ROCMI4 + sizeof(uint8_t), "wrong rocmi4 block size/padding");
_Static_assert(sizeof(block_rocmfpl) == 68, "wrong rocmfpl block size/padding");
_Static_assert(sizeof(block_rocmfpt) == 56, "wrong rocmfpt block size/padding");
#endif

GGML_API float  rocmfpx_ue4m3_to_fp32(uint8_t e);
GGML_API bool   rocmfpx_scale_is_valid(uint8_t e);
GGML_API size_t rocmfpx_row_size_fp2(int64_t k);
GGML_API size_t rocmfpx_row_size_fp3(int64_t k);
GGML_API size_t rocmfpx_row_size_fp6(int64_t k);
GGML_API size_t rocmfpx_row_size_fp8(int64_t k);

GGML_API void   rocmfpx_quantize_row_fp2_ref(const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp2(const block_rocmfp2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp3_ref(const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp3(const block_rocmfp3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp6_ref(const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp6(const block_rocmfp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp6(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp8_ref(const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp8(const block_rocmfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API bool rocmfpx_validate_row_data_fp2(const void * data, size_t nbytes);
GGML_API bool rocmfpx_validate_row_data_fp3(const void * data, size_t nbytes);
GGML_API bool rocmfpx_validate_row_data_fp6(const void * data, size_t nbytes);
GGML_API bool rocmfpx_validate_row_data_fp8(const void * data, size_t nbytes);

GGML_API size_t rocmfpx_row_size_i4(int64_t k);
GGML_API void   rocmfpx_quantize_row_i4_ref(const float * GGML_RESTRICT x, block_rocmi4 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_i4(const block_rocmi4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_i4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_i4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API bool   rocmfpx_validate_row_data_i4(const void * data, size_t nbytes);

GGML_API void   rocmfpx_quantize_row_fpl_ref(const float * GGML_RESTRICT x, block_rocmfpl * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fpl(const block_rocmfpl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fpl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fpl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API bool   rocmfpx_validate_row_data_fpl(const void * data, size_t nbytes);

GGML_API void   rocmfpx_quantize_row_fpt_ref(const float * GGML_RESTRICT x, block_rocmfpt * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fpt(const block_rocmfpt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fpt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fpt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API bool   rocmfpx_validate_row_data_fpt(const void * data, size_t nbytes);

// ROCmIQ: decode one block to int8 codes plus the four float sub scales (dot = sum_g sc[g] * sum_j q[64g+j] y[64g+j])
GGML_API void   rocmfpx_rocmiq2_unpack(const block_rocmiq2 * GGML_RESTRICT x, int8_t * GGML_RESTRICT q, float * GGML_RESTRICT sc);
GGML_API void   rocmfpx_rocmiq1_unpack(const block_rocmiq1 * GGML_RESTRICT x, int8_t * GGML_RESTRICT q, float * GGML_RESTRICT sc);

GGML_API void   rocmfpx_quantize_row_rocmiq2_ref(const float * GGML_RESTRICT x, block_rocmiq2 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_rocmiq2(const block_rocmiq2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_rocmiq2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_rocmiq2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API bool   rocmfpx_validate_row_data_rocmiq2(const void * data, size_t nbytes);

GGML_API void   rocmfpx_quantize_row_rocmiq1_ref(const float * GGML_RESTRICT x, block_rocmiq1 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_rocmiq1(const block_rocmiq1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_rocmiq1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_rocmiq1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API bool   rocmfpx_validate_row_data_rocmiq1(const void * data, size_t nbytes);

#ifdef __cplusplus
}
#endif
