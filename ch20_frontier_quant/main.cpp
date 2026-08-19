// ch20_frontier_quant/main.cpp
// 前沿量化：K-quants (Q2_K ~ Q6_K) 与 MXFP4 微观缩放
//
// 本章重点：
//   1. Q2_K 格式：256 元素 super-block，16 个子 block 共享 6-bit 量化尺度
//   2. Q6_K 格式：256 元素 super-block，6-bit 值打包，子 block 共享尺度
//   3. MXFP4 格式：OCP 标准 E2M1 微观缩放格式
//   4. 对比 FP32、Q4_0 (ch18)、Q2_K、Q6_K、MXFP4 的压缩比与精度
//   5. 展示 K-quant 系列在精度-压缩比之间的权衡

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdio>
#include <chrono>

// ============================================================
// 1. 数据类型定义
// ============================================================

// ---- Q2_K: 256 元素 super-block, 16 sub-blocks × 16 元素 ----
// 每个 super-block 84 字节（21 × uint32）
struct Q2K_Block {
    uint16_t d;           // fp16 super-block scale
    uint16_t dmin;        // fp16 super-block minimum
    uint8_t  scales[16];  // 16 × 6-bit sub-block scales（低 6 位有效）
    uint8_t  qs[64];      // 256 × 2-bit 值，packed 4/字节
    // 总大小 = 2 + 2 + 16 + 64 = 84 字节
};
static_assert(sizeof(Q2K_Block) == 84, "Q2K_Block must be 84 bytes");

struct Q2K_Weight {
    std::vector<uint8_t> data;
    uint32_t blocksPerCol;
    uint32_t K;
    uint32_t N;
};

// ---- Q6_K: 256 元素 super-block, 16 sub-blocks × 16 元素 ----
// 每个 super-block 212 字节（53 × uint32）
struct Q6K_Block {
    uint16_t d;           // fp16 super-block scale
    uint16_t dmin;        // fp16 super-block minimum
    uint8_t  scales[16];  // 16 × 6-bit sub-block scales
    uint8_t  qs[192];     // 256 × 6-bit 值，packed 4 值/3 字节
    // 总大小 = 2 + 2 + 16 + 192 = 212 字节
};
static_assert(sizeof(Q6K_Block) == 212, "Q6K_Block must be 212 bytes");

struct Q6K_Weight {
    std::vector<uint8_t> data;
    uint32_t blocksPerCol;
    uint32_t K;
    uint32_t N;
};

// ---- MXFP4: 32 元素 block, E2M1 格式 ----
// 每个 block 20 字节（5 × uint32）
struct MXFP4_Block {
    uint16_t d;           // fp16 scale
    uint8_t  qs[16];      // 32 × 4-bit E2M1 值，packed 2/字节
    uint8_t  padding[2];  // 填充到 20 字节
};
static_assert(sizeof(MXFP4_Block) == 20, "MXFP4_Block must be 20 bytes");

struct MXFP4_Weight {
    std::vector<uint8_t> data;
    uint32_t blocksPerCol;
    uint32_t K;
    uint32_t N;
};

// ---- Q4_0 格式（来自 ch18，用于对比）----
struct Q4_0_Block {
    uint16_t d;
    uint8_t  qs[16];
    uint8_t  padding[2];
};
static_assert(sizeof(Q4_0_Block) == 20, "Q4_0_Block must be 20 bytes");

struct Q4_0_Weight {
    std::vector<uint8_t> data;
    uint32_t blocksPerCol;
    uint32_t K;
    uint32_t N;
};

// Push constant 结构（与着色器一致）
struct PC20 {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t hasBias;
};

struct PC_GEMM {
    uint32_t M, N, K;
};

// ============================================================
// 2. FP16 转换辅助函数
// ============================================================

uint16_t fp32_to_fp16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp == 0) return (uint16_t)(sign << 15);
    if (exp == 255) {
        uint16_t h = (uint16_t)(sign << 15) | (31 << 10);
        if (mant) h |= 0x3FF;
        return h;
    }
    exp = exp - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign << 15) | (31 << 10);
    if (exp <= 0) {
        if (exp <= -10) return (uint16_t)(sign << 15);
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign << 15) | (uint16_t)(mant >> 13);
    }
    return (uint16_t)(sign << 15) | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

float fp16_to_fp32(uint16_t bits) {
    uint32_t sign = (uint32_t)((bits >> 15) & 1);
    uint32_t exp  = (uint32_t)((bits >> 10) & 0x1F);
    uint32_t mant = (uint32_t)(bits & 0x3FF);
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) { f32 = sign << 31; }
        else {
            int32_t e = -1; uint32_t m = mant;
            while (m & 0x400) { m <<= 1; e--; }
            m &= 0x3FF;
            f32 = (sign << 31) | ((uint32_t)(e + 127 + 10) << 24) | (m << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
        if (mant) f32 |= 0x7FFFFF;
    } else {
        f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

// ============================================================
// 3. CPU 端量化函数
// ============================================================

// ---- Q2_K 量化 ----
Q2K_Weight quantize_q2_k(const float* w, uint32_t K, uint32_t N) {
    const uint32_t superBlockSize = 256;
    const uint32_t subBlockSize = 16;
    const uint32_t numSubs = superBlockSize / subBlockSize;
    uint32_t blocksPerCol = (K + superBlockSize - 1) / superBlockSize;

    Q2K_Weight qw;
    qw.data.resize(blocksPerCol * N * sizeof(Q2K_Block));
    qw.blocksPerCol = blocksPerCol;
    qw.K = K;
    qw.N = N;

    Q2K_Block* blocks = reinterpret_cast<Q2K_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < blocksPerCol; ++b) {
            uint32_t kStart = b * superBlockSize;
            uint32_t blockEnd = std::min(kStart + superBlockSize, K);
            Q2K_Block& block = blocks[n * blocksPerCol + b];

            // 计算每个 sub-block 的最大绝对值
            float subMax[numSubs] = {0};
            for (uint32_t s = 0; s < numSubs; ++s) {
                uint32_t subStart = kStart + s * subBlockSize;
                uint32_t subEnd = std::min(subStart + subBlockSize, K);
                for (uint32_t k = subStart; k < subEnd; ++k) {
                    subMax[s] = std::max(subMax[s], std::fabs(w[k * N + n]));
                }
            }

            // super-block 的全局 scale
            float globalMax = 0.0f;
            for (uint32_t s = 0; s < numSubs; ++s) {
                globalMax = std::max(globalMax, subMax[s]);
            }
            float d = (globalMax == 0.0f) ? 1.0f : globalMax / 2.0f;  // 2-bit 范围 [-1, 2]
            block.d = fp32_to_fp16(d);

            float dmin = 1.0f;
            block.dmin = fp32_to_fp16(dmin);

            // 量化每个 sub-block
            for (uint32_t s = 0; s < numSubs; ++s) {
                uint32_t subStart = kStart + s * subBlockSize;
                uint32_t subEnd = std::min(subStart + subBlockSize, K);

                // 计算 6-bit scale
                float maxA = subMax[s];
                if (maxA == 0.0f) maxA = 0.001f;
                // scale6 = clamp(round(maxA / d * 4), 1, 63)
                float scaleRatio = maxA / d;
                uint32_t scale6 = (uint32_t)std::lroundf(scaleRatio * 32.0f);
                scale6 = std::max(1u, std::min(63u, scale6));
                block.scales[s] = (uint8_t)(scale6 & 0x3F);

                // 量化 2-bit 值
                float subScale = d * (float)scale6;
                for (uint32_t k = 0; k < subBlockSize; ++k) {
                    uint32_t ki = subStart + k;
                    float val = (ki < K) ? w[ki * N + n] : 0.0f;
                    // 2-bit 值范围 [0, 3]，中心化到 [-1, 2]
                    float qf = val / subScale + 1.0f;
                    int q = (int)std::lroundf(qf);
                    q = std::max(0, std::min(3, q));

                    // 打包 4 个值/字节
                    uint32_t elemInSuper = s * subBlockSize + k;
                    uint32_t byteIdx = elemInSuper / 4;
                    uint32_t shift = (elemInSuper % 4) * 2;
                    block.qs[byteIdx] &= ~(0x3 << shift);
                    block.qs[byteIdx] |= (uint8_t)(q << shift);
                }
            }
        }
    }
    return qw;
}

// ---- Q6_K 量化 ----
// 6-bit 值打包：4 个值 v0..v3 占 3 字节
// byte[0] = (v1 << 6) | v0
// byte[1] = (v2 << 4) | (v1 >> 2)
// byte[2] = (v3 << 2) | (v2 >> 4)
static void pack_q6k_4vals(uint8_t* dst, int v0, int v1, int v2, int v3) {
    dst[0] = (uint8_t)((v1 & 0x3F) << 6) | (uint8_t)(v0 & 0x3F);
    dst[1] = (uint8_t)((v2 & 0x0F) << 4) | (uint8_t)((v1 >> 2) & 0x0F);
    dst[2] = (uint8_t)((v3 & 0x3F) << 2) | (uint8_t)((v2 >> 4) & 0x03);
}

Q6K_Weight quantize_q6_k(const float* w, uint32_t K, uint32_t N) {
    const uint32_t superBlockSize = 256;
    const uint32_t subBlockSize = 16;
    const uint32_t numSubs = superBlockSize / subBlockSize;
    uint32_t blocksPerCol = (K + superBlockSize - 1) / superBlockSize;

    Q6K_Weight qw;
    qw.data.resize(blocksPerCol * N * sizeof(Q6K_Block));
    qw.blocksPerCol = blocksPerCol;
    qw.K = K;
    qw.N = N;

    Q6K_Block* blocks = reinterpret_cast<Q6K_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < blocksPerCol; ++b) {
            uint32_t kStart = b * superBlockSize;
            uint32_t blockEnd = std::min(kStart + superBlockSize, K);
            Q6K_Block& block = blocks[n * blocksPerCol + b];

            // 计算每个 sub-block 的最大绝对值
            float subMax[numSubs] = {0};
            for (uint32_t s = 0; s < numSubs; ++s) {
                uint32_t subStart = kStart + s * subBlockSize;
                uint32_t subEnd = std::min(subStart + subBlockSize, K);
                for (uint32_t k = subStart; k < subEnd; ++k) {
                    subMax[s] = std::max(subMax[s], std::fabs(w[k * N + n]));
                }
            }

            // super-block 的全局 scale
            float globalMax = 0.0f;
            for (uint32_t s = 0; s < numSubs; ++s) {
                globalMax = std::max(globalMax, subMax[s]);
            }
            float d = (globalMax == 0.0f) ? 1.0f : globalMax / 31.0f;  // 6-bit 范围 [-32, 31]
            block.d = fp32_to_fp16(d);
            block.dmin = fp32_to_fp16(1.0f);

            // 量化每个 sub-block
            for (uint32_t s = 0; s < numSubs; ++s) {
                uint32_t subStart = kStart + s * subBlockSize;
                uint32_t subEnd = std::min(subStart + subBlockSize, K);

                float maxA = subMax[s];
                if (maxA == 0.0f) maxA = 0.001f;
                float scaleRatio = maxA / d;
                uint32_t scale6 = (uint32_t)std::lroundf(scaleRatio);
                scale6 = std::max(1u, std::min(63u, scale6));
                block.scales[s] = (uint8_t)(scale6 & 0x3F);

                // 量化 6-bit 值
                float subScale = d * (float)scale6;
                int vals[subBlockSize];
                for (uint32_t k = 0; k < subBlockSize; ++k) {
                    uint32_t ki = subStart + k;
                    float val = (ki < K) ? w[ki * N + n] : 0.0f;
                    float qf = val / subScale;
                    int q = (int)std::lroundf(qf);
                    q = std::max(-32, std::min(31, q));
                    vals[k] = q & 0x3F;  // 6-bit 无符号表示
                }

                // 打包 4 值/3 字节
                uint32_t elemInSuper = s * subBlockSize;
                for (uint32_t k = 0; k < subBlockSize; k += 4) {
                    uint32_t quartetIdx = elemInSuper / 4 + k / 4;
                    pack_q6k_4vals(&block.qs[quartetIdx * 3],
                        vals[k], vals[k+1], vals[k+2], vals[k+3]);
                }
            }
        }
    }
    return qw;
}

// ---- MXFP4 量化 ----
// E2M1: 1 sign + 2 exp + 1 mant, bias=1
// 可表示值（正数）：0, 0.25, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
static uint8_t encode_e2m1(float v) {
    if (v == 0.0f) return 0;

    uint8_t sign = (v < 0.0f) ? 0x8 : 0x0;
    float absV = std::fabs(v);

    // 找到最近的 E2M1 可表示值
    const float e2m1_vals[8] = {0.0f, 0.25f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int bestIdx = 0;
    float bestErr = 1e10f;
    for (int i = 0; i < 8; ++i) {
        float err = std::fabs(absV - e2m1_vals[i]);
        if (err < bestErr) {
            bestErr = err;
            bestIdx = i;
        }
    }
    return sign | (uint8_t)bestIdx;
}

static float decode_e2m1(uint8_t val) {
    static const float e2m1_vals[8] = {0.0f, 0.25f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    uint8_t idx = val & 0x7;
    float f = e2m1_vals[idx];
    return (val & 0x8) ? -f : f;
}

MXFP4_Weight quantize_mxfp4(const float* w, uint32_t K, uint32_t N) {
    const uint32_t blockSize = 32;
    uint32_t blocksPerCol = (K + blockSize - 1) / blockSize;

    MXFP4_Weight qw;
    qw.data.resize(blocksPerCol * N * sizeof(MXFP4_Block));
    qw.blocksPerCol = blocksPerCol;
    qw.K = K;
    qw.N = N;

    MXFP4_Block* blocks = reinterpret_cast<MXFP4_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < blocksPerCol; ++b) {
            uint32_t kStart = b * blockSize;
            uint32_t blockEnd = std::min(kStart + blockSize, K);
            MXFP4_Block& block = blocks[n * blocksPerCol + b];

            // 找 block 内最大绝对值
            float maxAbs = 0.0f;
            for (uint32_t k = kStart; k < blockEnd; ++k) {
                maxAbs = std::max(maxAbs, std::fabs(w[k * N + n]));
            }

            // 计算 scale：E2M1 最大可表示 6.0
            float scale = (maxAbs == 0.0f) ? 1.0f : maxAbs / 6.0f;
            block.d = fp32_to_fp16(scale);

            // 量化
            for (uint32_t k = 0; k < blockSize; ++k) {
                uint32_t ki = kStart + k;
                float val = (ki < K) ? w[ki * N + n] : 0.0f;
                float quantized = val / scale;
                uint8_t e2m1 = encode_e2m1(quantized);

                // 打包 2 值/字节
                uint32_t byteIdx = k / 2;
                uint32_t shift = (k % 2) * 4;
                block.qs[byteIdx] &= ~(0xF << shift);
                block.qs[byteIdx] |= (e2m1 << shift);
            }
        }
    }
    return qw;
}

// ---- Q4_0 量化（来自 ch18，用于对比）----
Q4_0_Weight quantize_q4_0(const float* w, uint32_t K, uint32_t N) {
    const uint32_t blockSize = 32;
    uint32_t blocksPerCol = (K + blockSize - 1) / blockSize;
    Q4_0_Weight qw;
    qw.data.resize(blocksPerCol * N * sizeof(Q4_0_Block));
    qw.blocksPerCol = blocksPerCol;
    qw.K = K;
    qw.N = N;
    Q4_0_Block* blocks = reinterpret_cast<Q4_0_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < blocksPerCol; ++b) {
            uint32_t kStart = b * blockSize;
            uint32_t blockEnd = std::min(kStart + blockSize, K);
            float maxAbs = 0.0f;
            for (uint32_t k = kStart; k < blockEnd; ++k)
                maxAbs = std::max(maxAbs, std::fabs(w[k * N + n]));
            float scale = (maxAbs == 0.0f) ? 1.0f : maxAbs / 7.0f;
            Q4_0_Block& block = blocks[n * blocksPerCol + b];
            block.d = fp32_to_fp16(scale);
            for (uint32_t k = 0; k < blockSize; k += 2) {
                uint32_t ki0 = kStart + k, ki1 = kStart + k + 1;
                int q0 = 0, q1 = 0;
                if (ki0 < K) {
                    q0 = (int)std::lroundf(w[ki0 * N + n] / scale);
                    q0 = std::max(-8, std::min(7, q0));
                }
                if (ki1 < K) {
                    q1 = (int)std::lroundf(w[ki1 * N + n] / scale);
                    q1 = std::max(-8, std::min(7, q1));
                }
                block.qs[k / 2] = (uint8_t)(q0 & 0xF) | (uint8_t)((q1 & 0xF) << 4);
            }
        }
    }
    return qw;
}

// ============================================================
// 4. CPU 端反量化（用于验证）
// ============================================================

std::vector<float> dequantize_q2_k_cpu(const Q2K_Weight& qw) {
    uint32_t K = qw.K, N = qw.N;
    std::vector<float> result(K * N);
    const Q2K_Block* blocks = reinterpret_cast<const Q2K_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < qw.blocksPerCol; ++b) {
            const Q2K_Block& block = blocks[n * qw.blocksPerCol + b];
            float d = fp16_to_fp32(block.d);
            uint32_t kStart = b * 256;

            for (uint32_t s = 0; s < 16; ++s) {
                uint32_t scale6 = block.scales[s] & 0x3F;
                float subScale = d * (float)scale6;
                for (uint32_t k = 0; k < 16; ++k) {
                    uint32_t elemInSuper = s * 16 + k;
                    uint32_t byteIdx = elemInSuper / 4;
                    uint32_t shift = (elemInSuper % 4) * 2;
                    int q = (block.qs[byteIdx] >> shift) & 0x3;
                    float val = (float)(q - 1) * subScale;
                    if (kStart + elemInSuper < K)
                        result[(kStart + elemInSuper) * N + n] = val;
                }
            }
        }
    }
    return result;
}

std::vector<float> dequantize_q6_k_cpu(const Q6K_Weight& qw) {
    uint32_t K = qw.K, N = qw.N;
    std::vector<float> result(K * N);
    const Q6K_Block* blocks = reinterpret_cast<const Q6K_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < qw.blocksPerCol; ++b) {
            const Q6K_Block& block = blocks[n * qw.blocksPerCol + b];
            float d = fp16_to_fp32(block.d);
            uint32_t kStart = b * 256;

            for (uint32_t s = 0; s < 16; ++s) {
                uint32_t scale6 = block.scales[s] & 0x3F;
                float subScale = d * (float)scale6;
                for (uint32_t k = 0; k < 16; ++k) {
                    uint32_t elemInSuper = s * 16 + k;
                    // 解包 6-bit 值
                    uint32_t quartetIdx = elemInSuper / 4;
                    uint32_t elemInQuartet = elemInSuper % 4;
                    uint32_t byteOff = quartetIdx * 3;
                    int q;
                    uint8_t b0 = block.qs[byteOff];
                    uint8_t b1 = block.qs[byteOff + 1];
                    uint8_t b2 = block.qs[byteOff + 2];
                    if (elemInQuartet == 0) q = b0 & 0x3F;
                    else if (elemInQuartet == 1) q = ((b0 >> 6) & 0x3) | ((b1 & 0xF) << 2);
                    else if (elemInQuartet == 2) q = ((b1 >> 4) & 0xF) | ((b2 & 0x3) << 4);
                    else q = (b2 >> 2) & 0x3F;
                    if (q >= 32) q -= 64;
                    float val = (float)q * subScale;
                    if (kStart + elemInSuper < K)
                        result[(kStart + elemInSuper) * N + n] = val;
                }
            }
        }
    }
    return result;
}

std::vector<float> dequantize_mxfp4_cpu(const MXFP4_Weight& qw) {
    uint32_t K = qw.K, N = qw.N;
    std::vector<float> result(K * N);
    const MXFP4_Block* blocks = reinterpret_cast<const MXFP4_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < qw.blocksPerCol; ++b) {
            const MXFP4_Block& block = blocks[n * qw.blocksPerCol + b];
            float scale = fp16_to_fp32(block.d);
            uint32_t kStart = b * 32;
            for (uint32_t k = 0; k < 32; ++k) {
                uint32_t byteIdx = k / 2;
                uint32_t shift = (k % 2) * 4;
                uint8_t e2m1 = (block.qs[byteIdx] >> shift) & 0xF;
                float val = decode_e2m1(e2m1) * scale;
                if (kStart + k < K)
                    result[(kStart + k) * N + n] = val;
            }
        }
    }
    return result;
}

std::vector<float> dequantize_q4_0_cpu(const Q4_0_Weight& qw) {
    uint32_t K = qw.K, N = qw.N;
    std::vector<float> result(K * N);
    const Q4_0_Block* blocks = reinterpret_cast<const Q4_0_Block*>(qw.data.data());
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < qw.blocksPerCol; ++b) {
            const Q4_0_Block& block = blocks[n * qw.blocksPerCol + b];
            float scale = fp16_to_fp32(block.d);
            uint32_t kStart = b * 32;
            for (uint32_t k = 0; k < 32; k += 2) {
                uint8_t packed = block.qs[k / 2];
                int q0 = (int)(packed & 0xF), q1 = (int)((packed >> 4) & 0xF);
                if (q0 >= 8) q0 -= 16; if (q1 >= 8) q1 -= 16;
                if (kStart + k < K) result[(kStart + k) * N + n] = (float)q0 * scale;
                if (kStart + k + 1 < K) result[(kStart + k + 1) * N + n] = (float)q1 * scale;
            }
        }
    }
    return result;
}

// ============================================================
// 5. CPU 参考 GEMM 与验证
// ============================================================

std::vector<float> gemm_ref_fp32(const float* a, const float* b,
                                  uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> c(M * N, 0.0f);
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t col = 0; col < N; ++col) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                s += a[r * K + k] * b[k * N + col];
            c[r * N + col] = s;
        }
    return c;
}

struct VerifyResult {
    uint32_t errorCount;
    float maxError;
    float rmse;
};

VerifyResult verify(const float* gpu, const float* ref,
                    uint32_t M, uint32_t N, float relTol) {
    VerifyResult vr{0, 0.0f, 0.0f};
    double sqSum = 0.0;
    uint32_t total = M * N;
    for (uint32_t i = 0; i < total; ++i) {
        float diff = std::fabs(gpu[i] - ref[i]);
        float maxAllowed = std::max(1e-4f, std::fabs(ref[i]) * relTol);
        if (diff > maxAllowed) vr.errorCount++;
        vr.maxError = std::max(vr.maxError, diff);
        sqSum += (double)diff * diff;
    }
    vr.rmse = (float)std::sqrt(sqSum / (double)total);
    return vr;
}

// ============================================================
// 6. GPU 计时辅助
// ============================================================

double bench_gpu_ms(VulkanContext& ctx, ComputePass& pass,
                    const void* pc, size_t pcSize,
                    uint32_t gx, uint32_t gy, uint32_t gz) {
    vk::QueryPoolCreateInfo qpci{};
    qpci.queryType = vk::QueryType::eTimestamp;
    qpci.queryCount = 2;
    vk::QueryPool pool = ctx.device.createQueryPool(qpci);

    vk::CommandBuffer cmd =
        ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 0);
    pass.dispatch(cmd, pc, pcSize, gx, gy, gz);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 1);
    cmd.end();

    vk::SubmitInfo si; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    ctx.queue.submit(1, &si, ctx.fence);
    ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
    ctx.device.resetFences(1, &ctx.fence);

    std::vector<uint64_t> ts(2);
    ctx.device.getQueryPoolResults(pool, 0, 2, ts.size() * sizeof(uint64_t), ts.data(),
        sizeof(uint64_t), vk::QueryResultFlagBits::eWait | vk::QueryResultFlagBits::e64);
    double period = ctx.props.limits.timestampPeriod;
    double ms = (double)(ts[1] - ts[0]) * period / 1e6;

    ctx.device.destroyQueryPool(pool);
    ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
    return ms;
}

// ============================================================
// 7. Main
// ============================================================

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();

        // ---- 测试维度 ----
        uint32_t M = 256, K = 512, N = 256;
        std::mt19937 rng(18);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> hA(M * K);
        std::vector<float> hW(K * N);
        std::vector<float> hBias(N);
        if (tv.enabled()) {
            // 从测试向量读取形状并加载 A（输入）与 W（权重），在测试向量模式下运行 GEMM
            std::vector<uint32_t> sA, sW;
            tv.queryShape("A", sA);
            tv.queryShape("W", sW);
            if (sA.size() >= 2) { M = sA[0]; K = sA[1]; }
            if (sW.size() >= 2) N = sW[1];
            hA.resize(M * K); hW.resize(K * N); hBias.resize(N);
            tv.load("A", {M, K}, hA.data());
            tv.load("W", {K, N}, hW.data());
        } else {
            for (auto& v : hA)    v = dist(rng);
            for (auto& v : hW)    v = dist(rng);
            for (auto& v : hBias) v = dist(rng) * 0.1f;
        }

        // ---- CPU 参考结果 ----
        std::vector<float> hRef = gemm_ref_fp32(hA.data(), hW.data(), M, K, N);

        std::cout << i18n::tr("\n===== 第 20 章：前沿量化 (K-quants / MXFP4) =====\n",
                              "\n===== Chapter 20: Frontier Quantization (K-quants / MXFP4) =====\n");
        std::cout << i18n::tr("矩阵维度: M=", "Matrix: M=") << M << " K=" << K << " N=" << N << "\n";

        // ---- 量化 ----
        std::cout << i18n::tr("\n--- 量化 ---\n", "\n--- Quantization ---\n");
        Q2K_Weight   q2w  = quantize_q2_k(hW.data(), K, N);
        Q6K_Weight   q6w  = quantize_q6_k(hW.data(), K, N);
        MXFP4_Weight qmxf = quantize_mxfp4(hW.data(), K, N);
        Q4_0_Weight  q4w  = quantize_q4_0(hW.data(), K, N);

        size_t fp32Bytes = (size_t)K * N * 4;
        size_t q2kBytes  = q2w.data.size();
        size_t q6kBytes  = q6w.data.size();
        size_t mxfp4Bytes = qmxf.data.size();
        size_t q4Bytes   = q4w.data.size();

        auto printSize = [&](const char* name, size_t bytes, size_t refBytes) {
            std::cout << "  " << name << ": " << bytes
                      << i18n::tr(" 字节 (压缩比 ", " bytes (compression ")
                      << (refBytes / (double)bytes) << "x, "
                      << i18n::tr("比特率 ", "bits per element ")
                      << (bytes * 8.0 / (K * N)) << " bpe)\n";
        };

        printSize("FP32",   fp32Bytes,  fp32Bytes);
        printSize("Q4_0",   q4Bytes,    fp32Bytes);
        printSize("Q2_K",   q2kBytes,   fp32Bytes);
        printSize("Q6_K",   q6kBytes,   fp32Bytes);
        printSize("MXFP4",  mxfp4Bytes, fp32Bytes);

        // 验证量化误差
        auto quantRMSE = [&](const char* name, const std::vector<float>& deq) {
            double sqSum = 0.0;
            for (uint32_t i = 0; i < K * N; ++i) {
                double d = hW[i] - deq[i];
                sqSum += d * d;
            }
            double rmse = std::sqrt(sqSum / (K * N));
            std::cout << "  " << name << " " << i18n::tr("量化 RMSE: ", "quantization RMSE: ") << rmse << "\n";
        };

        std::vector<float> hWq2k  = dequantize_q2_k_cpu(q2w);
        std::vector<float> hWq6k  = dequantize_q6_k_cpu(q6w);
        std::vector<float> hWmxf  = dequantize_mxfp4_cpu(qmxf);
        std::vector<float> hWq4   = dequantize_q4_0_cpu(q4w);

        // 用反量化权重计算参考 GEMM（正确反映各量化格式的融合反量化精度）
        std::vector<float> hRefQ2K = gemm_ref_fp32(hA.data(), hWq2k.data(), M, K, N);
        std::vector<float> hRefQ6K = gemm_ref_fp32(hA.data(), hWq6k.data(), M, K, N);
        std::vector<float> hRefMxf = gemm_ref_fp32(hA.data(), hWmxf.data(), M, K, N);
        std::vector<float> hRefQ4  = gemm_ref_fp32(hA.data(), hWq4.data(), M, K, N);

        std::cout << i18n::tr("\n--- 量化 RMSE ---\n", "\n--- Quantization RMSE ---\n");
        quantRMSE("Q4_0",  hWq4);
        quantRMSE("Q2_K",  hWq2k);
        quantRMSE("Q6_K",  hWq6k);
        quantRMSE("MXFP4", hWmxf);

        // ---- 创建 GPU 资源 ----
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;

        Buffer bufA     = vkCreateBuffer(ctx, M * K * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWq2k  = vkCreateBuffer(ctx, q2w.data.size(), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWq6k  = vkCreateBuffer(ctx, q6w.data.size(), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWmxf  = vkCreateBuffer(ctx, qmxf.data.size(), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWq4   = vkCreateBuffer(ctx, q4w.data.size(), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufBias  = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufC     = vkCreateBuffer(ctx, M * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);

        vkUpload(ctx, bufA,    hA.data(), M * K * 4);
        vkUpload(ctx, bufWq2k, q2w.data.data(), q2w.data.size());
        vkUpload(ctx, bufWq6k, q6w.data.data(), q6w.data.size());
        vkUpload(ctx, bufWmxf, qmxf.data.data(), qmxf.data.size());
        vkUpload(ctx, bufWq4,  q4w.data.data(), q4w.data.size());
        vkUpload(ctx, bufBias, hBias.data(), N * 4);

        // ---- 准备 ComputePass ----
        auto makePass = [&](const std::string& spvName,
                            const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                            const std::vector<vk::PushConstantRange>& pcRanges) {
            ComputePass pass;
            pass.prepare(ctx, std::string(SHADER_DIR) + spvName, bindings, pcRanges);
            return pass;
        };

        auto desc = [](int b, vk::DescriptorType t) {
            return vk::DescriptorSetLayoutBinding{b, t, 1, vk::ShaderStageFlagBits::eCompute};
        };

        // Q2_K fused GEMM
        ComputePass passQ2K = makePass("q2_k_gemm.spv",
            {desc(0, vk::DescriptorType::eStorageBuffer),
             desc(1, vk::DescriptorType::eStorageBuffer),
             desc(2, vk::DescriptorType::eStorageBuffer),
             desc(3, vk::DescriptorType::eStorageBuffer)},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC20)}});

        // Q6_K fused GEMM
        ComputePass passQ6K = makePass("q6_k_gemm.spv",
            {desc(0, vk::DescriptorType::eStorageBuffer),
             desc(1, vk::DescriptorType::eStorageBuffer),
             desc(2, vk::DescriptorType::eStorageBuffer),
             desc(3, vk::DescriptorType::eStorageBuffer)},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC20)}});

        // MXFP4 fused GEMM
        ComputePass passMXFP4 = makePass("mxfp4_gemm.spv",
            {desc(0, vk::DescriptorType::eStorageBuffer),
             desc(1, vk::DescriptorType::eStorageBuffer),
             desc(2, vk::DescriptorType::eStorageBuffer),
             desc(3, vk::DescriptorType::eStorageBuffer)},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC20)}});

        // Q4_0 fused GEMM (from ch18)
        ComputePass passQ4 = makePass("q4_0_gemm.spv",
            {desc(0, vk::DescriptorType::eStorageBuffer),
             desc(1, vk::DescriptorType::eStorageBuffer),
             desc(2, vk::DescriptorType::eStorageBuffer),
             desc(3, vk::DescriptorType::eStorageBuffer)},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC20)}});

        // FP32 tiled GEMM
        ComputePass passF32 = makePass("gemm_tiled.spv",
            {desc(0, vk::DescriptorType::eStorageBuffer),
             desc(1, vk::DescriptorType::eStorageBuffer),
             desc(2, vk::DescriptorType::eStorageBuffer)},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_GEMM)}});

        uint32_t gx = (N + 15u) / 16u, gy = (M + 15u) / 16u;

        // ============================================================
        // 测试 1: Q2_K 融合反量化 GEMM
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 1: Q2_K 融合反量化 GEMM ---\n",
                              "\n--- Test 1: Q2_K Fused Dequant GEMM ---\n");
        {
            passQ2K.bindBuffers({{0, bufA.buf}, {1, bufWq2k.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC20 pc{M, N, K, 1};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passQ2K.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            std::vector<float> refWithBias = hRefQ2K;
            for (uint32_t r = 0; r < M; ++r)
                for (uint32_t col = 0; col < N; ++col)
                    refWithBias[r * N + col] += hBias[col];

            VerifyResult vr = verify(c, refWithBias.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  Q2_K fused GEMM 错误数=", "  Q2_K fused GEMM errors=") << vr.errorCount
                      << i18n::tr("（容差 15%），最大误差=", " (tol 15%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 测试 2: Q6_K 融合反量化 GEMM
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 2: Q6_K 融合反量化 GEMM ---\n",
                              "\n--- Test 2: Q6_K Fused Dequant GEMM ---\n");
        {
            passQ6K.bindBuffers({{0, bufA.buf}, {1, bufWq6k.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC20 pc{M, N, K, 1};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passQ6K.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            std::vector<float> refWithBias = hRefQ6K;
            for (uint32_t r = 0; r < M; ++r)
                for (uint32_t col = 0; col < N; ++col)
                    refWithBias[r * N + col] += hBias[col];

            VerifyResult vr = verify(c, refWithBias.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  Q6_K fused GEMM 错误数=", "  Q6_K fused GEMM errors=") << vr.errorCount
                      << i18n::tr("（容差 5%），最大误差=", " (tol 5%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 测试 3: MXFP4 融合反量化 GEMM
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 3: MXFP4 融合反量化 GEMM ---\n",
                              "\n--- Test 3: MXFP4 Fused Dequant GEMM ---\n");
        {
            passMXFP4.bindBuffers({{0, bufA.buf}, {1, bufWmxf.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC20 pc{M, N, K, 1};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passMXFP4.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            std::vector<float> refWithBias = hRefMxf;
            for (uint32_t r = 0; r < M; ++r)
                for (uint32_t col = 0; col < N; ++col)
                    refWithBias[r * N + col] += hBias[col];

            VerifyResult vr = verify(c, refWithBias.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  MXFP4 fused GEMM 错误数=", "  MXFP4 fused GEMM errors=") << vr.errorCount
                      << i18n::tr("（容差 15%），最大误差=", " (tol 15%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 测试 4: Q4_0 融合反量化 GEMM（对比）
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 4: Q4_0 融合反量化 GEMM (ch18 对比) ---\n",
                              "\n--- Test 4: Q4_0 Fused Dequant GEMM (ch18 comparison) ---\n");
        {
            passQ4.bindBuffers({{0, bufA.buf}, {1, bufWq4.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC20 pc{M, N, K, 1};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passQ4.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            std::vector<float> refWithBias = hRefQ4;
            for (uint32_t r = 0; r < M; ++r)
                for (uint32_t col = 0; col < N; ++col)
                    refWithBias[r * N + col] += hBias[col];

            VerifyResult vr = verify(c, refWithBias.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  Q4_0 fused GEMM 错误数=", "  Q4_0 fused GEMM errors=") << vr.errorCount
                      << i18n::tr("（容差 10%），最大误差=", " (tol 10%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 测试 5: FP32 GEMM 参考
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 5: FP32 GEMM 参考 ---\n",
                              "\n--- Test 5: FP32 GEMM Reference ---\n");
        {
            Buffer bufWf32 = vkCreateBuffer(ctx, K * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            vkUpload(ctx, bufWf32, hW.data(), K * N * 4);

            passF32.bindBuffers({{0, bufA.buf}, {1, bufWf32.buf}, {2, bufC.buf}});
            PC_GEMM pcF32{M, N, K};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passF32.dispatch(cmd, &pcF32, sizeof(pcF32), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            VerifyResult vr = verify(c, hRef.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  FP32 GEMM 错误数=", "  FP32 GEMM errors=") << vr.errorCount
                      << i18n::tr("（容差 0.1%），最大误差=", " (tol 0.1%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";

            vkDestroyBuffer(ctx, bufWf32);
        }

        // ---- 测试向量模式：写回 FP32 基准 GEMM 输出 ----
        if (tv.enabled()) {
            auto outC = vkDownload(ctx, bufC, M * N * 4);
            tv.save("C", {M, N}, reinterpret_cast<const float*>(outC.data()));
        }

        // ============================================================
        // 性能基准测试
        // ============================================================
        std::cout << i18n::tr("\n--- 性能基准 (GPU 时间戳) ---\n",
                              "\n--- Benchmark (GPU Timestamps) ---\n");

        const int warmupIters = 5;
        const int benchIters = 10;

        // 准备 FP32 权重 buffer
        Buffer bufWf32Bench = vkCreateBuffer(ctx, K * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufWf32Bench, hW.data(), K * N * 4);

        struct BenchEntry {
            const char* name;
            ComputePass* pass;
            const void* pc;
            size_t pcSize;
            double avgMs;
        };
        std::vector<BenchEntry> benchEntries;

        // FP32
        {
            passF32.bindBuffers({{0, bufA.buf}, {1, bufWf32Bench.buf}, {2, bufC.buf}});
            PC_GEMM pcF32{M, N, K};
            for (int i = 0; i < warmupIters; ++i)
                bench_gpu_ms(ctx, passF32, &pcF32, sizeof(pcF32), gx, gy, 1);
            double sum = 0.0;
            for (int i = 0; i < benchIters; ++i)
                sum += bench_gpu_ms(ctx, passF32, &pcF32, sizeof(pcF32), gx, gy, 1);
            benchEntries.push_back({"FP32 GEMM", &passF32, &pcF32, sizeof(pcF32), sum / benchIters});
        }

        // Q4_0
        {
            passQ4.bindBuffers({{0, bufA.buf}, {1, bufWq4.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC20 pc{M, N, K, 1};
            for (int i = 0; i < warmupIters; ++i)
                bench_gpu_ms(ctx, passQ4, &pc, sizeof(pc), gx, gy, 1);
            double sum = 0.0;
            for (int i = 0; i < benchIters; ++i)
                sum += bench_gpu_ms(ctx, passQ4, &pc, sizeof(pc), gx, gy, 1);
            benchEntries.push_back({"Q4_0 fused", &passQ4, &pc, sizeof(pc), sum / benchIters});
        }

        // Q2_K
        {
            passQ2K.bindBuffers({{0, bufA.buf}, {1, bufWq2k.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC20 pc{M, N, K, 1};
            for (int i = 0; i < warmupIters; ++i)
                bench_gpu_ms(ctx, passQ2K, &pc, sizeof(pc), gx, gy, 1);
            double sum = 0.0;
            for (int i = 0; i < benchIters; ++i)
                sum += bench_gpu_ms(ctx, passQ2K, &pc, sizeof(pc), gx, gy, 1);
            benchEntries.push_back({"Q2_K fused", &passQ2K, &pc, sizeof(pc), sum / benchIters});
        }

        // Q6_K
        {
            passQ6K.bindBuffers({{0, bufA.buf}, {1, bufWq6k.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC20 pc{M, N, K, 1};
            for (int i = 0; i < warmupIters; ++i)
                bench_gpu_ms(ctx, passQ6K, &pc, sizeof(pc), gx, gy, 1);
            double sum = 0.0;
            for (int i = 0; i < benchIters; ++i)
                sum += bench_gpu_ms(ctx, passQ6K, &pc, sizeof(pc), gx, gy, 1);
            benchEntries.push_back({"Q6_K fused", &passQ6K, &pc, sizeof(pc), sum / benchIters});
        }

        // MXFP4
        {
            passMXFP4.bindBuffers({{0, bufA.buf}, {1, bufWmxf.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC20 pc{M, N, K, 1};
            for (int i = 0; i < warmupIters; ++i)
                bench_gpu_ms(ctx, passMXFP4, &pc, sizeof(pc), gx, gy, 1);
            double sum = 0.0;
            for (int i = 0; i < benchIters; ++i)
                sum += bench_gpu_ms(ctx, passMXFP4, &pc, sizeof(pc), gx, gy, 1);
            benchEntries.push_back({"MXFP4 fused", &passMXFP4, &pc, sizeof(pc), sum / benchIters});
        }

        double flops = 2.0 * M * N * K;
        double fp32BaseMs = benchEntries[0].avgMs;

        for (const auto& e : benchEntries) {
            std::cout << "  " << e.name << ": " << e.avgMs
                      << i18n::tr(" ms (", " ms (") << (flops / e.avgMs / 1e6) << " GFLOP/s, "
                      << i18n::tr("速度比 FP32 的 ", "speedup vs FP32: ") << (fp32BaseMs / e.avgMs) << "x)\n";
        }

        // ---- 带宽分析 ----
        std::cout << i18n::tr("\n--- 带宽分析 ---\n", "\n--- Bandwidth Analysis ---\n");

        struct BWEntry {
            const char* name;
            size_t weightBytes;
        };
        std::vector<BWEntry> bwEntries = {
            {"FP32", fp32Bytes},
            {"Q4_0", q4Bytes},
            {"Q2_K", q2kBytes},
            {"Q6_K", q6kBytes},
            {"MXFP4", mxfp4Bytes},
        };

        for (const auto& e : bwEntries) {
            double totalBytes = (double)(M * K * 4 + e.weightBytes + M * N * 4);
            double ai = flops / totalBytes;
            std::cout << "  " << e.name << ": " << i18n::tr("访存 ", "traffic ") << (totalBytes / 1e6)
                      << i18n::tr(" MB，算术强度: ", " MB, AI: ") << ai << " FLOP/byte\n";
        }

        // ---- 压缩比与精度总结 ----
        std::cout << i18n::tr("\n===== 综合对比：压缩比 vs 精度 =====\n",
                              "\n===== Summary: Compression vs Accuracy =====\n");

        double q4RMSE = 0, q2kRMSE = 0, q6kRMSE = 0, mxfRMSE = 0;
        for (uint32_t i = 0; i < K * N; ++i) {
            double d; d = hW[i] - hWq4[i];  q4RMSE += d * d;
            d = hW[i] - hWq2k[i]; q2kRMSE += d * d;
            d = hW[i] - hWq6k[i]; q6kRMSE += d * d;
            d = hW[i] - hWmxf[i]; mxfRMSE += d * d;
        }
        q4RMSE = std::sqrt(q4RMSE / (K * N));
        q2kRMSE = std::sqrt(q2kRMSE / (K * N));
        q6kRMSE = std::sqrt(q6kRMSE / (K * N));
        mxfRMSE = std::sqrt(mxfRMSE / (K * N));

        auto printRow = [&](const char* name, double compRatio, double rmse, double bpe) {
            printf("  %-8s | 压缩比 %5.2fx | RMSE %.4f | %.2f bpe\n", name, compRatio, rmse, bpe);
        };

        printf("  %-8s | 压缩比 %5s | RMSE %5s | %s\n", "Format", "ratio", "value", "bpe");
        printf("  -------------------------------------------------\n");
        printRow("FP32",  1.0, 0.0, 32.0);
        printRow("Q4_0",  fp32Bytes / (double)q4Bytes,  q4RMSE,  q4Bytes * 8.0 / (K * N));
        printRow("Q2_K",  fp32Bytes / (double)q2kBytes, q2kRMSE, q2kBytes * 8.0 / (K * N));
        printRow("Q6_K",  fp32Bytes / (double)q6kBytes, q6kRMSE, q6kBytes * 8.0 / (K * N));
        printRow("MXFP4", fp32Bytes / (double)mxfp4Bytes, mxfRMSE, mxfp4Bytes * 8.0 / (K * N));

        // ---- 清理 ----
        passQ2K.destroy();
        passQ6K.destroy();
        passMXFP4.destroy();
        passQ4.destroy();
        passF32.destroy();
        vkDestroyBuffer(ctx, bufA);
        vkDestroyBuffer(ctx, bufWq2k);
        vkDestroyBuffer(ctx, bufWq6k);
        vkDestroyBuffer(ctx, bufWmxf);
        vkDestroyBuffer(ctx, bufWq4);
        vkDestroyBuffer(ctx, bufBias);
        vkDestroyBuffer(ctx, bufC);
        vkDestroyBuffer(ctx, bufWf32Bench);
        vkDestroyContext(ctx);

        std::cout << i18n::tr("\n第 20 章完成！\n", "\nChapter 20 complete!\n");

    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}