// ch20_frontier_quant/main_nvfp4.cpp
// 进阶 demo：NVFP4（E4M3 scale，两级微缩放）vs 本章 MXFP4（fp16 scale，单级微缩放）
//
// 这是 ch20 docs 里「把 MXFP4 的 E8M0/fp16 scale 换成 NVFP4 的 E4M3 scale，
// 对比同样的 E2M1 权重在不同缩放粒度下的误差」这个动手题的参考实现。
//
// 两个格式在本教学实现里都是 5 bits/element：
//   MXFP4 : 32 元素 / fp16 scale           → 20 字节 / 32 元素
//   NVFP4 : 16 元素 / E4M3 scale × 4，
//           64 元素共享 1 个 fp16 全局 scale → 40 字节 / 64 元素
// 因此压缩比相同，可以直接比较「缩放粒度 + scale 精度」带来的精度差异。
//
// 独立可执行目标，不改动 main.cpp，不影响 ch00~ch20 的既有验证结论。

#include "vk_base.h"
#include "i18n.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>

constexpr uint32_t M = 256, K = 512, N = 256;

// ============================================================
// 1. FP16 转换（与 ch20 main.cpp 一致）
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
// 2. E2M1（两种格式共用的 4-bit 值）
// ============================================================
static const float E2M1_VALS[8] = {0.0f, 0.25f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static uint8_t encode_e2m1(float v) {
    if (v == 0.0f) return 0;
    uint8_t sign = (v < 0.0f) ? 0x8 : 0x0;
    float absV = std::fabs(v);
    int best = 0;
    float bestErr = 1e10f;
    for (int i = 0; i < 8; ++i) {
        float err = std::fabs(absV - E2M1_VALS[i]);
        if (err < bestErr) { bestErr = err; best = i; }
    }
    return sign | (uint8_t)best;
}

static float decode_e2m1(uint8_t val) {
    float f = E2M1_VALS[val & 0x7];
    return (val & 0x8) ? -f : f;
}

// ============================================================
// 3. E4M3（NVFP4 的 micro-block scale，正值）
//    E4M3FN 简化：1 sign + 4 exp + 3 mant，bias=7，无 inf
//    exp=0 次正规：mant/8 × 2^-6 = mant × 2^-9
// ============================================================
static float decode_e4m3(uint8_t b) {
    uint32_t exp  = (uint32_t)((b >> 3) & 0xF);
    uint32_t mant = (uint32_t)(b & 0x7);
    if (exp == 0) return (float)mant * (1.0f / 512.0f);
    return (1.0f + (float)mant / 8.0f) * std::exp2((float)((int)exp - 7));
}

static uint8_t encode_e4m3(float v) {
    uint8_t best = 0;
    float bestErr = 1e30f;
    for (uint32_t b = 0; b < 128; ++b) {
        uint32_t exp = (b >> 3) & 0xF, mant = b & 7;
        if (exp == 15 && mant == 7) continue;   // NaN 编码
        float f = decode_e4m3((uint8_t)b);
        float err = std::fabs(f - v);
        if (err < bestErr) { bestErr = err; best = (uint8_t)b; }
    }
    return best;
}

// ============================================================
// 4. MXFP4：32 元素 / 1 个 fp16 scale（本章原实现）
// ============================================================
struct MXFP4_Block {
    uint16_t d;
    uint8_t  qs[16];
    uint8_t  padding[2];
};
static_assert(sizeof(MXFP4_Block) == 20, "MXFP4_Block must be 20 bytes");

struct WeightBuf {
    std::vector<uint8_t> data;
    uint32_t blocksPerCol;
};

static WeightBuf quantize_mxfp4(const float* w, uint32_t K, uint32_t N) {
    const uint32_t blockSize = 32;
    uint32_t bpc = (K + blockSize - 1) / blockSize;
    WeightBuf qw;
    qw.data.assign(bpc * N * sizeof(MXFP4_Block), 0);
    qw.blocksPerCol = bpc;
    auto* blocks = reinterpret_cast<MXFP4_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < bpc; ++b) {
            uint32_t kStart = b * blockSize;
            uint32_t blockEnd = std::min(kStart + blockSize, K);
            MXFP4_Block& blk = blocks[n * bpc + b];

            float maxAbs = 0.0f;
            for (uint32_t k = kStart; k < blockEnd; ++k)
                maxAbs = std::max(maxAbs, std::fabs(w[k * N + n]));

            float scale = (maxAbs == 0.0f) ? 1.0f : maxAbs / 6.0f;
            blk.d = fp32_to_fp16(scale);
            float scaleActual = fp16_to_fp32(blk.d);

            for (uint32_t k = 0; k < blockSize; ++k) {
                uint32_t ki = kStart + k;
                float val = (ki < K) ? w[ki * N + n] : 0.0f;
                uint8_t code = encode_e2m1(val / scaleActual);
                uint32_t byteIdx = k / 2, shift = (k % 2) * 4;
                blk.qs[byteIdx] &= (uint8_t)~(0xF << shift);
                blk.qs[byteIdx] |= (uint8_t)(code << shift);
            }
        }
    }
    return qw;
}

static std::vector<float> dequantize_mxfp4_cpu(const WeightBuf& qw, uint32_t K, uint32_t N) {
    std::vector<float> out(K * N);
    const auto* blocks = reinterpret_cast<const MXFP4_Block*>(qw.data.data());
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < qw.blocksPerCol; ++b) {
            const MXFP4_Block& blk = blocks[n * qw.blocksPerCol + b];
            float scale = fp16_to_fp32(blk.d);
            uint32_t kStart = b * 32;
            for (uint32_t k = 0; k < 32; ++k) {
                uint8_t code = (blk.qs[k / 2] >> ((k % 2) * 4)) & 0xF;
                if (kStart + k < K)
                    out[(kStart + k) * N + n] = decode_e2m1(code) * scale;
            }
        }
    }
    return out;
}

// ============================================================
// 5. NVFP4：16 元素 / E4M3 scale，64 元素共享 fp16 全局 scale
// ============================================================
struct NVFP4_SuperBlock {
    uint16_t global;      // fp16 两级缩放的全局部分（字节 0..1）
    uint8_t  scales[4];   // 4 × E4M3（字节 2..5）
    uint8_t  pad[2];      // 对齐（字节 6..7）
    uint8_t  qs[32];      // 64 × 4-bit E2M1（字节 8..39）
};
static_assert(sizeof(NVFP4_SuperBlock) == 40, "NVFP4_SuperBlock must be 40 bytes");

static WeightBuf quantize_nvfp4(const float* w, uint32_t K, uint32_t N) {
    const uint32_t SUPER = 64, MICRO = 16;
    uint32_t spc = (K + SUPER - 1) / SUPER;
    WeightBuf qw;
    qw.data.assign(spc * N * sizeof(NVFP4_SuperBlock), 0);
    qw.blocksPerCol = spc;
    auto* blocks = reinterpret_cast<NVFP4_SuperBlock*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < spc; ++s) {
            uint32_t kStart = s * SUPER;
            uint32_t superEnd = std::min(kStart + SUPER, K);
            NVFP4_SuperBlock& blk = blocks[n * spc + s];

            // 全局 scale：把 super-block 最大值映射到 E4M3 上限 448
            float maxAbsSuper = 0.0f;
            for (uint32_t k = kStart; k < superEnd; ++k)
                maxAbsSuper = std::max(maxAbsSuper, std::fabs(w[k * N + n]));
            float g = (maxAbsSuper == 0.0f) ? 1.0f : maxAbsSuper / 448.0f;
            blk.global = fp32_to_fp16(g);
            float gActual = fp16_to_fp32(blk.global);

            for (uint32_t m = 0; m < 4; ++m) {
                uint32_t microStart = kStart + m * MICRO;
                float maxAbsMicro = 0.0f;
                for (uint32_t e = 0; e < MICRO; ++e) {
                    uint32_t ki = microStart + e;
                    if (ki < K) maxAbsMicro = std::max(maxAbsMicro, std::fabs(w[ki * N + n]));
                }
                // E2M1 上限 6.0，故 micro scale = maxAbs / (6 × global)
                float sIdeal = maxAbsMicro / (6.0f * gActual);
                uint8_t e4 = encode_e4m3(sIdeal);
                blk.scales[m] = e4;
                float sActual = decode_e4m3(e4) * gActual;

                for (uint32_t e = 0; e < MICRO; ++e) {
                    uint32_t ki = microStart + e;
                    float val = (ki < K) ? w[ki * N + n] : 0.0f;
                    uint8_t code = encode_e2m1(val / std::max(sActual, 1e-12f));
                    uint32_t byteIdx = m * 8 + e / 2;
                    uint32_t shift = (e % 2) * 4;
                    blk.qs[byteIdx] &= (uint8_t)~(0xF << shift);
                    blk.qs[byteIdx] |= (uint8_t)(code << shift);
                }
            }
        }
    }
    return qw;
}

static std::vector<float> dequantize_nvfp4_cpu(const WeightBuf& qw, uint32_t K, uint32_t N) {
    std::vector<float> out(K * N);
    const auto* blocks = reinterpret_cast<const NVFP4_SuperBlock*>(qw.data.data());
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < qw.blocksPerCol; ++s) {
            const NVFP4_SuperBlock& blk = blocks[n * qw.blocksPerCol + s];
            float g = fp16_to_fp32(blk.global);
            for (uint32_t m = 0; m < 4; ++m) {
                float sm = decode_e4m3(blk.scales[m]);
                for (uint32_t e = 0; e < 16; ++e) {
                    uint32_t ki = s * 64 + m * 16 + e;
                    if (ki >= K) break;
                    uint8_t code = (blk.qs[m * 8 + e / 2] >> ((e % 2) * 4)) & 0xF;
                    out[ki * N + n] = decode_e2m1(code) * sm * g;
                }
            }
        }
    }
    return out;
}

// ============================================================
// 6. CPU GEMM 与验证
// ============================================================
static std::vector<float> gemm_ref_fp32(const float* a, const float* b) {
    std::vector<float> c(M * N, 0.0f);
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t col = 0; col < N; ++col) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k) s += a[r * K + k] * b[k * N + col];
            c[r * N + col] = s;
        }
    return c;
}

struct VerifyResult { uint32_t errorCount; float maxError; float rmse; };

static VerifyResult verify(const float* gpu, const float* ref, float relTol) {
    VerifyResult vr{0, 0.0f, 0.0f};
    double sqSum = 0.0;
    const uint32_t total = M * N;
    for (uint32_t i = 0; i < total; ++i) {
        float diff = std::fabs(gpu[i] - ref[i]);
        if (diff > std::max(1e-4f, std::fabs(ref[i]) * relTol)) vr.errorCount++;
        vr.maxError = std::max(vr.maxError, diff);
        sqSum += (double)diff * diff;
    }
    vr.rmse = (float)std::sqrt(sqSum / total);
    return vr;
}

static double rmseAgainst(const std::vector<float>& a, const std::vector<float>& b) {
    double sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double)a[i] - (double)b[i];
        sq += d * d;
    }
    return std::sqrt(sq / a.size());
}

// ============================================================
// 7. Main
// ============================================================
int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        VulkanContext ctx = vkInit();

        std::cout << i18n::tr(
            "\n===== 第 20 章 进阶：NVFP4 vs MXFP4 (两级 E4M3 缩放) =====\n",
            "\n===== Chapter 20 Advanced: NVFP4 vs MXFP4 (two-level E4M3 scales) =====\n");
        std::cout << i18n::tr("矩阵维度: M=", "Matrix: M=") << M << " K=" << K << " N=" << N << "\n";

        // 与 ch20 主程序完全相同的数据（种子 18）
        std::mt19937 rng(18);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> hA(M * K), hW(K * N), hBias(N);
        for (auto& v : hA) v = dist(rng);
        for (auto& v : hW) v = dist(rng);
        for (auto& v : hBias) v = dist(rng) * 0.1f;

        // ---- 量化 ----
        WeightBuf qmxf = quantize_mxfp4(hW.data(), K, N);
        WeightBuf qnvf = quantize_nvfp4(hW.data(), K, N);
        std::cout << i18n::tr("\n--- 存储开销 ---\n", "\n--- Storage ---\n");
        std::cout << "  FP32  : " << (size_t)K * N * 4 << " bytes (32.00 bpe)\n";
        std::cout << "  MXFP4 : " << qmxf.data.size()
                  << " bytes (" << (qmxf.data.size() * 8.0 / (K * N)) << " bpe, 32 elem/fp16 scale)\n";
        std::cout << "  NVFP4 : " << qnvf.data.size()
                  << " bytes (" << (qnvf.data.size() * 8.0 / (K * N)) << " bpe, 16 elem/E4M3 scale)\n";
        std::cout << i18n::tr("  两者比特率相同，可直接比较缩放粒度与 scale 精度。\n",
                              "  Same bitrate, so the comparison isolates scale granularity and precision.\n");

        // ---- 精度对比 ----
        std::vector<float> hWmxf = dequantize_mxfp4_cpu(qmxf, K, N);
        std::vector<float> hWnvf = dequantize_nvfp4_cpu(qnvf, K, N);
        std::vector<float> hRefFp32 = gemm_ref_fp32(hA.data(), hW.data());
        std::vector<float> hRefMxf  = gemm_ref_fp32(hA.data(), hWmxf.data());
        std::vector<float> hRefNvf  = gemm_ref_fp32(hA.data(), hWnvf.data());

        double wRmseMxf = rmseAgainst(hWmxf, hW);
        double wRmseNvf = rmseAgainst(hWnvf, hW);
        double cRmseMxf = rmseAgainst(hRefMxf, hRefFp32);
        double cRmseNvf = rmseAgainst(hRefNvf, hRefFp32);

        std::cout << i18n::tr("\n--- 精度对比 (RMSE) ---\n", "\n--- Accuracy (RMSE) ---\n");
        std::cout << "  MXFP4 " << i18n::tr("权重量化 RMSE: ", "weight quant RMSE: ") << wRmseMxf
                  << i18n::tr("，GEMM 输出 RMSE: ", ", GEMM output RMSE: ") << cRmseMxf << "\n";
        std::cout << "  NVFP4 " << i18n::tr("权重量化 RMSE: ", "weight quant RMSE: ") << wRmseNvf
                  << i18n::tr("，GEMM 输出 RMSE: ", ", GEMM output RMSE: ") << cRmseNvf << "\n";

        // ---- GPU ----
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        auto mk = [&](size_t bytes) {
            return vkCreateBuffer(ctx, bytes, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        };
        Buffer bufA    = mk(M * K * sizeof(float));
        Buffer bufWmxf = mk(qmxf.data.size());
        Buffer bufWnvf = mk(qnvf.data.size());
        Buffer bufBias = mk(N * sizeof(float));
        Buffer bufCmxf = mk(M * N * sizeof(float));
        Buffer bufCnvf = mk(M * N * sizeof(float));

        vkUpload(ctx, bufA,    hA.data(),        M * K * sizeof(float));
        vkUpload(ctx, bufWmxf, qmxf.data.data(), qmxf.data.size());
        vkUpload(ctx, bufWnvf, qnvf.data.data(), qnvf.data.size());
        vkUpload(ctx, bufBias, hBias.data(),     N * sizeof(float));

        auto bindings4 = []() {
            std::vector<vk::DescriptorSetLayoutBinding> v;
            for (uint32_t i = 0; i < 4; ++i)
                v.push_back({i, vk::DescriptorType::eStorageBuffer, 1,
                             vk::ShaderStageFlagBits::eCompute});
            return v;
        };
        struct PC { uint32_t M, N, K, hasBias; };
        auto pcs = []() {
            return std::vector<vk::PushConstantRange>{
                vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC)}};
        };

        ComputePass passMxf, passNvf;
        passMxf.prepare(ctx, std::string(SHADER_DIR) + "mxfp4_gemm.spv", bindings4(), pcs());
        passNvf.prepare(ctx, std::string(SHADER_DIR) + "nvfp4_gemm.spv", bindings4(), pcs());
        passMxf.bindBuffers({{0, bufA.buf}, {1, bufWmxf.buf}, {2, bufBias.buf}, {3, bufCmxf.buf}});
        passNvf.bindBuffers({{0, bufA.buf}, {1, bufWnvf.buf}, {2, bufBias.buf}, {3, bufCnvf.buf}});

        PC pc{M, N, K, 0};   // 不加 bias，CPU 参考同样不含 bias
        const uint32_t gx = (N + 15u) / 16u, gy = (M + 15u) / 16u;
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passMxf.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            passNvf.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
        });

        auto cMxf = vkDownload(ctx, bufCmxf, M * N * sizeof(float));
        auto cNvf = vkDownload(ctx, bufCnvf, M * N * sizeof(float));
        VerifyResult vrMxf = verify(reinterpret_cast<const float*>(cMxf.data()),
                                    hRefMxf.data(), 1e-3f);
        VerifyResult vrNvf = verify(reinterpret_cast<const float*>(cNvf.data()),
                                    hRefNvf.data(), 1e-3f);

        std::cout << i18n::tr("\n--- GPU 融合反量化 GEMM vs CPU 参考 ---\n",
                              "\n--- GPU fused-dequant GEMM vs CPU reference ---\n");
        std::cout << "  MXFP4 " << i18n::tr("错误数=", "errors=") << vrMxf.errorCount
                  << i18n::tr("，RMSE=", ", RMSE=") << vrMxf.rmse << "\n";
        std::cout << "  NVFP4 " << i18n::tr("错误数=", "errors=") << vrNvf.errorCount
                  << i18n::tr("，RMSE=", ", RMSE=") << vrNvf.rmse << "\n";

        // ---- 清理 ----
        passMxf.destroy(); passNvf.destroy();
        vkDestroyBuffer(ctx, bufA);    vkDestroyBuffer(ctx, bufWmxf);
        vkDestroyBuffer(ctx, bufWnvf); vkDestroyBuffer(ctx, bufBias);
        vkDestroyBuffer(ctx, bufCmxf); vkDestroyBuffer(ctx, bufCnvf);
        vkDestroyContext(ctx);

        bool ok = (vrMxf.errorCount == 0 && vrNvf.errorCount == 0);
        std::cout << i18n::tr("\nNVFP4 demo ", "\nNVFP4 demo ")
                  << i18n::tr(ok ? "通过！\n" : "存在错误！\n", ok ? "PASSED!\n" : "FAILED!\n");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
}
