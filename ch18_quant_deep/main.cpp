// ch18_quant_deep/main.cpp
// 深度量化：Q4_0 / Q8_0 格式的 GPU 融合反量化 GEMM
//
// 本章重点：
//   1. Q4_0 格式：32 元素/block，fp16 scale + 4-bit 有符号 int 打包
//   2. Q8_0 格式：32 元素/block，fp16 scale + int8
//   3. 融合反量化：在 GEMM 内核中就地解包反量化，避免物化 FP32 权重
//   4. 对比 fused 与 standalone 路径的性能差异
//   5. 对比 FP32、Q8_0、Q4_0 的压缩比与速度

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
#include <chrono>

// ============================================================
// 1. Q4_0 / Q8_0 数据结构定义
// ============================================================

// Q4_0 block: 32 个元素，20 字节（5 × uint32，GPU 对齐版）
struct Q4_0_Block {
    uint16_t d;          // fp16 scale
    uint8_t  qs[16];     // 16 字节 packed 4-bit 数据（2 元素/字节）
    uint8_t  padding[2]; // 填充到 20 字节
};
static_assert(sizeof(Q4_0_Block) == 20, "Q4_0_Block must be 20 bytes");

// Q8_0 block: 32 个元素，36 字节（9 × uint32，GPU 对齐版）
struct Q8_0_Block {
    uint16_t d;          // fp16 scale
    int8_t   qs[32];     // 32 字节 int8 数据
    uint8_t  padding[2]; // 填充到 36 字节
};
static_assert(sizeof(Q8_0_Block) == 36, "Q8_0_Block must be 36 bytes");

// 量化权重 buffer 的元信息
struct Q4_0_Weight {
    std::vector<uint8_t> data;  // 打包后的二进制数据
    uint32_t blocksPerCol;      // 每列多少个 block
    uint32_t K;                 // 原始行数
    uint32_t N;                 // 原始列数
};

struct Q8_0_Weight {
    std::vector<uint8_t> data;
    uint32_t blocksPerCol;
    uint32_t K;
    uint32_t N;
};

// Push constant 结构（与着色器一致）
struct PC18 {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t hasBias;  // 0 = 无 bias, 1 = 有 bias
};

// 普通 GEMM 的 push constant
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

    if (exp == 0) {
        // 零或次正规 → 零
        return (uint16_t)(sign << 15);
    }
    if (exp == 255) {
        // Inf 或 NaN
        uint16_t h = (uint16_t)(sign << 15) | (31 << 10);
        if (mant) h |= 0x3FF;  // NaN
        return h;
    }

    // 正规数：重定偏置
    exp = exp - 127 + 15;
    if (exp >= 31) {
        // 溢出 → Inf
        return (uint16_t)(sign << 15) | (31 << 10);
    }
    if (exp <= 0) {
        // 下溢 → 次正规或零
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
        // 次正规/零
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            // 次正规 → 正规
            int32_t e = -1;
            uint32_t m = mant;
            while (m & 0x400) { m <<= 1; e--; }
            m &= 0x3FF;
            f32 = (sign << 31) | ((uint32_t)(e + 127 + 10) << 24) | (m << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
        if (mant) f32 |= 0x7FFFFF;  // NaN
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

// Q4_0 量化：将 float 权重矩阵打包为 Q4_0 格式
// 权重矩阵尺寸 K×N，行优先存储
Q4_0_Weight quantize_q4_0(const float* w, uint32_t K, uint32_t N) {
    const uint32_t blockSize = 32;
    uint32_t blocksPerCol = (K + blockSize - 1) / blockSize;
    uint32_t totalBlocks = N * blocksPerCol;

    Q4_0_Weight qw;
    qw.data.resize(totalBlocks * sizeof(Q4_0_Block));
    qw.blocksPerCol = blocksPerCol;
    qw.K = K;
    qw.N = N;

    Q4_0_Block* blocks = reinterpret_cast<Q4_0_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < blocksPerCol; ++b) {
            uint32_t kStart = b * blockSize;
            uint32_t blockEnd = std::min(kStart + blockSize, K);

            // 找 block 内最大绝对值
            float maxAbs = 0.0f;
            for (uint32_t k = kStart; k < blockEnd; ++k) {
                maxAbs = std::max(maxAbs, std::fabs(w[k * N + n]));
            }

            // 计算 scale：int4 范围 [-8, 7]，使用 7 作为对称量化最大等级
            float scale = (maxAbs == 0.0f) ? 1.0f : maxAbs / 7.0f;

            Q4_0_Block& block = blocks[n * blocksPerCol + b];
            block.d = fp32_to_fp16(scale);

            // 打包 4-bit 数据（2 元素/字节）
            for (uint32_t k = 0; k < blockSize; k += 2) {
                uint32_t ki0 = kStart + k;
                uint32_t ki1 = kStart + k + 1;

                int q0 = 0, q1 = 0;
                if (ki0 < K) {
                    float v = w[ki0 * N + n] / scale;
                    q0 = (int)std::lroundf(v);
                    q0 = std::max(-8, std::min(7, q0));
                }
                if (ki1 < K) {
                    float v = w[ki1 * N + n] / scale;
                    q1 = (int)std::lroundf(v);
                    q1 = std::max(-8, std::min(7, q1));
                }

                // 有符号 int4 转无符号存储：负数转补码
                uint8_t u0 = (uint8_t)(q0 & 0xF);
                uint8_t u1 = (uint8_t)(q1 & 0xF);
                block.qs[k / 2] = u0 | (u1 << 4);
            }
        }
    }

    return qw;
}

// Q8_0 量化：将 float 权重矩阵打包为 Q8_0 格式
Q8_0_Weight quantize_q8_0(const float* w, uint32_t K, uint32_t N) {
    const uint32_t blockSize = 32;
    uint32_t blocksPerCol = (K + blockSize - 1) / blockSize;
    uint32_t totalBlocks = N * blocksPerCol;

    Q8_0_Weight qw;
    qw.data.resize(totalBlocks * sizeof(Q8_0_Block));
    qw.blocksPerCol = blocksPerCol;
    qw.K = K;
    qw.N = N;

    Q8_0_Block* blocks = reinterpret_cast<Q8_0_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < blocksPerCol; ++b) {
            uint32_t kStart = b * blockSize;
            uint32_t blockEnd = std::min(kStart + blockSize, K);

            float maxAbs = 0.0f;
            for (uint32_t k = kStart; k < blockEnd; ++k) {
                maxAbs = std::max(maxAbs, std::fabs(w[k * N + n]));
            }

            // int8 范围 [-127, 127]（对称量化，避免 -128）
            float scale = (maxAbs == 0.0f) ? 1.0f : maxAbs / 127.0f;

            Q8_0_Block& block = blocks[n * blocksPerCol + b];
            block.d = fp32_to_fp16(scale);

            for (uint32_t k = 0; k < blockSize; ++k) {
                uint32_t ki = kStart + k;
                int q = 0;
                if (ki < K) {
                    float v = w[ki * N + n] / scale;
                    q = (int)std::lroundf(v);
                    q = std::max(-127, std::min(127, q));
                }
                block.qs[k] = (int8_t)q;
            }
        }
    }

    return qw;
}

// CPU 端 Q4_0 反量化（用于验证）
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
                int q0 = (int)(packed & 0xF);
                int q1 = (int)((packed >> 4) & 0xF);
                if (q0 >= 8) q0 -= 16;
                if (q1 >= 8) q1 -= 16;

                if (kStart + k < K)
                    result[(kStart + k) * N + n] = (float)q0 * scale;
                if (kStart + k + 1 < K)
                    result[(kStart + k + 1) * N + n] = (float)q1 * scale;
            }
        }
    }
    return result;
}

std::vector<float> dequantize_q8_0_cpu(const Q8_0_Weight& qw) {
    uint32_t K = qw.K, N = qw.N;
    std::vector<float> result(K * N);
    const Q8_0_Block* blocks = reinterpret_cast<const Q8_0_Block*>(qw.data.data());

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < qw.blocksPerCol; ++b) {
            const Q8_0_Block& block = blocks[n * qw.blocksPerCol + b];
            float scale = fp16_to_fp32(block.d);
            uint32_t kStart = b * 32;
            for (uint32_t k = 0; k < 32; ++k) {
                uint32_t ki = kStart + k;
                if (ki < K)
                    result[ki * N + n] = (float)block.qs[k] * scale;
            }
        }
    }
    return result;
}

// CPU 端 FP32 参考 GEMM
std::vector<float> gemm_ref_fp32(const float* a, const float* b,
                                  uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> c(M * N, 0.0f);
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t col = 0; col < N; ++col) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                s += a[r * K + k] * b[k * N + col];
            }
            c[r * N + col] = s;
        }
    }
    return c;
}

// ============================================================
// 4. 验证辅助
// ============================================================

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
        if (diff > maxAllowed) {
            vr.errorCount++;
        }
        vr.maxError = std::max(vr.maxError, diff);
        sqSum += (double)diff * diff;
    }
    vr.rmse = (float)std::sqrt(sqSum / (double)total);
    return vr;
}

// ============================================================
// 5. GPU 计时辅助
// ============================================================

// 运行一次 dispatch 并返回 GPU 时间（毫秒）
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
// 6. Main
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
        std::vector<float> hRefBias(N);
        for (uint32_t i = 0; i < N; ++i) hRefBias[i] = hRef[i] + hBias[i];  // just first row

        std::cout << i18n::tr("\n===== 第 18 章：深度量化 (Q4_0 / Q8_0) =====\n",
                              "\n===== Chapter 18: Deep Quantization (Q4_0 / Q8_0) =====\n");
        std::cout << i18n::tr("矩阵维度: M=", "Matrix: M=") << M << " K=" << K << " N=" << N << "\n";

        // ---- 量化权重 ----
        std::cout << i18n::tr("\n--- 量化 ---\n", "\n--- Quantization ---\n");
        Q4_0_Weight q4w = quantize_q4_0(hW.data(), K, N);
        Q8_0_Weight q8w = quantize_q8_0(hW.data(), K, N);

        size_t fp32Bytes = (size_t)K * N * 4;
        size_t q4Bytes = q4w.data.size();
        size_t q8Bytes = q8w.data.size();

        std::cout << i18n::tr("  FP32 权重: ", "  FP32 weights: ") << fp32Bytes
                  << i18n::tr(" 字节\n", " bytes\n");
        std::cout << i18n::tr("  Q4_0 权重: ", "  Q4_0 weights: ") << q4Bytes
                  << i18n::tr(" 字节 (压缩比 ", " bytes (compression ") << (fp32Bytes / (double)q4Bytes)
                  << "x)\n";
        std::cout << i18n::tr("  Q8_0 权重: ", "  Q8_0 weights: ") << q8Bytes
                  << i18n::tr(" 字节 (压缩比 ", " bytes (compression ") << (fp32Bytes / (double)q8Bytes)
                  << "x)\n";

        // 验证量化误差
        std::vector<float> hWq4 = dequantize_q4_0_cpu(q4w);
        std::vector<float> hWq8 = dequantize_q8_0_cpu(q8w);
        double q4mse = 0.0;
        for (uint32_t i = 0; i < K * N; ++i) {
            double d = hW[i] - hWq4[i];
            q4mse += d * d;
        }
        q4mse = std::sqrt(q4mse / (K * N));
        std::cout << i18n::tr("  Q4_0 量化 RMSE: ", "  Q4_0 quantization RMSE: ") << q4mse << "\n";

        // 量化后的参考 GEMM（用反量化权重，而非原始 FP32 权重）
        std::vector<float> hRefQ4 = gemm_ref_fp32(hA.data(), hWq4.data(), M, K, N);
        std::vector<float> hRefQ8 = gemm_ref_fp32(hA.data(), hWq8.data(), M, K, N);

        // ---- 创建 GPU 资源 ----
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;

        Buffer bufA    = vkCreateBuffer(ctx, M * K * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWq4  = vkCreateBuffer(ctx, q4w.data.size(), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWq8  = vkCreateBuffer(ctx, q8w.data.size(), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufBias = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufC    = vkCreateBuffer(ctx, M * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWf32 = vkCreateBuffer(ctx, K * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);  // 用于 standalone 反量化

        vkUpload(ctx, bufA, hA.data(), M * K * 4);
        vkUpload(ctx, bufWq4, q4w.data.data(), q4w.data.size());
        vkUpload(ctx, bufWq8, q8w.data.data(), q8w.data.size());
        vkUpload(ctx, bufBias, hBias.data(), N * 4);

        // ---- 准备 ComputePass ----
        // Q4_0 融合反量化 GEMM
        ComputePass passQ4;
        passQ4.prepare(ctx, std::string(SHADER_DIR) + "q4_0_gemm.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC18)}});

        // Q8_0 融合反量化 GEMM
        ComputePass passQ8;
        passQ8.prepare(ctx, std::string(SHADER_DIR) + "q8_0_gemm.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC18)}});

        // Q4_0 独立反量化（standalone dequant）
        ComputePass passDeq;
        passDeq.prepare(ctx, std::string(SHADER_DIR) + "q4_0_dequant.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_GEMM)}});

        // FP32 分块 GEMM（用于参考和 standalone 路径）
        ComputePass passF32;
        passF32.prepare(ctx, std::string(SHADER_DIR) + "gemm_tiled.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_GEMM)}});

        uint32_t gx = (N + 15u) / 16u, gy = (M + 15u) / 16u;

        // ============================================================
        // 测试 1: Q4_0 融合反量化 GEMM
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 1: Q4_0 融合反量化 GEMM ---\n",
                              "\n--- Test 1: Q4_0 Fused Dequant GEMM ---\n");
        {
            passQ4.bindBuffers({{0, bufA.buf}, {1, bufWq4.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC18 pc{M, N, K, 1};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passQ4.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            // 用「反量化权重」的参考（含量化误差，正确反映融合反量化 GEMM 的精度）
            std::vector<float> refWithBias = hRefQ4;
            for (uint32_t r = 0; r < M; ++r)
                for (uint32_t col = 0; col < N; ++col)
                    refWithBias[r * N + col] += hBias[col];

            VerifyResult vr = verify(c, refWithBias.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  Q4_0 fused GEMM 错误数=", "  Q4_0 fused GEMM errors=") << vr.errorCount
                      << i18n::tr("（容差 0.1%），最大误差=", " (tol 0.1%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 测试 2: Q8_0 融合反量化 GEMM
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 2: Q8_0 融合反量化 GEMM ---\n",
                              "\n--- Test 2: Q8_0 Fused Dequant GEMM ---\n");
        {
            passQ8.bindBuffers({{0, bufA.buf}, {1, bufWq8.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC18 pc{M, N, K, 1};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passQ8.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            std::vector<float> refWithBias = hRefQ8;
            for (uint32_t r = 0; r < M; ++r)
                for (uint32_t col = 0; col < N; ++col)
                    refWithBias[r * N + col] += hBias[col];

            VerifyResult vr = verify(c, refWithBias.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  Q8_0 fused GEMM 错误数=", "  Q8_0 fused GEMM errors=") << vr.errorCount
                      << i18n::tr("（容差 0.1%），最大误差=", " (tol 0.1%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 测试 3: Standalone 反量化 + FP32 GEMM
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 3: 独立反量化 + FP32 GEMM ---\n",
                              "\n--- Test 3: Standalone Dequant + FP32 GEMM ---\n");
        {
            // 第一步：反量化 Q4_0 -> float32
            passDeq.bindBuffers({{0, bufWq4.buf}, {1, bufWf32.buf}});
            PC_GEMM pcDeq{K, N, 0};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                // 每个线程处理一个元素
                uint32_t totalElems = K * N;
                passDeq.dispatch(cmd, &pcDeq, sizeof(pcDeq), (totalElems + 255u) / 256u, 1, 1);
            });

            // 下载反量化后的权重，验证正确性
            auto deqOut = vkDownload(ctx, bufWf32, K * N * 4);
            const float* deqW = reinterpret_cast<const float*>(deqOut.data());

            // 验证反量化本身
            VerifyResult deqVr = verify(deqW, hWq4.data(), K, N, 1e-3f);
            std::cout << i18n::tr("  反量化正确性: 错误数=", "  Dequant correctness: errors=") << deqVr.errorCount
                      << i18n::tr("，最大误差=", ", max error=") << deqVr.maxError << "\n";

            // 第二步：用 FP32 GEMM 计算
            passF32.bindBuffers({{0, bufA.buf}, {1, bufWf32.buf}, {2, bufC.buf}});
            PC_GEMM pcF32{M, N, K};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passF32.dispatch(cmd, &pcF32, sizeof(pcF32), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            VerifyResult vr = verify(c, hRefQ4.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  Standalone 路径 GEMM 错误数=", "  Standalone path GEMM errors=") << vr.errorCount
                      << i18n::tr("（容差 0.1%），最大误差=", " (tol 0.1%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 测试 4: FP32 GEMM 参考
        // ============================================================
        std::cout << i18n::tr("\n--- 测试 4: FP32 GEMM 参考 ---\n",
                              "\n--- Test 4: FP32 GEMM Reference ---\n");
        {
            // 上传 FP32 权重
            Buffer bufWf32Upload = vkCreateBuffer(ctx, K * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            vkUpload(ctx, bufWf32Upload, hW.data(), K * N * 4);

            passF32.bindBuffers({{0, bufA.buf}, {1, bufWf32Upload.buf}, {2, bufC.buf}});
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

            vkDestroyBuffer(ctx, bufWf32Upload);
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

        // 准备 buffer：FP32 GEMM 需要单独上传 FP32 权重
        Buffer bufWf32Bench = vkCreateBuffer(ctx, K * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufWf32Bench, hW.data(), K * N * 4);

        // FP32 GEMM
        passF32.bindBuffers({{0, bufA.buf}, {1, bufWf32Bench.buf}, {2, bufC.buf}});
        PC_GEMM pcF32{M, N, K};
        for (int i = 0; i < warmupIters; ++i) {
            bench_gpu_ms(ctx, passF32, &pcF32, sizeof(pcF32), gx, gy, 1);
        }
        double fp32Sum = 0.0;
        for (int i = 0; i < benchIters; ++i) {
            fp32Sum += bench_gpu_ms(ctx, passF32, &pcF32, sizeof(pcF32), gx, gy, 1);
        }
        double fp32Avg = fp32Sum / benchIters;

        // Q8_0 融合 GEMM
        passQ8.bindBuffers({{0, bufA.buf}, {1, bufWq8.buf}, {2, bufBias.buf}, {3, bufC.buf}});
        PC18 pcQ8{M, N, K, 1};
        for (int i = 0; i < warmupIters; ++i) {
            bench_gpu_ms(ctx, passQ8, &pcQ8, sizeof(pcQ8), gx, gy, 1);
        }
        double q8Sum = 0.0;
        for (int i = 0; i < benchIters; ++i) {
            q8Sum += bench_gpu_ms(ctx, passQ8, &pcQ8, sizeof(pcQ8), gx, gy, 1);
        }
        double q8Avg = q8Sum / benchIters;

        // Q4_0 融合 GEMM
        passQ4.bindBuffers({{0, bufA.buf}, {1, bufWq4.buf}, {2, bufBias.buf}, {3, bufC.buf}});
        PC18 pcQ4{M, N, K, 1};
        for (int i = 0; i < warmupIters; ++i) {
            bench_gpu_ms(ctx, passQ4, &pcQ4, sizeof(pcQ4), gx, gy, 1);
        }
        double q4Sum = 0.0;
        for (int i = 0; i < benchIters; ++i) {
            q4Sum += bench_gpu_ms(ctx, passQ4, &pcQ4, sizeof(pcQ4), gx, gy, 1);
        }
        double q4Avg = q4Sum / benchIters;

        double flops = 2.0 * M * N * K;

        std::cout << i18n::tr("  FP32 GEMM  : ", "  FP32 GEMM  : ") << fp32Avg
                  << i18n::tr(" ms (", " ms (") << (flops / fp32Avg / 1e6) << " GFLOP/s)\n";
        std::cout << i18n::tr("  Q8_0 fused : ", "  Q8_0 fused : ") << q8Avg
                  << i18n::tr(" ms (", " ms (") << (flops / q8Avg / 1e6) << " GFLOP/s, "
                  << i18n::tr("速度比 FP32 的 ", " speedup vs FP32: ") << (fp32Avg / q8Avg) << "x)\n";
        std::cout << i18n::tr("  Q4_0 fused : ", "  Q4_0 fused : ") << q4Avg
                  << i18n::tr(" ms (", " ms (") << (flops / q4Avg / 1e6) << " GFLOP/s, "
                  << i18n::tr("速度比 FP32 的 ", " speedup vs FP32: ") << (fp32Avg / q4Avg) << "x)\n";

        // 带宽分析
        double fp32BytesPerOp = (double)(M * K + K * N + M * N) * 4.0;
        double q8BytesPerOp   = (double)(M * K * 4 + q8w.data.size() + M * N * 4);
        double q4BytesPerOp   = (double)(M * K * 4 + q4w.data.size() + M * N * 4);

        std::cout << i18n::tr("\n--- 带宽分析 ---\n", "\n--- Bandwidth Analysis ---\n");
        std::cout << i18n::tr("  FP32 访存: ", "  FP32 traffic: ") << (fp32BytesPerOp / 1e6)
                  << i18n::tr(" MB，算术强度: ", " MB, AI: ") << (flops / fp32BytesPerOp) << " FLOP/byte\n";
        std::cout << i18n::tr("  Q8_0 访存: ", "  Q8_0 traffic: ") << (q8BytesPerOp / 1e6)
                  << i18n::tr(" MB，算术强度: ", " MB, AI: ") << (flops / q8BytesPerOp) << " FLOP/byte\n";
        std::cout << i18n::tr("  Q4_0 访存: ", "  Q4_0 traffic: ") << (q4BytesPerOp / 1e6)
                  << i18n::tr(" MB，算术强度: ", " MB, AI: ") << (flops / q4BytesPerOp) << " FLOP/byte\n";

        // 压缩比总结
        std::cout << i18n::tr("\n--- 压缩比总结 ---\n", "\n--- Compression Summary ---\n");
        std::cout << i18n::tr("  FP32: 1.00x (基线)\n", "  FP32: 1.00x (baseline)\n");
        std::cout << i18n::tr("  Q8_0: ", "  Q8_0: ") << (fp32Bytes / (double)q8Bytes) << "x\n";
        std::cout << i18n::tr("  Q4_0: ", "  Q4_0: ") << (fp32Bytes / (double)q4Bytes) << "x\n";

        // ---- 清理 ----
        passQ4.destroy();
        passQ8.destroy();
        passDeq.destroy();
        passF32.destroy();
        vkDestroyBuffer(ctx, bufA);
        vkDestroyBuffer(ctx, bufWq4);
        vkDestroyBuffer(ctx, bufWq8);
        vkDestroyBuffer(ctx, bufBias);
        vkDestroyBuffer(ctx, bufC);
        vkDestroyBuffer(ctx, bufWf32);
        vkDestroyBuffer(ctx, bufWf32Bench);
        vkDestroyContext(ctx);

        std::cout << i18n::tr("\n第 18 章完成！\n", "\nChapter 18 complete!\n");

    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}