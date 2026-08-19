// ch11_quant/main.cpp
// 量化与融合反量化：int8、Q4_0、Q8_0 三种量化格式的 GPU 融合反量化 GEMM。
// 对照：若不融合，得先把量化权重解压成 float 的临时 buffer，既多吃显存又多一次搬运。
//
// Q4_0 格式：32 元素/block，fp16 scale + 4-bit 有符号 int 打包（2 元素/字节），20 字节/block
// Q8_0 格式：32 元素/block，fp16 scale + int8 数据，36 字节/block（对齐版）

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

// ---- Push constant 结构（与着色器一致） ----

// 用于 int8 q8_gemm.comp: { M, N, K, scale }
struct PC11 { uint32_t M, N, K; float scale; };

// 用于 Q4_0/Q8_0 gemm: { M, N, K, hasBias }
struct PC_Quant { uint32_t M, N, K, hasBias; };

// 用于 FP32 gemm_tiled: { M, N, K }
struct PC_GEMM { uint32_t M, N, K; };

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
        return (uint16_t)(sign << 15);
    }
    if (exp == 255) {
        uint16_t h = (uint16_t)(sign << 15) | (31 << 10);
        if (mant) h |= 0x3FF;
        return h;
    }

    exp = exp - 127 + 15;
    if (exp >= 31) {
        return (uint16_t)(sign << 15) | (31 << 10);
    }
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
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            int32_t e = -1;
            uint32_t m = mant;
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

// Q4_0 量化：将 float 权重矩阵打包为 Q4_0 格式（group-wise，每 32 元素一个 scale）
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

            // int4 范围 [-8, 7]，使用 7 作为对称量化最大等级
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

                uint8_t u0 = (uint8_t)(q0 & 0xF);
                uint8_t u1 = (uint8_t)(q1 & 0xF);
                block.qs[k / 2] = u0 | (u1 << 4);
            }
        }
    }

    return qw;
}

// Q8_0 量化：将 float 权重矩阵打包为 Q8_0 格式（group-wise，每 32 元素一个 scale）
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

// CPU 端 Q4_0 反量化（用于验证量化误差）
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
        uint32_t M = 256, K = 256, N = 256;

        std::mt19937 rng(11);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> hA(M * K), hW(K * N);
        if (tv.enabled()) {
            // 从测试向量读取形状并加载 A（输入）与 W（权重），以便在测试向量模式下运行 GEMM
            std::vector<uint32_t> sA, sW;
            tv.queryShape("A", sA);
            tv.queryShape("W", sW);
            if (sA.size() >= 2) { M = sA[0]; K = sA[1]; }
            if (sW.size() >= 2) N = sW[1];
            hA.resize(M * K); hW.resize(K * N);
            tv.load("A", {M, K}, hA.data());
            tv.load("W", {K, N}, hW.data());
        } else {
            for (auto& v : hA) v = dist(rng);
            for (auto& v : hW) v = dist(rng);
        }

        // ---- CPU 参考结果 ----
        std::vector<float> hRef = gemm_ref_fp32(hA.data(), hW.data(), M, K, N);

        // ============================================================
        // 6a. 对称 int8 量化（per-tensor）：w_q = round(w / scale)，scale = max|w|/127
        // ============================================================
        std::cout << i18n::tr("\n===== 第 11 章：量化与融合反量化 =====\n",
                              "\n===== Chapter 11: Quantization & Fused Dequant =====\n");
        std::cout << i18n::tr("矩阵维度: M=", "Matrix: M=") << M << " K=" << K << " N=" << N << "\n\n";

        float maxAbs = 0.0f;
        for (float v : hW) maxAbs = std::max(maxAbs, std::fabs(v));
        float scale = maxAbs / 127.0f;
        std::vector<int8_t> hWq(K * N);
        for (size_t i = 0; i < hW.size(); ++i) {
            int q = (int)std::lroundf(hW[i] / scale);
            q = std::max(-127, std::min(127, q));
            hWq[i] = (int8_t)q;
        }
        std::cout << i18n::tr("[int8] 权重 ", "[int8] weights ") << K * N
                  << i18n::tr(" 个：float32 需 ", " elems: float32 needs ") << (K * N * 4)
                  << i18n::tr(" B，", " B,")
                  << i18n::tr("int8 只需 ", "int8 needs only ") << (K * N * 1)
                  << i18n::tr(" B（省 4 倍）\n", " B (4x saving)\n");

        // ============================================================
        // 6b. Q4_0 / Q8_0 量化
        // ============================================================
        Q4_0_Weight q4w = quantize_q4_0(hW.data(), K, N);
        Q8_0_Weight q8w = quantize_q8_0(hW.data(), K, N);

        size_t fp32Bytes = (size_t)K * N * 4;
        size_t q4Bytes = q4w.data.size();
        size_t q8Bytes = q8w.data.size();

        std::cout << i18n::tr("[Q4_0] 权重 ", "[Q4_0] weights ") << q4Bytes
                  << i18n::tr(" 字节 (压缩比 ", " bytes (compression ") << (fp32Bytes / (double)q4Bytes)
                  << "x, " << i18n::tr("每权重 ", "bits/weight ") << (q4Bytes * 8.0 / (K * N)) << " bit)\n";
        std::cout << i18n::tr("[Q8_0] 权重 ", "[Q8_0] weights ") << q8Bytes
                  << i18n::tr(" 字节 (压缩比 ", " bytes (compression ") << (fp32Bytes / (double)q8Bytes)
                  << "x, " << i18n::tr("每权重 ", "bits/weight ") << (q8Bytes * 8.0 / (K * N)) << " bit)\n";

        // 量化误差
        std::vector<float> hWq4 = dequantize_q4_0_cpu(q4w);
        double q4mse = 0.0, q8mse = 0.0;
        for (uint32_t i = 0; i < K * N; ++i) {
            double d4 = hW[i] - hWq4[i];
            q4mse += d4 * d4;
            double d8 = hW[i] - (hWq[i] * scale);
            q8mse += d8 * d8;
        }
        q4mse = std::sqrt(q4mse / (K * N));
        q8mse = std::sqrt(q8mse / (K * N));
        std::cout << i18n::tr("  Q4_0 量化 RMSE: ", "  Q4_0 quantization RMSE: ") << q4mse << "\n";
        std::cout << i18n::tr("  Q8_0 量化 RMSE: ", "  Q8_0 quantization RMSE: ") << q8mse << "\n";
        std::cout << i18n::tr("  int8 量化 RMSE: ", "  int8 quantization RMSE: ") << q8mse << " (per-tensor)\n\n";

        // ============================================================
        // 6c. 创建 GPU 资源
        // ============================================================
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;

        // int8 路径
        Buffer bufA  = vkCreateBuffer(ctx, M * K * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWq = vkCreateBuffer(ctx, K * N * 1, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufC  = vkCreateBuffer(ctx, M * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufA, hA.data(), M * K * 4);
        vkUpload(ctx, bufWq, hWq.data(), K * N * 1);

        // Q4_0 / Q8_0 路径
        Buffer bufWq4  = vkCreateBuffer(ctx, q4w.data.size(), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWq8  = vkCreateBuffer(ctx, q8w.data.size(), usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufBias = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufWq4, q4w.data.data(), q4w.data.size());
        vkUpload(ctx, bufWq8, q8w.data.data(), q8w.data.size());
        // bias 全零（本测试中不使用偏置，但着色器需要绑定）
        std::vector<float> zeroBias(N, 0.0f);
        vkUpload(ctx, bufBias, zeroBias.data(), N * 4);

        // FP32 基准路径
        Buffer bufWf32 = vkCreateBuffer(ctx, K * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufWf32, hW.data(), K * N * 4);

        // ============================================================
        // 6d. 准备 ComputePass
        // ============================================================

        // 现有 int8 融合反量化 GEMM
        ComputePass passInt8;
        passInt8.prepare(ctx, std::string(SHADER_DIR) + "q8_gemm.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC11)}});

        // Q4_0 融合反量化 GEMM
        ComputePass passQ4;
        passQ4.prepare(ctx, std::string(SHADER_DIR) + "q4_0_gemm.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_Quant)}});

        // Q8_0 融合反量化 GEMM
        ComputePass passQ8;
        passQ8.prepare(ctx, std::string(SHADER_DIR) + "q8_0_gemm.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_Quant)}});

        // FP32 基准 GEMM（分块 tiled）
        ComputePass passF32;
        passF32.prepare(ctx, std::string(SHADER_DIR) + "gemm_fp32.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_GEMM)}});

        uint32_t gx = (N + 15u) / 16u, gy = (M + 15u) / 16u;

        // ============================================================
        // 6e. 验证：int8 融合反量化 GEMM
        // ============================================================
        std::cout << i18n::tr("--- 验证：int8 融合反量化 GEMM ---\n",
                              "--- Verify: int8 Fused Dequant GEMM ---\n");
        {
            passInt8.bindBuffers({{0, bufA.buf}, {1, bufWq.buf}, {2, bufC.buf}});
            PC11 pc{M, N, K, scale};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passInt8.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            // int8 参考：使用量化后的权重
            std::vector<float> refWq(K * N);
            for (uint32_t i = 0; i < K * N; ++i)
                refWq[i] = hWq[i] * scale;
            std::vector<float> refC = gemm_ref_fp32(hA.data(), refWq.data(), M, K, N);

            VerifyResult vr = verify(c, refC.data(), M, N, 0.05f);
            std::cout << i18n::tr("  int8 错误数=", "  int8 errors=") << vr.errorCount
                      << i18n::tr("（容差 5%），最大误差=", " (tol 5%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 6f. 验证：Q4_0 融合反量化 GEMM
        // ============================================================
        std::cout << i18n::tr("--- 验证：Q4_0 融合反量化 GEMM ---\n",
                              "--- Verify: Q4_0 Fused Dequant GEMM ---\n");
        {
            passQ4.bindBuffers({{0, bufA.buf}, {1, bufWq4.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC_Quant pc{M, N, K, 0};  // hasBias = 0
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passQ4.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            // Q4_0 参考：使用反量化后的权重
            std::vector<float> refWq4 = dequantize_q4_0_cpu(q4w);
            std::vector<float> refC = gemm_ref_fp32(hA.data(), refWq4.data(), M, K, N);

            VerifyResult vr = verify(c, refC.data(), M, N, 0.1f);
            std::cout << i18n::tr("  Q4_0 错误数=", "  Q4_0 errors=") << vr.errorCount
                      << i18n::tr("（容差 10%），最大误差=", " (tol 10%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 6g. 验证：Q8_0 融合反量化 GEMM
        // ============================================================
        std::cout << i18n::tr("--- 验证：Q8_0 融合反量化 GEMM ---\n",
                              "--- Verify: Q8_0 Fused Dequant GEMM ---\n");
        {
            passQ8.bindBuffers({{0, bufA.buf}, {1, bufWq8.buf}, {2, bufBias.buf}, {3, bufC.buf}});
            PC_Quant pc{M, N, K, 0};  // hasBias = 0
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passQ8.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            // Q8_0 参考：使用量化后的权重（反量化）
            std::vector<float> refWq8(K * N);
            const Q8_0_Block* blocks = reinterpret_cast<const Q8_0_Block*>(q8w.data.data());
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t b = 0; b < q8w.blocksPerCol; ++b) {
                    const Q8_0_Block& block = blocks[n * q8w.blocksPerCol + b];
                    float s = fp16_to_fp32(block.d);
                    uint32_t kStart = b * 32;
                    for (uint32_t k = 0; k < 32; ++k) {
                        uint32_t ki = kStart + k;
                        if (ki < K)
                            refWq8[ki * N + n] = (float)block.qs[k] * s;
                    }
                }
            }
            std::vector<float> refC = gemm_ref_fp32(hA.data(), refWq8.data(), M, K, N);

            VerifyResult vr = verify(c, refC.data(), M, N, 0.05f);
            std::cout << i18n::tr("  Q8_0 错误数=", "  Q8_0 errors=") << vr.errorCount
                      << i18n::tr("（容差 5%），最大误差=", " (tol 5%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 6h. 验证：FP32 基准 GEMM
        // ============================================================
        std::cout << i18n::tr("--- 验证：FP32 基准 GEMM ---\n",
                              "--- Verify: FP32 baseline GEMM ---\n");
        {
            passF32.bindBuffers({{0, bufA.buf}, {1, bufWf32.buf}, {2, bufC.buf}});
            PC_GEMM pc{M, N, K};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passF32.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
            });

            auto out = vkDownload(ctx, bufC, M * N * 4);
            const float* c = reinterpret_cast<const float*>(out.data());

            VerifyResult vr = verify(c, hRef.data(), M, N, 1e-3f);
            std::cout << i18n::tr("  FP32 错误数=", "  FP32 errors=") << vr.errorCount
                      << i18n::tr("（容差 0.1%），最大误差=", " (tol 0.1%), max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n\n";
        }

        // ---- 测试向量模式：写回 FP32 基准 GEMM 输出 ----
        if (tv.enabled()) {
            auto outC = vkDownload(ctx, bufC, M * N * 4);
            tv.save("C", {M, N}, reinterpret_cast<const float*>(outC.data()));
        }

        // ============================================================
        // 6i. 性能基准测试
        // ============================================================
        std::cout << i18n::tr("--- 性能基准 (GPU 时间戳) ---\n",
                              "--- Benchmark (GPU Timestamps) ---\n");

        const int warmupIters = 5;
        const int benchIters = 10;

        // FP32 GEMM
        passF32.bindBuffers({{0, bufA.buf}, {1, bufWf32.buf}, {2, bufC.buf}});
        PC_GEMM pcF32{M, N, K};
        for (int i = 0; i < warmupIters; ++i)
            bench_gpu_ms(ctx, passF32, &pcF32, sizeof(pcF32), gx, gy, 1);
        double fp32Sum = 0.0;
        for (int i = 0; i < benchIters; ++i)
            fp32Sum += bench_gpu_ms(ctx, passF32, &pcF32, sizeof(pcF32), gx, gy, 1);
        double fp32Avg = fp32Sum / benchIters;

        // int8 融合 GEMM
        passInt8.bindBuffers({{0, bufA.buf}, {1, bufWq.buf}, {2, bufC.buf}});
        PC11 pcInt8{M, N, K, scale};
        for (int i = 0; i < warmupIters; ++i)
            bench_gpu_ms(ctx, passInt8, &pcInt8, sizeof(pcInt8), gx, gy, 1);
        double int8Sum = 0.0;
        for (int i = 0; i < benchIters; ++i)
            int8Sum += bench_gpu_ms(ctx, passInt8, &pcInt8, sizeof(pcInt8), gx, gy, 1);
        double int8Avg = int8Sum / benchIters;

        // Q8_0 融合 GEMM
        passQ8.bindBuffers({{0, bufA.buf}, {1, bufWq8.buf}, {2, bufBias.buf}, {3, bufC.buf}});
        PC_Quant pcQ8{M, N, K, 0};
        for (int i = 0; i < warmupIters; ++i)
            bench_gpu_ms(ctx, passQ8, &pcQ8, sizeof(pcQ8), gx, gy, 1);
        double q8Sum = 0.0;
        for (int i = 0; i < benchIters; ++i)
            q8Sum += bench_gpu_ms(ctx, passQ8, &pcQ8, sizeof(pcQ8), gx, gy, 1);
        double q8Avg = q8Sum / benchIters;

        // Q4_0 融合 GEMM
        passQ4.bindBuffers({{0, bufA.buf}, {1, bufWq4.buf}, {2, bufBias.buf}, {3, bufC.buf}});
        PC_Quant pcQ4{M, N, K, 0};
        for (int i = 0; i < warmupIters; ++i)
            bench_gpu_ms(ctx, passQ4, &pcQ4, sizeof(pcQ4), gx, gy, 1);
        double q4Sum = 0.0;
        for (int i = 0; i < benchIters; ++i)
            q4Sum += bench_gpu_ms(ctx, passQ4, &pcQ4, sizeof(pcQ4), gx, gy, 1);
        double q4Avg = q4Sum / benchIters;

        double flops = 2.0 * M * N * K;

        auto fmtMs = [&](double ms) -> std::string {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.3f", ms);
            return buf;
        };
        auto fmtGflops = [&](double ms) -> std::string {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.1f", flops / ms / 1e6);
            return buf;
        };
        auto fmtRatio = [&](double a, double b) -> std::string {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.2f", a / b);
            return buf;
        };

        std::cout << i18n::tr("  FP32 GEMM  : ", "  FP32 GEMM  : ") << fmtMs(fp32Avg)
                  << i18n::tr(" ms (", " ms (") << fmtGflops(fp32Avg) << " GFLOP/s)\n";
        std::cout << i18n::tr("  int8 fused : ", "  int8 fused : ") << fmtMs(int8Avg)
                  << i18n::tr(" ms (", " ms (") << fmtGflops(int8Avg) << " GFLOP/s, "
                  << i18n::tr("速度比 FP32 的 ", " speedup vs FP32: ") << fmtRatio(fp32Avg, int8Avg) << "x)\n";
        std::cout << i18n::tr("  Q8_0 fused : ", "  Q8_0 fused : ") << fmtMs(q8Avg)
                  << i18n::tr(" ms (", " ms (") << fmtGflops(q8Avg) << " GFLOP/s, "
                  << i18n::tr("速度比 FP32 的 ", " speedup vs FP32: ") << fmtRatio(fp32Avg, q8Avg) << "x)\n";
        std::cout << i18n::tr("  Q4_0 fused : ", "  Q4_0 fused : ") << fmtMs(q4Avg)
                  << i18n::tr(" ms (", " ms (") << fmtGflops(q4Avg) << " GFLOP/s, "
                  << i18n::tr("速度比 FP32 的 ", " speedup vs FP32: ") << fmtRatio(fp32Avg, q4Avg) << "x)\n";

        // 带宽分析
        double fp32Traffic = (double)(M * K + K * N + M * N) * 4.0;
        double int8Traffic = (double)(M * K * 4 + K * N * 1 + M * N * 4);
        double q8Traffic   = (double)(M * K * 4 + q8w.data.size() + M * N * 4);
        double q4Traffic   = (double)(M * K * 4 + q4w.data.size() + M * N * 4);

        std::cout << i18n::tr("\n--- 带宽与算术强度 ---\n",
                              "\n--- Bandwidth & Arithmetic Intensity ---\n");
        std::cout << i18n::tr("  FP32: 访存 ", "  FP32: traffic ") << (fp32Traffic / 1e6)
                  << i18n::tr(" MB，算术强度 ", " MB, AI ") << (flops / fp32Traffic) << " FLOP/byte\n";
        std::cout << i18n::tr("  int8: 访存 ", "  int8: traffic ") << (int8Traffic / 1e6)
                  << i18n::tr(" MB，算术强度 ", " MB, AI ") << (flops / int8Traffic) << " FLOP/byte\n";
        std::cout << i18n::tr("  Q8_0: 访存 ", "  Q8_0: traffic ") << (q8Traffic / 1e6)
                  << i18n::tr(" MB，算术强度 ", " MB, AI ") << (flops / q8Traffic) << " FLOP/byte\n";
        std::cout << i18n::tr("  Q4_0: 访存 ", "  Q4_0: traffic ") << (q4Traffic / 1e6)
                  << i18n::tr(" MB，算术强度 ", " MB, AI ") << (flops / q4Traffic) << " FLOP/byte\n";

        // 压缩比与精度总结
        std::cout << i18n::tr("\n--- 压缩比与精度总结 ---\n",
                              "\n--- Compression & Accuracy Summary ---\n");
        std::cout << i18n::tr("  格式    压缩比   bits/weight   RMSE\n",
                              "  Format  Ratio   bits/weight   RMSE\n");
        std::cout << i18n::tr("  FP32    1.00x   32.0          -\n",
                              "  FP32    1.00x   32.0          -\n");
        std::cout << i18n::tr("  int8    ", "  int8    ") << (fp32Bytes / (double)(K * N)) << "x   "
                  << "8.0          " << q8mse << "\n";
        std::cout << i18n::tr("  Q8_0    ", "  Q8_0    ") << (fp32Bytes / (double)q8Bytes) << "x   "
                  << (q8Bytes * 8.0 / (K * N)) << "       " << q8mse << "\n";
        std::cout << i18n::tr("  Q4_0    ", "  Q4_0    ") << (fp32Bytes / (double)q4Bytes) << "x   "
                  << (q4Bytes * 8.0 / (K * N)) << "       " << q4mse << "\n";

        // ---- 清理 ----
        passInt8.destroy();
        passQ4.destroy();
        passQ8.destroy();
        passF32.destroy();
        vkDestroyBuffer(ctx, bufA);
        vkDestroyBuffer(ctx, bufWq);
        vkDestroyBuffer(ctx, bufWq4);
        vkDestroyBuffer(ctx, bufWq8);
        vkDestroyBuffer(ctx, bufBias);
        vkDestroyBuffer(ctx, bufWf32);
        vkDestroyBuffer(ctx, bufC);
        vkDestroyContext(ctx);

        std::cout << i18n::tr("\n第 11 章完成！\n", "\nChapter 11 complete!\n");

    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}