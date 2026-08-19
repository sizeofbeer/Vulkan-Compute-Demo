// ch17_flash_attn/main.cpp
// FlashAttention vs 朴素注意力：正确性验证 + 性能对比
//
// 实现两种注意力机制：
//   1. FlashAttention（融合）：online softmax + 分块 K/V，单 kernel
//   2. 朴素注意力（3 步）：先算所有分数，再 softmax，最后上下文
//
// 对多种序列长度（128, 256, 512, 1024）测时，用 GPU 时间戳精确对比。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

struct PC17 { uint32_t N, d, Br, Bc; };

// 创建一个 ComputePass 的辅助函数
static ComputePass makePass(VulkanContext& ctx, const std::string& spv, uint32_t nBind) {
    ComputePass p;
    std::vector<vk::DescriptorSetLayoutBinding> binds;
    for (uint32_t i = 0; i < nBind; ++i)
        binds.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
    p.prepare(ctx, std::string(SHADER_DIR) + spv, binds,
        {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC17)}});
    return p;
}

// GPU 单次 dispatch 计时（毫秒）
static double gpuTimeMs(VulkanContext& ctx, ComputePass& pass, const PC17& pc,
                        uint32_t gx, uint32_t gy, uint32_t gz) {
    vk::QueryPool pool = vkCreateTimestampPool(ctx, 2);

    vk::CommandBuffer cmd =
        ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmd.resetQueryPool(pool, 0, 2);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 0);
    pass.dispatch(cmd, &pc, sizeof(pc), gx, gy, gz);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 1);
    cmd.end();

    vk::SubmitInfo si; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    ctx.queue.submit(1, &si, ctx.fence);
    ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
    ctx.device.resetFences(1, &ctx.fence);

    uint64_t t0 = vkReadTimestamp(ctx, pool, 0);
    uint64_t t1 = vkReadTimestamp(ctx, pool, 1);

    ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
    ctx.device.destroyQueryPool(pool);

    return double(t1 - t0) / 1e6;  // ns → ms
}

// CPU 参考：softmax(QK^T / sqrt(d)) · V
static std::vector<float> cpuAttention(const std::vector<float>& Q,
                                       const std::vector<float>& K,
                                       const std::vector<float>& V,
                                       uint32_t N, uint32_t d) {
    std::vector<float> O(N * d, 0.0f);
    float inv_sqrt_d = 1.0f / std::sqrt((float)d);

    for (uint32_t qi = 0; qi < N; ++qi) {
        // 计算分数
        std::vector<float> scores(N);
        float m = -1e30f;
        for (uint32_t ki = 0; ki < N; ++ki) {
            float dot = 0.0f;
            for (uint32_t j = 0; j < d; ++j)
                dot += Q[qi * d + j] * K[ki * d + j];
            scores[ki] = dot * inv_sqrt_d;
            m = std::max(m, scores[ki]);
        }

        // softmax
        float sum = 0.0f;
        for (uint32_t ki = 0; ki < N; ++ki) {
            scores[ki] = std::exp(scores[ki] - m);
            sum += scores[ki];
        }
        for (uint32_t ki = 0; ki < N; ++ki) scores[ki] /= sum;

        // 上下文
        for (uint32_t j = 0; j < d; ++j) {
            float acc = 0.0f;
            for (uint32_t ki = 0; ki < N; ++ki)
                acc += scores[ki] * V[ki * d + j];
            O[qi * d + j] = acc;
        }
    }
    return O;
}

// 验证 GPU 输出与 CPU 参考的一致性
static uint32_t verify(const float* gpu, const float* cpu, uint32_t n, float eps) {
    uint32_t errs = 0;
    for (uint32_t i = 0; i < n; ++i) {
        float ref = cpu[i];
        if (std::fabs(gpu[i] - ref) > std::max(eps, std::fabs(ref) * 1e-3f))
            ++errs;
    }
    return errs;
}

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();

        // ---- 测试向量模式：加载外部 Q/K/V，跑一次 FlashAttention 并写回 out ----
        if (tv.enabled()) {
            std::vector<uint32_t> sQ;
            tv.queryShape("Q", sQ);
            if (sQ.size() < 2) { std::cerr << "[test_vectors] Q 形状无效\n"; return 1; }
            uint32_t N = sQ[0], d = sQ[1];
            std::vector<float> hQ(N * d), hK(N * d), hV(N * d);
            tv.load("Q", {N, d}, hQ.data());
            tv.load("K", {N, d}, hK.data());
            tv.load("V", {N, d}, hV.data());

            vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                       | vk::BufferUsageFlagBits::eTransferSrc
                                       | vk::BufferUsageFlagBits::eTransferDst;
            Buffer bQ = vkCreateBuffer(ctx, N * d * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            Buffer bK = vkCreateBuffer(ctx, N * d * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            Buffer bV = vkCreateBuffer(ctx, N * d * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            Buffer bO = vkCreateBuffer(ctx, N * d * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            vkUpload(ctx, bQ, hQ.data(), N * d * 4);
            vkUpload(ctx, bK, hK.data(), N * d * 4);
            vkUpload(ctx, bV, hV.data(), N * d * 4);

            ComputePass passFA = makePass(ctx, "flash_attn.spv", 4);
            passFA.bindBuffers({{0, bQ.buf}, {1, bK.buf}, {2, bV.buf}, {3, bO.buf}});
            PC17 pc{N, d, 1, 32};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passFA.dispatch(cmd, &pc, sizeof(pc), N, 1, 1);
            });
            auto outFA = vkDownload(ctx, bO, N * d * 4);
            tv.save("out", {N, d}, reinterpret_cast<const float*>(outFA.data()));

            passFA.destroy();
            for (auto b : {bQ, bK, bV, bO}) vkDestroyBuffer(ctx, b);
            vkDestroyContext(ctx);
            return 0;
        }

        const uint32_t d = 64;  // 头维度
        const uint32_t testNs[] = {128, 256, 512, 1024};
        const int numTests = 4;

        std::mt19937 rng(314159);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        auto mkBuf = [&](size_t n) {
            return vkCreateBuffer(ctx, n * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        };

        // 创建两个 ComputePass（复用管线）
        ComputePass passFA = makePass(ctx, "flash_attn.spv", 4);
        ComputePass passNaive = makePass(ctx, "flash_attn_naive.spv", 4);

        std::cout << i18n::tr("\n===== FlashAttention vs 朴素注意力 性能对比 =====\n",
                              "\n===== FlashAttention vs Naive Attention Benchmark =====\n");
        std::cout << i18n::tr("头维度 d = ", "Head dimension d = ") << d << "\n\n";

        for (int ti = 0; ti < numTests; ++ti) {
            uint32_t N = testNs[ti];
            std::cout << i18n::tr("--- N = ", "--- N = ") << N << " ---\n";

            // 生成随机 Q, K, V
            std::vector<float> hQ(N * d), hK(N * d), hV(N * d);
            for (auto& v : hQ) v = dist(rng);
            for (auto& v : hK) v = dist(rng);
            for (auto& v : hV) v = dist(rng);

            // 上传
            Buffer bQ = mkBuf(N * d), bK = mkBuf(N * d), bV = mkBuf(N * d);
            Buffer bO_FA = mkBuf(N * d), bO_Naive = mkBuf(N * d);
            vkUpload(ctx, bQ, hQ.data(), N * d * 4);
            vkUpload(ctx, bK, hK.data(), N * d * 4);
            vkUpload(ctx, bV, hV.data(), N * d * 4);

            // 绑 buffer
            passFA.bindBuffers({{0, bQ.buf}, {1, bK.buf}, {2, bV.buf}, {3, bO_FA.buf}});
            passNaive.bindBuffers({{0, bQ.buf}, {1, bK.buf}, {2, bV.buf}, {3, bO_Naive.buf}});

            PC17 pc{N, d, 1, 32};  // Br=1, Bc=32
            uint32_t gx = N;  // 每个 query 行一个工作组

            // ---- 计时（各跑 5 次取平均）----
            const int repeats = 5;

            // 先跑一次预热（GPU 管线冷启动不纳入统计）
            gpuTimeMs(ctx, passFA, pc, gx, 1, 1);
            gpuTimeMs(ctx, passNaive, pc, gx, 1, 1);
            double faTime = 0.0;
            for (int r = 0; r < repeats; ++r) {
                faTime += gpuTimeMs(ctx, passFA, pc, gx, 1, 1);
            }
            faTime /= repeats;

            double naiveTime = 0.0;
            for (int r = 0; r < repeats; ++r) {
                naiveTime += gpuTimeMs(ctx, passNaive, pc, gx, 1, 1);
            }
            naiveTime /= repeats;

            // ---- 验证正确性 ----
            auto outFA = vkDownload(ctx, bO_FA, N * d * 4);
            auto outNaive = vkDownload(ctx, bO_Naive, N * d * 4);
            const float* faData = reinterpret_cast<const float*>(outFA.data());
            const float* naiveData = reinterpret_cast<const float*>(outNaive.data());

            // CPU 参考
            std::vector<float> cpuRef = cpuAttention(hQ, hK, hV, N, d);

            uint32_t errFA = verify(faData, cpuRef.data(), N * d, 1e-2f);
            uint32_t errNaive = verify(naiveData, cpuRef.data(), N * d, 1e-2f);

            // FlashAttention vs 朴素 GPU（两者互相对比）
            uint32_t errCross = verify(faData, naiveData, N * d, 1e-2f);

            // ---- 输出 ----
            std::cout << i18n::tr("  FlashAttention : ", "  FlashAttention : ")
                      << faTime << " ms  "
                      << i18n::tr("错误=", "errors=") << errFA << "\n";
            std::cout << i18n::tr("  朴素注意力     : ", "  Naive Attention: ")
                      << naiveTime << " ms  "
                      << i18n::tr("错误=", "errors=") << errNaive << "\n";
            std::cout << i18n::tr("  交叉验证错误  : ", "  Cross-validate  : ")
                      << errCross << i18n::tr("（应为0）", " (should be 0)") << "\n";

            if (naiveTime > 0.0) {
                double speedup = naiveTime / faTime;
                std::cout << i18n::tr("  加速比        : ", "  Speedup        : ")
                          << speedup << "x\n";
            }

            std::cout << "\n";

            for (auto b : {bQ, bK, bV, bO_FA, bO_Naive})
                vkDestroyBuffer(ctx, b);
        }

        passFA.destroy();
        passNaive.destroy();
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}