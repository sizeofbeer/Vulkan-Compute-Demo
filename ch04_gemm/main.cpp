// ch04_gemm/main.cpp
// 矩阵乘 GEMM：C[M,N] = A[M,K] * B[K,N]。这是神经网络 90% 算力的来源（全连接/卷积/注意力都化归为它）。
// 本章首次使用 common 的 ComputePass 封装（mini-VkCompute），把「建管线/描述符/dispatch」收敛掉，
// 让 main 只表达「喂 3 个 buffer + 压 M/N/K + 分多少组」。
//
// 扩展：如果设备支持 VK_KHR_cooperative_matrix，额外使用 Cooperative Matrix（Tensor Core）
// 路径进行硬件加速，并与普通分块 GEMM 对比性能。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>
#include <cstdio>

// 与 gemm_tiled.comp push_constant 布局一致
struct GEMM_PC { uint32_t M, N, K; };

// 与 gemm_cm.comp push_constant 布局一致
struct CM_PC {
    uint32_t strideA;   // A 的行跨度（元素数），即 K
    uint32_t strideB;   // B 的行跨度（元素数），即 N
    uint32_t strideC;   // C 的行跨度（元素数），即 N
    uint32_t K_total;   // K 总维度
};

// 计时辅助：GPU timestamp query 测量单次 dispatch 耗时（毫秒）
static double measureMs(VulkanContext& ctx, ComputePass& pass,
                        const void* pc, size_t pcSize,
                        uint32_t gx, uint32_t gy, uint32_t gz = 1) {
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

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();

        uint32_t M = 256, N = 256, K = 256;
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> hA(M * K), hB(K * N);
        if (tv.enabled()) {
            // 从测试向量读取真实形状（A:{M,K}, B:{K,N}），并加载输入
            std::vector<uint32_t> sA, sB;
            tv.queryShape("A", sA);
            tv.queryShape("B", sB);
            if (sA.size() >= 2) { M = sA[0]; K = sA[1]; }
            if (sB.size() >= 2) N = sB[1];
            hA.resize(M * K); hB.resize(K * N);
            tv.load("A", {M, K}, hA.data());
            tv.load("B", {K, N}, hB.data());
        } else {
            for (auto& v : hA) v = dist(rng);
            for (auto& v : hB) v = dist(rng);
        }

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        Buffer bufA = vkCreateBuffer(ctx, M * K * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufB = vkCreateBuffer(ctx, K * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufC = vkCreateBuffer(ctx, M * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufA, hA.data(), M * K * 4);
        vkUpload(ctx, bufB, hB.data(), K * N * 4);

        // ---- CPU 参考（朴素三重循环，用于验证） ----
        std::vector<float> cpuRef(M * N, 0.0f);
        for (uint32_t r = 0; r < M; ++r)
            for (uint32_t col = 0; col < N; ++col)
                for (uint32_t k = 0; k < K; ++k)
                    cpuRef[r * N + col] += hA[r * K + k] * hB[k * N + col];

        // ============================================================
        // 路径1：分块 GEMM（gemm_tiled.comp）
        // ============================================================
        std::cout << i18n::tr("\n[ch04] 路径1：分块 GEMM（共享内存）",
                               "\n[ch04] Path 1: Tiled GEMM (shared memory)") << "\n";

        ComputePass passTiled;
        passTiled.prepare(ctx,
            std::string(SHADER_DIR) + "gemm_tiled.spv",
            {   // 3 个 storage buffer
                {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            },
            {   // push constant：M,N,K（3 个 uint = 12 字节）
                vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(GEMM_PC)}
            });
        passTiled.bindBuffers({{0, bufA.buf}, {1, bufB.buf}, {2, bufC.buf}});

        // 工作组尺寸 16x16，故分组数 = ceil(维/16)
        uint32_t gx = (N + 15u) / 16u;
        uint32_t gy = (M + 15u) / 16u;
        GEMM_PC pc{M, N, K};

        // 预热（vkSubmit 会分配命令缓冲并等待完成）
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) { passTiled.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1); });

        // 计时（5 次取平均）
        const int warmup = 1, samples = 5;
        double tiledMs = 0.0;
        for (int i = 0; i < warmup + samples; ++i) {
            double ms = measureMs(ctx, passTiled, &pc, sizeof(pc), gx, gy);
            if (i >= warmup) tiledMs += ms;
        }
        tiledMs /= samples;

        // 验证
        auto outTiled = vkDownload(ctx, bufC, M * N * 4);
        const float* cTiled = reinterpret_cast<const float*>(outTiled.data());
        uint32_t errTiled = 0;
        for (uint32_t r = 0; r < M; ++r)
            for (uint32_t col = 0; col < N; ++col) {
                float s = cpuRef[r * N + col];
                float v = cTiled[r * N + col];
                if (std::fabs(v - s) > std::max(1e-2f, std::fabs(s) * 1e-3f)) ++errTiled;
            }
        double tiledGflops = 2.0 * M * N * K / (tiledMs * 1e-3) / 1e9;
        std::cout << i18n::tr("  结果: ", "  Result: ")
                  << (errTiled == 0 ? i18n::tr("正确 ✓", "correct ✓") : i18n::tr("有错误", "has errors"))
                  << i18n::tr("，耗时=", ", time=") << tiledMs << " ms"
                  << i18n::tr("，GFLOPS=", ", GFLOPS=") << tiledGflops << "\n";

        // ============================================================
        // 路径2：Cooperative Matrix GEMM（gemm_cm.comp，仅当设备支持时）
        // ============================================================
        // 需设备支持 VK_KHR_cooperative_matrix 且 CM 着色器已成功编译
        std::string cmSpvPath = std::string(SHADER_DIR) + "gemm_cm.spv";
        FILE* cmProbe = std::fopen(cmSpvPath.c_str(), "rb");
        bool cmSpvReady = (cmProbe != nullptr);
        if (cmProbe) std::fclose(cmProbe);
        bool hasCM = ctx.features.cooperativeMatrixKHR && cmSpvReady;

        double cmMs = 0.0;
        double cmGflops = 0.0;
        uint32_t errCM = 0;
        ComputePass passCM;

        if (hasCM) {
            std::cout << i18n::tr("\n[ch04] 路径2：Cooperative Matrix GEMM",
                                   "\n[ch04] Path 2: Cooperative Matrix GEMM") << "\n";

            // 特化常量：tile 尺寸（16x16x16 是 Tensor Core 的常见规格）
            std::array<uint32_t, 3> specData = {16, 16, 16};
            auto specInfo = vkMakeSpecializationInfo(specData);

            passCM.prepareWithSpec(ctx,
                std::string(SHADER_DIR) + "gemm_cm.spv",
                {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                 {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                 {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
                {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CM_PC)}},
                specInfo.info);
            passCM.bindBuffers({{0, bufA.buf}, {1, bufB.buf}, {2, bufC.buf}});

            // CM 路径使用 stride + K_total 参数
            CM_PC pcCM{K, N, N, K};

            // 预热
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { passCM.dispatch(cmd, &pcCM, sizeof(pcCM), gx, gy, 1); });

            // 计时（5 次取平均）
            cmMs = 0.0;
            for (int i = 0; i < warmup + samples; ++i) {
                double ms = measureMs(ctx, passCM, &pcCM, sizeof(pcCM), gx, gy);
                if (i >= warmup) cmMs += ms;
            }
            cmMs /= samples;

            // 验证
            auto outCM = vkDownload(ctx, bufC, M * N * 4);
            const float* cCM = reinterpret_cast<const float*>(outCM.data());
            for (uint32_t r = 0; r < M; ++r)
                for (uint32_t col = 0; col < N; ++col) {
                    float s = cpuRef[r * N + col];
                    float v = cCM[r * N + col];
                    if (std::fabs(v - s) > std::max(1e-2f, std::fabs(s) * 1e-3f)) ++errCM;
                }
            cmGflops = 2.0 * M * N * K / (cmMs * 1e-3) / 1e9;
            std::cout << i18n::tr("  结果: ", "  Result: ")
                      << (errCM == 0 ? i18n::tr("正确 ✓", "correct ✓") : i18n::tr("有错误", "has errors"))
                      << i18n::tr("，耗时=", ", time=") << cmMs << " ms"
                      << i18n::tr("，GFLOPS=", ", GFLOPS=") << cmGflops << "\n";
        } else {
            std::cout << i18n::tr("\n[ch04] 路径2：Cooperative Matrix GEMM — 不支持（跳过）",
                                   "\n[ch04] Path 2: Cooperative Matrix GEMM — not supported (skipped)") << "\n";
        }

        // ============================================================
        // 性能对比总结
        // ============================================================
        std::cout << "\n" << i18n::tr("====== 性能总结 ======", "====== Performance Summary ======") << "\n";
        std::cout << i18n::tr("  矩阵尺寸: ", "  Matrix size: ") << M << "x" << K << "x" << N << "\n";
        std::cout << i18n::tr("  分块 GEMM: ", "  Tiled GEMM: ") << tiledMs << " ms, " << tiledGflops << " GFLOPS\n";

        if (hasCM) {
            double speedup = tiledMs / cmMs;
            std::cout << i18n::tr("  CM GEMM:   ", "  CM GEMM:   ") << cmMs << " ms, " << cmGflops << " GFLOPS\n";
            std::cout << i18n::tr("  CM 加速比: ", "  CM speedup: ") << speedup << "x";
            if (errCM == 0) std::cout << i18n::tr("（正确）", " (correct)");
            std::cout << "\n";
        } else {
            std::cout << i18n::tr("  CM GEMM:   不支持（设备无 VK_KHR_cooperative_matrix）",
                                   "  CM GEMM:   not supported (device lacks VK_KHR_cooperative_matrix)") << "\n";
        }
        std::cout << i18n::tr("  分块验证: ", "  Tiled verify: ")
                  << (errTiled == 0 ? i18n::tr("正确 ✓", "correct ✓") : i18n::tr("有错误", "has errors")) << "\n";
        std::cout << i18n::tr("  Subgroup 大小: ", "  Subgroup size: ") << ctx.features.subgroupSize << "\n";

        // ---- 测试向量模式：写回输出（取最后一次 dispatch 写入 bufC 的结果）----
        if (tv.enabled()) {
            auto outFinal = vkDownload(ctx, bufC, M * N * 4);
            const float* cFinal = reinterpret_cast<const float*>(outFinal.data());
            tv.save("C", {M, N}, cFinal);
        }

        // ---- 清理 ----
        passTiled.destroy();
        if (hasCM) passCM.destroy();
        vkDestroyBuffer(ctx, bufA); vkDestroyBuffer(ctx, bufB); vkDestroyBuffer(ctx, bufC);
        vkDestroyContext(ctx);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
}