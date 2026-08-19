// ch15_coopmat/main.cpp
// 协作矩阵（Cooperative Matrix）编程：利用 Tensor Core / Matrix Core 硬件加速 GEMM。
// 如果设备支持 VK_KHR_cooperative_matrix，使用 CM 路径；
// 否则回退到 ch04 的普通分块 GEMM。
// 最后对比两者性能，打印加速比。

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
#include <array>
#include <cstdio>

// 与着色器 push_constant 布局一致
struct PC_CM {
    uint32_t strideA;   // A 的行跨度（元素数）
    uint32_t strideB;   // B 的行跨度（元素数）
    uint32_t strideC;   // C 的行跨度（元素数）
    uint32_t K_total;   // K 总维度
};

// 与 ch04 gemm_tiled.comp push_constant 布局一致
struct PC_Tiled {
    uint32_t M, N, K;
};

// 打印 Cooperative Matrix 设备能力
static void printCoopMatCapabilities(const VulkanContext& ctx) {
    std::cout << "\n" << i18n::tr("===== Cooperative Matrix 设备能力 =====",
                                   "===== Cooperative Matrix Device Capabilities =====") << "\n";

    std::cout << i18n::tr("  扩展支持: ", "  Extension support: ");
    if (ctx.features.cooperativeMatrixKHR)  std::cout << "VK_KHR_cooperative_matrix ";
    if (ctx.features.cooperativeMatrixNV)   std::cout << "VK_NV_cooperative_matrix ";
    if (ctx.features.cooperativeMatrixMaintenance1)
        std::cout << "VK_EXT_cooperative_matrix_maintenance1 ";
    if (!ctx.features.cooperativeMatrixKHR && !ctx.features.cooperativeMatrixNV)
        std::cout << i18n::tr("无", "none");
    std::cout << "\n";

    // 查询 cooperative matrix 属性（仅当扩展支持时）
    if (ctx.features.cooperativeMatrixKHR) {
        // 注：本机 vulkan-hpp 为旧版 API，且链接的 Vulkan 库未导出
        // vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR，因此这里只打印
        // 通过 getProperties2 获取的设备级属性，不查询逐片段类型表。
        vk::PhysicalDeviceCooperativeMatrixPropertiesKHR cmProps;
        vk::PhysicalDeviceProperties2 props2;
        props2.pNext = &cmProps;
        ctx.physical.getProperties2(&props2);

        std::cout << i18n::tr("  CooperativeMatrixSupportedStages: ", "  CooperativeMatrixSupportedStages: ")
                  << vk::to_string(cmProps.cooperativeMatrixSupportedStages) << "\n";
    }
    std::cout << "\n";
}

// CPU 参考实现：C = A * B
static std::vector<float> cpuGemm(const float* a, const float* b,
                                   uint32_t M, uint32_t N, uint32_t K) {
    std::vector<float> c(M * N, 0.0f);
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t col = 0; col < N; ++col)
            for (uint32_t k = 0; k < K; ++k)
                c[r * N + col] += a[r * K + k] * b[k * N + col];
    return c;
}

// 验证正确性
static uint32_t verify(const float* gpu, const float* cpu,
                        uint32_t M, uint32_t N, uint32_t K) {
    uint32_t errs = 0;
    for (uint32_t r = 0; r < M; ++r)
        for (uint32_t col = 0; col < N; ++col) {
            float ref = cpu[r * N + col];
            float val = gpu[r * N + col];
            float absErr = std::fabs(val - ref);
            float relErr = std::fabs(ref) * 1e-3f;
            if (absErr > std::max(1e-2f, relErr)) ++errs;
        }
    return errs;
}

// 计时辅助：单次 dispatch + 同步
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
        VulkanContext ctx = vkInitWithExtensions({});

        // ------ 打印设备信息 ------
        std::cout << i18n::tr("[ch15] 设备: ", "[ch15] Device: ") << ctx.props.deviceName << "\n";
        printCoopMatCapabilities(ctx);

        // 设备需支持 CM 扩展，且 CM 着色器已成功编译（toolchain 不支持时跳过）
        std::string cmSpvPath = std::string(SHADER_DIR) + "gemm_cm.spv";
        FILE* cmProbe = std::fopen(cmSpvPath.c_str(), "rb");
        bool cmSpvReady = (cmProbe != nullptr);
        if (cmProbe) std::fclose(cmProbe);
        bool hasCM = ctx.features.cooperativeMatrixKHR && cmSpvReady;
        std::cout << i18n::tr("  Cooperative Matrix 支持: ", "  Cooperative Matrix support: ")
                  << (hasCM ? i18n::tr("是 ✓", "yes ✓") : i18n::tr("否 ✗（将使用回退路径）", "no ✗ (will use fallback path)"))
                  << "\n\n";

        // ------ 矩阵尺寸 ------
        uint32_t M = 256, N = 256, K = 256;

        // ------ 生成随机数据 ------
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> hA(M * K), hB(K * N);
        if (tv.enabled()) {
            // 从测试向量读取真实形状并加载输入（A:{M,K}, B:{K,N}）
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

        // CPU 参考
        std::cout << i18n::tr("[ch15] 计算 CPU 参考...", "[ch15] Computing CPU reference...") << "\n";
        auto cpuRef = cpuGemm(hA.data(), hB.data(), M, N, K);

        // ------ 创建 GPU 缓冲区 ------
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        Buffer bufA = vkCreateBuffer(ctx, M * K * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufB = vkCreateBuffer(ctx, K * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufC = vkCreateBuffer(ctx, M * N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufA, hA.data(), M * K * 4);
        vkUpload(ctx, bufB, hB.data(), K * N * 4);

        // ------ 回退路径：普通分块 GEMM（ch04 风格）------
        std::cout << i18n::tr("[ch15] 回退路径：普通分块 GEMM...",
                               "[ch15] Fallback path: tiled GEMM...") << "\n";
        ComputePass passTiled;
        passTiled.prepare(ctx,
            std::string(SHADER_DIR) + "gemm_tiled.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_Tiled)}});
        passTiled.bindBuffers({{0, bufA.buf}, {1, bufB.buf}, {2, bufC.buf}});

        // 分块 GEMM 的参数
        PC_Tiled pcTiled{M, N, K};
        uint32_t gx = (N + 15u) / 16u;
        uint32_t gy = (M + 15u) / 16u;

        // 预热（vkSubmit 会分配命令缓冲并等待完成）
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) { passTiled.dispatch(cmd, &pcTiled, sizeof(pcTiled), gx, gy, 1); });

        // 计时（取 5 次平均）
        double tiledMs = 0.0;
        const int warmup = 1, samples = 5;
        for (int i = 0; i < warmup + samples; ++i) {
            double ms = measureMs(ctx, passTiled, &pcTiled, sizeof(pcTiled), gx, gy);
            if (i >= warmup) tiledMs += ms;
        }
        tiledMs /= samples;

        // 验证回退路径结果
        auto outTiled = vkDownload(ctx, bufC, M * N * 4);
        const float* cTiled = reinterpret_cast<const float*>(outTiled.data());
        uint32_t errTiled = verify(cTiled, cpuRef.data(), M, N, K);
        std::cout << "  " << i18n::tr("回退路径错误数=", "Fallback errors=") << errTiled
                  << i18n::tr("（应为0）", " (should be 0)")
                  << i18n::tr("，平均耗时=", ", avg time=") << tiledMs << " ms\n";

        // 累加器 / 回退路径的 GFLOPS
        double tiledFlops = 2.0 * M * N * K / (tiledMs * 1e-3) / 1e9;
        std::cout << "  " << i18n::tr("回退路径 FP32 GFLOPS: ", "Fallback FP32 GFLOPS: ") << tiledFlops << "\n";

        // ------ CM 路径（如果支持）------
        double cmMs = 0.0;
        uint32_t errCM = 0;
        bool cmWorked = false;

        if (hasCM) {
            std::cout << i18n::tr("\n[ch15] CM 路径：Cooperative Matrix GEMM...",
                                   "\n[ch15] CM path: Cooperative Matrix GEMM...") << "\n";

            // 特化常量：tile 尺寸
            // 注：实际硬件可能有约束（如 NVIDIA Tensor Core 要求 16x16x16）
            std::array<uint32_t, 3> specData = {16, 16, 16};
            auto specInfo = vkMakeSpecializationInfo(specData);

            ComputePass passCM;
            passCM.prepareWithSpec(ctx,
                std::string(SHADER_DIR) + "gemm_cm.spv",
                {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                 {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                 {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
                {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_CM)}},
                specInfo.info);
            passCM.bindBuffers({{0, bufA.buf}, {1, bufB.buf}, {2, bufC.buf}});

            // CM 路径的参数：使用 stride 和 K_total
            PC_CM pcCM{K, N, N, K};

            // 预热
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { passCM.dispatch(cmd, &pcCM, sizeof(pcCM), gx, gy, 1); });

            // 计时
            cmMs = 0.0;
            for (int i = 0; i < warmup + samples; ++i) {
                double ms = measureMs(ctx, passCM, &pcCM, sizeof(pcCM), gx, gy);
                if (i >= warmup) cmMs += ms;
            }
            cmMs /= samples;

            // 验证 CM 路径结果
            auto outCM = vkDownload(ctx, bufC, M * N * 4);
            const float* cCM = reinterpret_cast<const float*>(outCM.data());
            errCM = verify(cCM, cpuRef.data(), M, N, K);
            std::cout << "  " << i18n::tr("CM 路径错误数=", "CM path errors=") << errCM
                      << i18n::tr("（应为0）", " (should be 0)")
                      << i18n::tr("，平均耗时=", ", avg time=") << cmMs << " ms\n";

            double cmFlops = 2.0 * M * N * K / (cmMs * 1e-3) / 1e9;
            std::cout << "  " << i18n::tr("CM 路径 FP32 GFLOPS: ", "CM path FP32 GFLOPS: ") << cmFlops << "\n";

            if (errCM == 0) {
                cmWorked = true;
                double speedup = tiledMs / cmMs;
                std::cout << "\n" << i18n::tr("===== 加速比 =====", "===== Speedup =====") << "\n";
                std::cout << i18n::tr("  CM / 回退路径: ", "  CM / fallback: ")
                          << speedup << "x\n";
                std::cout << i18n::tr("  回退: ", "  Fallback: ") << tiledMs
                          << i18n::tr(" ms, CM: ", " ms, CM: ") << cmMs
                          << i18n::tr(" ms\n", " ms\n");
            }

            passCM.destroy();
        }

        // ------ 总结 ------
        std::cout << "\n" << i18n::tr("====== 总结 ======", "====== Summary ======") << "\n";
        std::cout << i18n::tr("  矩阵尺寸: ", "  Matrix size: ") << M << "x" << K << "x" << N << "\n";

        // 打印 CM 能力摘要
        std::cout << i18n::tr("  Cooperative Matrix: ", "  Cooperative Matrix: ")
                  << (hasCM ? i18n::tr("支持", "supported") : i18n::tr("不支持", "not supported")) << "\n";
        if (ctx.features.cooperativeMatrixMaintenance1)
            std::cout << i18n::tr("  Maintenance1: ", "  Maintenance1: ") << "yes\n";
        std::cout << i18n::tr("  Subgroup 大小: ", "  Subgroup size: ") << ctx.features.subgroupSize << "\n";

        if (cmWorked) {
            std::cout << i18n::tr("  CM 路径: ", "  CM path: ")
                      << i18n::tr("正确 ✓", "correct ✓") << "\n";
        }
        std::cout << i18n::tr("  回退路径: ", "  Fallback path: ")
                  << (errTiled == 0 ? i18n::tr("正确 ✓", "correct ✓") : i18n::tr("有错误", "has errors"))
                  << "\n";

        // ------ 测试向量模式：写回输出 ------
        if (tv.enabled()) {
            auto outFinal = vkDownload(ctx, bufC, M * N * 4);
            tv.save("C", {M, N}, reinterpret_cast<const float*>(outFinal.data()));
        }

        // ------ 清理 ------
        passTiled.destroy();
        vkDestroyBuffer(ctx, bufA); vkDestroyBuffer(ctx, bufB); vkDestroyBuffer(ctx, bufC);
        vkDestroyContext(ctx);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
}