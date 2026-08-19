// ch09_profiling/main.cpp
// 性能分析入门：用 GPU 硬件时间戳（timestamp query）精确测量一次 GEMM 的耗时，
// 并与「CPU 墙钟（submit→等 fence）」对比——后者测到的多半是提交/同步开销，而非真实算力。
// 最后用 Roofline 模型估算该算子的「算术强度」，判断瓶颈在带宽还是算力。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>

struct PC9 { uint32_t M, N, K; };

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();
        uint32_t M = 512, N = 512, K = 512;
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> hA(M * K), hB(K * N);
        if (tv.enabled()) {
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

        ComputePass pass;
        pass.prepare(ctx, std::string(SHADER_DIR) + "gemm_tiled.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC9)}});
        pass.bindBuffers({{0, bufA.buf}, {1, bufB.buf}, {2, bufC.buf}});
        PC9 pc{M, N, K};
        uint32_t gx = (N + 15u) / 16u, gy = (M + 15u) / 16u;

        // ---- 时间戳查询池（2 个 slot：起点/终点）----
        vk::QueryPoolCreateInfo qpci{};
        qpci.queryType = vk::QueryType::eTimestamp;
        qpci.queryCount = 2;
        vk::QueryPool pool = ctx.device.createQueryPool(qpci);

        vk::CommandBuffer cmd =
            ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 0);  // 起点
        pass.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 1);  // 终点
        cmd.end();

        // ---- 两种计时对比 ----
        auto cpuT0 = std::chrono::high_resolution_clock::now();
        vk::SubmitInfo si; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        ctx.queue.submit(1, &si, ctx.fence);
        ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
        // 复位 fence：vkDownload 内部会复用同一个 fence，若未复位会因已 signaled 而提交失败
        ctx.device.resetFences(1, &ctx.fence);
        auto cpuT1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpuT1 - cpuT0).count();

        std::vector<uint64_t> ts(2);
        ctx.device.getQueryPoolResults(pool, 0, 2, ts.size() * sizeof(uint64_t), ts.data(),
            sizeof(uint64_t), vk::QueryResultFlagBits::eWait | vk::QueryResultFlagBits::e64);
        double period = ctx.props.limits.timestampPeriod;   // ns / tick
        double gpu_ms = (double)(ts[1] - ts[0]) * period / 1e6;

        std::cout << "[ch09] GEMM " << M << "x" << K << "x" << N << "\n";
        std::cout << i18n::tr("  GPU 时间戳实测 : ", "  GPU timestamp measured: ") << gpu_ms
                  << i18n::tr(" ms（这才是真实计算耗时）\n", " ms (this is the real computation time)\n");
        std::cout << i18n::tr("  CPU 墙钟(含提交): ", "  CPU wall clock (incl. submit): ") << cpu_ms
                  << i18n::tr(" ms（含队列/同步开销，通常偏大）\n", " ms (incl. queue/sync overhead, usually inflated)\n");

        // ---- Roofline：算术强度 = FLOPs / 访存量 ----
        double flops = 2.0 * M * N * K;
        double bytes = (double)(M * K + K * N + M * N) * 4.0;
        double ai = flops / bytes;   // FLOP/byte
        double tput_fp32 = flops / (gpu_ms * 1e-3) / 1e9;   // 实测 GFLOP/s
        std::cout << "  FLOPs = " << flops / 1e9
                  << i18n::tr(" G，访存量 = ", " G, memory traffic = ") << bytes / 1e6
                  << i18n::tr(" MB，算术强度 = ", " MB, arithmetic intensity = ") << ai << " FLOP/byte\n";
        std::cout << "  实测算力 = " << tput_fp32
                  << i18n::tr(" GFLOP/s（对比你 GPU 的 FP32 峰值，看打了几折）\n", " GFLOP/s (compare with your GPU's FP32 peak to see the discount)\n");
        std::cout << i18n::tr("  判读：算术强度高（>~10）→ 算力受限；低 → 带宽受限。GEMM 通常偏算力受限。\n",
                              "  Interpretation: high arithmetic intensity (>~10) -> compute-bound; low -> bandwidth-bound. GEMM is usually compute-bound.\n");

        // 正确性抽查
        auto out = vkDownload(ctx, bufC, M * N * 4);
        const float* c = reinterpret_cast<const float*>(out.data());
        uint32_t errs = 0;
        for (uint32_t s = 0; s < 100; ++s) {
            uint32_t r = rng() % M, col = rng() % N;
            float ref = 0.0f;
            for (uint32_t k = 0; k < K; ++k) ref += hA[r * K + k] * hB[k * N + col];
            if (std::fabs(c[r * N + col] - ref) > std::max(1e-2f, std::fabs(ref) * 1e-3f)) ++errs;
        }
        std::cout << "  正确性抽查错误数=" << errs << i18n::tr("（应为0）\n", "(should be 0)\n");

        if (tv.enabled()) tv.save("C", {M, N}, c);

        ctx.device.destroyQueryPool(pool);
        ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
        ctx.device.resetFences(1, &ctx.fence);
        pass.destroy();
        vkDestroyBuffer(ctx, bufA); vkDestroyBuffer(ctx, bufB); vkDestroyBuffer(ctx, bufC);
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
