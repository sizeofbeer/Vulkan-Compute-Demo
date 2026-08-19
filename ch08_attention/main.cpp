// ch08_attention/main.cpp
// 简化单头注意力：O = softmax(QKᵀ/√D) · V。
// 拆成 3 个 kernel（分数 → softmax → 上下文），串起来跑通，并验证结果。
// 这清晰展示「注意力 = 两次矩阵乘夹一个 softmax」，正是 Transformer 的核心。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct PC8 { uint32_t N, D; };

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();
        uint32_t N = 64, D = 64;     // 序列长 64，头维度 64
        std::vector<float> Q(N * D), K(N * D), V(N * D);
        if (tv.enabled()) {
            // 从测试向量读取真实形状（Q/K/V 均为 {N, D}），并加载输入
            std::vector<uint32_t> sQ;
            tv.queryShape("Q", sQ);
            if (sQ.size() >= 2) { N = sQ[0]; D = sQ[1]; }
            Q.resize(N * D); K.resize(N * D); V.resize(N * D);
            tv.load("Q", {N, D}, Q.data());
            tv.load("K", {N, D}, K.data());
            tv.load("V", {N, D}, V.data());
        } else {
            std::mt19937 rng(314);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : Q) v = dist(rng);
            for (auto& v : K) v = dist(rng);
            for (auto& v : V) v = dist(rng);
        }

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        auto mk = [&](size_t n) { return vkCreateBuffer(ctx, n * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal); };
        Buffer bQ = mk(N * D), bK = mk(N * D), bV = mk(N * D);
        Buffer bS = mk(N * N), bP = mk(N * N), bO = mk(N * D);
        vkUpload(ctx, bQ, Q.data(), N * D * 4);
        vkUpload(ctx, bK, K.data(), N * D * 4);
        vkUpload(ctx, bV, V.data(), N * D * 4);

        auto run = [&](const std::string& spv, const std::vector<std::pair<uint32_t, vk::Buffer>>& binds,
                       const PC8& pc, uint32_t gx, uint32_t gy) {
            ComputePass pass;
            std::vector<vk::DescriptorSetLayoutBinding> binds3;
            for (uint32_t i = 0; i < (uint32_t)binds.size(); ++i)
                binds3.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
            pass.prepare(ctx, std::string(SHADER_DIR) + spv, binds3,
                {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC8)}});
            pass.bindBuffers(binds);
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1); });
            pass.destroy();
        };

        // 1) 分数 S = QKᵀ/√D
        run("attn_scores.spv", {{0, bQ.buf}, {1, bK.buf}, {2, bS.buf}}, PC8{N, D}, 8, 8);
        // 2) softmax 按行
        run("softmax.spv", {{0, bS.buf}, {1, bP.buf}}, PC8{N, 0}, 1, N);
        // 3) 上下文 O = P·V
        run("attn_context.spv", {{0, bP.buf}, {1, bV.buf}, {2, bO.buf}}, PC8{N, D}, 8, 8);

        auto out = vkDownload(ctx, bO, N * D * 4);
        const float* o = reinterpret_cast<const float*>(out.data());

        // CPU 参考
        std::vector<float> S(N * N), P(N * N);
        for (uint32_t qi = 0; qi < N; ++qi)
            for (uint32_t ki = 0; ki < N; ++ki) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < D; ++d) dot += Q[qi * D + d] * K[ki * D + d];
                S[qi * N + ki] = dot / std::sqrt((float)D);
            }
        uint32_t errs = 0;
        for (uint32_t qi = 0; qi < N; ++qi) {
            float m = -1e30f, sum = 0.0f;
            for (uint32_t ki = 0; ki < N; ++ki) m = std::max(m, S[qi * N + ki]);
            for (uint32_t ki = 0; ki < N; ++ki) { P[qi * N + ki] = std::exp(S[qi * N + ki] - m); sum += P[qi * N + ki]; }
            for (uint32_t ki = 0; ki < N; ++ki) P[qi * N + ki] /= sum;
            for (uint32_t d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (uint32_t ki = 0; ki < N; ++ki) acc += P[qi * N + ki] * V[ki * D + d];
                if (std::fabs(o[qi * D + d] - acc) > std::max(1e-2f, std::fabs(acc) * 1e-3f)) ++errs;
            }
        }
        std::cout << i18n::tr("[ch08] 简化注意力 O=softmax(QKᵀ/√D)·V 错误数=", "[ch08] simplified attention O=softmax(QKᵀ/√D)·V errors=") << errs
                  << i18n::tr("（应为0）", " (should be 0)") << "\n";

        // ============================================================
        // FlashAttention 融合 kernel（在线 softmax + 单 kernel fused）
        // ============================================================
        std::cout << "\n";

        // 创建 flash_attn 输出 buffer
        Buffer bO_FA = mk(N * D);

        // 创建 flash_attn 计算管线
        ComputePass passFA;
        {
            std::vector<vk::DescriptorSetLayoutBinding> faBindings;
            for (uint32_t i = 0; i < 4; ++i)
                faBindings.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
            passFA.prepare(ctx, std::string(SHADER_DIR) + "flash_attn.spv", faBindings,
                {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC8)}});
        }
        passFA.bindBuffers({{0, bQ.buf}, {1, bK.buf}, {2, bV.buf}, {3, bO_FA.buf}});

        // 正确性验证：FlashAttention 输出 vs 朴素 GPU 输出
        PC8 pcFA{N, D};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passFA.dispatch(cmd, &pcFA, sizeof(PC8), N, 1, 1);
        });

        auto outFA = vkDownload(ctx, bO_FA, N * D * 4);
        const float* oFA = reinterpret_cast<const float*>(outFA.data());

        uint32_t faErrs = 0;
        for (uint32_t i = 0; i < N * D; ++i) {
            if (std::fabs(oFA[i] - o[i]) > std::max(1e-2f, std::fabs(o[i]) * 1e-3f)) ++faErrs;
        }
        std::cout << i18n::tr("[ch08] FlashAttention vs naive GPU 错误数=", "[ch08] FlashAttention vs naive GPU errors=")
                  << faErrs << i18n::tr("（应为0）", " (should be 0)") << "\n";

        // ---- 性能对比：GPU 时间戳 ----
        // 为 3-kernel 创建持久管线用于计时
        ComputePass passScores, passSoftmax, passContext;
        {
            auto mkBind = [](uint32_t n) {
                std::vector<vk::DescriptorSetLayoutBinding> b;
                for (uint32_t i = 0; i < n; ++i)
                    b.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
                return b;
            };
            auto pcs = std::vector<vk::PushConstantRange>{
                vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC8)}};
            passScores.prepare(ctx, std::string(SHADER_DIR) + "attn_scores.spv", mkBind(3), pcs);
            passSoftmax.prepare(ctx, std::string(SHADER_DIR) + "softmax.spv", mkBind(2), pcs);
            passContext.prepare(ctx, std::string(SHADER_DIR) + "attn_context.spv", mkBind(3), pcs);
            passScores.bindBuffers({{0, bQ.buf}, {1, bK.buf}, {2, bS.buf}});
            passSoftmax.bindBuffers({{0, bS.buf}, {1, bP.buf}});
            passContext.bindBuffers({{0, bP.buf}, {1, bV.buf}, {2, bO.buf}});
        }

        // 计时辅助函数：单次 dispatch 计时
        auto timeDispatch = [&](ComputePass& pass, const PC8& pc, uint32_t gx, uint32_t gy) -> double {
            vk::QueryPool pool = vkCreateTimestampPool(ctx, 2);
            auto cmd = ctx.device.allocateCommandBuffers(
                {ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
            cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cmd.resetQueryPool(pool, 0, 2);
            cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 0);
            pass.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);
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
        };

        // 计时辅助函数：3-kernel 总计时
        auto time3Kernel = [&]() -> double {
            vk::QueryPool pool = vkCreateTimestampPool(ctx, 2);
            auto cmd = ctx.device.allocateCommandBuffers(
                {ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
            cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cmd.resetQueryPool(pool, 0, 2);
            cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 0);
            PC8 pcScores{N, D}, pcSoftmax{N, 0}, pcContext{N, D};
            passScores.dispatch(cmd, &pcScores, sizeof(PC8), 8, 8, 1);
            passSoftmax.dispatch(cmd, &pcSoftmax, sizeof(PC8), 1, N, 1);
            passContext.dispatch(cmd, &pcContext, sizeof(PC8), 8, 8, 1);
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
        };

        // 预热（GPU 管线冷启动不纳入统计）
        timeDispatch(passFA, PC8{N, D}, N, 1);
        time3Kernel();

        // 正式计时（跑 5 次取平均）
        const int repeats = 5;
        double faTime = 0.0, naiveTime = 0.0;
        for (int r = 0; r < repeats; ++r) {
            faTime += timeDispatch(passFA, PC8{N, D}, N, 1);
            naiveTime += time3Kernel();
        }
        faTime /= repeats;
        naiveTime /= repeats;

        std::cout << i18n::tr("\n性能对比（GPU 时间戳，", "\nPerformance (GPU timestamps, ")
                  << repeats << i18n::tr("次平均）：", " runs avg):\n");
        std::cout << i18n::tr("  FlashAttention : ", "  FlashAttention : ")
                  << faTime << " ms\n";
        std::cout << i18n::tr("  3-kernel 朴素  : ", "  3-kernel Naive : ")
                  << naiveTime << " ms\n";
        if (naiveTime > 0.0) {
            std::cout << i18n::tr("  加速比         : ", "  Speedup        : ")
                      << (naiveTime / faTime) << "x\n";
        }

        // 清理 FlashAttention 资源
        passFA.destroy();
        passScores.destroy();
        passSoftmax.destroy();
        passContext.destroy();
        vkDestroyBuffer(ctx, bO_FA);

        for (auto b : {bQ, bK, bV, bS, bP, bO}) vkDestroyBuffer(ctx, b);
        vkDestroyContext(ctx);

        // ---- 测试向量模式：写回输出（注意在上下文销毁后也能安全执行）----
        if (tv.enabled()) {
            tv.save("O", {N, D}, o);
            tv.save("O_FA", {N, D}, oFA);
        }
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
