// ch12_graph/main.cpp
// 计算图与调度：把「matmul -> 加偏置 -> ReLU -> matmul」串成一张图，录进同一条命令缓冲，
// 依赖处插 pipeline barrier（写后读），最后只 submit 一次。对比「每算子一次 submit」省掉队列开销。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct PC12 { uint32_t a, b, c, d; };

static ComputePass makePass(VulkanContext& ctx, const std::string& spv, uint32_t nBind) {
    ComputePass p;
    std::vector<vk::DescriptorSetLayoutBinding> binds;
    for (uint32_t i = 0; i < nBind; ++i)
        binds.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
    p.prepare(ctx, std::string(SHADER_DIR) + spv, binds,
        {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC12)}});
    return p;
}

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();
        uint32_t M = 64, N = 64, K = 64;   // 所有中间张量 64x64
        std::mt19937 rng(21);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> A(M * K), B(K * N), C(N * K), bias(N);
        if (tv.enabled()) {
            std::vector<uint32_t> sA, sB, sC;
            tv.queryShape("A", sA);
            tv.queryShape("B", sB);
            tv.queryShape("C", sC);
            if (sA.size() >= 2) { M = sA[0]; K = sA[1]; }
            if (sB.size() >= 2) N = sB[1];
            A.resize(M * K); B.resize(K * N); C.resize(N * K); bias.resize(N);
            tv.load("A", {M, K}, A.data());
            tv.load("B", {K, N}, B.data());
            tv.load("C", {N, K}, C.data());
            tv.load("bias", {N}, bias.data());
        } else {
            for (auto& v : A) v = dist(rng);
            for (auto& v : B) v = dist(rng);
            for (auto& v : C) v = dist(rng);
            for (auto& v : bias) v = dist(rng);
        }

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        auto mk = [&](size_t n) { return vkCreateBuffer(ctx, n * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal); };
        Buffer bA = mk(M * K), bB = mk(K * N), bC = mk(N * K), bBias = mk(N);
        Buffer bT1 = mk(M * N), bT2 = mk(M * N), bT3 = mk(M * N), bOUT = mk(M * N);
        vkUpload(ctx, bA, A.data(), M * K * 4);
        vkUpload(ctx, bB, B.data(), K * N * 4);
        vkUpload(ctx, bC, C.data(), N * K * 4);
        vkUpload(ctx, bBias, bias.data(), N * 4);

        // 4 个算子（节点）
        ComputePass pMM1  = makePass(ctx, "gemm_tiled.spv", 3);
        ComputePass pBias = makePass(ctx, "bias_add.spv", 3);
        ComputePass pRelu = makePass(ctx, "elemwise.spv", 2);
        ComputePass pMM2  = makePass(ctx, "gemm_tiled.spv", 3);
        pMM1.bindBuffers({{0, bA.buf}, {1, bB.buf}, {2, bT1.buf}});
        pBias.bindBuffers({{0, bT1.buf}, {1, bBias.buf}, {2, bT2.buf}});
        pRelu.bindBuffers({{0, bT2.buf}, {1, bT3.buf}});
        pMM2.bindBuffers({{0, bT3.buf}, {1, bC.buf}, {2, bOUT.buf}});

        PC12 pcMM{M, N, K, 0};
        PC12 pcBias{M * N, N, 0, 0};          // n=M*N 元素，n_bias=N 偏置按列广播
        PC12 pcRelu{M * N, 0, 0, 0};

        // ---- 录成一张图，一次提交 ----
        Graph g; g.ctx = &ctx; g.begin();
        g.node(pMM1,  &pcMM,   sizeof(pcMM),   (N+15)/16, (M+15)/16, 1, nullptr);   // 写 bT1
        g.node(pBias, &pcBias, sizeof(pcBias), (M*N+255)/256, 1, 1, bT1.buf);      // 读 bT1 写 bT2
        g.node(pRelu, &pcRelu, sizeof(pcRelu), (M*N+255)/256, 1, 1, bT2.buf);      // 读 bT2 写 bT3
        g.node(pMM2,  &pcMM,   sizeof(pcMM),   (N+15)/16, (M+15)/16, 1, bT3.buf);   // 读 bT3 写 bOUT
        g.end();

        // ---- 校验 ----
        auto out = vkDownload(ctx, bOUT, M * N * 4);
        const float* o = reinterpret_cast<const float*>(out.data());
        std::vector<float> T1(M * N), T2(M * N), T3(M * N);
        for (uint32_t r = 0; r < M; ++r)
            for (uint32_t c = 0; c < N; ++c) {
                float s = 0; for (uint32_t k = 0; k < K; ++k) s += A[r*K+k]*B[k*N+c]; T1[r*N+c]=s;
                T2[r*N+c] = T1[r*N+c] + bias[c];
                T3[r*N+c] = T2[r*N+c] > 0 ? T2[r*N+c] : 0;
            }
        uint32_t errs = 0;
        for (uint32_t r = 0; r < M; ++r)
            for (uint32_t c = 0; c < N; ++c) {
                float s = 0; for (uint32_t k = 0; k < K; ++k) s += T3[r*K+k]*C[k*N+c];
                if (std::fabs(o[r*N+c] - s) > std::max(1e-2f, std::fabs(s)*1e-3f)) ++errs;
            }
        std::cout << i18n::tr("[ch12] 计算图( matmul->bias->relu->matmul )错误数=", "[ch12] compute graph ( matmul->bias->relu->matmul ) errors=") << errs << i18n::tr("（应为0）\n", " (should be 0)\n");
        std::cout << i18n::tr("[ch12] 4 个算子只 submit 1 次；若拆成 4 次 vkSubmit，队列开销会显著放大（ch09）。\n", "[ch12] 4 operators submit only once; if split into 4 vkSubmit, queue overhead grows significantly (ch09).\n");

        if (tv.enabled()) tv.save("out", {M, N}, o);

        for (auto p : {pMM1, pBias, pRelu, pMM2}) p.destroy();
        for (auto b : {bA, bB, bC, bBias, bT1, bT2, bT3, bOUT}) vkDestroyBuffer(ctx, b);
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
