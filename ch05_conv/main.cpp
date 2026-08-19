// ch05_conv/main.cpp
// 卷积的两条路：① 直接卷积（直白但访存差）；② im2col + GEMM（工业界主流）。
// 我们用同一组随机权重/输入，分别跑两种，并验证结果一致——证明「卷积本质上就是矩阵乘」。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

// 通用 push constant 包（按各 shader 需要取字段）
struct PC5 { uint32_t a, b, c, d, e, f; };

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();

        // 网络形状（教学小尺寸）：单输入通道、2 输出通道、16x16 特征图、3x3 卷积、same padding
        uint32_t Cin = 1, Cout = 2, H = 16, W = 16, K = 3;
        uint32_t OH = H, OW = W;                 // stride=1,pad=1 -> 同尺寸
        uint32_t inpN = Cin * H * W;
        uint32_t wN   = Cout * Cin * K * K;
        uint32_t outN = Cout * OH * OW;

        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> inp(inpN), w(wN);
        if (tv.enabled()) {
            // 从测试向量读取形状：x:{1,Cin,H,W}，w:{Cout,Cin,K,K}
            std::vector<uint32_t> sx, sw;
            tv.queryShape("x", sx);
            tv.queryShape("w", sw);
            if (sx.size() >= 4) { Cin = sx[1]; H = sx[2]; W = sx[3]; }
            if (sw.size() >= 4) { Cout = sw[0]; K = sw[3]; }
            OH = H; OW = W;
            inpN = Cin * H * W; wN = Cout * Cin * K * K; outN = Cout * OH * OW;
            inp.resize(inpN); w.resize(wN);
            tv.load("x", {1, Cin, H, W}, inp.data());
            tv.load("w", {Cout, Cin, K, K}, w.data());
        } else {
            for (auto& v : inp) v = dist(rng);
            for (auto& v : w)   v = dist(rng);
        }

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;

        // ---------- 路径①：直接卷积 ----------
        Buffer bInp  = vkCreateBuffer(ctx, inpN * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bW    = vkCreateBuffer(ctx, wN * 4,   usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bOut1 = vkCreateBuffer(ctx, outN * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bInp, inp.data(), inpN * 4);
        vkUpload(ctx, bW,   w.data(),   wN * 4);

        auto runConv = [&](const std::string& spv, Buffer in, Buffer wt, Buffer out,
                           const PC5& pc, uint32_t gx, uint32_t gy) {
            ComputePass pass;
            pass.prepare(ctx, std::string(SHADER_DIR) + spv,
                {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                 {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
                 {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
                {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC5)}});
            pass.bindBuffers({{0, in.buf}, {1, wt.buf}, {2, out.buf}});
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1); });
            pass.destroy();
        };
        PC5 pcDirect{H, W, K, Cout, Cin, 0};
        runConv("direct_conv.spv", bInp, bW, bOut1, pcDirect,
                (W + 15u) / 16u, H * Cout);
        auto out1 = vkDownload(ctx, bOut1, outN * 4);

        // ---------- 测试向量模式：仅运行直接卷积并写回输出 ----------
        if (tv.enabled()) {
            tv.save("out", {Cout, OH, OW}, reinterpret_cast<const float*>(out1.data()));
            std::cout << i18n::tr("[ch05] 测试向量模式：直接卷积结果已写回 out\n",
                                  "[ch05] test-vector mode: direct conv result written to out\n");
            for (auto b : {bInp, bW, bOut1}) vkDestroyBuffer(ctx, b);
            vkDestroyContext(ctx);
            return 0;
        }

        // ---------- 路径②：im2col + GEMM ----------
        const uint32_t totalP = OH * OW;     // 256
        const uint32_t totalQ = Cin * K * K; // 9
        Buffer bCol = vkCreateBuffer(ctx, totalP * totalQ * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        PC5 pcIm2col{H, W, K, Cin, OW, OH};
        runConv("im2col.spv", bInp, bInp, bCol, pcIm2col,
                (totalP + 15u) / 16u, totalQ);   // im2col 只用 binding0/2；binding1 用 bInp 占位（shader 不引用）

        // 把权重 reshape 成 B[k*N + oc] = w[oc*totalQ + k]（GEMM 需要的列主序）
        std::vector<float> B(totalQ * Cout);
        for (uint32_t oc = 0; oc < Cout; ++oc)
            for (uint32_t k = 0; k < totalQ; ++k)
                B[k * Cout + oc] = w[oc * totalQ + k];
        Buffer bB = vkCreateBuffer(ctx, B.size() * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bOut2 = vkCreateBuffer(ctx, outN * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bB, B.data(), B.size() * 4);

        PC5 pcGemm{totalP, Cout, totalQ, 0, 0, 0};   // M=totalP, N=Cout, K=totalQ
        runConv("gemm_naive.spv", bCol, bB, bOut2, pcGemm,
                (Cout + 15u) / 16u, (totalP + 15u) / 16u);
        auto out2 = vkDownload(ctx, bOut2, outN * 4);

        // ---------- 路径③：Winograd F(2×2, 3×3) ----------
        // Winograd 将 3×3 卷积分解为：输入变换 → 逐元素乘加 → 输出变换，
        // 用更少的乘法换取额外加法，适合 3×3 卷积这种小核场景。
        {
            const uint32_t num_tiles_x = (W + 1) / 2;   // 每行 2×2 输出 tile 数
            const uint32_t num_tiles_y = (H + 1) / 2;
            const uint32_t total_tiles = num_tiles_x * num_tiles_y;

            // Winograd 临时 buffer
            // bU: Cout × Cin × 16 —— 变换后的滤波器
            // bV: Cin × total_tiles × 16 —— 变换后的输入 tile
            // bM: total_tiles × Cout × 16 —— Winograd 域中间结果
            // bOut3: Cout × OH × OW —— 最终输出
            Buffer bU    = vkCreateBuffer(ctx, Cout * Cin * 16 * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            Buffer bV    = vkCreateBuffer(ctx, Cin * total_tiles * 16 * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            Buffer bM    = vkCreateBuffer(ctx, total_tiles * Cout * 16 * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            Buffer bOut3 = vkCreateBuffer(ctx, outN * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);

            // ① 滤波器变换：U = G @ g @ G^T（3×3 → 4×4 Winograd 域）
            PC5 pcFilter{Cout, Cin, 0, 0, 0, 0};
            runConv("winograd_filter_transform.spv", bW, bW, bU, pcFilter, Cin, Cout);

            // ② 输入变换：V = BT @ d @ B（4×4 输入 tile → 4×4 Winograd 域）
            PC5 pcInput{H, W, num_tiles_x, Cin, 0, 0};
            runConv("winograd_transform_input.spv", bInp, bInp, bV, pcInput, total_tiles, Cin);

            // ③ Winograd 域逐元素乘加：M[oc] = sum_cin V[cin] ⊙ U[oc][cin]
            PC5 pcGemmWino{total_tiles, Cout, Cin, 0, 0, 0};
            runConv("winograd_gemm.spv", bV, bU, bM, pcGemmWino, total_tiles, Cout);

            // ④ 输出变换：Y = AT @ M @ A（4×4 Winograd 域 → 2×2 输出）
            PC5 pcOutput{total_tiles, Cout, num_tiles_x, OH, OW, 0};
            runConv("winograd_transform_output.spv", bM, bM, bOut3, pcOutput, total_tiles, Cout);

            auto out3 = vkDownload(ctx, bOut3, outN * 4);

            // 验证 Winograd 结果与直接卷积一致
            const float* c1_ref = reinterpret_cast<const float*>(out1.data());
            const float* c3 = reinterpret_cast<const float*>(out3.data());
            uint32_t errs_wino = 0;
            for (uint32_t oc = 0; oc < Cout; ++oc)
                for (uint32_t oy = 0; oy < OH; ++oy)
                    for (uint32_t ox = 0; ox < OW; ++ox) {
                        uint32_t i = oc * OH * OW + oy * OW + ox;
                        if (std::fabs(c1_ref[i] - c3[i]) > std::max(1e-3f, std::fabs(c1_ref[i]) * 1e-4f)) ++errs_wino;
                    }
            std::cout << i18n::tr("[ch05] 直接卷积 vs Winograd 结果一致？错误数=", "[ch05] direct conv vs Winograd match? errors=") << errs_wino
                      << i18n::tr("（应为0）", " (should be 0)") << "\n";

            // 释放 Winograd 临时 buffer
            for (auto b : {bU, bV, bM, bOut3}) vkDestroyBuffer(ctx, b);
        }

        // ---------- 验证两路结果一致 ----------
        // 注意：direct_conv 输出是 oc-major 布局 [oc][oy][ox]；gemm 输出是 position-major 布局
        //       [oy*OW+ox][oc]（C=M×N, M=位置数, N=Cout）。两者内存布局不同，不能直接线性对比，
        //       必须按 (oc,oy,ox) 语义索引对齐后再逐元素比较，否则会系统性错位（全部错误）。
        const float* c1 = reinterpret_cast<const float*>(out1.data());
        const float* c2 = reinterpret_cast<const float*>(out2.data());
        uint32_t errs = 0;
        for (uint32_t oc = 0; oc < Cout; ++oc)
            for (uint32_t oy = 0; oy < OH; ++oy)
                for (uint32_t ox = 0; ox < OW; ++ox) {
                    uint32_t i1 = oc * OH * OW + oy * OW + ox;    // direct_conv 布局
                    uint32_t i2 = (oy * OW + ox) * Cout + oc;     // gemm 布局
                    if (std::fabs(c1[i1] - c2[i2]) > std::max(1e-3f, std::fabs(c1[i1]) * 1e-4f)) ++errs;
                }
        std::cout << i18n::tr("[ch05] 直接卷积 vs im2col+GEMM 结果一致？错误数=", "[ch05] direct conv vs im2col+GEMM match? errors=") << errs
                  << i18n::tr("（应为0）", " (should be 0)") << "\n";

        // 释放
        for (auto b : {bInp, bW, bOut1, bCol, bB, bOut2}) vkDestroyBuffer(ctx, b);
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
