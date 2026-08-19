// ch06_elemwise/main.cpp
// 逐元素算子（ReLU/GELU/SiLU）：每个元素独立运算，计算密度极低——典型「带宽受限」算子。
// 本章演示两件事：① 一个参数化 kernel 搞定三种激活；② 这类算子应「融合」进相邻算子省带宽。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <cctype>

struct PC6 { uint32_t n, mode; };

static float ref_act(float v, uint32_t mode) {
    if (mode == 0) return v > 0.0f ? v : 0.0f;
    if (mode == 1) { float c = 0.79788456f * (v + 0.044715f * v * v * v); return v * 0.5f * (1.0f + std::tanh(c)); }
    return v / (1.0f + std::exp(-v));
}

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();
        uint32_t N = 1u << 18;     // 26 万多个元素
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        std::vector<float> x(N);
        if (tv.enabled()) {
            std::vector<uint32_t> sx;
            tv.queryShape("x", sx);
            if (!sx.empty()) N = sx[0];
            x.resize(N);
            tv.load("x", {N}, x.data());
        } else {
            for (auto& v : x) v = dist(rng);
        }

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        Buffer bX = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bX, x.data(), N * 4);

        ComputePass pass;
        pass.prepare(ctx, std::string(SHADER_DIR) + "elemwise.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC6)}});

        const char* names[] = {"ReLU", "GELU", "SiLU"};
        for (uint32_t mode = 0; mode < 3; ++mode) {
            Buffer bY = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
            pass.bindBuffers({{0, bX.buf}, {1, bY.buf}});
            PC6 pc{N, mode};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                pass.dispatch(cmd, &pc, sizeof(pc), (N + 255u) / 256u, 1, 1);
            });
            auto out = vkDownload(ctx, bY, N * 4);
            const float* y = reinterpret_cast<const float*>(out.data());
            uint32_t errs = 0;
            for (uint32_t i = 0; i < N; ++i)
                if (std::fabs(y[i] - ref_act(x[i], mode)) > 1e-3f) ++errs;
            std::cout << i18n::tr("[ch06] ", "[ch06] ") << names[mode]
                      << i18n::tr(" 错误数=", " errors=") << errs
                      << i18n::tr("（应为0）", " (should be 0)") << "\n";
            if (tv.enabled()) {
                // 按激活类型保存输出，Python 侧按测试向量名取用
                std::string key = names[mode];
                for (auto& ch : key) ch = (char)std::tolower(ch);  // relu/gelu/silu
                tv.save("out_" + key, {N}, y);
            }
            vkDestroyBuffer(ctx, bY);
        }

        pass.destroy();
        vkDestroyBuffer(ctx, bX);
        vkDestroyContext(ctx);
        std::cout << i18n::tr("[ch06] 提示：这类算子瓶颈在带宽而非算力。把激活『融合』进前一个 matmul（同一次 dispatch 内算），\n",
                              "[ch06] Hint: this operator's bottleneck is bandwidth, not compute. Fuse the activation into the preceding matmul (computed within the same dispatch),\n")
                     << i18n::tr("         能省掉一次完整的『写回显存 + 再读出来』，延迟可降一大截——见 docs 的『算子融合』。\n",
                                 "         saving a full 'write-back to memory + read-out' round trip, and cutting latency substantially — see 'Operator Fusion' in docs.\n");
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
