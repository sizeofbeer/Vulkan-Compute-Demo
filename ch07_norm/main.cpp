// ch07_norm/main.cpp
// 归一化（RMSNorm）：一行特征需要先「跨元素求和」得到 RMS，再逐元素归一化。
// 用「工作组内共享内存归约」算出平方和，演示 GPU 上跨线程协作做 reduction。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

struct PC7 { uint32_t R, F; };

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();
        uint32_t R = 64, F = 256;     // 64 个样本，每个 256 维特征
        std::mt19937 rng(2024);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        std::vector<float> x(R * F), g(F, 1.0f);   // gamma 取 1，便于核对
        if (tv.enabled()) {
            std::vector<uint32_t> sx, sg;
            tv.queryShape("x", sx);
            tv.queryShape("weight", sg);
            if (sx.size() >= 2) { R = sx[0]; F = sx[1]; }
            if (sg.size() >= 1) F = std::max(F, sg[0]);
            x.resize(R * F); g.resize(F);
            tv.load("x", {R, F}, x.data());
            tv.load("weight", {F}, g.data());
        } else {
            for (auto& v : x) v = dist(rng);
        }

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        Buffer bX = vkCreateBuffer(ctx, R * F * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bG = vkCreateBuffer(ctx, F * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bY = vkCreateBuffer(ctx, R * F * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bX, x.data(), R * F * 4);
        vkUpload(ctx, bG, g.data(), F * 4);

        ComputePass pass;
        pass.prepare(ctx, std::string(SHADER_DIR) + "rmsnorm.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC7)}});
        pass.bindBuffers({{0, bX.buf}, {1, bY.buf}, {2, bG.buf}});
        PC7 pc{R, F};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pc, sizeof(pc), 1, R, 1); });

        auto out = vkDownload(ctx, bY, R * F * 4);
        const float* y = reinterpret_cast<const float*>(out.data());

        // 逐行 CPU 参考
        uint32_t errs = 0;
        for (uint32_t r = 0; r < R; ++r) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < F; ++i) ss += x[r * F + i] * x[r * F + i];
            float rms = std::sqrt(ss / (float)F);
            for (uint32_t i = 0; i < F; ++i) {
                float ref = (x[r * F + i] / rms) * g[i];
                if (std::fabs(y[r * F + i] - ref) > 1e-3f) ++errs;
            }
        }
        std::cout << i18n::tr("[ch07] RMSNorm R=", "[ch07] RMSNorm R=") << R << i18n::tr(" F=", " F=") << F
                  << i18n::tr(" 错误数=", " errors=") << errs << i18n::tr("（应为0）", " (should be 0)") << "\n";

        if (tv.enabled()) tv.save("out", {R, F}, y);

        pass.destroy();
        vkDestroyBuffer(ctx, bX); vkDestroyBuffer(ctx, bG); vkDestroyBuffer(ctx, bY);
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
