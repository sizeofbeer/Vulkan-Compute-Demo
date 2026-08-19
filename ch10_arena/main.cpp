// ch10_arena/main.cpp
// 显存管理进阶：用一块大的 device-local buffer 当「显存池」，从中切分多个 tensor（bump 分配器）。
// 好处：① 避免每次 tiny 分配都调 Vulkan（分配本身有开销）；② 权重/激活常驻池中，推理时只 reset 不重分；
//       ③ buffer 复用，显存碎片可控。对应 ncnn 的 VkBlobAllocator 与 ggml 的 arena。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

struct PC10 { uint32_t n, p1, p2, p3; };   // fill: {n, base, seed, _}; add: {n, baseA, baseB, baseC}

static void runAndCheck(VulkanContext& ctx, DeviceArena& arena, Buffer& bOut, uint32_t N,
                        uint32_t seedA, uint32_t seedB) {
    // 从池中切三个子区域（返回字节偏移），各自按 float 对齐到元素数
    vk::DeviceSize ra = arena.alloc(N * 4);
    vk::DeviceSize rb = arena.alloc(N * 4);
    vk::DeviceSize rc = arena.alloc(N * 4);
    std::cout << i18n::tr("  [分配] A@", "  [alloc] A@") << ra << " B@" << rb << " C@" << rc
              << i18n::tr(" （下一次 reset 后这些偏移会被原样复用）\n", " (these offsets are reused as-is after the next reset)\n");

    auto fill = [&](uint32_t base, uint32_t seed) {
        ComputePass pass;
        pass.prepare(ctx, std::string(SHADER_DIR) + "fill.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC10)}});
        pass.bindBuffers({{0, arena.buf.buf}});
        PC10 pc{N, base, seed, 0};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pc, sizeof(pc), (N+255u)/256u, 1, 1); });
        pass.destroy();
    };
    fill((uint32_t)ra, seedA);
    fill((uint32_t)rb, seedB);

    ComputePass pass;
    pass.prepare(ctx, std::string(SHADER_DIR) + "add_regions.spv",
        {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
         {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
        {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC10)}});
    pass.bindBuffers({{0, arena.buf.buf}, {1, bOut.buf}});
    PC10 pc{N, (uint32_t)ra, (uint32_t)rb, (uint32_t)rc};
    vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pc, sizeof(pc), (N+255u)/256u, 1, 1); });
    pass.destroy();

    auto out = vkDownload(ctx, bOut, N * 4);
    const float* o = reinterpret_cast<const float*>(out.data());
    uint32_t errs = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float ref = float(i * 2u + seedA) + float(i * 2u + seedB);   // = 4i + seedA + seedB
        if (std::fabs(o[i] - ref) > 1e-3f) ++errs;
    }
    std::cout << i18n::tr("  [校验] 错误数=", "  [check] errors=") << errs << i18n::tr("（应为0）\n", "(should be 0)\n");
}

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);   // 消费 --test-vectors/--output（本章无匹配测试向量，仅保证参数不报错）
        (void)tv;
        VulkanContext ctx = vkInit();
        const uint32_t N = 1024;

        DeviceArena arena;
        arena.init(ctx, 4 * 1024 * 1024);   // 4MB 大池
        Buffer bOut = vkCreateBuffer(ctx, N * 4,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        std::cout << i18n::tr("[ch10] 第 1 轮：从池分配并使用\n", "[ch10] Round 1: allocate from pool and use\n");
        runAndCheck(ctx, arena, bOut, N, 1, 2);

        std::cout << i18n::tr("[ch10] reset 池，第 2 轮：偏移原样复用，零新增分配\n", "[ch10] reset pool, Round 2: offsets reused as-is, zero new allocations\n");
        arena.reset();                       // bump 归零，所有上一轮的子区域作废
        runAndCheck(ctx, arena, bOut, N, 5, 7);

        vkDestroyBuffer(ctx, bOut);
        arena.destroy();
        vkDestroyContext(ctx);
        std::cout << i18n::tr("[ch10] 启示：真实推理引擎用同一个池放「一帧的所有激活值」，每帧 reset 复用，\n",
                                "[ch10] Insight: real inference engines use one pool for all activations of a frame, reusing them via reset each frame,\n")
                     << i18n::tr("         权重则放进另一个常驻池（ch03 思路），避免重复上传。\n",
                                "        while weights go into a separate resident pool (ch03 idea) to avoid re-uploading.\n");
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
