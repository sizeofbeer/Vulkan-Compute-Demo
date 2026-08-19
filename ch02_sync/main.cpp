// ch02_sync/main.cpp
// 同步模型：两个有依赖的算子放在「同一个命令缓冲」里，用 pipeline barrier 串起来。
// 这一步做： b = a + 1  →(barrier)→  c = b * 2
// 如果没有 barrier，第二个 dispatch 可能读到还没写好的 b，结果随机。
// 本章故意不用 common 的 vkSubmit，而是手写命令缓冲 / fence / barrier，把同步讲透。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>

static vk::Pipeline makePipeline(const VulkanContext& ctx, vk::ShaderModule mod,
                                 vk::PipelineLayout layout) {
    vk::PipelineShaderStageCreateInfo stage{
        vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, mod, "main"};
    vk::ComputePipelineCreateInfo ci{{}, stage, layout};
    return ctx.device.createComputePipeline({}, ci).value;
}

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();
        uint32_t N = 1u << 15;

        std::vector<float> a(N);
        if (tv.enabled()) {
            // 从测试向量读取真实形状，并加载输入
            std::vector<uint32_t> sa;
            tv.queryShape("a", sa);
            if (!sa.empty()) N = sa[0];
            a.resize(N);
            tv.load("a", {N}, a.data());
        } else {
            for (uint32_t i = 0; i < N; ++i) a[i] = (float)i;
        }

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        Buffer bufA = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufB = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufC = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufA, a.data(), N * 4);

        // ---- 两个管线共享同一套描述符布局 / push constant ----
        vk::DescriptorSetLayoutBinding binding{
            0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute};
        // 注意：两个 shader 合起来用到 binding 0/1/2，但布局要覆盖最大 binding。
        // 直接声明 3 个 binding（A,B,C）最简单。
        vk::DescriptorSetLayoutBinding bindings[3] = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        vk::DescriptorSetLayout dsl =
            ctx.device.createDescriptorSetLayout({vk::DescriptorSetLayoutCreateFlags{}, 3, bindings});
        vk::PushConstantRange pc{vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)};
        vk::PipelineLayout layout =
            ctx.device.createPipelineLayout({vk::PipelineLayoutCreateFlags{}, 1, &dsl, 1, &pc});

        vk::ShaderModule m1 = vkLoadShader(ctx, std::string(SHADER_DIR) + "vec_add_one.spv");
        vk::ShaderModule m2 = vkLoadShader(ctx, std::string(SHADER_DIR) + "vec_scale.spv");
        vk::Pipeline p1 = makePipeline(ctx, m1, layout);
        vk::Pipeline p2 = makePipeline(ctx, m2, layout);

        vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 3};
        vk::DescriptorPool pool = ctx.device.createDescriptorPool(
            {vk::DescriptorPoolCreateFlags{}, 1, 1, &ps});
        vk::DescriptorSet set = ctx.device.allocateDescriptorSets({pool, 1, &dsl})[0];
        vk::DescriptorBufferInfo infos[3] = {
            {bufA.buf, 0, VK_WHOLE_SIZE}, {bufB.buf, 0, VK_WHOLE_SIZE}, {bufC.buf, 0, VK_WHOLE_SIZE}};
        vk::WriteDescriptorSet writes[3] = {
            {set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[0]},
            {set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[1]},
            {set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[2]},
        };
        ctx.device.updateDescriptorSets(3, writes, 0, nullptr);

        // ---- 手写命令缓冲：两个 dispatch + 中间 barrier ----
        vk::CommandBuffer cmd =
            ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p1);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, 1, &set, 0, nullptr);
        cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), &N);
        cmd.dispatch((N + 63u) / 64u, 1, 1);

        // 关键：等 b 全部写完，第二个 kernel 才能读 b（写后读 hazard）
        // 用 buffer-specific barrier 精确同步 bufB 的「写后读」。
        vk::BufferMemoryBarrier bmb{
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
            0, 0, bufB.buf, 0, VK_WHOLE_SIZE};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            vk::DependencyFlags{}, 0, nullptr, 1, &bmb, 0, nullptr);

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p2);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, 1, &set, 0, nullptr);
        cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), &N);
        cmd.dispatch((N + 63u) / 64u, 1, 1);

        cmd.end();

        // ---- fence：等 GPU 真正算完 ----
        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(1, &si, ctx.fence);
        ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
        // 复位 fence：vkDownload 内部会再次 submit 使用同一个 fence，
        // 若仍处于 signaled 状态会导致 vkQueueSubmit 失败、读到未初始化数据。
        ctx.device.resetFences(1, &ctx.fence);

        auto out = vkDownload(ctx, bufC, N * 4);
        const float* c = reinterpret_cast<const float*>(out.data());
        uint32_t errs = 0;
        for (uint32_t i = 0; i < N; ++i)
            if (std::abs(c[i] - (a[i] + 1.0f) * 2.0f) > 1e-3f) ++errs;
        std::cout << i18n::tr("[ch02] b=a+1, c=b*2 完成 N=", "[ch02] b=a+1, c=b*2 done N=") << N
                  << i18n::tr(" 错误数=", " errors=") << errs
                  << i18n::tr("（应为0）", " (should be 0)") << "\n";
        std::cout << i18n::tr("[ch02] 提示：删掉 pipelineBarrier 再跑，部分 GPU 上会看到错误数>0",
                              "[ch02] tip: remove pipelineBarrier and rerun, on some GPUs you may see errors>0") << "\n";

        // ---- 测试向量模式：写回输出 ----
        if (tv.enabled()) tv.save("c", {N}, c);

        // cleanup
        ctx.device.destroyPipeline(p1); ctx.device.destroyPipeline(p2);
        ctx.device.destroyPipelineLayout(layout);
        ctx.device.destroyDescriptorSetLayout(dsl);
        ctx.device.destroyDescriptorPool(pool);
        ctx.device.destroyShaderModule(m1); ctx.device.destroyShaderModule(m2);
        ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
        vkDestroyBuffer(ctx, bufA); vkDestroyBuffer(ctx, bufB); vkDestroyBuffer(ctx, bufC);
        ctx.device.resetFences(1, &ctx.fence);
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
