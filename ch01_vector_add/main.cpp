// ch01_vector_add/main.cpp
// 最小计算闭环：在 GPU 上做 c = a + b（向量加法）。
// 这一章只关心「一条算子怎么跑通」：建 buffer -> 上传 -> 建管线 -> dispatch -> 读回 -> 校验。
// 所有样板（实例/设备/队列/内存/上传下载）都交给 common/vk_base。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);
        VulkanContext ctx = vkInit();

        uint32_t N = 1u << 16;   // 65536 个元素
        std::vector<float> a(N), b(N);
        if (tv.enabled()) {
            // 从测试向量读取真实形状，并加载输入
            std::vector<uint32_t> sa, sb;
            tv.queryShape("a", sa);
            tv.queryShape("b", sb);
            if (!sa.empty()) N = sa[0];
            if (!sb.empty() && sb[0] > N) N = sb[0];
            a.resize(N); b.resize(N);
            tv.load("a", {N}, a.data());
            tv.load("b", {N}, b.data());
        } else {
            for (uint32_t i = 0; i < N; ++i) { a[i] = (float)i; b[i] = (float)(N - i); }
        }

        // 三个 storage buffer（device-local）。注意加 TRANSFER 用途，方便上传/读回。
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        Buffer bufA = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufB = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufC = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);

        // 上传输入（device-local 时内部自动走 staging 中转，见 ch03）
        vkUpload(ctx, bufA, a.data(), N * 4);
        vkUpload(ctx, bufB, b.data(), N * 4);

        // ---- 计算管线 ----
        vk::ShaderModule module = vkLoadShader(ctx, std::string(SHADER_DIR) + "vec_add.spv");

        // 描述符布局：3 个 storage buffer（只读 a、只读 b、只写 c）
        vk::DescriptorSetLayoutBinding bindings[3] = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        vk::DescriptorSetLayout dsl =
            ctx.device.createDescriptorSetLayout({vk::DescriptorSetLayoutCreateFlags{}, 3, bindings});

        // push constant：把元素个数 n 直接压进指令流（比描述符更轻量）
        vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)};
        vk::PipelineLayout pipelineLayout =
            ctx.device.createPipelineLayout({vk::PipelineLayoutCreateFlags{}, 1, &dsl, 1, &pcRange});

        vk::PipelineShaderStageCreateInfo stageCi{
            vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, module, "main"};
        vk::ComputePipelineCreateInfo pipeCi{{}, stageCi, pipelineLayout};
        vk::Pipeline pipeline = ctx.device.createComputePipeline({}, pipeCi).value;

        // ---- 描述符集：把真实 buffer 绑到 binding 0/1/2 ----
        vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 3};
        vk::DescriptorPool pool = ctx.device.createDescriptorPool(
            {vk::DescriptorPoolCreateFlags{}, 1, 1, &ps});
        vk::DescriptorSet set = ctx.device.allocateDescriptorSets({pool, 1, &dsl})[0];
        vk::DescriptorBufferInfo infos[3] = {
            {bufA.buf, 0, VK_WHOLE_SIZE},
            {bufB.buf, 0, VK_WHOLE_SIZE},
            {bufC.buf, 0, VK_WHOLE_SIZE},
        };
        vk::WriteDescriptorSet writes[3] = {
            {set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[0]},
            {set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[1]},
            {set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[2]},
        };
        ctx.device.updateDescriptorSets(3, writes, 0, nullptr);

        // ---- 调度：工作组数 = ceil(N / 64) ----
        uint32_t groups = (N + 63u) / 64u;
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, 1, &set, 0, nullptr);
            cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), &N);
            cmd.dispatch(groups, 1, 1);
        });

        // ---- 读回并校验 ----
        auto out = vkDownload(ctx, bufC, N * 4);
        const float* c = reinterpret_cast<const float*>(out.data());
        uint32_t errs = 0;
        for (uint32_t i = 0; i < N; ++i)
            if (std::abs(c[i] - (a[i] + b[i])) > 1e-3f) ++errs;
        std::cout << i18n::tr("[ch01] 向量加法完成 N=", "[ch01] vector add done N=") << N
                  << i18n::tr(" 错误数=", " errors=") << errs
                  << i18n::tr("（应为0）", " (should be 0)") << "\n";

        // ---- 测试向量模式：写回输出 ----
        if (tv.enabled()) tv.save("c", {N}, c);

        // ---- 释放 ----
        ctx.device.destroyPipeline(pipeline);
        ctx.device.destroyPipelineLayout(pipelineLayout);
        ctx.device.destroyDescriptorSetLayout(dsl);
        ctx.device.destroyDescriptorPool(pool);
        ctx.device.destroyShaderModule(module);
        vkDestroyBuffer(ctx, bufA); vkDestroyBuffer(ctx, bufB); vkDestroyBuffer(ctx, bufC);
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
