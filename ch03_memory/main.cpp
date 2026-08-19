// ch03_memory/main.cpp
// 显存管理入门：看懂 GPU 显存的三种形态、staging 中转上传、以及 FP16 存储 buffer（省一半带宽）。
// 本章用 common 的 vkUpload/vkDownload（它们已经把 staging 中转封装好了），并亲眼看一次 FP16 往返。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

// ---- 主机侧 float<->half（IEEE 754 half）----
// 仅用于演示与校验；工业级会用 SIMD/查表加速（见 ch11）。
static uint16_t float_to_half(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign     = (bits >> 16) & 0x8000u;
    int32_t  exponent = (int32_t)((bits >> 23) & 0xFFu);
    uint32_t mantissa =  bits & 0x7FFFFFu;
    uint16_t half;
    if (exponent == 0xFF) {                       // NaN / Inf
        half = (mantissa ? 0x7E00u : 0x7C00u) | sign;
    } else {
        int32_t e = exponent - 127 + 15;
        if (e >= 0x1F) {                          // 溢出 -> Inf
            half = 0x7C00u | sign;
        } else if (e <= 0) {                      // 非规 / 下溢
            if (e < -10) { half = (uint16_t)sign; }
            else {
                mantissa |= 0x800000u;
                uint32_t shift = 14u - (uint32_t)e;
                half = (uint16_t)(sign | (mantissa >> shift));
                if ((mantissa >> (shift - 1)) & 1u) half += 1u;   // 四舍五入
            }
        } else {
            half = (uint16_t)(sign | (uint32_t(e) << 10) | (mantissa >> 13));
            if ((mantissa >> 12) & 1u) half += 1u;                // 四舍五入
        }
    }
    return half;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign << 31; }
        else {
            uint32_t e = 0;
            while (!(mant & 0x400u)) { mant <<= 1; ++e; }
            mant &= 0x3FFu;
            bits = (sign << 31) | ((127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, 4);
    return f;
}

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);   // 消费 --test-vectors/--output（本章无匹配测试向量，仅保证参数不报错）
        (void)tv;
        VulkanContext ctx = vkInit();

        // ---------- 1. 内存类型速览 ----------
        auto mem = ctx.physical.getMemoryProperties();
        std::cout << i18n::tr("\n[ch03] 内存类型（共 ", "\n[ch03] memory types (total ") << mem.memoryTypeCount << i18n::tr(" 种）：\n", " types):\n");
        for (uint32_t t = 0; t < mem.memoryTypeCount; ++t) {
            auto f = mem.memoryTypes[t].propertyFlags;
            std::cout << i18n::tr("  类型#", "  type#") << t << i18n::tr(" 堆", " heap") << mem.memoryTypes[t].heapIndex << " [";
            if (f & vk::MemoryPropertyFlagBits::eDeviceLocal)  std::cout << "device-local ";
            if (f & vk::MemoryPropertyFlagBits::eHostVisible)  std::cout << "host-visible ";
            if (f & vk::MemoryPropertyFlagBits::eHostCoherent) std::cout << "coherent ";
            if (f & vk::MemoryPropertyFlagBits::eHostCached)   std::cout << "cached ";
            std::cout << "]\n";
        }
        std::cout << i18n::tr("  要点：device-local=显存(快，CPU 不可直接写)；host-visible=CPU 可映射(慢)。\n\n",
                             "  Note: device-local = VRAM (fast, CPU cannot write directly); host-visible = CPU-mappable (slow).\n\n");

        const uint32_t N = 1u << 16;

        // ---------- 2. staging 中转 vs 直接映射：上传耗时对比 ----------
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        std::vector<float> dummy(N);
        for (uint32_t i = 0; i < N; ++i) dummy[i] = (float)i;

        // 2a. 直接映射（host-visible + coherent）：CPU 写即可，无需 GPU 搬运
        auto t0 = std::chrono::high_resolution_clock::now();
        Buffer bufHost = vkCreateBuffer(ctx, N * 4, usage,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        vkUpload(ctx, bufHost, dummy.data(), N * 4);
        auto t1 = std::chrono::high_resolution_clock::now();

        // 2b. device-local：vkUpload 内部会先写 host staging，再 GPU copyBuffer 到显存
        Buffer bufDev = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufDev, dummy.data(), N * 4);
        auto t2 = std::chrono::high_resolution_clock::now();

        double msHost = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double msDev  = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << i18n::tr("[ch03] 上传 ", "[ch03] upload ") << N << i18n::tr(" 个 float：\n", " floats:\n");
        std::cout << i18n::tr("  host-visible 直接映射 : ", "  host-visible direct map: ") << msHost << i18n::tr(" ms（纯 CPU 拷贝）\n", " ms (pure CPU copy)\n");
        std::cout << "  device-local(staging): " << msDev  << i18n::tr(" ms（含 GPU 搬运 + 同步）\n", " ms (incl. GPU copy + sync)\n");
        std::cout << i18n::tr("  结论：device-local 计算最快，但上传要走 staging；常驻权重应一次性传好，别每帧重传。\n\n",
                             "  Conclusion: device-local compute is fastest, but upload goes through staging; resident weights should be uploaded once, not re-uploaded each frame.\n\n");

        vkDestroyBuffer(ctx, bufHost);
        vkDestroyBuffer(ctx, bufDev);

        // ---------- 3. FP16 存储 buffer 往返：省一半带宽 ----------
        // 输入用 half（2 字节）存，GPU 读出转 float32 计算后写回 float32 输出。
        std::vector<uint16_t> a16(N);
        for (uint32_t i = 0; i < N; ++i) a16[i] = float_to_half((float)i);

        Buffer bufA16 = vkCreateBuffer(ctx, N * 2, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufC32 = vkCreateBuffer(ctx, N * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vkUpload(ctx, bufA16, a16.data(), N * 2);

        vk::ShaderModule module = vkLoadShader(ctx, std::string(SHADER_DIR) + "fp16_roundtrip.spv");
        vk::DescriptorSetLayoutBinding binds[2] = {
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        };
        vk::DescriptorSetLayout dsl =
            ctx.device.createDescriptorSetLayout({vk::DescriptorSetLayoutCreateFlags{}, 2, binds});
        vk::PipelineLayout pl = ctx.device.createPipelineLayout(
            {vk::PipelineLayoutCreateFlags{}, 1, &dsl, 0, nullptr});
        vk::Pipeline pipeline = ctx.device.createComputePipeline({},
            vk::ComputePipelineCreateInfo{{},
                vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eCompute, module, "main"}, pl})
            .value;
        vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, 2};
        vk::DescriptorPool pool = ctx.device.createDescriptorPool(
            {vk::DescriptorPoolCreateFlags{}, 1, 1, &ps});
        vk::DescriptorSet set = ctx.device.allocateDescriptorSets({pool, 1, &dsl})[0];
        vk::DescriptorBufferInfo infos[2] = {{bufA16.buf, 0, VK_WHOLE_SIZE}, {bufC32.buf, 0, VK_WHOLE_SIZE}};
        vk::WriteDescriptorSet writes[2] = {
            {set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[0]},
            {set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[1]}};
        ctx.device.updateDescriptorSets(2, writes, 0, nullptr);

        uint32_t groups = (N + 63u) / 64u;
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pl, 0, 1, &set, 0, nullptr);
            cmd.pushConstants(pl, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t), &N);
            cmd.dispatch(groups, 1, 1);
        });

        auto out = vkDownload(ctx, bufC32, N * 4);
        const float* c = reinterpret_cast<const float*>(out.data());
        uint32_t errs = 0;
        double maxDiff = 0.0;
        for (uint32_t i = 0; i < N; ++i) {
            // CPU 参考也走「fp16 往返」路径，与 GPU 完全对等：
            //   a16[i] = half(i)  →  v = half_to_float(a16[i])  →  expected = v * 2.0f
            // 而不是用无损的 float i，否则两者起点不同（GPU 从量化后的 half 出发）。
            float v = half_to_float(a16[i]);
            float expected = v * 2.0f;
            float tol = std::max(1e-6f, std::fabs(expected) * 1e-5f);  // 仅余 fp16 乘法的舍入差异
            double d = std::fabs((double)c[i] - (double)expected);
            if (d > maxDiff) maxDiff = d;
            if (d > tol) ++errs;
        }
        std::cout << i18n::tr("[ch03] FP16 存储往返（CPU 亦按 fp16 参考）：c[i]=2*i，错误数=",
                              "[ch03] FP16 storage roundtrip (CPU ref also in fp16): c[i]=2*i, errors=") << errs
                  << i18n::tr("（应为0；CPU 参考同样先量化到 half 再计算）\n",
                              " (should be 0; CPU ref quantizes to half first, then computes)\n")
                  << i18n::tr("  max 偏差=", "  max diff=") << maxDiff << "\n";
        std::cout << i18n::tr("  输入 buffer 只占 ", "  input buffer occupies only ") << (N*2/1024) << i18n::tr(" KB，而同样 float32 要 ", " KB, while the same float32 needs ")
                  << (N*4/1024) << i18n::tr(" KB —— 内存与带宽都省一半。\n", " KB — memory and bandwidth both halved.\n");

        // 释放
        ctx.device.destroyPipeline(pipeline);
        ctx.device.destroyPipelineLayout(pl);
        ctx.device.destroyDescriptorSetLayout(dsl);
        ctx.device.destroyDescriptorPool(pool);
        ctx.device.destroyShaderModule(module);
        vkDestroyBuffer(ctx, bufA16); vkDestroyBuffer(ctx, bufC32);
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
