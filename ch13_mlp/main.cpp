// ch13_mlp/main.cpp —— 路径 A：加载真实 ncnn 模型并 GPU 推理 + CPU 对拍
//
// 这是课程「综合项目」的实战版：把 ch13 之前硬编码的随机权重 MLP，
// 换成加载你导出的真实 Qwen3-ASR ncnn 子图（默认 lm_head.ncnn）。
//
// 链路：解析标准 ncnn param -> 读 fp32 权重流 -> 上传权重常驻显存 ->
//       一次 gemm_tiled dispatch（hidden=1024 -> vocab=151936）-> 读回 ->
//       与 CPU 参考前向逐元素对拍。
//
// 用法：
//   ch13_mlp                                  # 默认加载 F:/vulkan/model/qwen3_asr_0.6b/lm_head.ncnn.*
//   ch13_mlp <xxx.ncnn.param> <xxx.ncnn.bin>  # 指定其它纯 InnerProduct 子图
//
// 说明：lm_head 是「单 InnerProduct、无偏置」的全连接层（vocab 投影头），
//       是 ch13 当前引擎（只认 InnerProduct + 无 bias）能直接跑的通子图。
//       其它含 Gemm/RMSNorm/Swish 的子图（decoder_*/audio_*）需路径 B 扩算子。

#include "vk_base.h"
#include "model_loader.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdlib>

struct PC13 { uint32_t p0, p1, p2, p3; };

static ComputePass makePass(VulkanContext& ctx, const std::string& spv, uint32_t nBind) {
    ComputePass p;
    std::vector<vk::DescriptorSetLayoutBinding> binds;
    for (uint32_t i = 0; i < nBind; ++i)
        binds.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
    p.prepare(ctx, std::string(SHADER_DIR) + spv, binds,
        {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC13)}});
    return p;
}

int main(int argc, char** argv) {
    i18n::init(argc, argv);
    TestVectorConfig tv;
    tv.parse(argc, argv);   // 消费 --test-vectors/--output（本章依赖外部 ncnn 模型，仅保证参数不报错）
    (void)tv;
    try {
        // 默认加载 Qwen3-ASR lm_head 子图；也可命令行指定其它纯 InnerProduct 子图。
        // 跨平台：路径优先取命令行参数，其次环境变量 VULKAN_NN_MODEL_DIR（模型目录），
        // 最后才回退到 Windows 专属默认路径（Linux 上请通过环境变量或命令行指定）。
        std::string modelDir;
        if (const char* e = std::getenv("VULKAN_NN_MODEL_DIR")) modelDir = e;
        std::string paramPath, binPath;
        if (argc > 2) {
            paramPath = argv[1]; binPath = argv[2];
        } else if (!modelDir.empty()) {
            paramPath = modelDir + "/qwen3_asr_0.6b/lm_head.ncnn.param";
            binPath   = modelDir + "/qwen3_asr_0.6b/lm_head.ncnn.bin";
        } else {
            paramPath = "F:/vulkan/model/qwen3_asr_0.6b/lm_head.ncnn.param";
            binPath   = "F:/vulkan/model/qwen3_asr_0.6b/lm_head.ncnn.bin";
        }

        VulkanContext ctx = vkInit();

        // 1) 解析标准 ncnn param + 读权重（fp32 流）
        NcnnModel model = NcnnModel::load(paramPath);
        auto W = readNcnnBin(binPath, model);

        // 2) 找到唯一的 InnerProduct（lm_head 只有一个）
        NcnnLayer* ip = nullptr;
        for (auto& L : model.layers) if (L.type == "InnerProduct") { ip = &L; break; }
        if (!ip) throw std::runtime_error("param 中没有 InnerProduct 层，路径 A 暂只支持纯全连接子图");
        long long numOut = ip->iParams[0];              // vocab = 151936
        long long wElems = ip->iParams[2];              // weight 元素数 = 155582464
        long long numIn  = wElems / numOut;             // hidden = 1024（weight 为 [numOut][numIn] 行主）
        bool hasBias = (ip->iParams[1] == 1);
        std::cout << i18n::tr("[ch13/路径A] 加载 ncnn 模型: ", "[ch13/Path A] loading ncnn model: ") << ip->name
                  << "  numOut(vocab)=" << numOut
                  << "  numIn(hidden)=" << numIn
                  << "  bias=" << (hasBias ? i18n::tr("有", "yes") : i18n::tr("无", "no")) << "\n";

        const auto& weight = W[ip->name + ".weight"];
        std::vector<float> bias;
        if (hasBias) bias = W[ip->name + ".bias"];

        // 3) 分配 device-local buffer 并上传权重（只上传一次，推理时常驻）
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        auto mk = [&](size_t n) { return vkCreateBuffer(ctx, n * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal); };
        Buffer bW = mk(numOut * numIn), bX = mk(numIn), bY = mk(numOut);
        vkUpload(ctx, bW, weight.data(), weight.size() * 4);   // 622MB 权重一次上传
        std::cout << i18n::tr("[ch13/路径A] 权重已上传显存（", "[ch13/Path A] weights uploaded to VRAM (") << (weight.size() * 4 / 1024 / 1024) << i18n::tr(" MB 常驻）", " MB resident)") << "\n";

        // 4) 准备 GEMM pass（复用 ch04 的分块 GEMM）
        ComputePass pMM = makePass(ctx, "gemm_tiled.spv", 3);
        pMM.bindBuffers({{0, bW.buf}, {1, bX.buf}, {2, bY.buf}});

        // 5) 随机输入（固定种子，结果可复现）
        std::mt19937 rng(20260812);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> x(numIn);
        for (auto& v : x) v = dist(rng);

        // 推理若干次：权重常驻，每个样本只重传输入 x、跑图、读回输出 y
        const int S = 3;
        long long totalErrs = 0;
        for (int s = 0; s < S; ++s) {
            vkUpload(ctx, bX, x.data(), numIn * 4);                 // 只重传输入
            PC13 pc{ (uint32_t)numOut, 1, (uint32_t)numIn, 0 };      // M=vocab, N=1, K=hidden
            Graph g; g.ctx = &ctx; g.begin();                         // 注意：Graph 必须显式挂上 ctx
            g.node(pMM, &pc, sizeof(pc), 1, (uint32_t)((numOut + 15) / 16), 1, nullptr);
            g.end();

            auto out = vkDownload(ctx, bY, numOut * 4);
            const float* y = reinterpret_cast<const float*>(out.data());

            // CPU 参考：y[o] = sum_k W[o][k] * x[k] (+ bias[o])
            std::vector<float> yref(numOut, 0.f);
            for (long long o = 0; o < numOut; ++o) {
                float acc = 0.f;
                const float* wr = &weight[(size_t)o * (size_t)numIn];
                for (long long k = 0; k < numIn; ++k) acc += wr[k] * x[(size_t)k];
                if (hasBias) acc += bias[(size_t)o];
                yref[(size_t)o] = acc;
            }
            double maxDiff = 0.0; long long errs = 0;
            for (long long o = 0; o < numOut; ++o) {
                double d = std::fabs((double)y[o] - (double)yref[o]);
                if (d > std::max(1e-2, std::fabs(yref[o]) * 1e-2)) ++errs;   // 容差 1%
                if (d > maxDiff) maxDiff = d;
            }
            int gpuArg = (int)(std::max_element(y, y + numOut) - y);
            int cpuArg = (int)(std::max_element(yref.begin(), yref.end()) - yref.begin());
            std::cout << i18n::tr("  样本#", "  sample#") << s
                      << "  maxAbsDiff=" << maxDiff
                      << i18n::tr("  错误数=", "  errors=") << errs << "/" << numOut
                      << i18n::tr("  argmax一致=", "  argmax match=") << (gpuArg == cpuArg ? i18n::tr("是", "yes") : i18n::tr("否", "no")) << "\n";
            totalErrs += errs;
        }
        std::cout << i18n::tr("[ch13/路径A] 完成：用自研 Vulkan 引擎加载真实 ncnn 权重并推理，", "[ch13/Path A] done: loaded real ncnn weights and ran inference with the self-built Vulkan engine,")
                  << i18n::tr("与 CPU 参考一致（总错误数=", "matched CPU reference (total errors=") << totalErrs << i18n::tr("，应为0）。", ", should be 0).") << "\n";

        pMM.destroy();
        for (auto b : {bW, bX, bY}) vkDestroyBuffer(ctx, b);
        vkDestroyContext(ctx);
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
