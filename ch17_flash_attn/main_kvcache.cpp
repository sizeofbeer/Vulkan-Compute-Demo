// ch17_flash_attn/main_kvcache.cpp
// 进阶 demo：KV cache 增量解码 + decode 两阶段（split / reduce）
//
// 主程序 `main.cpp` 的 flash_attn.comp 是「prefill 思路」：1 个 workgroup
// 串行扫完整个序列。真实 LLM 解码时每步只有 1 个新 query，却要读很长的
// KV cache，于是工业引擎把 decode 拆成不同 kernel：
//   append : 把新 token 的 K/V 追加到 cache（ncnn sdpa_kvcache_append.comp）
//   phase1 : cache 按 tile 分给多个 workgroup，各自算局部 online softmax
//   phase2 : 全局归约（ggml flash_attn_decode_phase_1/2、split_k_reduce）
//
// 本 demo 同时跑「单 kernel 基线」（复用本章 flash_attn.spv）与「两阶段」，
// 用 CPU 参考 + 交叉验证确认两者等价，并对比并行度。
// 独立可执行目标，不改动 main.cpp，不影响 ch00~ch20 的既有验证结论。

#include "vk_base.h"
#include "i18n.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>

constexpr uint32_t D           = 64;
constexpr uint32_t PROMPT_LEN  = 256;
constexpr uint32_t DECODE_STEPS= 4;
constexpr uint32_t MAX_LEN     = PROMPT_LEN + DECODE_STEPS;
constexpr uint32_t TILE        = 32;
constexpr uint32_t MAX_TILES   = (MAX_LEN + TILE - 1) / TILE;

struct PC_Append { uint32_t writePos, d; };
struct PC_Phase  { uint32_t seqLen, d, tileSize; };
struct PC17      { uint32_t N, d, Br, Bc; };   // 本章 flash_attn.spv 的 push constant

// ============================================================
// 1. 辅助
// ============================================================
static ComputePass makePass(VulkanContext& ctx, const std::string& spv,
                            uint32_t nBind, uint32_t pcSize) {
    ComputePass p;
    std::vector<vk::DescriptorSetLayoutBinding> binds;
    for (uint32_t i = 0; i < nBind; ++i)
        binds.push_back({i, vk::DescriptorType::eStorageBuffer, 1,
                         vk::ShaderStageFlagBits::eCompute});
    p.prepare(ctx, std::string(SHADER_DIR) + spv, binds,
        {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, pcSize}});
    return p;
}

static std::vector<float> downloadFloats(const VulkanContext& ctx, const Buffer& b, size_t count) {
    auto bytes = vkDownload(ctx, b, count * sizeof(float));
    std::vector<float> out(count);
    std::memcpy(out.data(), bytes.data(), count * sizeof(float));
    return out;
}

// CPU 参考：单 query 对 seqLen 个 cache key 做标准 attention
static std::vector<float> cpuDecodeAttention(const std::vector<float>& q,
                                             const std::vector<float>& K,
                                             const std::vector<float>& V,
                                             uint32_t seqLen, uint32_t d) {
    std::vector<float> scores(seqLen);
    float inv = 1.0f / std::sqrt((float)d);
    float mx = -1e30f;
    for (uint32_t k = 0; k < seqLen; ++k) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < d; ++j) dot += q[j] * K[k * d + j];
        scores[k] = dot * inv;
        mx = std::max(mx, scores[k]);
    }
    float sum = 0.0f;
    for (uint32_t k = 0; k < seqLen; ++k) {
        scores[k] = std::exp(scores[k] - mx);
        sum += scores[k];
    }
    for (uint32_t k = 0; k < seqLen; ++k) scores[k] /= sum;

    std::vector<float> O(d, 0.0f);
    for (uint32_t k = 0; k < seqLen; ++k)
        for (uint32_t j = 0; j < d; ++j)
            O[j] += scores[k] * V[k * d + j];
    return O;
}

static uint32_t verify(const float* gpu, const float* ref, uint32_t n, float eps) {
    uint32_t errs = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (std::fabs(gpu[i] - ref[i]) > std::max(eps, std::fabs(ref[i]) * 1e-3f)) ++errs;
    }
    return errs;
}

// ============================================================
// 2. Main
// ============================================================
int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        VulkanContext ctx = vkInit();

        std::cout << i18n::tr(
            "\n===== 第 17 章 进阶：KV cache 增量解码 (split / reduce) =====\n",
            "\n===== Chapter 17 Advanced: KV Cache Incremental Decode (split / reduce) =====\n");
        std::cout << i18n::tr("头维度 d=", "Head dim d=") << D
                  << i18n::tr("，prefill 长度=", ", prefill length=") << PROMPT_LEN
                  << i18n::tr("，decode 步数=", ", decode steps=") << DECODE_STEPS
                  << i18n::tr("，tile=", ", tile=") << TILE << "\n";

        std::mt19937 rng(314159);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // CPU 侧完整 cache（前 PROMPT_LEN 行是 prefill，后续由每步 append 填入）
        std::vector<float> hKcache(MAX_LEN * D), hVcache(MAX_LEN * D);
        for (size_t i = 0; i < (size_t)PROMPT_LEN * D; ++i) {
            hKcache[i] = dist(rng);
            hVcache[i] = dist(rng);
        }

        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        auto mk = [&](size_t floats) {
            return vkCreateBuffer(ctx, floats * sizeof(float), usage,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal);
        };

        Buffer bufCacheK = mk(MAX_LEN * D), bufCacheV = mk(MAX_LEN * D);
        Buffer bufNewK   = mk(D),           bufNewV   = mk(D);
        Buffer bufQ      = mk(D);
        Buffer bufPartO  = mk(MAX_TILES * D), bufPartM = mk(MAX_TILES), bufPartL = mk(MAX_TILES);
        Buffer bufO2     = mk(D),           bufOBase  = mk(D);

        // prefill：只上传前 PROMPT_LEN 行，其余位置留待 append shader 写入
        vkUpload(ctx, bufCacheK, hKcache.data(), PROMPT_LEN * D * sizeof(float));
        vkUpload(ctx, bufCacheV, hVcache.data(), PROMPT_LEN * D * sizeof(float));

        // ---- 四个/六个绑定的 pass ----
        ComputePass passAppend = makePass(ctx, "kvcache_append.spv", 4, sizeof(PC_Append));
        ComputePass passP1     = makePass(ctx, "flash_attn_decode_phase1.spv", 6, sizeof(PC_Phase));
        ComputePass passP2     = makePass(ctx, "flash_attn_decode_phase2.spv", 4, sizeof(PC_Phase));
        ComputePass passBase   = makePass(ctx, "flash_attn.spv", 4, sizeof(PC17));

        passAppend.bindBuffers({{0, bufNewK.buf}, {1, bufNewV.buf},
                                {2, bufCacheK.buf}, {3, bufCacheV.buf}});
        passP1.bindBuffers({{0, bufQ.buf}, {1, bufCacheK.buf}, {2, bufCacheV.buf},
                            {3, bufPartO.buf}, {4, bufPartM.buf}, {5, bufPartL.buf}});
        passP2.bindBuffers({{0, bufPartO.buf}, {1, bufPartM.buf},
                            {2, bufPartL.buf}, {3, bufO2.buf}});
        passBase.bindBuffers({{0, bufQ.buf}, {1, bufCacheK.buf},
                              {2, bufCacheV.buf}, {3, bufOBase.buf}});

        std::cout << i18n::tr(
            "\n--- 每步: append → phase1(split) → phase2(reduce) vs 单 kernel 基线 ---\n",
            "\n--- Each step: append → phase1(split) → phase2(reduce) vs single-kernel baseline ---\n");

        uint32_t totalErrors = 0;
        for (uint32_t step = 0; step < DECODE_STEPS; ++step) {
            // 本步的新 token
            std::vector<float> hNewK(D), hNewV(D), hQ(D);
            for (auto& v : hNewK) v = dist(rng);
            for (auto& v : hNewV) v = dist(rng);
            for (auto& v : hQ)    v = dist(rng);

            uint32_t writePos = PROMPT_LEN + step;
            std::memcpy(&hKcache[(size_t)writePos * D], hNewK.data(), D * sizeof(float));
            std::memcpy(&hVcache[(size_t)writePos * D], hNewV.data(), D * sizeof(float));

            vkUpload(ctx, bufNewK, hNewK.data(), D * sizeof(float));
            vkUpload(ctx, bufNewV, hNewV.data(), D * sizeof(float));
            vkUpload(ctx, bufQ,    hQ.data(),    D * sizeof(float));

            uint32_t seqLen   = writePos + 1;
            uint32_t numTiles = (seqLen + TILE - 1) / TILE;

            // 1) append 新 K/V 到 cache
            PC_Append pcAppend{writePos, D};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passAppend.dispatch(cmd, &pcAppend, sizeof(pcAppend), (D + 31u) / 32u, 1, 1);
            });

            // 2) phase1：每个 workgroup 处理一个 key tile
            PC_Phase pcPhase{seqLen, D, TILE};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passP1.dispatch(cmd, &pcPhase, sizeof(pcPhase), numTiles, 1, 1);
            });

            // 3) phase2：全局归约（独立 submit，跨 submit 隐式同步 phase1 的写入）
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passP2.dispatch(cmd, &pcPhase, sizeof(pcPhase), 1, 1, 1);
            });

            // 4) 单 kernel 基线（复用本章 flash_attn.spv，只 dispatch 1 个 query 行）
            PC17 pcBase{seqLen, D, 1, TILE};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passBase.dispatch(cmd, &pcBase, sizeof(pcBase), 1, 1, 1);
            });

            // ---- 下载与验证 ----
            auto gpuO2   = downloadFloats(ctx, bufO2,   D);
            auto gpuOBase= downloadFloats(ctx, bufOBase,D);
            auto gpuCK   = downloadFloats(ctx, bufCacheK, MAX_LEN * D);
            auto gpuCV   = downloadFloats(ctx, bufCacheV, MAX_LEN * D);
            std::vector<float> cpuRef = cpuDecodeAttention(hQ, hKcache, hVcache, seqLen, D);

            uint32_t errAppend = 0;
            for (uint32_t k = 0; k < seqLen; ++k)
                for (uint32_t j = 0; j < D; ++j) {
                    if (std::fabs(gpuCK[k * D + j] - hKcache[k * D + j]) > 1e-5f) ++errAppend;
                    if (std::fabs(gpuCV[k * D + j] - hVcache[k * D + j]) > 1e-5f) ++errAppend;
                }

            uint32_t errTwo   = verify(gpuO2.data(),    cpuRef.data(),       D, 1e-2f);
            uint32_t errBase  = verify(gpuOBase.data(), cpuRef.data(),       D, 1e-2f);
            uint32_t errCross = verify(gpuO2.data(),    gpuOBase.data(),     D, 1e-2f);
            uint32_t stepErr  = errAppend + errTwo + errBase + errCross;
            totalErrors += stepErr;

            std::cout << i18n::tr("  step ", "  step ") << step
                      << ": seqLen=" << seqLen << ", tiles=" << numTiles
                      << i18n::tr(" | append错误=", " | append errors=") << errAppend
                      << i18n::tr(" 两阶段vsCPU=", " two-phase vs CPU=") << errTwo
                      << i18n::tr(" 单kernelvsCPU=", " single-kernel vs CPU=") << errBase
                      << i18n::tr(" 交叉=", " cross=") << errCross << "\n";
        }

        std::cout << i18n::tr("\n--- 并行度对比（结构性，不依赖计时）---\n",
                              "\n--- Parallelism (structural, not a timing claim) ---\n");
        std::cout << i18n::tr("  单 kernel 基线: 1 个 workgroup 串行遍历 seqLen 个 key\n",
                              "  Single-kernel baseline: 1 workgroup serially scans seqLen keys\n");
        std::cout << i18n::tr("  两阶段 decode  : phase1 有 numTiles 个 workgroup 并行 + phase2 归约\n",
                              "  Two-phase decode: phase1 runs numTiles workgroups in parallel + phase2 reduce\n");
        std::cout << i18n::tr("  这正是长上下文下 ggml 用 decode_phase_1/2 + split_k_reduce 的原因；\n"
                              "  短序列时多一次 dispatch 与全局同步反而可能更慢（本 demo 不宣称加速比）。\n",
                              "  This is why ggml uses decode_phase_1/2 + split_k_reduce for long contexts;\n"
                              "  for short sequences the extra dispatch/reduction may even be slower\n"
                              "  (this demo makes no speedup claim).\n");

        // ---- 清理 ----
        passAppend.destroy(); passP1.destroy(); passP2.destroy(); passBase.destroy();
        vkDestroyBuffer(ctx, bufCacheK); vkDestroyBuffer(ctx, bufCacheV);
        vkDestroyBuffer(ctx, bufNewK);   vkDestroyBuffer(ctx, bufNewV);
        vkDestroyBuffer(ctx, bufQ);      vkDestroyBuffer(ctx, bufPartO);
        vkDestroyBuffer(ctx, bufPartM);  vkDestroyBuffer(ctx, bufPartL);
        vkDestroyBuffer(ctx, bufO2);     vkDestroyBuffer(ctx, bufOBase);
        vkDestroyContext(ctx);

        bool ok = (totalErrors == 0);
        std::cout << i18n::tr("\nKV cache decode demo ", "\nKV cache decode demo ")
                  << i18n::tr(ok ? "通过！\n" : "存在错误！\n", ok ? "PASSED!\n" : "FAILED!\n");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
}
