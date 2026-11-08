// ch19_moe/main_sparse.cpp
// 进阶 demo：稀疏 MoE（Gather → Compute → Scatter），练习 #3 的参考实现。
//
// 与 main.cpp（dense 教学版）的区别：
//   main.cpp       : 8 个专家 × 32 个 token 全部算一遍 GEMM，combine 时再筛掉 3/4 的结果
//   main_sparse.cpp: 先 count → prefix sum → gather 成紧凑列表，只算被选中的
//                    num_tokens × top_k = 64 行，再按反查表 combine
//
// 这条链路对应现行工业级实现：
//   ncnn-MoE-Runtime : router_select.comp → expert_gather.comp → expertbank_* → moe_combine.comp
//   ggml-vulkan      : topk_moe.comp + count_experts.comp → ggml_vk_mul_mat_id()
//
// 本文件是独立可执行目标，不改动 main.cpp，因此不影响 ch00~ch20 的既有验证结论。

#include "vk_base.h"
#include "i18n.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <numeric>
#include <functional>

// ============================================================
// 1. 维度定义（与 main.cpp 完全一致，便于直接对比）
// ============================================================
constexpr uint32_t NUM_TOKENS     = 32;
constexpr uint32_t NUM_EXPERTS    = 8;
constexpr uint32_t TOP_K          = 2;
constexpr uint32_t HIDDEN_DIM     = 64;
constexpr uint32_t TOTAL_ENTRIES  = NUM_TOKENS * TOP_K;   // 紧凑列表长度 = 槽位数

// ============================================================
// 2. Push constant（与着色器一一对应）
// ============================================================
struct PC_Router         { uint32_t num_tokens, num_experts, top_k; };
struct PC_Count          { uint32_t total_entries, num_experts; };
struct PC_Gather         { uint32_t total_entries, top_k; };
struct PC_SparseGemm     { uint32_t dim, total_slots; };
struct PC_SparseCombine  { uint32_t num_tokens, top_k, dim; };
struct PC_DenseGemm      { uint32_t num_tokens, N, K, expert_id; };

// ============================================================
// 3. CPU 参考（与 ch19 main.cpp 相同，作为 golden）
// ============================================================
void softmax_ref(const float* logits, uint32_t num_tokens,
                 uint32_t num_experts, float* out) {
    for (uint32_t t = 0; t < num_tokens; ++t) {
        float max_val = -1e30f;
        for (uint32_t e = 0; e < num_experts; ++e)
            max_val = std::max(max_val, logits[t * num_experts + e]);
        float sum = 0.0f;
        for (uint32_t e = 0; e < num_experts; ++e) {
            float v = std::exp(logits[t * num_experts + e] - max_val);
            out[t * num_experts + e] = v;
            sum += v;
        }
        for (uint32_t e = 0; e < num_experts; ++e)
            out[t * num_experts + e] /= sum;
    }
}

void topk_ref(const float* softmax_out, uint32_t num_tokens,
              uint32_t num_experts, uint32_t top_k,
              uint32_t* out_indices, float* out_weights) {
    for (uint32_t t = 0; t < num_tokens; ++t) {
        std::vector<float> vals(num_experts);
        std::vector<uint32_t> idxs(num_experts);
        for (uint32_t e = 0; e < num_experts; ++e) {
            vals[e] = softmax_out[t * num_experts + e];
            idxs[e] = e;
        }
        for (uint32_t k = 0; k < top_k; ++k) {
            uint32_t best_idx = k;
            float best_val = vals[k];
            for (uint32_t e = k + 1; e < num_experts; ++e) {
                if (vals[e] > best_val) { best_val = vals[e]; best_idx = e; }
            }
            std::swap(vals[k], vals[best_idx]);
            std::swap(idxs[k], idxs[best_idx]);
            out_indices[t * top_k + k] = idxs[k];
            out_weights[t * top_k + k] = vals[k];
        }
    }
}

void expert_gemm_ref(const float* A, const float* W,
                     uint32_t M, uint32_t K, uint32_t N, float* C) {
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k) s += A[r * K + k] * W[k * N + c];
            C[r * N + c] = s;
        }
    }
}

void combine_ref(const float* expert_outputs, const uint32_t* indices,
                 const float* weights, uint32_t num_tokens, uint32_t num_experts,
                 uint32_t top_k, uint32_t hidden_dim, float* final_out) {
    std::fill(final_out, final_out + num_tokens * hidden_dim, 0.0f);
    for (uint32_t t = 0; t < num_tokens; ++t) {
        for (uint32_t k = 0; k < top_k; ++k) {
            uint32_t e = indices[t * top_k + k];
            float w = weights[t * top_k + k];
            for (uint32_t d = 0; d < hidden_dim; ++d) {
                final_out[t * hidden_dim + d] +=
                    w * expert_outputs[e * num_tokens * hidden_dim + t * hidden_dim + d];
            }
        }
    }
}

// ============================================================
// 4. 验证 / 计时辅助
// ============================================================
struct VerifyResult { uint32_t errorCount; float maxError; float rmse; };

VerifyResult verify(const float* gpu, const float* ref,
                    uint32_t count, float relTol) {
    VerifyResult vr{0, 0.0f, 0.0f};
    double sqSum = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        float diff = std::fabs(gpu[i] - ref[i]);
        float maxAllowed = std::max(1e-4f, std::fabs(ref[i]) * relTol);
        if (diff > maxAllowed) vr.errorCount++;
        vr.maxError = std::max(vr.maxError, diff);
        sqSum += (double)diff * diff;
    }
    vr.rmse = (float)std::sqrt(sqSum / (double)count);
    return vr;
}

template <typename T>
std::vector<T> downloadAs(const VulkanContext& ctx, const Buffer& b, size_t count) {
    auto bytes = vkDownload(ctx, b, count * sizeof(T));
    std::vector<T> out(count);
    std::memcpy(out.data(), bytes.data(), count * sizeof(T));
    return out;
}

// 用 GPU timestamp 测一段录制命令的耗时（毫秒）
double bench_gpu_ms(VulkanContext& ctx,
                    const std::function<void(vk::CommandBuffer)>& recorder,
                    uint32_t repeats = 1) {
    vk::QueryPoolCreateInfo qpci{};
    qpci.queryType = vk::QueryType::eTimestamp;
    qpci.queryCount = 2;
    vk::QueryPool pool = ctx.device.createQueryPool(qpci);

    vk::CommandBuffer cmd =
        ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 0);
    for (uint32_t r = 0; r < repeats; ++r) recorder(cmd);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 1);
    cmd.end();

    vk::SubmitInfo si; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    ctx.queue.submit(1, &si, ctx.fence);
    ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
    ctx.device.resetFences(1, &ctx.fence);

    std::vector<uint64_t> ts(2);
    ctx.device.getQueryPoolResults(pool, 0, 2, ts.size() * sizeof(uint64_t), ts.data(),
        sizeof(uint64_t), vk::QueryResultFlagBits::eWait | vk::QueryResultFlagBits::e64);
    double ms = (double)(ts[1] - ts[0]) * ctx.props.limits.timestampPeriod / 1e6
              / (double)repeats;

    ctx.device.destroyQueryPool(pool);
    ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
    return ms;
}

// ============================================================
// 5. Main
// ============================================================
int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        VulkanContext ctx = vkInit();

        std::cout << i18n::tr(
            "\n===== 第 19 章 进阶：稀疏 MoE (Gather / Scatter) =====\n",
            "\n===== Chapter 19 Advanced: Sparse MoE (Gather / Scatter) =====\n");

        const uint32_t num_tokens  = NUM_TOKENS;
        const uint32_t num_experts = NUM_EXPERTS;
        const uint32_t top_k       = TOP_K;
        const uint32_t dim         = HIDDEN_DIM;
        const uint32_t total       = TOTAL_ENTRIES;

        std::cout << i18n::tr("  Token 数: ", "  Tokens: ") << num_tokens
                  << i18n::tr("，Expert 数: ", ", Experts: ") << num_experts
                  << i18n::tr("，top_k: ", ", top_k: ") << top_k
                  << i18n::tr("，隐藏维度: ", ", hidden dim: ") << dim << "\n";
        std::cout << i18n::tr("  路由条目数 (槽位): ", "  Routing entries (slots): ")
                  << total << "\n";

        // ---- 与 ch19 主程序完全相同的数据（种子 42） ----
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist_logits(-2.0f, 2.0f);
        std::uniform_real_distribution<float> dist_weights(-1.0f, 1.0f);

        std::vector<float> hLogits(num_tokens * num_experts);
        for (auto& v : hLogits) v = dist_logits(rng);
        std::vector<float> hAct(num_tokens * dim);
        for (auto& v : hAct) v = dist_weights(rng);
        std::vector<float> hExpertWeights(num_experts * dim * dim);
        for (auto& v : hExpertWeights) v = dist_weights(rng);

        // ---- CPU golden（dense 语义，与 ch19 主程序一致） ----
        std::vector<float> hSoftmax(num_tokens * num_experts);
        softmax_ref(hLogits.data(), num_tokens, num_experts, hSoftmax.data());
        std::vector<uint32_t> hRefIndices(total);
        std::vector<float> hRefWeights(total);
        topk_ref(hSoftmax.data(), num_tokens, num_experts, top_k,
                 hRefIndices.data(), hRefWeights.data());

        std::vector<float> hRefExpertOut(num_experts * num_tokens * dim);
        for (uint32_t e = 0; e < num_experts; ++e) {
            expert_gemm_ref(hAct.data(), &hExpertWeights[e * dim * dim],
                            num_tokens, dim, dim,
                            &hRefExpertOut[e * num_tokens * dim]);
        }
        std::vector<float> hRefFinal(num_tokens * dim);
        combine_ref(hRefExpertOut.data(), hRefIndices.data(), hRefWeights.data(),
                    num_tokens, num_experts, top_k, dim, hRefFinal.data());

        // CPU 侧期望的每专家条目数（用于校验 GPU count / prefix sum）
        std::vector<uint32_t> hExpectedCounts(num_experts, 0);
        for (uint32_t i = 0; i < total; ++i) hExpectedCounts[hRefIndices[i]]++;
        std::vector<uint32_t> hExpectedOffsets(num_experts, 0);
        uint32_t running = 0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            hExpectedOffsets[e] = running;
            running += hExpectedCounts[e];
        }

        // ---- GPU 资源 ----
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;
        auto mk = [&](size_t bytes) {
            return vkCreateBuffer(ctx, bytes, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        };

        Buffer bufLogits     = mk(num_tokens * num_experts * sizeof(float));
        Buffer bufIndices    = mk(total * sizeof(uint32_t));
        Buffer bufWeights    = mk(total * sizeof(float));
        Buffer bufAct        = mk(num_tokens * dim * sizeof(float));
        Buffer bufExpertWts  = mk(num_experts * dim * dim * sizeof(float));
        Buffer bufCounts     = mk(num_experts * sizeof(uint32_t));
        Buffer bufOffsets    = mk(num_experts * sizeof(uint32_t));
        Buffer bufCursor     = mk(num_experts * sizeof(uint32_t));
        Buffer bufGToken     = mk(total * sizeof(uint32_t));
        Buffer bufGExpert    = mk(total * sizeof(uint32_t));
        Buffer bufGSlot      = mk(total * sizeof(uint32_t));
        Buffer bufPartial    = mk(total * dim * sizeof(float));
        Buffer bufFinal      = mk(num_tokens * dim * sizeof(float));
        Buffer bufDenseOut   = mk(num_experts * num_tokens * dim * sizeof(float));

        vkUpload(ctx, bufLogits,    hLogits.data(),        num_tokens * num_experts * sizeof(float));
        vkUpload(ctx, bufAct,       hAct.data(),           num_tokens * dim * sizeof(float));
        vkUpload(ctx, bufExpertWts, hExpertWeights.data(), num_experts * dim * dim * sizeof(float));

        // ---- ComputePass ----
        auto bindings = [](uint32_t n) {
            std::vector<vk::DescriptorSetLayoutBinding> v;
            for (uint32_t i = 0; i < n; ++i)
                v.push_back({i, vk::DescriptorType::eStorageBuffer, 1,
                             vk::ShaderStageFlagBits::eCompute});
            return v;
        };
        auto pcs = [](size_t sz) {
            return std::vector<vk::PushConstantRange>{
                vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, (uint32_t)sz}};
        };

        ComputePass passRouter, passCount, passGather, passSparseGemm,
                    passSparseCombine, passDenseGemm;
        passRouter.prepare(ctx, std::string(SHADER_DIR) + "router.spv",
                           bindings(3), pcs(sizeof(PC_Router)));
        passCount.prepare(ctx, std::string(SHADER_DIR) + "count_experts.spv",
                          bindings(4), pcs(sizeof(PC_Count)));
        passGather.prepare(ctx, std::string(SHADER_DIR) + "expert_gather.spv",
                           bindings(5), pcs(sizeof(PC_Gather)));
        passSparseGemm.prepare(ctx, std::string(SHADER_DIR) + "expert_gemm_sparse.spv",
                               bindings(5), pcs(sizeof(PC_SparseGemm)));
        passSparseCombine.prepare(ctx, std::string(SHADER_DIR) + "moe_combine_sparse.spv",
                                  bindings(4), pcs(sizeof(PC_SparseCombine)));
        passDenseGemm.prepare(ctx, std::string(SHADER_DIR) + "expert_gemm.spv",
                              bindings(3), pcs(sizeof(PC_DenseGemm)));

        passRouter.bindBuffers({{0, bufLogits.buf}, {1, bufIndices.buf}, {2, bufWeights.buf}});
        passCount.bindBuffers({{0, bufIndices.buf}, {1, bufCounts.buf},
                               {2, bufOffsets.buf}, {3, bufCursor.buf}});
        passGather.bindBuffers({{0, bufIndices.buf}, {1, bufCursor.buf},
                                {2, bufGToken.buf}, {3, bufGExpert.buf}, {4, bufGSlot.buf}});
        passSparseGemm.bindBuffers({{0, bufAct.buf}, {1, bufExpertWts.buf},
                                    {2, bufGToken.buf}, {3, bufGExpert.buf}, {4, bufPartial.buf}});
        passSparseCombine.bindBuffers({{0, bufPartial.buf}, {1, bufGSlot.buf},
                                       {2, bufWeights.buf}, {3, bufFinal.buf}});
        passDenseGemm.bindBuffers({{0, bufAct.buf}, {1, bufExpertWts.buf},
                                   {2, bufDenseOut.buf}});

        const uint32_t gx16 = (dim + 15u) / 16u;
        const uint32_t gyTokens = (num_tokens + 15u) / 16u;
        const uint32_t gxEntries = (total + 63u) / 64u;

        // ---- 稀疏链路（逐 pass 提交；跨 submit 有隐式同步，与 ch19 主程序一致） ----
        std::cout << i18n::tr("\n--- 稀疏链路: router → count → gather → sparse GEMM → combine ---\n",
                              "\n--- Sparse chain: router → count → gather → sparse GEMM → combine ---\n");

        PC_Router pcRouter{num_tokens, num_experts, top_k};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passRouter.dispatch(cmd, &pcRouter, sizeof(pcRouter), num_tokens, 1, 1);
        });
        std::cout << i18n::tr("  1) router.spv            (softmax + top-k)\n",
                              "  1) router.spv            (softmax + top-k)\n");

        PC_Count pcCount{total, num_experts};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passCount.dispatch(cmd, &pcCount, sizeof(pcCount), 1, 1, 1);
        });
        std::cout << i18n::tr("  2) count_experts.spv     (计数 + prefix sum)\n",
                              "  2) count_experts.spv     (count + prefix sum)\n");

        PC_Gather pcGather{total, top_k};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passGather.dispatch(cmd, &pcGather, sizeof(pcGather), gxEntries, 1, 1);
        });
        std::cout << i18n::tr("  3) expert_gather.spv     (scatter 到紧凑槽位)\n",
                              "  3) expert_gather.spv     (scatter into compact slots)\n");

        PC_SparseGemm pcSparseGemm{dim, total};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passSparseGemm.dispatch(cmd, &pcSparseGemm, sizeof(pcSparseGemm),
                                    total, 1, 1);
        });
        std::cout << i18n::tr("  4) expert_gemm_sparse.spv(只算被选中的 64 行)\n",
                              "  4) expert_gemm_sparse.spv(selected 64 rows only)\n");

        PC_SparseCombine pcCombine{num_tokens, top_k, dim};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passSparseCombine.dispatch(cmd, &pcCombine, sizeof(pcCombine),
                                       gx16, gyTokens, 1);
        });
        std::cout << i18n::tr("  5) moe_combine_sparse.spv(反查表加权和)\n",
                              "  5) moe_combine_sparse.spv(gather-weighted sum)\n");

        // ---- 验证 1：count / offsets / cursor ----
        auto gpuCounts  = downloadAs<uint32_t>(ctx, bufCounts,  num_experts);
        auto gpuOffsets = downloadAs<uint32_t>(ctx, bufOffsets, num_experts);
        auto gpuCursor  = downloadAs<uint32_t>(ctx, bufCursor,  num_experts);

        uint32_t planErrors = 0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            if (gpuCounts[e]  != hExpectedCounts[e])  ++planErrors;
            if (gpuOffsets[e] != hExpectedOffsets[e]) ++planErrors;
            // gather 结束后 cursor 应恰好走到该专家区间末尾
            if (gpuCursor[e]  != hExpectedOffsets[e] + hExpectedCounts[e]) ++planErrors;
        }
        std::cout << i18n::tr("\n--- 验证 1: 专家计数 / prefix sum ---\n",
                              "\n--- Check 1: expert counts / prefix sum ---\n");
        std::cout << i18n::tr("  计划表错误数 = ", "  Plan errors = ") << planErrors
                  << i18n::tr("，槽位总数 = ", ", total slots = ")
                  << std::accumulate(gpuCounts.begin(), gpuCounts.end(), 0u)
                  << " / " << total << "\n";
        for (uint32_t e = 0; e < num_experts; ++e) {
            std::cout << i18n::tr("  专家 ", "  Expert ") << e
                      << ": count=" << gpuCounts[e]
                      << ", offset=" << gpuOffsets[e]
                      << ", cursor=" << gpuCursor[e] << "\n";
        }

        // ---- 验证 2：gather 映射（槽位唯一、落在正确区间、token/expert 对得上） ----
        auto gpuIndices = downloadAs<uint32_t>(ctx, bufIndices, total);
        auto gpuGToken  = downloadAs<uint32_t>(ctx, bufGToken,  total);
        auto gpuGExpert = downloadAs<uint32_t>(ctx, bufGExpert, total);
        auto gpuGSlot   = downloadAs<uint32_t>(ctx, bufGSlot,   total);

        std::vector<uint8_t> used(total, 0);
        uint32_t mapErrors = 0;
        for (uint32_t entry = 0; entry < total; ++entry) {
            uint32_t slot = gpuGSlot[entry];
            if (slot >= total) { ++mapErrors; continue; }
            if (used[slot]) ++mapErrors;
            used[slot] = 1;
            uint32_t e = gpuIndices[entry];
            uint32_t t = entry / top_k;
            if (gpuGToken[slot] != t || gpuGExpert[slot] != e) ++mapErrors;
            if (e >= num_experts ||
                slot < gpuOffsets[e] || slot >= gpuOffsets[e] + gpuCounts[e]) ++mapErrors;
        }
        for (uint32_t s = 0; s < total; ++s) if (!used[s]) ++mapErrors;

        std::cout << i18n::tr("\n--- 验证 2: Gather / Scatter 映射 ---\n",
                              "\n--- Check 2: Gather / Scatter mapping ---\n");
        std::cout << i18n::tr("  映射错误数 = ", "  Mapping errors = ") << mapErrors
                  << i18n::tr("（槽位唯一性 / 区间 / 反向索引）\n",
                              "  (slot uniqueness / ranges / reverse index)\n");

        // ---- 验证 3：最终输出 vs dense CPU golden ----
        auto gpuFinal = downloadAs<float>(ctx, bufFinal, num_tokens * dim);
        VerifyResult vrFinal = verify(gpuFinal.data(), hRefFinal.data(),
                                      num_tokens * dim, 1e-3f);
        std::cout << i18n::tr("\n--- 验证 3: 稀疏输出 vs dense CPU 参考 ---\n",
                              "\n--- Check 3: sparse output vs dense CPU reference ---\n");
        std::cout << i18n::tr("  错误数=", "  Errors=") << vrFinal.errorCount
                  << i18n::tr("，最大误差=", ", max error=") << vrFinal.maxError
                  << i18n::tr("，RMSE=", ", RMSE=") << vrFinal.rmse << "\n";

        // ---- 计算量对比（理论） ----
        uint64_t denseMAC  = (uint64_t)num_experts * num_tokens * dim * dim;
        uint64_t sparseMAC = (uint64_t)total * dim * dim;
        std::cout << i18n::tr("\n--- 计算量对比 ---\n", "\n--- Arithmetic comparison ---\n");
        std::cout << i18n::tr("  dense 专家 GEMM: ", "  dense expert GEMM: ") << denseMAC << " MAC\n";
        std::cout << i18n::tr("  sparse 专家 GEMM: ", "  sparse expert GEMM: ") << sparseMAC << " MAC\n";
        std::cout << i18n::tr("  理论节省倍数: ", "  Theoretical reduction: ")
                  << (double)denseMAC / (double)sparseMAC << "x\n";

        // ---- GPU 实测：dense 8 次 dispatch vs sparse 1 次 dispatch ----
        auto denseRecorder = [&](vk::CommandBuffer cmd) {
            for (uint32_t e = 0; e < num_experts; ++e) {
                PC_DenseGemm pc{num_tokens, dim, dim, e};
                passDenseGemm.dispatch(cmd, &pc, sizeof(pc), gx16, gyTokens, 1);
            }
        };
        auto sparseRecorder = [&](vk::CommandBuffer cmd) {
            passSparseGemm.dispatch(cmd, &pcSparseGemm, sizeof(pcSparseGemm),
                                    total, 1, 1);
        };
        // 单条命令缓冲内重复多组，压低微秒级 dispatch / 频率抖动带来的噪声
        const uint32_t repeats = 20;
        for (int i = 0; i < 3; ++i) {
            bench_gpu_ms(ctx, denseRecorder, repeats);
            bench_gpu_ms(ctx, sparseRecorder, repeats);
        }
        double denseMs = 0.0, sparseMs = 0.0;
        for (int i = 0; i < 10; ++i) {
            denseMs  += bench_gpu_ms(ctx, denseRecorder, repeats);
            sparseMs += bench_gpu_ms(ctx, sparseRecorder, repeats);
        }
        denseMs /= 10.0; sparseMs /= 10.0;
        std::cout << i18n::tr("\n--- GPU 时间戳实测 (每组重复 20 次，10 组平均) ---\n",
                              "\n--- GPU timestamps (20 repeats per sample, 10 samples) ---\n");
        std::cout << i18n::tr("  dense  (8 次 dispatch): ", "  dense  (8 dispatches): ")
                  << denseMs << " ms\n";
        std::cout << i18n::tr("  sparse (1 次 dispatch): ", "  sparse (1 dispatch):   ")
                  << sparseMs << " ms\n";
        std::cout << i18n::tr("  实测加速比: ", "  Measured speedup: ")
                  << (sparseMs > 0.0 ? denseMs / sparseMs : 0.0) << "x\n";
        std::cout << i18n::tr(
            "\n  解读：理论倍数只是算术量上界。本教学规模 (32 token × 64 维) 下，\n"
            "        workgroup 调度、权重访存与 count / gather 两次额外 dispatch 都会\n"
            "        摊薄收益，实测加速比低于理论上界且会随 GPU 频率 / 缓存状态波动。\n"
            "        把 NUM_EXPERTS 调大、TOP_K 调小、HIDDEN_DIM 放大，或把稀疏 GEMM\n"
            "        换成 tiled / coopmat 实现，才会逐步逼近理论上界。\n",
            "\n  Reading: the theoretical factor is only an arithmetic upper bound. At this\n"
            "        teaching scale (32 tokens x 64 dims), workgroup scheduling, weight\n"
            "        traffic and the two extra count/gather dispatches all dilute the gain,\n"
            "        so the measured speedup stays below 4x and varies with GPU clocks/cache.\n"
            "        Increase NUM_EXPERTS, lower TOP_K, enlarge HIDDEN_DIM, or replace the\n"
            "        sparse GEMM with a tiled/coopmat kernel to approach the upper bound.\n");

        // ---- 清理 ----
        passRouter.destroy(); passCount.destroy(); passGather.destroy();
        passSparseGemm.destroy(); passSparseCombine.destroy(); passDenseGemm.destroy();
        vkDestroyBuffer(ctx, bufLogits);    vkDestroyBuffer(ctx, bufIndices);
        vkDestroyBuffer(ctx, bufWeights);   vkDestroyBuffer(ctx, bufAct);
        vkDestroyBuffer(ctx, bufExpertWts); vkDestroyBuffer(ctx, bufCounts);
        vkDestroyBuffer(ctx, bufOffsets);   vkDestroyBuffer(ctx, bufCursor);
        vkDestroyBuffer(ctx, bufGToken);    vkDestroyBuffer(ctx, bufGExpert);
        vkDestroyBuffer(ctx, bufGSlot);     vkDestroyBuffer(ctx, bufPartial);
        vkDestroyBuffer(ctx, bufFinal);     vkDestroyBuffer(ctx, bufDenseOut);
        vkDestroyContext(ctx);

        bool ok = (planErrors == 0 && mapErrors == 0 && vrFinal.errorCount == 0);
        std::cout << i18n::tr("\n稀疏 MoE demo ", "\nSparse MoE demo ")
                  << i18n::tr(ok ? "通过！\n" : "存在错误！\n", ok ? "PASSED!\n" : "FAILED!\n");
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
}
