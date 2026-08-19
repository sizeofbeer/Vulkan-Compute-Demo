// ch19_moe/main.cpp
// MoE (Mixture of Experts) Routing: Router → ExpertGroup → Combine
//
// 本章演示 MoE 层的核心机制：
//   1. Router：Softmax + top-k 选择（subgroup 加速）
//   2. Expert GEMM：每个专家独立的前向传播
//   3. Combine：加权求和输出
//   4. 负载均衡分析：统计每个专家被分配的 token 数量

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <chrono>
#include <numeric>

// ============================================================
// 1. 维度定义
// ============================================================
constexpr uint32_t NUM_TOKENS   = 32;
constexpr uint32_t NUM_EXPERTS  = 8;
constexpr uint32_t TOP_K        = 2;
constexpr uint32_t HIDDEN_DIM   = 64;   // 输入/输出维度
constexpr uint32_t FFN_DIM      = 128;  // 专家中间维度（用于 2 层 FFN，但本章只用单层）

// ============================================================
// 2. Push constant 结构（与着色器一致）
// ============================================================

struct PC_Router {
    uint32_t num_tokens;
    uint32_t num_experts;
    uint32_t top_k;
};

struct PC_GEMM {
    uint32_t num_tokens;
    uint32_t N;
    uint32_t K;
    uint32_t expert_id;
};

struct PC_Combine {
    uint32_t num_tokens;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t hidden_dim;
};

// ============================================================
// 3. CPU 参考实现
// ============================================================

// Softmax 沿 expert 维度
void softmax_ref(const float* logits, uint32_t num_tokens,
                 uint32_t num_experts, float* out) {
    for (uint32_t t = 0; t < num_tokens; ++t) {
        float max_val = -1e30f;
        for (uint32_t e = 0; e < num_experts; ++e) {
            max_val = std::max(max_val, logits[t * num_experts + e]);
        }
        float sum = 0.0f;
        for (uint32_t e = 0; e < num_experts; ++e) {
            float v = std::exp(logits[t * num_experts + e] - max_val);
            out[t * num_experts + e] = v;
            sum += v;
        }
        for (uint32_t e = 0; e < num_experts; ++e) {
            out[t * num_experts + e] /= sum;
        }
    }
}

// Top-k 选择
void topk_ref(const float* softmax_out, uint32_t num_tokens,
              uint32_t num_experts, uint32_t top_k,
              uint32_t* out_indices, float* out_weights) {
    for (uint32_t t = 0; t < num_tokens; ++t) {
        // 拷贝 softmax 值到临时数组
        std::vector<float> vals(num_experts);
        std::vector<uint32_t> idxs(num_experts);
        for (uint32_t e = 0; e < num_experts; ++e) {
            vals[e] = softmax_out[t * num_experts + e];
            idxs[e] = e;
        }

        // 选择 top-k（简单冒泡，k 和 num_experts 都很小）
        for (uint32_t k = 0; k < top_k; ++k) {
            uint32_t best_idx = k;
            float best_val = vals[k];
            for (uint32_t e = k + 1; e < num_experts; ++e) {
                if (vals[e] > best_val) {
                    best_val = vals[e];
                    best_idx = e;
                }
            }
            // 交换到位置 k
            std::swap(vals[k], vals[best_idx]);
            std::swap(idxs[k], idxs[best_idx]);
            out_indices[t * top_k + k] = idxs[k];
            out_weights[t * top_k + k] = vals[k];
        }
    }
}

// 专家 GEMM 参考：C = A * W
void expert_gemm_ref(const float* A, const float* W,
                     uint32_t M, uint32_t K, uint32_t N,
                     float* C) {
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                s += A[r * K + k] * W[k * N + c];
            }
            C[r * N + c] = s;
        }
    }
}

// Combine 参考：加权求和
void combine_ref(const float* expert_outputs,  // [num_experts, num_tokens, hidden_dim]
                 const uint32_t* indices,       // [num_tokens, top_k]
                 const float* weights,          // [num_tokens, top_k]
                 uint32_t num_tokens, uint32_t num_experts,
                 uint32_t top_k, uint32_t hidden_dim,
                 float* final_out) {
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
// 4. 验证辅助
// ============================================================

struct VerifyResult {
    uint32_t errorCount;
    float maxError;
    float rmse;
};

VerifyResult verify(const float* gpu, const float* ref,
                    uint32_t count, float relTol) {
    VerifyResult vr{0, 0.0f, 0.0f};
    double sqSum = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        float diff = std::fabs(gpu[i] - ref[i]);
        float maxAllowed = std::max(1e-4f, std::fabs(ref[i]) * relTol);
        if (diff > maxAllowed) {
            vr.errorCount++;
        }
        vr.maxError = std::max(vr.maxError, diff);
        sqSum += (double)diff * diff;
    }
    vr.rmse = (float)std::sqrt(sqSum / (double)count);
    return vr;
}

// ============================================================
// 5. GPU 计时辅助
// ============================================================

double bench_gpu_ms(VulkanContext& ctx, ComputePass& pass,
                    const void* pc, size_t pcSize,
                    uint32_t gx, uint32_t gy, uint32_t gz) {
    vk::QueryPoolCreateInfo qpci{};
    qpci.queryType = vk::QueryType::eTimestamp;
    qpci.queryCount = 2;
    vk::QueryPool pool = ctx.device.createQueryPool(qpci);

    vk::CommandBuffer cmd =
        ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 0);
    pass.dispatch(cmd, pc, pcSize, gx, gy, gz);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, pool, 1);
    cmd.end();

    vk::SubmitInfo si; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    ctx.queue.submit(1, &si, ctx.fence);
    ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
    ctx.device.resetFences(1, &ctx.fence);

    std::vector<uint64_t> ts(2);
    ctx.device.getQueryPoolResults(pool, 0, 2, ts.size() * sizeof(uint64_t), ts.data(),
        sizeof(uint64_t), vk::QueryResultFlagBits::eWait | vk::QueryResultFlagBits::e64);
    double period = ctx.props.limits.timestampPeriod;
    double ms = (double)(ts[1] - ts[0]) * period / 1e6;

    ctx.device.destroyQueryPool(pool);
    ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
    return ms;
}

// ============================================================
// 6. Main
// ============================================================

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);   // 消费 --test-vectors/--output（本章使用编译期常量维度，无直接匹配向量）
        (void)tv;
        VulkanContext ctx = vkInit();

        std::cout << i18n::tr("\n===== 第 19 章：MoE 路由 (Mixture of Experts) =====\n",
                              "\n===== Chapter 19: MoE Routing (Mixture of Experts) =====\n");

        // ---- 维度信息 ----
        const uint32_t num_tokens   = NUM_TOKENS;
        const uint32_t num_experts  = NUM_EXPERTS;
        const uint32_t top_k        = TOP_K;
        const uint32_t hidden_dim   = HIDDEN_DIM;

        std::cout << i18n::tr("  Token 数: ", "  Tokens: ") << num_tokens << "\n";
        std::cout << i18n::tr("  Expert 数: ", "  Experts: ") << num_experts << "\n";
        std::cout << i18n::tr("  top_k: ", "  top_k: ") << top_k << "\n";
        std::cout << i18n::tr("  隐藏维度: ", "  Hidden dim: ") << hidden_dim << "\n";

        // ============================================================
        // 7. 生成随机数据
        // ============================================================
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist_logits(-2.0f, 2.0f);
        std::uniform_real_distribution<float> dist_weights(-1.0f, 1.0f);

        // router_logits: [num_tokens, num_experts]
        std::vector<float> hLogits(num_tokens * num_experts);
        for (auto& v : hLogits) v = dist_logits(rng);

        // token_activations: [num_tokens, hidden_dim]
        std::vector<float> hAct(num_tokens * hidden_dim);
        for (auto& v : hAct) v = dist_weights(rng);

        // expert_weights: [num_experts, hidden_dim, hidden_dim]
        std::vector<float> hExpertWeights(num_experts * hidden_dim * hidden_dim);
        for (auto& v : hExpertWeights) v = dist_weights(rng);

        // ============================================================
        // 8. CPU 参考结果
        // ============================================================
        // 8a. Softmax + top-k
        std::vector<float> hSoftmaxOut(num_tokens * num_experts);
        softmax_ref(hLogits.data(), num_tokens, num_experts, hSoftmaxOut.data());

        std::vector<uint32_t> hRefIndices(num_tokens * top_k);
        std::vector<float> hRefWeights(num_tokens * top_k);
        topk_ref(hSoftmaxOut.data(), num_tokens, num_experts, top_k,
                 hRefIndices.data(), hRefWeights.data());

        // 8b. Expert GEMM: 每个专家对所有 token 做 GEMM
        std::vector<float> hRefExpertOutputs(num_experts * num_tokens * hidden_dim);
        for (uint32_t e = 0; e < num_experts; ++e) {
            expert_gemm_ref(hAct.data(),
                            &hExpertWeights[e * hidden_dim * hidden_dim],
                            num_tokens, hidden_dim, hidden_dim,
                            &hRefExpertOutputs[e * num_tokens * hidden_dim]);
        }

        // 8c. Combine: 加权求和
        std::vector<float> hRefFinal(num_tokens * hidden_dim);
        combine_ref(hRefExpertOutputs.data(), hRefIndices.data(), hRefWeights.data(),
                    num_tokens, num_experts, top_k, hidden_dim,
                    hRefFinal.data());

        // 打印一些参考路由信息
        std::cout << i18n::tr("\n--- 参考路由结果（前 8 个 token） ---\n",
                              "\n--- Reference Routing (first 8 tokens) ---\n");
        for (uint32_t t = 0; t < std::min(8u, num_tokens); ++t) {
            std::cout << i18n::tr("  Token ", "  Token ") << t << ": ";
            for (uint32_t k = 0; k < top_k; ++k) {
                std::cout << i18n::tr("专家", "expert") << hRefIndices[t * top_k + k]
                          << "(" << hRefWeights[t * top_k + k] << ") ";
            }
            std::cout << "\n";
        }

        // ============================================================
        // 9. 创建 GPU 资源
        // ============================================================
        vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer
                                   | vk::BufferUsageFlagBits::eTransferSrc
                                   | vk::BufferUsageFlagBits::eTransferDst;

        // 路由器输入/输出
        Buffer bufLogits   = vkCreateBuffer(ctx, num_tokens * num_experts * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufIndices  = vkCreateBuffer(ctx, num_tokens * top_k * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufWeights  = vkCreateBuffer(ctx, num_tokens * top_k * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Token 激活和专家权重
        Buffer bufAct         = vkCreateBuffer(ctx, num_tokens * hidden_dim * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufExpertWts   = vkCreateBuffer(ctx, num_experts * hidden_dim * hidden_dim * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);

        // 专家输出和最终输出
        Buffer bufExpertOut   = vkCreateBuffer(ctx, num_experts * num_tokens * hidden_dim * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
        Buffer bufFinal       = vkCreateBuffer(ctx, num_tokens * hidden_dim * 4, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);

        // 上传数据
        vkUpload(ctx, bufLogits,   hLogits.data(),         num_tokens * num_experts * 4);
        vkUpload(ctx, bufAct,      hAct.data(),            num_tokens * hidden_dim * 4);
        vkUpload(ctx, bufExpertWts, hExpertWeights.data(),  num_experts * hidden_dim * hidden_dim * 4);

        // ============================================================
        // 10. 准备 ComputePass
        // ============================================================

        // ── Router ──
        ComputePass passRouter;
        passRouter.prepare(ctx, std::string(SHADER_DIR) + "router.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_Router)}});

        // ── Expert GEMM ──
        ComputePass passExpertGEMM;
        passExpertGEMM.prepare(ctx, std::string(SHADER_DIR) + "expert_gemm.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_GEMM)}});

        // ── Combine ──
        ComputePass passCombine;
        passCombine.prepare(ctx, std::string(SHADER_DIR) + "combine.spv",
            {{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
             {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
            {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC_Combine)}});

        // ============================================================
        // 11. Step 1: Dispatch Router
        // ============================================================
        std::cout << i18n::tr("\n--- Step 1: Router (Softmax + Top-",  "\n--- Step 1: Router (Softmax + Top-")
                  << top_k << ") ---\n";

        passRouter.bindBuffers({{0, bufLogits.buf}, {1, bufIndices.buf}, {2, bufWeights.buf}});
        PC_Router pcRouter{num_tokens, num_experts, top_k};
        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passRouter.dispatch(cmd, &pcRouter, sizeof(pcRouter), num_tokens, 1, 1);
        });

        // 下载 router 结果并验证
        {
            auto indicesData = vkDownload(ctx, bufIndices, num_tokens * top_k * 4);
            auto weightsData = vkDownload(ctx, bufWeights, num_tokens * top_k * 4);
            const uint32_t* gpuIndices = reinterpret_cast<const uint32_t*>(indicesData.data());
            const float*    gpuWeights = reinterpret_cast<const float*>(weightsData.data());

            VerifyResult vrIdx = verify(
                reinterpret_cast<const float*>(gpuIndices),
                reinterpret_cast<const float*>(hRefIndices.data()),
                num_tokens * top_k, 0.0f);
            VerifyResult vrWts = verify(gpuWeights, hRefWeights.data(),
                                        num_tokens * top_k, 1e-3f);

            std::cout << i18n::tr("  Router indices 错误数=", "  Router indices errors=") << vrIdx.errorCount
                      << i18n::tr("，权重 RMSE=", ", weights RMSE=") << vrWts.rmse << "\n";
        }

        // ============================================================
        // 12. Step 2: Dispatch Expert GEMM for each expert
        // ============================================================
        std::cout << i18n::tr("\n--- Step 2: Expert GEMM (", "\n--- Step 2: Expert GEMM (")
                  << num_experts << i18n::tr(" 个专家) ---\n", " experts) ---\n");

        // 统计每个专家被分配的 token 数
        std::vector<uint32_t> expertLoad(num_experts, 0);
        for (uint32_t t = 0; t < num_tokens; ++t) {
            for (uint32_t k = 0; k < top_k; ++k) {
                expertLoad[hRefIndices[t * top_k + k]]++;
            }
        }

        // 为每个专家执行 GEMM
        // 输出写入 bufExpertOut 的不同区域（按 expert_id 索引）
        // expert_gemm 绑定: 0=act, 1=expert_weights, 2=expert_out
        // 但 expert_weights 和 expert_out 是每个专家独立区域，
        // 在同一个大 buffer 中通过 expert_id 偏移
        passExpertGEMM.bindBuffers({{0, bufAct.buf}, {1, bufExpertWts.buf}, {2, bufExpertOut.buf}});

        uint32_t gemmGx = (hidden_dim + 15u) / 16u;
        uint32_t gemmGy = (num_tokens + 15u) / 16u;

        for (uint32_t e = 0; e < num_experts; ++e) {
            PC_GEMM pc{num_tokens, hidden_dim, hidden_dim, e};
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                passExpertGEMM.dispatch(cmd, &pc, sizeof(pc), gemmGx, gemmGy, 1);
            });

            std::cout << i18n::tr("  专家 ", "  Expert ") << e
                      << i18n::tr(" 已计算 (负载=", " computed (load=") << expertLoad[e] << ")\n";
        }

        // 验证专家输出
        {
            auto expertData = vkDownload(ctx, bufExpertOut, num_experts * num_tokens * hidden_dim * 4);
            const float* gpuExpertOut = reinterpret_cast<const float*>(expertData.data());

            VerifyResult vr = verify(gpuExpertOut, hRefExpertOutputs.data(),
                                     num_experts * num_tokens * hidden_dim, 1e-3f);
            std::cout << i18n::tr("  Expert GEMM 错误数=", "  Expert GEMM errors=") << vr.errorCount
                      << i18n::tr("，最大误差=", ", max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 13. Step 3: Dispatch Combine
        // ============================================================
        std::cout << i18n::tr("\n--- Step 3: Combine ---\n",
                              "\n--- Step 3: Combine ---\n");

        passCombine.bindBuffers({{0, bufExpertOut.buf}, {1, bufIndices.buf},
                                 {2, bufWeights.buf}, {3, bufFinal.buf}});
        PC_Combine pcCombine{num_tokens, num_experts, top_k, hidden_dim};

        uint32_t combineGx = (hidden_dim + 15u) / 16u;
        uint32_t combineGy = (num_tokens + 15u) / 16u;

        vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
            passCombine.dispatch(cmd, &pcCombine, sizeof(pcCombine), combineGx, combineGy, 1);
        });

        // 验证最终输出
        {
            auto finalData = vkDownload(ctx, bufFinal, num_tokens * hidden_dim * 4);
            const float* gpuFinal = reinterpret_cast<const float*>(finalData.data());

            VerifyResult vr = verify(gpuFinal, hRefFinal.data(),
                                     num_tokens * hidden_dim, 1e-3f);
            std::cout << i18n::tr("  Combine 错误数=", "  Combine errors=") << vr.errorCount
                      << i18n::tr("，最大误差=", ", max error=") << vr.maxError
                      << i18n::tr("，RMSE=", ", RMSE=") << vr.rmse << "\n";
        }

        // ============================================================
        // 14. 专家负载分布分析
        // ============================================================
        std::cout << i18n::tr("\n--- 专家负载分布 ---\n",
                              "\n--- Expert Load Distribution ---\n");

        uint32_t totalAssignments = std::accumulate(expertLoad.begin(), expertLoad.end(), 0u);
        float idealLoad = (float)totalAssignments / (float)num_experts;

        std::cout << i18n::tr("  总分配次数: ", "  Total assignments: ") << totalAssignments
                  << i18n::tr(" (=", " (=)") << num_tokens << " × " << top_k << ")\n";
        std::cout << i18n::tr("  理想平均负载: ", "  Ideal average load: ") << idealLoad << "\n";

        for (uint32_t e = 0; e < num_experts; ++e) {
            float ratio = (float)expertLoad[e] / idealLoad;
            std::cout << i18n::tr("  专家 ", "  Expert ") << e << ": "
                      << expertLoad[e] << i18n::tr(" 个 token (", " tokens (")
                      << (ratio * 100.0f) << i18n::tr("% 均衡)", "% balanced)\n");
        }

        // 计算负载不均衡度（标准差）
        double mean = (double)totalAssignments / (double)num_experts;
        double variance = 0.0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            double diff = (double)expertLoad[e] - mean;
            variance += diff * diff;
        }
        variance /= num_experts;
        double stddev = std::sqrt(variance);
        std::cout << i18n::tr("  负载标准差: ", "  Load stddev: ") << stddev << "\n";
        std::cout << i18n::tr("  负载变异系数: ", "  Load CV: ") << (stddev / mean * 100.0) << "%\n";

        // ============================================================
        // 15. 性能基准测试
        // ============================================================
        std::cout << i18n::tr("\n--- 性能基准 (GPU 时间戳) ---\n",
                              "\n--- Benchmark (GPU Timestamps) ---\n");

        const int warmupIters = 5;
        const int benchIters  = 10;

        // 15a. Router 基准
        {
            passRouter.bindBuffers({{0, bufLogits.buf}, {1, bufIndices.buf}, {2, bufWeights.buf}});
            for (int i = 0; i < warmupIters; ++i) {
                bench_gpu_ms(ctx, passRouter, &pcRouter, sizeof(pcRouter), num_tokens, 1, 1);
            }
            double sum = 0.0;
            for (int i = 0; i < benchIters; ++i) {
                sum += bench_gpu_ms(ctx, passRouter, &pcRouter, sizeof(pcRouter), num_tokens, 1, 1);
            }
            std::cout << i18n::tr("  Router 平均: ", "  Router avg: ") << (sum / benchIters)
                      << i18n::tr(" ms\n", " ms\n");
        }

        // 15b. Expert GEMM 基准（所有专家合计）
        {
            double totalSum = 0.0;
            for (int i = 0; i < warmupIters; ++i) {
                for (uint32_t e = 0; e < num_experts; ++e) {
                    PC_GEMM pc{num_tokens, hidden_dim, hidden_dim, e};
                    bench_gpu_ms(ctx, passExpertGEMM, &pc, sizeof(pc), gemmGx, gemmGy, 1);
                }
            }
            for (int i = 0; i < benchIters; ++i) {
                for (uint32_t e = 0; e < num_experts; ++e) {
                    PC_GEMM pc{num_tokens, hidden_dim, hidden_dim, e};
                    totalSum += bench_gpu_ms(ctx, passExpertGEMM, &pc, sizeof(pc), gemmGx, gemmGy, 1);
                }
            }
            double expertAvg = totalSum / benchIters;
            std::cout << i18n::tr("  Expert GEMM 合计: ", "  Expert GEMM total: ") << expertAvg
                      << i18n::tr(" ms (", " ms (") << (expertAvg / num_experts)
                      << i18n::tr(" ms/专家)\n", " ms/expert)\n");
        }

        // 15c. Combine 基准
        {
            passCombine.bindBuffers({{0, bufExpertOut.buf}, {1, bufIndices.buf},
                                     {2, bufWeights.buf}, {3, bufFinal.buf}});
            for (int i = 0; i < warmupIters; ++i) {
                bench_gpu_ms(ctx, passCombine, &pcCombine, sizeof(pcCombine), combineGx, combineGy, 1);
            }
            double sum = 0.0;
            for (int i = 0; i < benchIters; ++i) {
                sum += bench_gpu_ms(ctx, passCombine, &pcCombine, sizeof(pcCombine), combineGx, combineGy, 1);
            }
            std::cout << i18n::tr("  Combine 平均: ", "  Combine avg: ") << (sum / benchIters)
                      << i18n::tr(" ms\n", " ms\n");
        }

        // 15d. 路由开销分析
        {
            // 对比：路由 + 组合 开销 vs 专家计算开销
            std::cout << i18n::tr("\n--- 路由开销分析 ---\n",
                                  "\n--- Routing Overhead Analysis ---\n");
            // 使用单次测量
            double routerMs = 0.0, combineMs = 0.0, gemmTotalMs = 0.0;
            {
                PC_GEMM pc{num_tokens, hidden_dim, hidden_dim, 0};
                gemmTotalMs = bench_gpu_ms(ctx, passExpertGEMM, &pc, sizeof(pc), gemmGx, gemmGy, 1) * num_experts;
            }
            {
                routerMs = bench_gpu_ms(ctx, passRouter, &pcRouter, sizeof(pcRouter), num_tokens, 1, 1);
            }
            {
                combineMs = bench_gpu_ms(ctx, passCombine, &pcCombine, sizeof(pcCombine), combineGx, combineGy, 1);
            }
            double totalMs = routerMs + gemmTotalMs + combineMs;

            std::cout << i18n::tr("  Router: ", "  Router: ") << routerMs
                      << i18n::tr(" ms (", " ms (") << (routerMs / totalMs * 100.0) << "%)\n";
            std::cout << i18n::tr("  Expert GEMM: ", "  Expert GEMM: ") << gemmTotalMs
                      << i18n::tr(" ms (", " ms (") << (gemmTotalMs / totalMs * 100.0) << "%)\n";
            std::cout << i18n::tr("  Combine: ", "  Combine: ") << combineMs
                      << i18n::tr(" ms (", " ms (") << (combineMs / totalMs * 100.0) << "%)\n";
            std::cout << i18n::tr("  Total: ", "  Total: ") << totalMs << " ms\n";
        }

        // ============================================================
        // 16. 清理
        // ============================================================
        passRouter.destroy();
        passExpertGEMM.destroy();
        passCombine.destroy();
        vkDestroyBuffer(ctx, bufLogits);
        vkDestroyBuffer(ctx, bufIndices);
        vkDestroyBuffer(ctx, bufWeights);
        vkDestroyBuffer(ctx, bufAct);
        vkDestroyBuffer(ctx, bufExpertWts);
        vkDestroyBuffer(ctx, bufExpertOut);
        vkDestroyBuffer(ctx, bufFinal);
        vkDestroyContext(ctx);

        std::cout << i18n::tr("\n第 19 章完成！\n", "\nChapter 19 complete!\n");

    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "Error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}