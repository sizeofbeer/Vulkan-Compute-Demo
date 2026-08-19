// ch16_subgroup/main.cpp
// Subgroup 编程：展示 Vulkan subgroup 操作的 6 种模式，
// 并对比 subgroup softmax 与共享内存 softmax 的性能。

#include "vk_base.h"
#include "i18n.h"
#include "test_vectors.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

struct PC16 { uint32_t N; };

// 便捷函数：创建 storage buffer
static Buffer makeStorage(const VulkanContext& ctx, size_t bytes) {
    auto usage = vk::BufferUsageFlagBits::eStorageBuffer
               | vk::BufferUsageFlagBits::eTransferSrc
               | vk::BufferUsageFlagBits::eTransferDst;
    return vkCreateBuffer(ctx, bytes, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
}

// 便捷函数：创建 compute pass（带 specialization constant）
static ComputePass makePass(VulkanContext& ctx, const std::string& spv,
                            int opMode, const PC16& pc) {
    ComputePass pass;
    std::vector<vk::DescriptorSetLayoutBinding> binds = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
    };
    std::array<int, 1> specData = {opMode};
    auto specInfo = vkMakeSpecializationInfo(specData);
    pass.prepareWithSpec(ctx, std::string(SHADER_DIR) + spv, binds,
        {vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PC16)}},
        specInfo.info);
    return pass;
}

static void checkErr(uint32_t errs, const std::string& label) {
    std::cout << "  " << label << ": "
              << (errs == 0 ? i18n::tr("通过", "PASS") : i18n::tr("失败 (错误数=", "FAIL (errors=") + std::to_string(errs) + ")")
              << "\n";
}

int main(int argc, char** argv) {
    try {
        i18n::init(argc, argv);
        TestVectorConfig tv;
        tv.parse(argc, argv);   // 消费 --test-vectors/--output（本章为硬件相关多模式演示，无直接匹配向量）
        (void)tv;
        VulkanContext ctx = vkInit();

        const auto& feat = ctx.features;
        const uint32_t SG_SIZE = feat.subgroupSize;

        // ── 打印 subgroup 属性 ──────────────────────────────────────────
        std::cout << i18n::tr("\n=== Subgroup 属性 ===\n", "\n=== Subgroup Properties ===\n");
        std::cout << i18n::tr("  设备: ", "  Device: ") << ctx.props.deviceName << "\n";
        std::cout << i18n::tr("  Subgroup 大小: ", "  Subgroup Size: ") << SG_SIZE << "\n";
        std::cout << i18n::tr("  Basic:          ", "  Basic:          ") << (feat.subgroupBasic          ? "✓" : "✗") << "\n";
        std::cout << i18n::tr("  Vote:           ", "  Vote:           ") << (feat.subgroupVote           ? "✓" : "✗") << "\n";
        std::cout << i18n::tr("  Arithmetic:     ", "  Arithmetic:     ") << (feat.subgroupArithmetic     ? "✓" : "✗") << "\n";
        std::cout << i18n::tr("  Ballot:         ", "  Ballot:         ") << (feat.subgroupBallot         ? "✓" : "✗") << "\n";
        std::cout << i18n::tr("  Shuffle:        ", "  Shuffle:        ") << (feat.subgroupShuffle        ? "✓" : "✗") << "\n";
        std::cout << i18n::tr("  ShuffleRelative:", "  ShuffleRelative:") << (feat.subgroupShuffleRelative ? "✓" : "✗") << "\n";
        std::cout << i18n::tr("  Clustered:      ", "  Clustered:      ") << (feat.subgroupClustered      ? "✓" : "✗") << "\n";
        std::cout << i18n::tr("  Quad:           ", "  Quad:           ") << (feat.subgroupQuad           ? "✓" : "✗") << "\n";
        if (feat.subgroupSizeControl) {
            std::cout << i18n::tr("  Subgroup Size Control: ✓ (min=", "  Subgroup Size Control: ✓ (min=")
                      << feat.minSubgroupSize << i18n::tr(", max=", ", max=") << feat.maxSubgroupSize << ")\n";
        } else {
            std::cout << i18n::tr("  Subgroup Size Control: ✗\n", "  Subgroup Size Control: ✗\n");
        }

        // ── 准备测试数据 ────────────────────────────────────────────────
        const uint32_t N = 64;   // 大多数模式使用 64
        const uint32_t SG_N = 32; // softmax 使用 32（必须 ≤ subgroupSize）
        std::mt19937 rng(314);
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
        std::uniform_real_distribution<float> dist11(-1.0f, 1.0f);

        std::vector<float> testData(N);
        for (auto& v : testData) v = dist01(rng);

        std::vector<float> sgData(SG_N);
        for (auto& v : sgData) v = dist11(rng);

        Buffer bIn  = makeStorage(ctx, N * 4);
        Buffer bOut = makeStorage(ctx, N * 4 * 4); // 每线程 4 个 float
        Buffer bSgIn  = makeStorage(ctx, SG_N * 4);
        Buffer bSgOut = makeStorage(ctx, SG_N * 4);

        // 命名 push_constant（dispatch 需要稳定地址，不能取临时量地址）
        const PC16 pcN{N};
        const PC16 pcSg{SG_N};

        std::cout << i18n::tr("\n=== 测试 Subgroup 操作 ===\n", "\n=== Testing Subgroup Operations ===\n");

        // ── Mode 0: Basic Info ──────────────────────────────────────────
        if (feat.subgroupBasic) {
            vkUpload(ctx, bIn, testData.data(), N * 4);
            auto pass = makePass(ctx, "subgroup_ops.spv", 0, PC16{N});
            pass.bindBuffers({{0, bIn.buf}, {1, bOut.buf}});
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pcN, sizeof(PC16), 1, 1, 1); });
            auto raw = vkDownload(ctx, bOut, N * 4 * 4);
            const float* out = reinterpret_cast<const float*>(raw.data());

            // 验证：所有线程的 subgroupSize 应一致 = SG_SIZE
            uint32_t errs = 0;
            for (uint32_t i = 0; i < N; ++i) {
                float sgsz = out[i * 4 + 0];
                if ((uint32_t)sgsz != SG_SIZE) ++errs;
            }
            checkErr(errs, i18n::tr("Mode 0 Basic", "Mode 0 Basic"));
            std::cout << "    gl_SubgroupSize=" << (uint32_t)out[0]
                      << "  gl_NumSubgroups=" << (uint32_t)out[2] << "\n";
            pass.destroy();
        } else {
            std::cout << i18n::tr("  Mode 0 Basic: 不支持\n", "  Mode 0 Basic: not supported\n");
        }

        // ── Mode 1: Vote ────────────────────────────────────────────────
        if (feat.subgroupVote) {
            vkUpload(ctx, bIn, testData.data(), N * 4);
            auto pass = makePass(ctx, "subgroup_ops.spv", 1, PC16{N});
            pass.bindBuffers({{0, bIn.buf}, {1, bOut.buf}});
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pcN, sizeof(PC16), 1, 1, 1); });
            auto raw = vkDownload(ctx, bOut, N * 4 * 4);
            const float* out = reinterpret_cast<const float*>(raw.data());

            // 验证：每个线程的 ballot 应相同（同一 subgroup）
            uint32_t refBallot = (uint32_t)out[0];
            uint32_t errs = 0;
            for (uint32_t i = 0; i < N; ++i) {
                // 同一 subgroup 内 ballot 应一致
                // 对于 64 线程工作组，subgroupSize=32 时有 2 个 subgroup
                // 我们只检查第一个 subgroup 内的线程
                if (i < SG_SIZE) {
                    if ((uint32_t)out[i * 4] != refBallot) ++errs;
                }
            }
            checkErr(errs, i18n::tr("Mode 1 Vote", "Mode 1 Vote"));
            std::cout << "    ballot=" << (uint32_t)out[0]
                      << "  all=" << (out[1] > 0.0f ? "true" : "false")
                      << "  any=" << (out[2] > 0.0f ? "true" : "false") << "\n";
            pass.destroy();
        } else {
            std::cout << i18n::tr("  Mode 1 Vote: 不支持\n", "  Mode 1 Vote: not supported\n");
        }

        // ── Mode 2: Arithmetic ──────────────────────────────────────────
        if (feat.subgroupArithmetic) {
            vkUpload(ctx, bIn, testData.data(), N * 4);
            auto pass = makePass(ctx, "subgroup_ops.spv", 2, PC16{N});
            pass.bindBuffers({{0, bIn.buf}, {1, bOut.buf}});
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pcN, sizeof(PC16), 1, 1, 1); });
            auto raw = vkDownload(ctx, bOut, N * 4 * 4);
            const float* out = reinterpret_cast<const float*>(raw.data());

            // CPU 参考：第一个 subgroup 内求和
            float refSum = 0.0f, refMin = 1e30f, refMax = -1e30f;
            for (uint32_t i = 0; i < std::min(N, SG_SIZE); ++i) {
                refSum += testData[i];
                refMin = std::min(refMin, testData[i]);
                refMax = std::max(refMax, testData[i]);
            }
            uint32_t errs = 0;
            if (std::fabs(out[0] - refSum) > 1e-4f * std::max(1.0f, refSum)) ++errs;
            if (std::fabs(out[1] - refMin) > 1e-6f) ++errs;
            if (std::fabs(out[2] - refMax) > 1e-6f) ++errs;
            checkErr(errs, i18n::tr("Mode 2 Arithmetic", "Mode 2 Arithmetic"));
            std::cout << "    subgroupAdd=" << out[0] << "  (ref=" << refSum << ")\n"
                      << "    subgroupMin=" << out[1] << "  subgroupMax=" << out[2] << "\n";
            pass.destroy();
        } else {
            std::cout << i18n::tr("  Mode 2 Arithmetic: 不支持\n", "  Mode 2 Arithmetic: not supported\n");
        }

        // ── Mode 3: Ballot ──────────────────────────────────────────────
        if (feat.subgroupBallot) {
            vkUpload(ctx, bIn, testData.data(), N * 4);
            auto pass = makePass(ctx, "subgroup_ops.spv", 3, PC16{N});
            pass.bindBuffers({{0, bIn.buf}, {1, bOut.buf}});
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pcN, sizeof(PC16), 1, 1, 1); });
            auto raw = vkDownload(ctx, bOut, N * 4 * 4);
            const float* out = reinterpret_cast<const float*>(raw.data());

            // 验证：统计 > 0 的线程数应与 bitCount 一致
            uint32_t refCnt = 0;
            for (uint32_t i = 0; i < std::min(N, SG_SIZE); ++i) {
                if (testData[i] > 0.0f) ++refCnt;
            }
            uint32_t errs = 0;
            if ((uint32_t)out[1] != refCnt) ++errs;
            checkErr(errs, i18n::tr("Mode 3 Ballot", "Mode 3 Ballot"));
            std::cout << "    ballot=" << (uint32_t)out[0]
                      << "  bitCount=" << (uint32_t)out[1]
                      << "  (ref=" << refCnt << ")\n";
            pass.destroy();
        } else {
            std::cout << i18n::tr("  Mode 3 Ballot: 不支持\n", "  Mode 3 Ballot: not supported\n");
        }

        // ── Mode 4: Shuffle ─────────────────────────────────────────────
        if (feat.subgroupShuffle) {
            vkUpload(ctx, bIn, testData.data(), N * 4);
            auto pass = makePass(ctx, "subgroup_ops.spv", 4, PC16{N});
            pass.bindBuffers({{0, bIn.buf}, {1, bOut.buf}});
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pcN, sizeof(PC16), 1, 1, 1); });
            auto raw = vkDownload(ctx, bOut, N * 4 * 4);
            const float* out = reinterpret_cast<const float*>(raw.data());

            // 验证：广播值 = testData[0]（第一个 subgroup 的 lane 0）
            // 洗牌：out[2] 应等于下一个线程的原始值
            uint32_t errs = 0;
            for (uint32_t i = 0; i < std::min(N, SG_SIZE); ++i) {
                if (std::fabs(out[i * 4 + 1] - testData[0]) > 1e-6f) ++errs;
                // subgroupShuffle(v, (sgid+1)%sgsz): out[i*4+2] should equal testData[(i+1)%SG_SIZE]
                uint32_t next = (i + 1) % SG_SIZE;
                if (std::fabs(out[i * 4 + 2] - testData[next]) > 1e-6f) ++errs;
            }
            checkErr(errs, i18n::tr("Mode 4 Shuffle", "Mode 4 Shuffle"));
            std::cout << "    [0] original=" << out[0]
                      << "  broadcast=" << out[1]
                      << "  shuffle(next)=" << out[2] << "\n";
            pass.destroy();
        } else {
            std::cout << i18n::tr("  Mode 4 Shuffle: 不支持\n", "  Mode 4 Shuffle: not supported\n");
        }

        // ── Mode 5: Softmax (subgroup) ──────────────────────────────────
        if (feat.subgroupArithmetic) {
            vkUpload(ctx, bSgIn, sgData.data(), SG_N * 4);
            auto pass = makePass(ctx, "subgroup_ops.spv", 5, PC16{SG_N});
            pass.bindBuffers({{0, bSgIn.buf}, {1, bSgOut.buf}});
            vkSubmit(ctx, [&](vk::CommandBuffer cmd) { pass.dispatch(cmd, &pcSg, sizeof(PC16), 1, 1, 1); });
            auto raw = vkDownload(ctx, bSgOut, SG_N * 4);
            const float* out = reinterpret_cast<const float*>(raw.data());

            // CPU 参考
            std::vector<float> ref(SG_N);
            float m = -1e30f, sum = 0.0f;
            for (uint32_t i = 0; i < SG_N; ++i) m = std::max(m, sgData[i]);
            for (uint32_t i = 0; i < SG_N; ++i) { ref[i] = std::exp(sgData[i] - m); sum += ref[i]; }
            for (uint32_t i = 0; i < SG_N; ++i) ref[i] /= sum;

            uint32_t errs = 0;
            for (uint32_t i = 0; i < SG_N; ++i) {
                if (std::fabs(out[i] - ref[i]) > 1e-4f) ++errs;
            }
            checkErr(errs, i18n::tr("Mode 5 Softmax (subgroup)", "Mode 5 Softmax (subgroup)"));
            pass.destroy();
        } else {
            std::cout << i18n::tr("  Mode 5 Softmax: 不支持 (需要 Arithmetic)\n", "  Mode 5 Softmax: not supported (needs Arithmetic)\n");
        }

        // ── 性能对比：Subgroup Softmax vs Shared Memory Softmax ─────────
        std::cout << i18n::tr("\n=== 性能对比: Subgroup Softmax vs 共享内存 Softmax ===\n",
                              "\n=== Benchmark: Subgroup Softmax vs Shared Memory Softmax ===\n");
        const uint32_t BENCH_N = 32;
        const uint32_t ITERS = 1000;
        const PC16 pcBench{BENCH_N};

        if (feat.subgroupArithmetic) {
            // 准备数据
            std::vector<float> benchData(BENCH_N);
            for (auto& v : benchData) v = dist11(rng);
            Buffer bBenchIn  = makeStorage(ctx, BENCH_N * 4);
            Buffer bBenchOut = makeStorage(ctx, BENCH_N * 4);
            vkUpload(ctx, bBenchIn, benchData.data(), BENCH_N * 4);

            // ── Subgroup Softmax ──
            auto passSG = makePass(ctx, "subgroup_ops.spv", 5, PC16{BENCH_N});
            passSG.bindBuffers({{0, bBenchIn.buf}, {1, bBenchOut.buf}});

            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t iter = 0; iter < ITERS; ++iter) {
                vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                    passSG.dispatch(cmd, &pcBench, sizeof(PC16), 1, 1, 1);
                });
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            auto sgUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            std::cout << i18n::tr("  Subgroup softmax:  ", "  Subgroup softmax:  ")
                      << sgUs / (double)ITERS << " μs/iter\n";
            passSG.destroy();

            // ── Shared Memory Softmax ──
            auto passSM = makePass(ctx, "softmax_smem.spv", 0, PC16{BENCH_N});
            passSM.bindBuffers({{0, bBenchIn.buf}, {1, bBenchOut.buf}});

            auto t2 = std::chrono::high_resolution_clock::now();
            for (uint32_t iter = 0; iter < ITERS; ++iter) {
                vkSubmit(ctx, [&](vk::CommandBuffer cmd) {
                    passSM.dispatch(cmd, &pcBench, sizeof(PC16), 1, 1, 1);
                });
            }
            auto t3 = std::chrono::high_resolution_clock::now();
            auto smUs = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            std::cout << i18n::tr("  共享内存 softmax: ", "  Shared mem softmax: ")
                      << smUs / (double)ITERS << " μs/iter\n";

            if (sgUs < smUs) {
                double ratio = (double)smUs / (double)sgUs;
                std::cout << i18n::tr("  Subgroup 版本快了 ", "  Subgroup version is ")
                          << ratio << i18n::tr(" 倍\n", "x faster\n");
            } else {
                double ratio = (double)sgUs / (double)smUs;
                std::cout << i18n::tr("  共享内存版本快了 ", "  Shared mem version is ")
                          << ratio << i18n::tr(" 倍\n", "x faster\n");
            }

            // 验证结果一致
            auto raw2 = vkDownload(ctx, bBenchOut, BENCH_N * 4);
            const float* out2 = reinterpret_cast<const float*>(raw2.data());
            // 用最后一个 dispatch 的结果
            float mCPU = -1e30f, sumCPU = 0.0f;
            for (uint32_t i = 0; i < BENCH_N; ++i) mCPU = std::max(mCPU, benchData[i]);
            for (uint32_t i = 0; i < BENCH_N; ++i) { float e = std::exp(benchData[i] - mCPU); sumCPU += e; }
            uint32_t errs = 0;
            for (uint32_t i = 0; i < BENCH_N; ++i) {
                float refVal = std::exp(benchData[i] - mCPU) / sumCPU;
                if (std::fabs(out2[i] - refVal) > 1e-4f) ++errs;
            }
            std::cout << i18n::tr("  共享内存版结果验证: ", "  Shared mem result check: ")
                      << (errs == 0 ? i18n::tr("通过", "PASS") : i18n::tr("失败", "FAIL"))
                      << " (" << i18n::tr("错误数=", "errors=") << errs << ")\n";

            vkDestroyBuffer(ctx, bBenchIn);
            vkDestroyBuffer(ctx, bBenchOut);
        } else {
            std::cout << i18n::tr("  跳过 benchmark (不支持 Arithmetic)\n",
                                  "  Skipping benchmark (Arithmetic not supported)\n");
        }

        // ── 清理 ────────────────────────────────────────────────────────
        vkDestroyBuffer(ctx, bIn);
        vkDestroyBuffer(ctx, bOut);
        vkDestroyBuffer(ctx, bSgIn);
        vkDestroyBuffer(ctx, bSgOut);
        vkDestroyContext(ctx);
        std::cout << i18n::tr("\n[ch16] Done.\n", "\n[ch16] Done.\n");
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}