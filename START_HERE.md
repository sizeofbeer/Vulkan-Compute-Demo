# Vulkan 计算加速课程 · 入门导读（从零到自研推理引擎）

> 本文档是仓库的"使用说明 + 设计哲学"合并读本。目标读者：准备学这套 21 章课程、想理解**为什么这么编排**、以及**怎么开始**的人。
>
> 读完本文档，你应该能：判断先学哪几章、跳过哪几章、之后翻工业代码时该去哪一份对照。
>
> **2026-11-08 同步**：第 7 节的「读源码导航表」与全课程 docs 已按 ncnn `20261020-52-g00357e2f`、ncnn-compiler `867c6ab`、ncnn-MoE-Runtime `63745f1`、llama.cpp `a691b23` 重新核对；旧版的 `init_device()` / `VkCompute::dispatch()` / `vk-device.cpp` / `iq-quants.c` / `ExpertDispatcher` 等导航均已修正。版本对照与完整新旧映射见 `README.md` 第 4 节。

---

## 0. 一句话目标

把"连 Vulkan 是什么都不清楚"的人带到**能自研一个类 ncnn / 类 ggml-vulkan 的推理引擎**。21 章 = 4 阶段递进，每章 = `main.cpp + shaders/*.comp + docs/<章>.md` 三件套，独立可编译运行。

---

## 1. 为什么编排是「4 阶段 × 21 章」而不是别的方式

设计者要解决一件反直觉的事：**「Vulkan 是显式 API」+「神经网络是大量算子」**两个特性叠加，新手 90% 卡在"不知道下一步该调哪个 API"，而不是"写不出算法"。所以编排服从一个原则：

> **先把 Vulkan 计算的"硬件骨架"跑通，再让骨架长出神经网络的"算子肌肉"，最后把肌肉训练成"推理引擎的关节"。**

四个阶段就是这条主链：

| 阶段 | 章节 | 解决的核心问题 | 终极交付 |
|------|------|----------------|---------|
| **一 · 骨架** | ch00–ch03 | **能不能跑**：Vulkan 上下文、buffer、同步、显存类型 | 一条命令能把数据送进 GPU、跑算子、读回来 |
| **二 · 肌肉** | ch04–ch08 | **算得对不对**：GEMM、卷积、激活、归一化、注意力 | 覆盖 90% NN 算子的"自研 kernel" |
| **三 · 关节** | ch09–ch13 | **跑得快不快、可不可工程化**：profile、显存池、量化、计算图、综合 | 一个能跑通小模型的前向推理 demo |
| **四 · 前沿** | ch14–ch20 | **做到工业级深度**：LLM 引擎、TensorCore、subgroup、FlashAttention、K-quant、MoE | 能对照读 ncnn / llama.cpp 代码的能力 |

### 编排逻辑要点（为什么这样排序，不是反过来）

- **ch00 单独作为"探测章"**——不写一行算子，因为后面所有章节都会被 `maxComputeWorkGroupInvocations / sharedMemorySize / shaderFloat16 / timestampPeriod` 这些**设备能力**卡住。先看你的卡能干什么，再决定后面怎么打。
- **ch01 必须紧跟 ch02**——ch01 跑通的是"裸 dispatch"，你不知道"为什么 GPU 的写后再读可能错"。ch02 把 fence + pipeline barrier 这件事讲透。之后所有章节再没回到同步——因为已经会了。
- **ch03 在 ch01 之后立刻讲"上传/读回"**——因为 ch01 用的是 `vkUpload` 这个黑盒，但你跑大模型时一定会卡在"上传好慢 / 显存不够"上，必须知道背后的内存类型与 staging。
- **ch04 是分水岭**——从这一章起，kernel 进入"有共享内存、有 tile 划分"的真优化阶段；同时引入 `ComputePass`（mini-VkCompute），把前面的样板再封装一层。所以**ch04 之前是"会用 API"，ch04 之后是"写高性能 kernel"**。
- **ch05–ch08 是"算子大家族"**——把这几章串起来就是**一整个 Transformer Block**（卷积 ch05 可选；激活/归一化/注意力是 LLM 核心）。ch08 第一次出现"在线 softmax / 3-kernel simplified attn"，是 ch17 的铺垫。
- **ch09 是"性能眼"**——从这一章起，每章都要测时、画 Roofline、做对比。不开 profile 永远不会优化。
- **ch10 显存池** + **ch11 量化** + **ch12 计算图** 是工程化三件套——任何真实推理引擎都要这三件事。ch13 综合项目把"小模型前向"跑通，**课程的"贯通点"就在 ch13**。
- **ch14 才是 LLM**——而且是**CPU 参考实现**，用来当"数值标尺"。后续 Vulkan kernel 一行一行对拍它。所以**不要跳过 ch14**，跳过它等于没有 ground truth。
- **ch15–ch20 是"对标工业"**——这六章每一章都对应一个 ncnn / ggml 内部的真实子系统（Tensor Core / warp 级 / FlashAttn / 量化 / MoE / K-quants）。到这一层你才算"会读 llama.cpp 后端源码"。

---

## 2. 章间依赖图

```
                            [ch00 环境探测]
                                  │ 设备能力决定后续上限
                                  ▼
                            [ch01 向量加法]
                                  │ 最小闭环：buffer→dispatch→读回
                                  ▼
                          [ch02 同步：fence/barrier]
                                  │ 写后读 hazard 必须解决
                                  ▼
                       [ch03 显存：内存类型/staging]
                                  │
              ┌───────────────────┼───────────────────────┐
              ▼                   ▼                       ▼
   [ch04 GEMM · tiling]   [ch05 卷积 · 直接/im2col]   [ch06 逐元素激活]
              │                   │                       │
              └───────────────────┼───────────────────────┘
                                  ▼
                       [ch07 归一化 RMSNorm]
                                  │
                                  ▼
              [ch08 简化注意力：softmax + QKᵀ]
                                  │  ┌─── (ch17 深入 FlashAttention)
                                  ▼  │
                       [ch09 性能分析] ◄── 这之后每章都带 profile
                                  │
              ┌───────────────────┼────────────────────────┐
              ▼                   ▼                        ▼
     [ch10 显存池 arena]   [ch11 量化 int8/Q4/Q8]   [ch12 计算图 Graph]
              │                   │                        │
              └───────────────────┼────────────────────────┘
                                  ▼
                       [ch13 综合：小 MLP 前向]
                                  │  ← 课程主线"贯通"
                                  ▼
                       [ch14 LLM 推理 CPU 参考]  ← 后续 Vulkan kernel 的数值标尺
                                  │
              ┌───────────────────┼────────────────────────┐
              ▼                   ▼                        ▼
   [ch15 CoopMat]   [ch16 Subgroup]   [ch17 FlashAttn]
              │           │                │
              └───────────┼────────────────┘
                              ▼
               [ch18 量化深入 Q4/Q8]
                              │
                              ▼
               [ch19 MoE 引擎]
                              │
                              ▼
               [ch20 K-quant / MXFP4]
```

实线 = 强依赖（必须按顺序读），分支 = 任选（学完主链后回头补）。

---

## 3. 五条"心智模型"（先记住，再看章节）

学 Vulkan 神经网络的人最容易在 5 个地方迷糊。把这 5 句话先刻进脑子，后面每章都会反复印证：

1. **Vulkan 没有"自动 buffer 管理"**——你要显式选"在 CPU 侧可见的 host memory"还是"在 GPU 显存里快但 CPU 看不见的 device-local"，并自己解决两者之间的搬运。**这条决定 ch01 到 ch13 的所有工程问题**。
2. **GPU 计算 = "录一条命令 + 一次 dispatch"**——`vkCmdBindPipeline` → `vkCmdBindDescriptorSets` → `vkCmdPushConstants` → `vkCmdDispatch` → `vkEndCommandBuffer` → `vkQueueSubmit`。**ch01–ch03 全在做这件事，ch04 起开始优化它**。
3. **工作组（workgroup）= 协作单元**——`local_size_x` 决定一个 group 内多少线程。组内可以共享 `shared` 变量、`barrier()`、用 subgroup 操作（ch16）。**跨组只能靠 pipeline barrier（ch02）+ 全局 buffer + dispatch 顺序**。
4. **Tensor Core 不是"自动的"**——你要显式写 `coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>`（ch15），并确认硬件 + 驱动 + glslc 都支持。**这是 ncnn `gemm_cm.comp` / ggml `mul_mm_cm2.comp` 的核心**。
5. **量化不是"减少精度"，是"减少带宽"**——int8/Q4/Q8 不换算力，省的是**显存带宽**和**模型常驻显存**。这正是 LLM 能跑在 4GB 显卡的根因。**融合反量化（ch11/ch18）是核心技巧：绝不物化 FP32 权重**。

---

## 4. 起步 30 分钟：环境 → ch00 → ch01

### 第 1 步 · 装工具链（10 分钟）

**Windows（你的环境）**
- 装 [Vulkan SDK ≥ 1.2](https://vulkan.lunarg.com/) → 自动设 `VULKAN_SDK` 环境变量
- 装显卡厂商驱动（NVIDIA/AMD/Intel 任一，要支持 Vulkan 1.2+）
- `cmake --version`（要 ≥ 3.16）；VS 2019/2022 含 MSVC

**Linux/macOS**：装 SDK + `apt install vulkan-tools mesa-vulkan-drivers` 之类。

### 第 2 步 · 全量构建（5 分钟）

```bash
cd c:/msys64/home/14616/Vulkan-Compute-Demo
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

顶层 `cmake/helpers.cmake` 的 `add_chapter()` 会**自动用 glslc 把每章 `shaders/*.comp` 编译成 `.spv`**；找不到 glslc 时退化为复制已存在的 `.spv`。

### 第 3 步 · 跑 ch00 探测自己 GPU（2 分钟）

```bash
./build/ch00_environment/ch00_environment
```

记下三个数：
- `maxComputeWorkGroupInvocations`（线程上限，常 256/1024）
- `maxComputeSharedMemorySize`（共享内存，常 32/64 KB）
- `timestampPeriod`（ch09 用）

看一眼是否支持 `shaderFloat16`（决定 ch03/ch11 FP16 路径能否走）。

### 第 4 步 · 跑 ch01 跑通最小算子（5 分钟）

```bash
./build/ch01_vector_add/ch01_vector_add
```

预期输出：`a[i]+b[i] == c[i]` 全匹配，0 错误。

### 第 5 步 · 立刻做 1 个实验（5 分钟）

打开 `ch01_vector_add/shaders/vec_add.comp`，把加法改成 `a[i] * 2.0f + 1.0f`：

```bash
cmake --build build --target ch01_vector_add   # 自动 recompile .comp
./build/ch01_vector_add/ch01_vector_add
```

如果还能匹配，恭喜，你已经"读懂 GPU 着色器输入"了。**这就是本章 docs 里"动手实验 #4"的核心动作**。

> **整个课程都建议开启 Validation Layers 跑**（ch00 默认开）。它是 Vulkan 的"编译器 warning"，90% 的"结果错"都是 barrier 漏插、buffer 越界、descriptor 错绑——Layers 会直接报出来。这是这条路线里**最重要的调试姿势**。

---

## 5. 21 章速览表

每一章的「一句话获得」与「对照工业代码」：

| # | 主题 | 一句话获得 | 对照工业代码（2026-11 现行） |
|---|------|------------|-------------|
| **ch00** | 环境探测 | 拿到你的设备能力报告 | ncnn `create_gpu_instance()` + `get_gpu_device()->info`；ggml `vk_device_struct` |
| **ch01** | 向量加法 | 跑通第一条 GPU 算子 | 最小 kernel（无直接对应） |
| **ch02** | fence / barrier | 解决 GPU 写后读 hazard | ncnn `VkCompute::record_pipeline()` → `submit_and_wait()` |
| **ch03** | 显存管理 | 理解 device-local / host-visible / staging | ncnn `VkStagingAllocator::fastMalloc()` |
| **ch04** | GEMM | tile + 共享内存 + 时间戳 | ncnn `shader/gemm.comp` / `gemm_cm.comp` / `gemm_sg.comp` |
| **ch05** | 卷积 | 直接 / im2col+GEMM / Winograd F(2×2,3×3) | ncnn `convolution_vulkan.cpp` + `shader/convolution_*winograd*.comp` |
| **ch06** | 逐元素算子 | ReLU / GELU / SiLU + 融合动机 | 激活融合 |
| **ch07** | RMSNorm | 工作组内共享内存归约 | ncnn `shader/reduction.comp`（fp32 累加 + subgroup） |
| **ch08** | 简化注意 | 数值稳定 softmax + 3-kernel attn | （ch17 FlashAttn 铺垫） |
| **ch09** | 性能分析 | GPU timestamp + Roofline | 算子级 profiling |
| **ch10** | 显存池 | `DeviceArena` bump 分配器 + 权重常驻 | ncnn `VkBlobAllocator`（16MB 池）/ `VkWeightAllocator`（8MB） |
| **ch11** | 量化 | int8 / Q4_0 / Q8_0 + 融合反量化 GEMM | ggml `dequant_funcs.glsl` + `ggml-quants.c` |
| **ch12** | 计算图 | 多算子录一条命令 + 自动 barrier | ggml `ggml_backend_vk_graph_compute()`；ncnn-compiler fixed-run AOT |
| **ch13** | 综合 MLP | 权重常驻 + GEMM+bias+ReLU+图 | 完整推理引擎雏形 |
| **ch14** | LLM 引擎 | Qwen3-0.6B 端到端 CPU 自回归 | 数值 ground truth |
| **ch15** | CoopMat | `coopmatMulAdd`、Tensor Core 编程 | ncnn 17 个 `*_cm.comp` / ggml `mul_mm_cm2.comp` |
| **ch16** | Subgroup | shuffle / broadcast / ballot / warp-softmax | ncnn `reduction.comp` / ggml `flash_attn*.comp` |
| **ch17** | FlashAttn | 在线 softmax + 分块 + 融合 kernel + KV cache 两阶段 decode demo | ncnn `sdpa_fa{,_cm}.comp` / ggml `flash_attn{,_cm1,_cm2,decode_phase_*,split_k_reduce}.comp` |
| **ch18** | 量化深入 | Q4_0 / Q8_0 block-wise + 融合 GEMM | ggml `dequant_funcs{,_cm2}.glsl` + `dequant_*.comp` |
| **ch19** | MoE | Router top-k + Expert dispatch + Combine + 稀疏 Gather/Scatter demo | ncnn-MoE-Runtime `forward_router/forward_moe` / ggml `ggml_vk_mul_mat_id` |
| **ch20** | K-quant | Q2_K ~ Q6_K super-block + MXFP4 E2M1 + NVFP4 E4M3 两级缩放 demo | ggml 44 种类型：NVFP4 / TQ / Q1_0 / Q2_0 / F8_E4M3 |

---

## 6. 建议学习路径（三档）

### A.「我只想跑通最小推理引擎」（最短贯通路径）

```
ch00 → ch01 → ch02 → ch03 → ch04 → ch07 → ch08 → ch11 → ch10 → ch12 → ch13 → ch14（看 docs）
```

- **课时**：约 2 周（每天 2 小时）
- **目标产物**：能改 ch13 的 MLP，跑自己的小权重，对照 ncnn 解释为什么它这么写
- **跳过代价**：缺 ch05 卷积、ch06 激活、ch09 profile、ch15+ 前沿——但主线已经通

### B.「能自研类 ggml-vulkan 的推理引擎」（推荐路径）

```
A + ch05 → ch06 → ch09（养成 profile 习惯） → ch15 → ch16 → ch17 → ch18 → ch20
```

- **课时**：约 6–8 周
- **目标产物**：对照读 llama.cpp 后端源码，能改它的 kernel、能加新算子

### C.「能对标 ncnn 的 mobile 推理框架」

```
B + ch19（MoE 是大模型特化，ncnn MoE-Runtime 用得到）+ 持续对照 ncnn source 调试
```

- **目标产物**：理解为什么 llama.cpp 把设备、管线、图调度分别收敛到 `ggml-vulkan.cpp` / `ggml-vulkan-types.h` / `ggml_backend_vk_graph_compute()`，以及 ncnn-compiler 为何能把整张静态图 AOT 成模型专用 `.so`（见 ch12 新增小节）

---

## 7. 自研「类 ncnn / 类 ggml-vulkan」目标落地点

如果你最终要写自己的推理引擎，这套课程的每一章都对应工业代码的一个具体子系统。给你一张"读源码的导航表"——读完对应章节后，去读这个仓库的这份代码，会豁然开朗：

| 你的目标 | 该看的工业代码（现行路径） | 对照课程章节 |
|----------|----------------|--------------|
| 设备探测 / 能力决定 kernel 变体 | `ncnn/src/gpu.cpp::create_gpu_instance()`、`VulkanDevice::info`（`src/gpu.h`）；ggml `ggml-vulkan-types.h::vk_device_struct` | ch00 |
| 录制 + 一次性提交 | `ncnn/src/command.h::VkCompute::record_pipeline()` / `submit_and_wait()`（实现见 `command.cpp` 的 `delayed_records`） | ch02、ch04 起 |
| 上传 / 读回的 staging 隐藏 | `ncnn/src/allocator.h::VkStagingAllocator::fastMalloc()`；`VkWeightAllocator` 权重常驻 | ch03、ch10 |
| 权重常驻 + 显存分配 | `VkBlobAllocator`（`preferred_block_size` 默认 16MB）/ `VkWeightAllocator`（8MB） | ch10 |
| 计算图录制 + barrier | `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_backend_vk_graph_compute()` | ch12 |
| 静态图 AOT（无运行时解释器） | `ncnn-compiler/docs/ncnn-vulkan-backend-contract.md` + `tools/VulkanLifecycleCodegen.cpp`、`lib/Support/VulkanHostProgram.cpp` | ch12 |
| Tensor Core GEMM | `ncnn/src/layer/vulkan/shader/gemm_cm.comp`（17 个 `*_cm.comp` 之一）；`ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_cm2.comp` | ch15 |
| FlashAttention kernel | 本课程 `ch17_flash_attn/main_kvcache.cpp` + `shaders/{kvcache_append,flash_attn_decode_phase1,flash_attn_decode_phase2}.comp`；工业侧 `ncnn/.../shader/sdpa_fa.comp` + `sdpa_fa_cm.comp`、`ggml/.../vulkan-shaders/flash_attn*.comp` | ch17 |
| 量化反量化 + 融合 | 本课程 `ch20_frontier_quant/main_nvfp4.cpp` + `shaders/nvfp4_gemm.comp`；工业侧 `ggml/.../vulkan-shaders/dequant_funcs.glsl`、`dequant_funcs_cm2.glsl`、`dequant_*.comp`，结构定义 `ggml/src/ggml-quants.c` | ch11 / ch18 / ch20 |
| MoE 调度 | 本课程 `ch19_moe/main_sparse.cpp` + `shaders/{count_experts,expert_gather,expert_gemm_sparse,moe_combine_sparse}.comp`；工业侧 `ggml-vulkan.cpp::ggml_vk_mul_mat_id()` + `topk_moe.comp` / `count_experts.comp`，ncnn-MoE-Runtime `src/graph/router.h` + `src/engine/expert.h` + `src/kernels/vulkan/` | ch19 |
| K-quant / MXFP4 / NVFP4 | `ggml/include/ggml.h` 的 `GGML_TYPE_*`（44 种）+ `dequant_{q*,k*,iq*,mxfp4,nvfp4,tq*}.comp`；本课程 `ch20_frontier_quant/shaders/nvfp4_gemm.comp` | ch20 |
| 多队列 / 流水线缓存 | ncnn `VulkanDevice::acquire_queue()`、`PipelineCache::save_cache()/load_cache()` | ch09 / ch12 |

**判断你是否"出师"的硬指标（README 里也写了）：**

> 当你能用自己的引擎跑通一个小模型前向、并对照 ncnn / llama.cpp 比延迟、且能解释每条 kernel 设计的取舍——课程目标达成。

---

## 8. 调试与工具链速记

- **Validation Layers**：`VK_LAYER_KHRONOS_validation`（ch00 默认开） → barrier 漏插 / 越界 / 描述符绑错时直接报错。**永远开着调试**。
- **GPU 时间戳**：`vkCmdWriteTimestamp` + `vkGetQueryPoolResults`，单位换算成 ns 用设备的 `timestampPeriod`。ch09 详细说。
- **Nsight（NVIDIA）/ RGP（AMD）**：超过 ch09 后建议交叉验证课程的自测时间。
- **glslc**：Vulkan SDK 自带，CMake 已自动调用。改 `.comp` 后用 `cmake --build build --target <章>` 即重编译。
- **多语言**：所有可执行支持 `--lang en`。

---

## 9. 几条贯穿全课程的"为什么这么写"判断

- **不要跳过 ch01–ch03**——它们建的"命令缓冲 / 同步 / 内存"心智模型，后面每一章都在用。跳了就会在 ch04 翻车。
- **永远开 Validation Layers**——验证过的 barrier / 内存错误比手动 print 调试快 10 倍。
- **"工业级对应"是这门课的隐藏礼物**——每章 docs 末尾都点了对照代码，**那是"读完本章后去翻仓库"的导航**，别只看 docs 不翻 ncnn / ggml。
- **ch13 / ch14 是真正的毕业考**——ch13 检验你"组合能力"，ch14 检验你"读 + 改工业代码"能力。

---

## 10. 下一步动作

### 今天就要动起来（30 分钟）

1. `cmake -B build && cmake --build build -j` —— 看 24 个目标都成功（21 章主程序 + `ch19_moe_sparse` / `ch17_kvcache` / `ch20_nvfp4`）
2. `./build/ch00_environment/ch00_environment` —— 截屏你的设备能力（存 `device_report.txt`）
3. `./build/ch01_vector_add/ch01_vector_add` —— 看 `a+b==c` 全对
4. 打开 `ch01_vector_add/docs/ch01_vector_add.md` —— "动手实验 #4" 改 `.comp`，重 build，重跑

### 这一周想贯通主链

- 按 A 档路径走完 ch00–ch13（每天 2–3 章）
- 在 ch13 把 MLP 的权重换成自己的，跑出"过拟合玩具"
- 再回到 ch14 看 CPU 参考实现，对自己 Vulkan 路径做 A/B 对拍

### 想先纵深某一块

- **优化卷积**：跳到 ch05 + ch15（Winograd + CoopMat）
- **做 LLM 量化**：ch11 → ch18 → ch20
- **做 attention 极致**：ch08 → ch17 → ch16

---

## 附录 · 排错速查

| 现象 | 大概率原因 | 对照章节 |
|------|-----------|----------|
| GPU 结果错乱、偶尔对偶尔错 | 漏 pipeline barrier（写后读 hazard） | ch02、ch12 |
| Validation 报 "buffer too small" / "offset+size exceeds" | push constant 形状与 shader 不一致 | ch01、ch04 |
| 共享内存 kernel 链接失败 | `maxComputeSharedMemorySize` 不够；或 `local_size_x*y*z` 超 `maxComputeWorkGroupInvocations` | ch00、ch04 |
| 跑 ch03/ch11 FP16 路径挂 | `shaderFloat16` 未启用，需在 `VkPhysicalDeviceVulkan12Features` 里开 | ch00、ch03 |
| ch15 CoopMat 链接不过 | `GL_KHR_cooperative_matrix` 未 require，或驱动不支持 | ch00、ch15 |
| 性能差但 profile 显示带宽未满 | 没融合反量化、没复用 arena buffer | ch11、ch10 |
| 模型前向 logits 整体飘一个量级 | 量化 scale / zero-point 设错或漏 | ch11、ch18 |

---

需要我把其中某章再展开讲吗？比如：
- 「ch04 GEMM 共享内存 tiling 是怎么写出来的 + 手把手读 `.comp`」
- 「ch15 CoopMat 跟 ch04 GEMM 的本质区别」
- 「ch17 在线 softmax 的数学推导」

挑一个我接着展开。
