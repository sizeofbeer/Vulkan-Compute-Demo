# Vulkan 神经网络计算加速 · 从零到精通课程

> 一套「代码 + 文档」配套、逐章递进的实战课程。每一章 = 一个可独立编译运行的工程 + 一份讲解文档。
> 目标：从「连 Vulkan 是什么都不清楚」带到「能自研一个类 ncnn / 类 ggml-vulkan 的推理引擎」。

---

## 多语言支持 / Bilingual

- **文档**：每章文档均有中文（`<章>.md`）与英文（`<章>.en.md`）两个版本；本 README 也有 [英文版 `README.en.md`](README.en.md)。
- **程序输出**：所有章节的可执行程序默认输出**中文**；加 `--lang en`（或设置环境变量 `COURSE_LANG=en`）即切换为**英文**。例如：

  ```bash
  # Windows 下可执行文件带 .exe，Linux/macOS 下无后缀
  ./build/ch01_vector_add/ch01_vector_add            # 中文输出
  ./build/ch01_vector_add/ch01_vector_add --lang en  # 英文输出
  ```

## 0. 课程定位与前提

- **语言**：C++17 + GLSL（计算着色器）。C++ 用 `vulkan.hpp`（Vulkan 官方 C++ 封装）。
- **平台**：Windows / Linux 均可，需 **Vulkan SDK ≥ 1.2** 与一块支持 compute 的 GPU。
- **工具链**：CMake ≥ 3.16、`glslc`（Vulkan SDK 自带，编译 `.comp` → `.spv`）、Validation Layers（调试必开）。
- **参考对比**：每一阶段都会对照 **ncnn**（移动端 CV）与 **llama.cpp / ggml-vulkan**（桌面 LLM）的同类实现，告诉你工业级代码怎么写。

### 工程结构（统一模板）

```
course/
├── README.md                 # 本文件：课程总纲
├── CMakeLists.txt            # 顶层：拉起所有章节
├── cmake/helpers.cmake       # add_chapter() 宏：自动编译着色器、链接 Vulkan
├── common/                   # 共享 Vulkan 封装（随课程演进，越往后越像一个小引擎）
│   ├── vk_base.h / .cpp      # instance/device/buffer/shader/上传下载/一键提交
│   └── CMakeLists.txt
├── ch01_vector_add/          # 第 N 章
│   ├── CMakeLists.txt
│   ├── main.cpp              # 本章主程序（聚焦概念，样板交给 common）
│   ├── shaders/*.comp        # 本章计算着色器
│   └── docs/chNN_xxx.md      # 本章文档（概念 + 代码讲解 + 练习）
└── ...
```

构建方式（任意章节或整体）：

```bash
cd course
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# 运行某一章（着色器已自动编译到 build 目录）
./build/ch01_vector_add/ch01_vector_add
```

---

## 1. 课程大纲（零基础 → 精通）

### 第一部分：环境与最小闭环（上手）
| 章节 | 主题 | 你将学会 | 对应工业级 |
|------|------|----------|------------|
| **ch00** | 环境搭建与设备枚举 | 装 SDK、开 Validation Layers、枚举 GPU/队列、打印设备能力 | 一切的起点 |
| **ch01** | 第一个计算着色器：向量加法 | buffer 创建、shader 编译/加载、dispatch、读回结果、push constant | 最小可运行 kernel |
| **ch02** | 同步模型：fence 与 barrier | 手写命令缓冲、submit+等 fence、算子间 pipeline barrier（解决读写冒险） | ncnn 的 `VkCompute::submit_and_wait` |
| **ch03** | 显存管理入门 | 内存类型（host-visible / device-local / coherent）、staging 中转上传下载、FP16 存储 | ncnn `VkStagingAllocator` |

### 第二部分：神经网络算子（常用算子实现）
| 章节 | 主题 | 你将学会 | 对应工业级 |
|------|------|----------|------------|
| **ch04** | 矩阵乘 GEMM | 2D block tiling、共享内存、合并访存、时间戳测时、**Cooperative Matrix 加速** | ncnn/llama 的 matmul 核心 |
| **ch05** | 卷积 | 直接卷积 → im2col+GEMM → **Winograd F(2×2,3×3)** 三种实现、NC4HW4 打包 | ncnn 卷积族 |
| **ch06** | 逐元素算子 | ReLU / GELU / SiLU，带宽受限算子优化 | 融合激活 |
| **ch07** | 归一化 | LayerNorm / RMSNorm，reduction 与 workgroup 协作 | Transformer 归一化 |
| **ch08** | Softmax 与注意力 | 数值稳定 softmax、3-kernel 简化 attention、**FlashAttention 融合 kernel** | FlashAttention |

### 第三部分：性能、量化与工程化（精通）
| 章节 | 主题 | 你将学会 | 对应工业级 |
|------|------|----------|------------|
| **ch09** | 性能分析方法论 | GPU timestamp query、Roofline 模型、Nsight/RGP 实测 | 算子级 profiling |
| **ch10** | 显存管理进阶 | 内存池 / blob 分配器、权重常驻、buffer 复用 | ncnn `VkBlobAllocator` |
| **ch11** | 量化与融合反量化 | int8 + **Q4_0/Q8_0** 权重、GPU 上就地反量化、融合 matmul | llama.cpp `dequant_funcs` |
| **ch12** | 图执行与调度 | 录制命令、延迟 barrier、批提交、多线程队列 | ggml 计算图 |
| **ch13** | 综合项目 | 用自研引擎跑通一个小模型（MNIST 或 TinyLLM），对比 ncnn/llama.cpp | 完整推理引擎 |

### 第四部分：GPU 高级编程与前沿技术（精通 → 专家）
| 章节 | 主题 | 你将学会 | 对应工业级 |
|------|------|----------|------------|
| **ch14** | LLM 推理引擎 | 完整 LLM 推理 pipeline（tokenizer→embedding→transformer→输出），CPU 参考实现 | ncnn / llama.cpp 完整引擎 |
| **ch15** | **Cooperative Matrix 编程** | Tensor Core 编程、`coopmatMulAdd`、矩阵片段、特化常量握手、硬件约束 | ncnn `gemm_cm.comp` / ggml `mul_mm_cm2.comp` |
| **ch16** | **Subgroup 编程** | warp 级操作：shuffle/broadcast/reduce/ballot、warp-level softmax、subgroup size control | ncnn/ggml 的 subgroup 优化 |
| **ch17** | **FlashAttention 深入** | 在线 softmax、分块策略、subgroup 协作加载、融合 kernel 设计 | ncnn `sdpa_fa.comp` / ggml `flash_attn.comp` |
| **ch18** | **量化体系深入** | Q4_0/Q8_0 block-wise 量化、融合反量化 GEMM、压缩比与精度权衡 | ggml `dequant_funcs` |
| **ch19** | **MoE 推理引擎** | Router (top-k)、Expert dispatch、Combine、负载均衡、GPU 上 MoE 推理 | ncnn-MoE-Runtime / ggml `MUL_MAT_ID` |
| **ch20** | **前沿量化格式** | K-quant 家族 (Q2_K~Q6_K)、MXFP4 微观缩放 (E2M1)、超块结构 | ggml 最新量化格式 |

---

## 2. 当前进度（已全部完成 ✅）

- ✅ **课程设计总纲**（本文件）
- ✅ **common/ 共享封装库**（随课程演进：`vkInit/Buffer/上传下载` → `ComputePass`（mini-VkCompute）→ `DeviceArena`（显存池）→ `Graph`（计算图））
- ✅ **ch00 环境枚举**（设备/队列/内存/limits/特性探测报告）
- ✅ **ch01 向量加法**（最小计算闭环：buffer→上传→管线→dispatch→读回）
- ✅ **ch02 同步模型**（fence + pipeline barrier 手写，解决读写冒险）
- ✅ **ch03 显存管理**（内存类型 / staging 中转 / FP16 存储 buffer 往返）
- ✅ **ch04 GEMM**（2D block tiling + 共享内存，首次用 ComputePass）
- ✅ **ch05 卷积**（直接卷积 vs im2col+GEMM，结果一致验证）
- ✅ **ch06 逐元素算子**（ReLU/GELU/SiLU 参数化 kernel + 融合动机）
- ✅ **ch07 归一化**（RMSNorm，工作组内共享内存归约）
- ✅ **ch08 Softmax/注意力**（数值稳定 softmax + 3-kernel 简化注意力）
- ✅ **ch09 性能分析**（GPU 时间戳测时 + Roofline + 厂商工具）
- ✅ **ch10 显存管理进阶**（DeviceArena bump 分配器，权重常驻/激活复用）
- ✅ **ch11 量化与融合反量化**（int8 权重 + 内核内就地反量化 GEMM）
- ✅ **ch12 图执行与调度**（多算子录进一条命令缓冲 + barrier + 一次提交）
- ✅ **ch13 综合项目**（两层 MLP 前向：常驻权重 + GEMM + 偏置 + ReLU + 图）
- ✅ **ch14 LLM 推理引擎**（Tokenizer + Embedding + Transformer Block + 输出采样，CPU 参考实现）
- ✅ **ch15 Cooperative Matrix 编程**（Tensor Core GEMM，`coopmatMulAdd`，特化常量握手）
- ✅ **ch16 Subgroup 编程**（warp 级操作：shuffle/broadcast/reduce/ballot，warp-level softmax）
- ✅ **ch17 FlashAttention 深入**（在线 softmax，分块策略，融合 kernel）
- ✅ **ch18 量化体系深入**（Q4_0/Q8_0 block-wise 量化，融合反量化 GEMM）
- ✅ **ch19 MoE 推理引擎**（Router→Expert→Combine，top-k 路由，GPU 上 MoE 推理）
- ✅ **ch20 前沿量化格式**（K-quant 家族 Q2_K~Q6_K，MXFP4 微观缩放 E2M1）

> 全部 **20 章**均通过 **glslc 着色器编译** 与 **CMake 全量 C++ 构建**（`cmake -B build && cmake --build build` 产出 20 个可执行文件）。运行需一块支持 compute 的 Vulkan GPU 与驱动；调试请开启 Validation Layers（见 ch00）。

每一章的 `docs/` 都包含「概念讲解 → 代码逐段拆解 → 练习/进阶」三段式内容，建议**先读 docs 再读 main.cpp，然后自己改一处参数重跑**。

---

## 3. 学习建议

1. 不要跳过 ch01~ch03：它们建立的「命令缓冲 / 同步 / 内存」心智模型，后面每一章都在用。
2. 每章务必**开着 Validation Layers 跑一遍**——90% 的「结果错」都是 barrier 或内存可见性问题，Layers 会直接报出来。
3. 把自研实现和同章节标注的「工业级对应」对照读：你会看懂 ncnn / llama.cpp 为什么那么写。
4. 真正的精通来自 ch13 综合项目：当你能把自己的引擎跑通一个小模型，并和 ncnn/llama.cpp 比延迟，课程目标就达成了。

---

## 4. 参考资源

### 参考工业级实现

本课程每章都标注了对应的工业级实现。以下是调研并参考的本地仓库版本：

| 仓库 | 版本 | 关键特性 |
|------|------|---------|
| **ncnn** | `20260526-33-g946fe3fb` | 15+ 个 Cooperative Matrix 着色器（GEMM/卷积/Winograd/SDPA）、FlashAttention、Winograd F(2×2)/F(4×4)、PipelineCache、多级分配器、多队列 |
| **ncnn MoE Runtime** | `3240ff4` | 独立 MoE 推理引擎：MoE IR、Expert 多级缓存（ARC+Vulkan Victim）、异构调度（Dense GPU + Expert CPU/GPU）、跨 Session 微批、MXFP4 多平台内核 |
| **llama.cpp / ggml-vulkan** | 持续演进 | 20+ 量化格式（Q4_0~Q8_0、K-quant、I-quant、MXFP4、NVFP4）、Cooperative Matrix 2（decode vector）、FlashAttention 三路径（CM1/CM2）、MoE、SSM、Subgroup 深度使用 |

### 本课程中源自调研的前沿技术

| 技术 | 来源 | 课程章节 |
|------|------|---------|
| Cooperative Matrix | ncnn `gemm_cm.comp` / ggml `mul_mm_cm2.comp` | ch04（扩展）+ ch15 |
| Subgroup 操作 | ncnn/ggml 的 subgroup 优化 | ch16 |
| FlashAttention | ncnn `sdpa_fa.comp` / ggml `flash_attn.comp` | ch08（扩展）+ ch17 |
| Q4_0/Q8_0 量化 | ggml `dequant_funcs` | ch11（扩展）+ ch18 |
| K-quant/MXFP4 | ggml 最新量化格式 | ch20 |
| Winograd 卷积 | ncnn 卷积族 | ch05（扩展） |
| MoE 推理 | ncnn-MoE-Runtime / ggml `MUL_MAT_ID` | ch19 |
