# 第 15 章 · 协作矩阵（Cooperative Matrix）编程

> 目标：理解 **VK_KHR_cooperative_matrix** 扩展，学会利用 GPU 上的 Tensor Core / Matrix Core / XMX 硬件单元加速矩阵乘法。本章实现一个同时包含 CM 路径和传统分块 GEMM 回退路径的矩阵乘，并对比两者的性能差异。

---

## 15.1 什么是 Cooperative Matrix / Tensor Core 硬件

### 15.1.1 传统 ALU 的瓶颈

传统 GPU 计算核心（CUDA Core / SP）执行的是 **标量** 运算：一个线程一次只能做一次乘加（FMA = 1 次乘法 + 1 次加法）。对于矩阵乘 `C[M,N] = A[M,K] * B[K,N]`，需要 `M * N * K` 次乘法和 `M * N * K` 次加法——总共 `2 * M * N * K` 次 FMA 操作。

当矩阵规模很大时（例如 LLM 的 `4096x4096`），即使有几千个 CUDA Core，也需要极多的指令周期。

### 15.1.2 专用矩阵乘法硬件

从 NVIDIA Volta 架构（2017）开始，GPU 引入了 **Tensor Core**——一种专门为 **小矩阵乘加**（`D = A * B + C`）设计的硬件单元。它在一个时钟周期内可以完成一个 **4×4 矩阵乘加**（64 个 FMA 操作），相当于 64 个标量 FMA 的吞吐。

类似地：
- **AMD**：CDNA 架构的 **Matrix Core**（Matrix FMA 指令）
- **Intel**：Xe 架构的 **XMX**（Xe Matrix eXtension）
- **Apple**：Metal 的 **Matrix Multiply** 支持

### 15.1.3 Vulkan 中的抽象：Cooperative Matrix

Vulkan 通过 **VK_KHR_cooperative_matrix** 扩展（以及早期的 **VK_NV_cooperative_matrix**）将这些硬件单元暴露给开发者。其核心思想是：

> 把一次矩阵乘加操作，从「N 个线程各自做 N 次标量 FMA」变成「一个 subgroup 协作完成一次小矩阵乘加」。

这个「小矩阵」在着色器中被称为 **矩阵片段（Matrix Fragment）**，由 Subgroup 内所有线程共同持有——每个线程只持有片段的一小部分数据。

---

## 15.2 VK_KHR_cooperative_matrix 扩展概述

### 15.2.1 扩展版本

| 扩展 | 状态 | 说明 |
|------|------|------|
| `VK_NV_cooperative_matrix` | 专有（NVIDIA） | 最早的 CM 扩展，仅 NVIDIA 支持 |
| `VK_KHR_cooperative_matrix` | 跨厂商标准 | 被 NVIDIA / AMD / Intel 广泛支持 |
| `VK_EXT_cooperative_matrix_maintenance1` | 补充 | 增加 flexible dimensions 等能力 |

### 15.2.2 核心概念

- **Cooperative Matrix（协作矩阵）**：一种特殊的 SPIR-V 类型，表示一个分布在 Subgroup 所有线程上的矩阵片段。
- **Matrix Fragment（矩阵片段）**：一个 `M_tile × N_tile` 的小矩阵，其数据由 Subgroup 内的所有线程协作持有。
- **coopmatMulAdd**：`C = C + A * B`，由硬件一次性完成，等价于 `M_tile * K_tile * N_tile` 次标量 FMA。
- **coopmatLoad / coopmatStore**：在全局内存和矩阵片段之间加载/存储数据，也是 Subgroup 集体操作。

### 15.2.3 GLSL 语法

```glsl
#extension GL_KHR_cooperative_matrix : require

// 定义片段类型
cooperative_matrix<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> fragA;
cooperative_matrix<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> fragB;
cooperative_matrix<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> fragC;

// 加载片段
cooperativeMatrixLoad(fragA, buffer, stride, gl_MatrixRowMajor, offset);
cooperativeMatrixLoad(fragB, buffer, stride, gl_MatrixRowMajor, offset);

// 矩阵乘加
fragC = cooperativeMatrixMulAdd(fragA, fragB, fragC);

// 存储结果
cooperativeMatrixStore(buffer, fragC, stride, gl_MatrixRowMajor, offset);
```

---

## 15.3 矩阵片段概念（Subgroup 作用域分布式存储）

### 15.3.1 为什么需要 Subgroup

矩阵乘 `C = A * B` 中，每个输出元素 `C[r][c]` 依赖 A 的第 r 行和 B 的第 c 列——这天然是 **跨线程的依赖**。如果让单个线程计算整个 16×16 的输出块，它需要读取 16×K 个 A 元素和 K×16 个 B 元素，这远远超出了单个线程的寄存器容量。

Cooperative Matrix 的解法是：**让 Subgroup 内所有线程协作**，每个线程只持有片段的一小部分，但 collective 地完成加载、计算和存储。

### 15.3.2 数据如何分布

假设一个 `16×16` 的矩阵片段，Subgroup 有 32 个线程：

- 每个线程持有 `16×16 / 32 = 8` 个元素
- 具体的分布方式由硬件实现决定（对开发者透明）
- 使用 `coopmatLoad` 和 `coopmatStore` 时，所有线程协作完成数据传输

### 15.3.3 与共享内存的对比

| 方面 | 传统 Tiling（ch04） | Cooperative Matrix |
|------|-------------------|-------------------|
| 数据存储 | 共享内存（显式搬入） | 寄存器（隐式分布） |
| 同步 | `barrier()` 显式同步 | Subgroup 集体操作隐式同步 |
| 计算 | 循环内标量 FMA | 单条 `coopmatMulAdd` 指令 |
| 硬件 | CUDA Core / SP | Tensor Core / Matrix Core |
| 编程复杂度 | 高（手动 tile + 边界） | 低（集体操作封装） |

---

## 15.4 内存布局（行主序 vs 列主序，stride）

### 15.4.1 行主序（Row-Major）

这是 C/C++ 的默认布局：`A[i][j]` 在内存中连续存储，`&A[i][j+1] = &A[i][j] + 1`。

对于 CM 加载/存储：

```glsl
cooperativeMatrixLoad(fragA, buffer, strideA, gl_MatrixRowMajor, offset);
```

- `strideA`：相邻两行之间的元素数（即 A 的列数 K）
- `offset`：起始位置的字节偏移（或元素索引，取决于具体语法）

### 15.4.2 列主序（Column-Major）

Fortran / MATLAB 的默认布局。在 CM 中通过 `gl_MatrixColumnMajor` 指定：

```glsl
cooperativeMatrixStore(buffer, fragC, strideC, gl_MatrixColumnMajor, offset);
```

### 15.4.3 Stride 对齐要求

硬件通常要求 stride 是某个对齐值（如 16、32、64 字节）的整数倍。如果 stride 不满足对齐，可能无法使用 CM 路径，需要填充（padding）或改用回退路径。

---

## 15.5 特化常量握手（Specialization Constants for Tile Sizes）

### 15.5.1 为什么需要特化常量

Cooperative Matrix 的片段尺寸（`M_tile × N_tile × K_tile`）必须编译期确定，因为不同的硬件支持不同的 tile 组合：

- **NVIDIA Tensor Core（Ampere+）**：16×16×16（FP16），16×8×16（BF16），16×8×8（INT8）
- **AMD Matrix Core（RDNA3+）**：16×16×16（FP16）
- **Intel XMX（Xe）**：8×8×8（FP16），8×8×8（INT8）

通过特化常量，可以在不重新编译着色器的情况下，为不同硬件选择不同的 tile 尺寸：

```glsl
layout(constant_id = 0) const uint TILE_M = 16;
layout(constant_id = 1) const uint TILE_N = 16;
layout(constant_id = 2) const uint TILE_K = 16;
```

### 15.5.2 C++ 端设置

```cpp
std::array<uint32_t, 3> specData = {16, 16, 16};
auto specInfo = vkMakeSpecializationInfo(specData);
pass.prepareWithSpec(ctx, spv, bindings, pcs, specInfo.info);
```

### 15.5.3 硬件驱动的 tile 选择

在工业级引擎中（如 ncnn、ggml），会先查询设备的 `VkCooperativeMatrixPropertiesKHR`，得到硬件支持的 tile 列表，然后选最优的：

```cpp
// 查询支持的片段类型
uint32_t propCount = 0;
ctx.physical.getCooperativeMatrixPropertiesKHR(&propCount, nullptr);
std::vector<vk::CooperativeMatrixPropertiesKHR> props(propCount);
ctx.physical.getCooperativeMatrixPropertiesKHR(&propCount, props.data());
```

---

## 15.6 GPU 硬件约束

### 15.6.1 NVIDIA Tensor Core

| 架构 | Tensor Core 版本 | 支持的数据类型 | Tile 尺寸 |
|------|-----------------|---------------|-----------|
| Volta (V100) | 第 1 代 | FP16 | 16×16×16 |
| Turing (T4) | 第 2 代 | FP16, INT8 | 16×16×16, 8×8×8 |
| Ampere (A100) | 第 3 代 | FP16, BF16, TF32, INT8, INT4 | 16×16×16, 16×8×16, 8×8×8 |
| Hopper (H100) | 第 4 代 | 同上 + FP8 | 同上 |
| Ada Lovelace (RTX 40xx) | 第 4 代 | 同上 | 同上 |

### 15.6.2 AMD Matrix Core

| 架构 | Matrix Core 版本 | 支持的数据类型 | Tile 尺寸 |
|------|-----------------|---------------|-----------|
| RDNA3 (RX 7000) | 第 1 代 | FP16, BF16 | 16×16×16 |
| CDNA2 (MI250) | Matrix Core | FP16, BF16, INT8 | 16×16×16 |
| CDNA3 (MI300) | Matrix Core | 同上 + FP8 | 同上 |

### 15.6.3 Intel XMX

| 架构 | XMX 版本 | 支持的数据类型 | Tile 尺寸 |
|------|---------|---------------|-----------|
| Xe-LP (Tiger Lake) | 第 1 代 | INT8 | 8×8×8 |
| Xe-HPG (Arc Alchemist) | XMX | FP16, INT8, INT4 | 8×8×8 |
| Xe-HPC (Ponte Vecchio) | XMX | FP16, BF16, INT8 | 8×8×8, 16×16×16 |

### 15.6.4 关键约束

1. **Subgroup 大小**：CM 操作要求 Subgroup 大小与硬件支持的 CM 尺寸匹配。通常 subgroupSize 必须 ≥ tile 的一个维度。
2. **Stride 对齐**：stride 必须是 16 字节（或更大）的倍数。
3. **数据类型对齐**：float16 通常需要 16 字节对齐，int8 需要 8 字节对齐。
4. **Workgroup 大小**：某些实现要求 workgroup 大小与 subgroup 大小一致，或为 subgroup 大小的整数倍。

---

## 15.7 何时使用 CM vs 传统分块 GEMM

### 15.7.1 使用 CM 的场景

- **矩阵规模大**：M、N、K 都 ≥ 128（充分利用 Tensor Core 的吞吐优势）
- **数据类型为 FP16/BF16/INT8**：Tensor Core 的 FP16 吞吐是 FP32 的 4~16 倍
- **批量推理**：batch size 较大时，CM 的加速比更明显
- **硬件支持 CM**：当然，这是前提条件

### 15.7.2 使用传统分块 GEMM 的场景

- **矩阵规模小**：M、N、K 很小（如 ≤ 64），CM 的启动开销可能超过收益
- **FP32 精度要求**：某些 Tensor Core 不支持 FP32 的 CM（需要使用 TF32 或模拟）
- **硬件不支持 CM**：老 GPU（Pascal、GCN 等）没有 CM 硬件
- **不规则的矩阵形状**：CM 的 tile 尺寸固定，对不规则形状的填充开销大

### 15.7.3 工业级实践

**ncnn**：使用 `gemm_cm.comp` 作为 CM 路径，回退到 `gemm_tiled.comp`。它通过 `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` 查询支持的 tile 尺寸，为不同硬件选择最优参数。

**llama.cpp / ggml-vulkan**：`mul_mm_cm2.comp` 是大矩阵路径的首选。对于小矩阵，使用 `mul_mm.comp`（传统 tiling）。它还通过 `VK_EXT_cooperative_matrix_maintenance1` 的 flexible dimensions 支持更灵活的 tile 选择。

---

## 15.8 与工业级实现的对照

### 15.8.1 ncnn 的 gemm_cm.comp

ncnn 的 CM 实现特点：
- 使用 `#extension GL_NV_cooperative_matrix : enable`（早期版本）或 `GL_KHR_cooperative_matrix`
- 支持 FP16 和 INT8 的 CM 路径
- 配合 `elempack=4` 打包，减少访存次数
- 通过 `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` 自动选择 tile 尺寸
- 回退路径与 CM 路径共享同一个 pipeline 选择逻辑

### 15.8.2 ggml 的 mul_mm_cm2.comp

ggml 的 CM 实现特点：
- 专门为大矩阵（≥ 1024）优化
- 使用 `VK_KHR_cooperative_matrix` 扩展
- 支持 FP16、BF16 数据类型
- 通过 `VK_EXT_subgroup_size_control` 控制 subgroup 大小以匹配 CM 需求
- 与传统 `mul_mm.comp` 共享 buffer 管理，但使用不同的 pipeline

### 15.8.3 本章的对比思路

本章的 `gemm_cm.comp` 同时包含了 CM 路径和 fallback 路径：
- **CM 路径**：使用 `cooperativeMatrixMulAdd` 单条指令完成矩阵乘加，由 Tensor Core 硬件执行
- **回退路径**：使用传统共享内存 + 标量 FMA（与 ch04 相同）

通过 `main.cpp` 中的计时对比，你可以直观地看到 CM 的加速效果。

---

## 15.9 代码解读

### 15.9.1 着色器结构

`gemm_cm.comp` 的两个路径通过编译时宏 `USE_COOPMAT` 选择：

```glsl
#ifdef USE_COOPMAT
// CM 路径：使用 cooperative matrix 类型和函数
#extension GL_KHR_cooperative_matrix : require
// ...
void path_cm() {
    cooperativeMatrixLoad(fragA, ...);
    cooperativeMatrixLoad(fragB, ...);
    fragC = cooperativeMatrixMulAdd(fragA, fragB, fragC);
    cooperativeMatrixStore(..., fragC, ...);
}
#else
// 回退路径：普通分块 GEMM（使用共享内存）
void path_fallback() {
    // 与 ch04 相同的 tiling 逻辑
}
#endif
```

### 15.9.2 主程序流程

1. 调用 `vkInitWithExtensions` 初始化，自动启用 CM 扩展
2. 通过 `printCoopMatCapabilities` 打印设备 CTM 能力
3. 生成随机矩阵，计算 CPU 参考
4. 运行回退路径（分块 GEMM），计时并验证正确性
5. 如果设备支持 CM，运行 CM 路径，计时并验证
6. 打印加速比

### 15.9.3 特化常量的使用

```cpp
std::array<uint32_t, 3> specData = {16, 16, 16};
auto specInfo = vkMakeSpecializationInfo(specData);
pass.prepareWithSpec(ctx, spv, bindings, pcs, specInfo.info);
```

对应的着色器端：

```glsl
layout(constant_id = 0) const uint TILE_M = 16;
layout(constant_id = 1) const uint TILE_N = 16;
layout(constant_id = 2) const uint TILE_K = 16;
```

---

## 15.10 动手实验

### 基础练习

1. **跑通 CM 路径**：在有 Tensor Core 的 GPU 上运行，观察 CM 路径是否比回退路径快。
2. **修改矩阵尺寸**：将 M、N、K 从 256 改为 512、1024，观察加速比的变化趋势。
3. **修改 Tile 尺寸**：尝试不同的 tile 尺寸（如 8×8×8、32×32×32），观察哪些能通过硬件验证。

### 进阶练习

4. **FP16 路径**：将着色器改为使用 `cooperative_matrix<float16_t, ...>`（需要 `shaderFloat16` 支持），对比 FP16 CM 与 FP32 CM 的性能。
5. **查询硬件能力**：修改主程序，打印所有支持的 `VkCooperativeMatrixPropertiesKHR`，理解不同硬件的 tile 约束。
6. **与 ncnn 对比**：下载 ncnn 源码，查看 `gemm_cm.comp` 的实现，对比与我们实现的异同。
7. **Maintenance1 扩展**：如果设备支持 `VK_EXT_cooperative_matrix_maintenance1`，尝试使用 `cooperativeMatrixFlexibleDimensions` 来支持更灵活的 tile 尺寸。

### 思考题

8. 为什么 CM 路径的 push constant 使用 stride 和 K_total，而不是 M/N/K？
9. 当 K 不是 tile 尺寸的整数倍时，边界处理应该怎么做？
10. 为什么 Tensor Core 的 FP16 吞吐远高于 FP32？这对神经网络推理意味着什么？

---

## 15.11 本章总结

| 概念 | 说明 |
|------|------|
| Cooperative Matrix | 分布在 Subgroup 线程上的矩阵片段 |
| Tensor Core / Matrix Core / XMX | 专用矩阵乘加硬件单元 |
| 特化常量 | 编译期确定 tile 尺寸，运行时选择 |
| pass.prepareWithSpec | 带特化常量的管线创建 |
| coopmatMulAdd | 单条指令完成矩阵乘加 |
| 硬件约束 | Subgroup 大小、Stride 对齐、Tile 尺寸限制 |

下一章（ch16）我们将探索 Subgroup 大小控制和 Shuffle 指令，进一步理解 Subgroup 在 Vulkan 计算中的作用。