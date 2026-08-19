# 第 4 章 · 矩阵乘 GEMM（神经网络的算力心脏）

> 目标：亲手写出一个**分块 + 共享内存**的矩阵乘。它是全连接层、卷积（im2col 后）、注意力（QKᵀ、PV）共同的计算内核——搞懂这一章，后面 80% 的算子都是它的变体。

---

## 4.1 为什么 GEMM 是重中之重

几乎一切神经网络运算最终都化归为矩阵乘：

- 全连接层：`Y = X·W`
- 卷积：展开成 im2col 后变成 GEMM（ch05）
- 自注意力：`QKᵀ`、`softmax(QKᵀ)·V`、`X·W_qkv`

所以推理引擎的性能，基本等于「GEMM 有多快」。ncnn 和 llama.cpp 各自写了十几套 GEMM kernel（按矩阵形状、量化类型、硬件特性分派），足见其地位。

## 4.2 朴素 GEMM 的问题：带宽瓶颈

最直白的写法：`c[r][col] = Σ_k a[r][k]*b[k][col]`。问题：B 的列元素 `b[k][col]` 在内存里是**不连续**的（列主序 vs 行主序），每个 `k` 都要跳着读——访存不合并，带宽被打满却算力闲置。这就是 Roofline 模型里的「带宽受限」。

## 4.3 解法：2D Block Tiling + 共享内存

核心思想：把 C 切成 16×16 的小块，每个**工作组**负责一块。工作组内 256 个线程协作，先把 A、B 对应的 tile **整块搬进共享内存（shared memory）**——共享内存是片上高速存储，比全局显存快一个数量级，且搬运时天然**合并访存**；再在共享内存里做乘加。

对应 `gemm_tiled.comp`：

1. `local_size_x=16, local_size_y=16` → 每工作组 256 线程。
2. `gl_GlobalInvocationID.y/x` 是 C 的全局行列；`gl_LocalInvocationID` 是工作组内坐标。
3. 循环 `numTiles` 次：每次把 A/B 的一个 tile 搬进 `As`/`Bs`（共享内存），`barrier()` 等齐，再做 16 次乘加，再 `barrier()`。
4. 累加器 `acc` 用 **float32**，避免 half/float16 误差累积（ch03 提过）。

> 为什么用 `barrier()`？共享内存是工作组内共享的。必须等所有线程都写完后才能读，算完一个 tile 也必须等齐再加载下一个——否则会读到半成品。这正是 ch02 讲的「同步」在计算着色器内部的版本（`barrier()` ≈ 工作组内 barrier，比 pipeline barrier 更轻）。

## 4.4 第一次用 ComputePass（mini-VkCompute）

本章起，样板被收进 `common/vk_base.cpp` 的 `ComputePass`：

```cpp
ComputePass pass;
pass.prepare(ctx, spv, {3个binding}, {pushConstant范围}); // 建管线+描述符集
pass.bindBuffers({{0,bufA},{1,bufB},{2,bufC}});           // 绑 buffer
pass.dispatch(cmd, &pc, sizeof(pc), gx, gy, 1);           // 录制 dispatch
pass.destroy();                                           // 释放
```

这直接对应 **ncnn 的 `VkCompute`**（它把 pipeline + descriptor + 一次 dispatch 封成一个对象）和 **llama.cpp 的 `vk_pipeline_wrapper`**。你之前在 ch01 手写的十几行建管线代码，现在一行搞定——但请记住它们背后还是那些 Vulkan 调用。

分组数：`gx = ceil(N/16)`、`gy = ceil(M/16)`，因为工作组是 16×16。

## 4.5 Cooperative Matrix（协作矩阵）加速

> 本节展示如何利用 GPU 的 Tensor Core / Matrix Core 硬件，通过 `VK_KHR_cooperative_matrix` 扩展对 GEMM 做硬件加速。

### 4.5.1 什么是 Cooperative Matrix

Cooperative Matrix（协作矩阵，简称 coopmat）是 Vulkan 的一种扩展类型，允许 Subgroup 内的所有线程**协作持有**一个矩阵片段。每个线程只持有片段的一小部分元素，通过 `coopmatLoad` / `coopmatStore` / `coopmatMulAdd` 等集体操作完成加载、存储和矩阵乘加。

与传统分块 GEMM 的关键区别：

| 特性 | 分块 GEMM（gemm_tiled） | Cooperative Matrix（gemm_cm） |
|------|------------------------|-------------------------------|
| 数据搬运 | 手动搬进共享内存，用 `barrier()` 同步 | `coopmatLoad` 由硬件自动调度 |
| 计算单元 | 标量 ALU 逐元素乘加 | Tensor Core 单指令完成 `M×K×N` 次 FMA |
| 编程模型 | 每个线程独立计算 | Subgroup 集体操作，线程协作持有片段 |
| 同步 | 需要 `barrier()` 保证共享内存正确 | 集体操作内部隐式同步 |

### 4.5.2 CM GEMM 着色器设计

对应 `gemm_cm.comp`，核心思路：

1. **片段声明**：在文件作用域声明三个 `coopmat` 片段——A（`TILE_M×TILE_K`）、B（`TILE_K×TILE_N`）、C（累加器，`TILE_M×TILE_N`），维度通过**特化常量**（Specialization Constants）传入，以便在不同硬件上选择最优 tile 尺寸。

2. **集体加载**：`coopmatLoad` 从全局内存加载数据到片段，所有 Subgroup 线程协作完成，无需手动 `barrier()`。

3. **硬件加速乘加**：`coopmatMulAdd` 在单条指令中完成 `TILE_M × TILE_K × TILE_N` 次浮点乘加，由 Tensor Core 硬件执行。

4. **集体存储**：`coopmatStore` 将结果写回全局内存。

着色器使用 `push_constant` 传递 stride（步长）和 `K_total`（K 总维度），而非直接传 M/N/K，因为 `coopmatLoad/Store` 需要的是行跨度（以元素数计）。

### 4.5.3 编译与运行时检测

编译需要 `-DUSE_COOPMAT` 标志启用扩展：

```bash
glslc -DUSE_COOPMAT gemm_cm.comp -o gemm_cm.spv
```

CMakeLists.txt 中通过独立的 `add_custom_command` 处理，不干扰 `gemm_tiled.comp` 的正常编译。

运行时通过 `ctx.features.cooperativeMatrixKHR` 检测设备是否支持 `VK_KHR_cooperative_matrix`。只有在支持时才创建 CM 管线并 dispatch，否则跳过。

### 4.5.4 CM 路径的 C++ 集成

在 `main.cpp` 中，CM 路径使用 `prepareWithSpec` 传入特化常量（tile 尺寸），push constant 使用 stride 结构体：

```cpp
struct CM_PC {
    uint32_t strideA;   // = K
    uint32_t strideB;   // = N
    uint32_t strideC;   // = N
    uint32_t K_total;   // = K
};

std::array<uint32_t, 3> specData = {16, 16, 16};
auto specInfo = vkMakeSpecializationInfo(specData);
passCM.prepareWithSpec(ctx, "gemm_cm.spv", bindings, pcs, specInfo.info);
```

### 4.5.5 CM vs 分块 GEMM 性能对比

通过 GPU timestamp query 分别测量两种路径的耗时，计算 GFLOPS 和加速比：

```
====== 性能总结 ======
  矩阵尺寸: 256x256x256
  分块 GEMM: 0.XX ms, XX.X GFLOPS
  CM GEMM:   0.XX ms, XX.X GFLOPS
  CM 加速比: 1.XXx（正确）
```

在支持 Tensor Core 的硬件上，CM 路径通常可获得 2-5 倍加速。你可以将 `M/N/K` 改为 `512` 或 `1024` 观察更大矩阵下的加速效果。

## 4.6 动手实验

1. 跑通后，把 `M/N/K` 改成 `512` 或 `1024`，观察仍正确。
2. 把 `local_size_x/y` 从 16 改成 8（同时改 shader 里的 `As[8][8]`/`Bs[8][8]` 和 `numTiles` 里的 15→7），对比两者结果——理解「工作组尺寸是可调旋钮」。
3. （进阶）去掉共享内存，改成朴素按列读 B 的版本，用 ch09 的计时方法对比带宽差异——你会亲眼看到 tiling 的价值。
4. 如果设备支持 Cooperative Matrix，对比分块 GEMM 和 CM GEMM 的性能差异。注意 CM 路径的 tile 尺寸（16×16×16）是 Tensor Core 的常见规格，尝试改为 32×32×16 或 8×8×16 观察变化。
5. 思考：当 K 不是 16 的整数倍时，`numTiles` 的向上取整 + `if (aCol<K)` 边界保护为什么是必需的？

## 4.7 对照工业级

- **ncnn**：`gemm` 系列用 `optimal_local_size_xyz` 自动选工作组，并针对 ARM Mali / Adreno / 桌面 GPU 写了不同 tile 策略；还用 `elempack=4` 把 4 个元素打包进一个 vec4 进一步合并访存。
- **llama.cpp / ggml-vulkan**：`mul_mm.comp` 按矩阵大小分 **large / medium / small** 三套 kernel；大矩阵用 cooperatice matrix（coopmat，相当于 Tensor Core），中小矩阵用手写 tiling；并通过 `vkCmdPushConstants` 传 M/N/K。

下一章（ch05）我们把卷积拆成「直接卷积」和「im2col+GEMM」两条路，看清卷积为什么最终也变成 GEMM。
