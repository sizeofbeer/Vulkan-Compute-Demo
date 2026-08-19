# 第 16 章 · Subgroup 编程（Warp 级并行）

> 目标：理解 GPU 上最底层的硬件执行单元——**subgroup（warp/wavefront）**，学会用 subgroup 操作替代共享内存，实现更高效的归约、广播、洗牌等通信模式。这是编译器/框架（ncnn、ggml）做极致优化的关键手段。

---

## 16.1 什么是 Subgroup

### 硬件视角

GPU 的「线程」不是完全独立的。硬件实际以固定大小的 **warp（NVIDIA）** 或 **wavefront（AMD）** 为单位执行指令——同一 warp 内的线程在同一个时钟周期执行同一条指令（SIMD 风格）。Vulkan 将这种硬件执行单元抽象为 **subgroup**。

- **NVIDIA GPU**：warp = 32 线程，subgroupSize = 32
- **AMD GPU**：wavefront = 64 线程（GCN）或 32 线程（RDNA），subgroupSize = 32 或 64
- **Intel GPU**：subgroupSize = 8 或 16 或 32

> 关键：**subgroup 是硬件原生的并发粒度，subgroup 内的线程之间可以零开销地交换数据**（通过寄存器），无需经过共享内存。

### 工作组 vs Subgroup

```
┌─────────────────────────────────────┐
│         工作组 (Workgroup)           │
│  ┌──────┬──────┬──────┬──────┐      │
│  │ SG 0 │ SG 1 │ SG 2 │ SG 3 │ ...  │  ← Subgroup（硬件决定大小）
│  └──────┴──────┴──────┴──────┘      │
│   共享内存 / barrier 可跨 SG 通信    │
│   Subgroup 操作：寄存器级通信，零延迟  │
└─────────────────────────────────────┘
```

- **工作组**是程序员指定的逻辑分组，大小可变（如 64、128、256），通过共享内存 + barrier 实现组内协作。
- **Subgroup** 是硬件固定的物理执行单元，大小由设备决定，subgroup 内用 **subgroup 操作**实现寄存器级通信，无需共享内存，无需 barrier。

### Subgroup 的优势

| 特性 | 共享内存 | Subgroup 操作 |
|------|---------|---------------|
| 延迟 | 数十周期（SRAM 访问） | 1-2 周期（寄存器旁路） |
| 同步 | 需要 barrier | 隐式同步（同一指令内完成） |
| 占用率 | 占用共享内存，降低活跃线程数 | 不占共享内存，不影响占用率 |
| 代码量 | 多行（shmem 声明 + 树形归约 + barrier） | 一行 |

---

## 16.2 Subgroup 操作分类

### 16.2.1 Basic（基础信息）

| GLSL 内建 | 含义 |
|-----------|------|
| `gl_SubgroupSize` | 当前 subgroup 的线程数（设备常量） |
| `gl_SubgroupInvocationID` | 当前线程在 subgroup 内的索引（0 ~ size-1） |
| `gl_NumSubgroups` | 当前工作组内 subgroup 的数量 |
| `gl_SubgroupID` | 当前 subgroup 在工作组内的索引 |

**用途**：查询硬件特性、动态调整算法。

### 16.2.2 Vote（投票）

| 函数 | 含义 |
|------|------|
| `subgroupAll(cond)` | 本 subgroup 所有线程 cond 都为真？ |
| `subgroupAny(cond)` | 本 subgroup 任一线程 cond 为真？ |
| `subgroupBallot(cond)` | 返回一个 uint32 位掩码，第 i 位表示 lane i 的 cond |
| `subgroupAllEqual(value)` | 所有线程的 value 都相等？ |

**用途**：检测收敛性（如循环中所有线程是否都满足退出条件）、条件分支统计。

### 16.2.3 Arithmetic（算术归约）

| 函数 | 含义 |
|------|------|
| `subgroupAdd(value)` | 跨 subgroup 求和 |
| `subgroupMin(value)` | 跨 subgroup 求最小值 |
| `subgroupMax(value)` | 跨 subgroup 求最大值 |
| `subgroupMul(value)` | 跨 subgroup 求积 |
| `subgroupAnd(value)` | 按位与 |
| `subgroupOr(value)` | 按位或 |
| `subgroupXor(value)` | 按位异或 |
| `subgroupInclusiveAdd(value)` | 前缀和（包含当前线程） |
| `subgroupExclusiveAdd(value)` | 前缀和（不包含当前线程） |

**用途**：替代共享内存树形归约，一行代码完成 reduction。

### 16.2.4 Ballot（位掩码统计）

| 函数 | 含义 |
|------|------|
| `subgroupBallot(cond)` | 返回 32/64 位掩码 |
| `subgroupBallotBitCount(mask)` | 统计掩码中 1 的个数 |
| `subgroupBallotFindLSB(mask)` | 找到最低位 1 的位置 |
| `subgroupBallotFindMSB(mask)` | 找到最高位 1 的位置 |
| `bitCount(mask)` | GLSL 内建，标准位计数 |

**用途**：高效统计满足条件的线程数、动态 warp 管理。

### 16.2.5 Shuffle（寄存器直接交换）

| 函数 | 含义 |
|------|------|
| `subgroupBroadcast(value, id)` | 将 lane `id` 的值广播给所有线程 |
| `subgroupShuffle(value, id)` | 从 lane `id` 获取值 |
| `subgroupShuffleXor(value, mask)` | 按 mask 做 XOR 寻址交换 |
| `subgroupShuffleUp(value, delta)` | 从 `id - delta` 获取值 |
| `subgroupShuffleDown(value, delta)` | 从 `id + delta` 获取值 |

**用途**：在不使用共享内存的情况下实现任意数据交换模式，是许多高级算法的基础。

---

## 16.3 Subgroup 广播：一线程共享，全组受益

`subgroupBroadcast` 是最常用的 shuffle 操作之一：让 subgroup 内的某个线程把自己的值分享给所有其他线程。

```glsl
// lane 0 持有归约结果，广播给所有线程
float result = subgroupBroadcast(partial_sum, 0);
```

**典型场景**：树形归约后，`lane 0` 持有最终结果，广播给所有线程做后续处理。用共享内存的话需要写 smem + barrier + 读 smem，至少 3 条指令多拍延迟；subgroup broadcast 只需要一条指令。

---

## 16.4 Subgroup 洗牌：寄存器间数据交换

`subgroupShuffle` 系列函数允许线程直接读取另一个线程的寄存器值，**不需要经过共享内存**。

```glsl
// 从下一个线程取值（环形）
float next_val = subgroupShuffle(my_val, (gl_SubgroupInvocationID + 1) % gl_SubgroupSize);

// 从上一个线程取值
float prev_val = subgroupShuffleUp(my_val, 1);

// XOR 洗牌：蝴蝶网络交换
float partner = subgroupShuffleXor(my_val, 1u << stage);
```

**为什么快**：现代 GPU 的寄存器文件是一个统一的物理存储，shuffle 指令通过寄存器旁路网络直接路由数据，延迟仅 1-2 个周期，远低于共享内存（~20-30 周期 + barrier 同步开销）。

---

## 16.5 Subgroup 算术归约

传统共享内存归约（ch07）需要：
1. 声明 `shared float smem[256]`
2. 每个线程写 `smem[tid] = value`
3. `barrier()`
4. 树形循环：`for (s=128; s>0; s>>=1) { if (tid < s) smem[tid] += smem[tid+s]; barrier(); }`

Subgroup 归约只需一行：
```glsl
float sum = subgroupAdd(value);
```

**性能差异**：subgroup 版本没有共享内存访问延迟，没有 barrier 同步开销，不占用共享内存资源（意味着更高占用率）。

### 示例：Warp-Level 归约

```glsl
// 每个线程持有自己的值
float val = in[gl_GlobalInvocationID.x];

// 一行代码替代整个共享内存树形归约
float sum = subgroupAdd(val);
float max_val = subgroupMax(val);
float min_val = subgroupMin(val);
```

---

## 16.6 Softmax 优化：用 Subgroup 替代共享内存

ch08 中 softmax 使用共享内存做两步归约（求 max、求和）。有了 subgroup 操作，可以完全消除共享内存：

```glsl
// 数值稳定的 warp-level softmax
float v = in[idx];                  // 每个线程持有一个元素
float m = subgroupMax(v);           // 一步求行 max
float e = exp(v - m);               // 指数
float sum = subgroupAdd(e);         // 一步求和
out[idx] = e / sum;                 // 归一化
```

**4 行代码替代了共享内存版 ~20 行**，且省去了 barrier 和共享内存访问。

### 性能对比

运行 `ch16_subgroup` 的 benchmark 会显示 subgroup 版本通常比共享内存版本快 **2-5 倍**（取决于 GPU 架构）：

```
Subgroup softmax:   0.15 μs/iter
Shared mem softmax: 0.45 μs/iter
Subgroup 版本快了 3.0 倍
```

> 注意：这个 benchmark 仅测量单次 kernel 启动，实际推理中差异会被更大的 kernel 摊销。但在热点路径（如注意力分数计算）中，这些微秒级差异会累积成显著的加速。

---

## 16.7 Subgroup Size Control 扩展

`VK_EXT_subgroup_size_control` 允许着色器指定期望的 subgroup 大小，对跨平台优化非常重要：

- **NVIDIA**：强制 subgroupSize = 32
- **AMD RDNA**：支持 32 或 64，通过扩展可选择
- **Intel**：支持 8、16、32

```cpp
// C++ 端检查
if (ctx.features.subgroupSizeControl) {
    // 可以要求编译器使用特定的 subgroup 大小
    // 配合 VK_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT
}
```

着色器端：
```glsl
#extension GL_EXT_subgroup_size_control : require
layout(local_size_x = 32) in;
layout(subgroup_size = 32) in;  // 要求 32
```

---

## 16.8 工业级对照

### ncnn 中的 Subgroup 使用

ncnn 在 Vulkan 后端广泛使用 subgroup 操作加速：

- **packing 转换**：用 `subgroupShuffle` 实现高效的数据重排（如 AoS ↔ SoA）
- **Winograd 卷积**：用 subgroup 加速 tile 内的数据搬移
- **Reduction 算子**：`subgroupAdd` 替代共享内存归约
- **Softmax**：warp-level softmax 用于注意力头内的归一化

### ggml (llama.cpp) 中的 Subgroup 使用

ggml 在 `flash_attn` 和 `ggml_mul_mat` 等热路径中大量使用 subgroup：

- **Flash Attention**：使用 `subgroupShuffle` 和 `subgroupAdd` 在 warp 内完成在线 softmax 的局部归约，避免共享内存瓶颈
- **Matmul**：用 `subgroupBroadcast` 在 warp 内广播 A 矩阵元素，用 `subgroupShuffleXor` 实现 butterfly 归约
- **RMSNorm**：`subgroupAdd` 一步求平方和，替代共享内存归约

关键洞见：**工业级推理引擎中，所有涉及「组内协作」的算子，只要有 subgroup 支持，都会优先使用 subgroup 操作而非共享内存**。

---

## 16.9 动手实验

1. **修改 subgroup 大小**：你的设备 subgroupSize 是多少？如果支持 `subgroupSizeControl`，尝试强制使用不同的 subgroup 大小。
2. **实现 warp-level reduce sum**：用 `subgroupInclusiveAdd` 或 `subgroupExclusiveAdd` 实现一个并行前缀和（scan）。
3. **蝴蝶归约**：用 `subgroupShuffleXor` 实现一个不依赖 `subgroupAdd` 的手动归约，体会 shuffle 的灵活性。
4. **扩展 softmax**：当 N > subgroupSize 时，如何扩展 warp-level softmax？（提示：每个线程处理多个元素，或让多个 subgroup 协作）
5. **对比 ncnn 源码**：阅读 ncnn Vulkan 后端的 `softmax` 实现，看它是否使用了 subgroup 操作。

---

## 16.10 本章总结

- Subgroup 是 GPU 硬件原生的执行单元（warp/wavefront），subgroup 内线程共享程序计数器，执行同一条指令。
- Subgroup 操作提供寄存器级的通信原语，延迟远低于共享内存，且不占用共享内存资源。
- 五大操作类别：**Basic**（信息查询）、**Vote**（条件判断）、**Arithmetic**（归约）、**Ballot**（位掩码）、**Shuffle**（数据交换）。
- **Warp-level softmax** 只需 4 行代码即可完成归约 + 归一化，是 subgroup 优化的经典案例。
- 工业级推理引擎（ncnn、ggml）在热路径中广泛使用 subgroup 操作替代共享内存。

下一章（ch17）我们将探索 **Cooperative Matrix（协同矩阵）**——Vulkan 中接近 Tensor Core 的矩阵乘法加速硬件。