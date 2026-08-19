# 第 17 章 · FlashAttention：融合注意力与在线 Softmax

> 目标：理解标准注意力 O(N²) 显存复杂度的瓶颈，并实现 **FlashAttention**——通过分块（tiling）和在线 softmax（online softmax）将显存复杂度降至 O(N) 的融合 kernel。这是长上下文 LLM 推理的核心优化。

---

## 17.1 问题：注意力为什么「贵」

标准缩放点积注意力（Scaled Dot-Product Attention）：

```
S = Q · Kᵀ / √d          # N×N 分数矩阵
P = softmax(S, dim=-1)    # N×N 概率矩阵（按行归一化）
O = P · V                 # N×d 输出
```

问题在于中间矩阵 `S` 和 `P` 的大小是 **N×N**。当序列长度 N 增长时，显存占用以 O(N²) 增长：

| N | S/P 大小（FP32） | 显存占用 |
|---|-----------------|---------|
| 512 | 512×512 = 262K | 1 MB/head |
| 1024 | 1024×1024 = 1M | 4 MB/head |
| 4096 | 4096×4096 = 16M | 64 MB/head |
| 32768 | 32768×32768 = 1G | 4 GB/head |

对于 32 头、32K 序列长度，仅注意力分数就需要 128 GB 显存——远超任何单 GPU 的容量。这就是 **FlashAttention** 要解决的问题。

### 显存 vs 计算

注意力的**计算量**也是 O(N²·d)，但现代 GPU 的算力足以支撑。真正的瓶颈是**显存带宽**：把 N×N 的分数矩阵写回显存再读出来，比计算它本身慢得多。

FlashAttention 的核心洞察：**不物化 N×N 的中间矩阵**，而是在片上（寄存器和共享内存）边算分数边做 softmax 边累加输出，最终只写回 O(N×d) 的结果。

---

## 17.2 FlashAttention 核心算法

### 17.2.1 分块策略（Tiling）

FlashAttention 将 Q、K、V 分成 tile：

- **Q tile**：`Br × d`（通常 Br = 1，即每次处理一个 query 行）
- **K/V tile**：`Bc × d`（通常 Bc = 32 或 64，即一个 subgroup 的线程数）

外层循环遍历 K/V tile，内层循环遍历 tile 内的每个 key 位置。

### 17.2.2 在线 Softmax（Online Softmax）

标准 softmax 需要知道所有分数后才能归一化：
```
m = max(scores)          # 先求最大值
p_i = exp(s_i - m)       # 再求指数
sum = Σ p_i              # 再求和
softmax_i = p_i / sum    # 最后归一化
```

**在线 softmax** 可以在不预先知道所有分数的情况下逐步更新：

```
初始化: m = -inf, d = 0, O = 0

对于每个 key 位置 k:
  score = Q[q] · K[k] / √d
  m_new = max(m, score)
  rescale = exp(m - m_new)        # 重缩放因子 (≤1)
  p = exp(score - m_new)
  d = d * rescale + p
  O = O * rescale + p * V[k]
  m = m_new

结束后: O = O / d
```

关键性质：
1. **数值稳定**：始终减去当前最大值，`exp` 永不溢出
2. **数学等价**：最终结果与标准 softmax 完全一致
3. **逐步更新**：可以在不存储所有分数的情况下完成 softmax

### 17.2.3 数值推导

为什么在线 softmax 与标准 softmax 等价？

标准 softmax 输出：
```
O_std = Σ_k exp(s_k - m_global) · V[k] / Σ_k exp(s_k - m_global)
```

其中 `m_global = max(all scores)`。

在线 softmax 维护一个运行最大值 m。当处理到第 j 个 key 时：
```
m_j = max(s_1, s_2, ..., s_j)
d_j = Σ_{i=1}^{j} exp(s_i - m_j)
O_j = Σ_{i=1}^{j} exp(s_i - m_j) · V[i]
```

当处理第 j+1 个 key 时，如果 `s_{j+1} > m_j`，则需要**重缩放**之前的所有项：
```
m_{j+1} = s_{j+1}
d_{j+1} = d_j · exp(m_j - m_{j+1}) + exp(s_{j+1} - m_{j+1})
O_{j+1} = O_j · exp(m_j - m_{j+1}) + exp(s_{j+1} - m_{j+1}) · V[j+1]
```

这等价于把之前所有项除以 `exp(m_{j+1} - m_j)`，然后加上新项。

最终：
```
O = O_N / d_N = Σ_k exp(s_k - m_N) · V[k] / Σ_k exp(s_k - m_N)
```

与标准 softmax 输出一致，因为 `m_N = m_global`。

---

## 17.3 实现详解

### 17.3.1 着色器设计：`flash_attn.comp`

```
工作组结构:
  local_size_x = 32    ← 一个 subgroup
  Br = 1               ← 每个工作组处理 1 个 query 行
  Bc = 32              ← K/V tile 大小

每个线程的责任:
  - 处理 d/32 个输出维度（列）
  - 对每个 K/V tile，计算部分点积并归约
  - 维护在线 softmax 状态
```

**状态变量**（每个线程独有，寄存器中）：
- `q_reg[elemsPerThread]`：Q 的本地元素
- `m`：运行最大值
- `d_sum`：运行 exp 和
- `o_acc[elemsPerThread]`：输出累加器

**主循环**：
```
for each K/V tile (k_start = 0, 32, 64, ..., N-1):
  for each key in tile (k = k_start ... k_start+31):
    score = dot(Q[q_idx], K[k]) / √d     # subgroupAdd 归约
    m_new = max(m, score)
    rescale = exp(m - m_new)
    p = exp(score - m_new)
    d_sum = d_sum * rescale + p
    o_acc = o_acc * rescale + p * V[k]
    m = m_new
```

**最终归一化**：
```
O[q_idx][col] = o_acc[col] / d_sum
```

### 17.3.2 Subgroup 归约

点积计算使用 `subgroupAdd` 在 32 线程间归约：

```glsl
float partial = 0.0;
for (uint i = 0; i < elemsPerThread; ++i)
    partial += q_reg[i] * K[k_idx * d + colBase + i];
float score = subgroupAdd(partial) * inv_sqrt_d;
```

每个线程贡献 d/32 个元素的部分点积，`subgroupAdd` 一步完成全部 32 线程的累加。这需要 `GL_KHR_shader_subgroup_arithmetic` 扩展。

### 17.3.3 朴素对照：`flash_attn_naive.comp`

朴素版本先完整计算所有 N 个分数（存入共享内存 `scores[128]`），再做 softmax，最后累加 V。这需要 O(N) 共享内存，N 大时不可行。

```
步骤 1: 计算所有 scores[q, k] = Q[q]·K[k] / √d
步骤 2: 按行 softmax（求 max → exp → sum → 归一化）
步骤 3: 计算 O[q, d] = Σ_k P[q, k] · V[k, d]
```

---

## 17.4 性能分析

### 17.4.1 显存复杂度对比

| 方法 | 中间显存 | 额外读写 |
|------|---------|---------|
| 标准注意力（3 kernel） | N×N（分数 + 概率） | 2 × N² 元素 |
| 朴素单 kernel | N（存储在共享内存） | 0（片上） |
| FlashAttention | O(1)（寄存器） | 0（片上） |

### 17.4.2 计算量对比

两种方法的计算量都是 O(N²·d)，但 FlashAttention 的访存模式更优：
- **标准注意力**：写 2×N² 元素到显存（分数 + 概率），再读回来
- **FlashAttention**：仅读写 Q、K、V、O（总共 4×N×d 元素），分块加载 K/V tile

对于 N=1024, d=64:
- 标准注意力：额外 2×1024² = 2M 元素 = 8 MB 显存读写
- FlashAttention：无额外显存读写

### 17.4.3 算术强度

FlashAttention 的算术强度（FLOP / byte）远高于标准注意力，因为每次从全局显存加载的 K/V 数据被多次使用（用于多个 query 行的分数计算）。

---

## 17.5 工业级对照

### 17.5.1 ncnn 的 `sdpa_fa.comp`

ncnn 的 FlashAttention 实现（`sdpa_fa.comp`）使用 2×2 unroll 策略，即每个工作组同时处理 2 个 query 行和 2 个 key 位置，通过 local memory 缓存 tile 数据。它支持 cooperative matrix 加速版本（`sdpa_fa_cm.comp`），利用 Tensor Core 硬件加速矩阵乘法部分。

### 17.5.2 ggml (llama.cpp) 的 `flash_attn.comp`

ggml 的 FlashAttention 实现使用完整的分块策略 + subgroup 操作，与本章实现类似。它提供多套变体：
- **标量路径**：基本分块，适用于小序列
- **CM 路径**：使用 cooperative matrix 加速（`flash_attn_cm1.comp`）
- **CM2 路径**：NVIDIA 专属的 coopmat2 加速（`flash_attn_cm2.comp`）
- **Split-K 路径**：将 K 维度分到多个工作组，最后归约（适用于大序列）

### 17.5.3 对比总结

| 特性 | 本章实现 | ncnn sdpa_fa | ggml flash_attn |
|------|---------|-------------|-----------------|
| Subgroup 归约 | ✓ (subgroupAdd) | ✓ | ✓ |
| 共享内存 K/V tile | ✗（直接读全局） | ✓ (local mem) | ✓ |
| Online softmax | ✓ | ✓ | ✓ |
| CM 加速 | ✗ | ✓ (KHR+NV) | ✓ (KHR+NV+CM2) |
| Split-K | ✗ | ✗ | ✓ |
| KV cache 支持 | ✗ | ✓ | ✓ |

---

## 17.6 动手实验

1. **改变 tile 大小**：将 `Bc` 从 32 改为 64（需调整线程数），观察性能变化。更大的 tile 意味着更少的全局内存加载次数，但共享内存占用增加。

2. **对比不同序列长度**：运行 `ch17_flash_attn`，观察 N=128、256、512、1024 时 FlashAttention 与朴素注意力的加速比变化。理论上 N 越大，FlashAttention 优势越明显。

3. **实现 KV cache 版本**：修改着色器，使 K 和 V 从独立的 KV cache buffer 中读取（支持增量解码）。

4. **实现多 query 行**：将 `Br` 从 1 改为 4（每个工作组处理 4 个 query 行），通过共享内存缓存 Q tile，减少 K/V 的重复加载。

5. **添加 causal mask**：在分数计算时加入因果关系掩码（causal mask），使 `score[q,k] = -inf` 当 `k > q`。

6. **对比 ch08 的 3-kernel 注意力**：用 GPU 时间戳对比本章 FlashAttention 与 ch08 的 3-kernel 注意力的性能差异。

---

## 17.7 本章总结

- 标准注意力的显存复杂度是 O(N²)，长序列时不可行
- **FlashAttention** 通过分块（tiling）和在线 softmax，将显存复杂度降至 O(N)
- **在线 softmax** 使用运行最大值和重缩放因子，在不存储所有分数的情况下完成数值稳定的 softmax
- Subgroup 操作（`subgroupAdd`）用于高效的 warp 级点积归约
- 本章实现的是简化的 Br=1 版本，实际工业实现支持多 query 行 tile、cooperative matrix 加速、causal mask 等扩展
- 工业级实现（ncnn、ggml）在 FlashAttention 基础上增加了 CM 加速、Split-K、KV cache 等特性

下一章（ch18）我们将探索 **量化体系深入**——从 int8 到 Q4_0/Q8_0/K-quant 再到 MXFP4 的 GPU 端反量化与融合推理。