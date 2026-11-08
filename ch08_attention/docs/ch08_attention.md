# 第 8 章 · Softmax 与注意力（Transformer 核心）

> 目标：把前几章的积木（矩阵乘、归约）拼成**简化自注意力**。你会看到注意力不过是「两次矩阵乘夹一个 softmax」，这也是为什么 GPU 上的 matmul 优化能直接加速大模型。

---

## 8.1 单头注意力的数学

```
S = Q · Kᵀ / √D        # 分数，Q,K ∈ ℝ^{N×D}
P = softmax(S)         # 按行归一化成概率
O = P · V              # 上下文，O ∈ ℝ^{N×D}
```

`N` 是序列长度，`D` 是头维度。整条链路就是 **两个矩阵乘 + 一个按行 softmax**。Flash Attention 的种种优化，本质是不把巨大的 `S`(N×N) 和 `P` 完整写回显存，而是在片上边算边归一化——但数学等价。

## 8.2 三个 kernel 串起来

本章 `main.cpp` 把注意力拆成 3 个独立 kernel，中间用 `vkSubmit`（含隐式 barrier/fence）衔接：

1. **`attn_scores.comp`**：每个线程算一个 `(qi, ki)` 的分数 `Σ_d Q[qi,d]·K[ki,d]/√D`——这就是 ch04 矩阵乘的「一个元素」。
2. **`softmax.comp`**：按行做**数值稳定** softmax（先减行最大值防止 `exp` 溢出，再归一化）。归约求 `max` 和 `sum` 复用 ch07 的共享内存树形归约。每行一个工作组（`N≤64`）。
3. **`attn_context.comp`**：每个线程算一个 `(qi, d)` 的 `Σ_k P[qi,k]·V[k,d]`——又一次矩阵乘。

> 数值稳定为何重要：`exp(大数)` 会溢出成 `inf`，导致 softmax 全 `nan`。减掉行最大值后，最大指数变 0，`exp` 永不溢出，结果不变（softmax 对整体平移不变）。

## 8.3 为什么注意力「贵」

分数的中间矩阵 `S`、`P` 是 `N×N`。当序列长 `N=4096`，`S` 就有 1600 万个元素（64MB，FP32）——而且注意力要算**很多头**。这就是长上下文推理显存爆炸的根源，也是 Flash Attention（不物化 S/P）成为标配的原因。

## 8.4 动手实验

1. 把 `N` 改成 128（注意 `softmax.comp` 的 `local_size_x=64` 与 `N≤64` 假设——需相应改成 128 并调整归约步长或分两趟）。
2. 加一个「scale」观察：去掉 `/√D`，看分数绝对值变大、softmax 更「尖锐」，对结果有无影响（数值上略有）。
3. （进阶）实现 Flash Attention 雏形：在 `attn_scores` 里**边算分数边在线 softmax**，直接产出 `O`，避免物化 N×N 的 `S`/`P`。这是 llama.cpp `flash_attn` 系列的核心思想。

## 8.5 对照工业级

- **llama.cpp / ggml-vulkan**：注意力是 LLM 推理最热路径。它提供多套实现：
  - 标量 Flash Attention（小序列）；
  - 基于 **cooperative matrix**（coopmat，近似 Tensor Core）的 Flash Attention；
  - NVIDIA 专属的 **coopmat2** 路径。
  共同目标是「不把 N×N 分数矩阵写回显存」。本章的 3-kernel 版本正是不做此优化的「朴素注意力」，便于理解。
- **ncnn**：CV 模型里注意力（如 ViT、MobileViT）同样拆成 matmul+softmax+matmul，并尽量融合。
- **2026-11 同步**：工业级 FlashAttention 已不是「一个 `flash_attn.comp`」。ncnn `shader/` 下有 `sdpa_fa.comp` / `sdpa_fa_cm.comp` / `sdpa_cross*.comp` / `sdpa_kvcache_{append,copy}.comp`；ggml `vulkan-shaders/` 下有 `flash_attn{,_cm1,_cm2}.comp`、`flash_attn_decode_phase_{1,2}.comp`、`flash_attn_split_k_reduce.comp`、`flash_attn_sparse_compact.comp`。prefill 与 decode 分开成不同 kernel 是当前的主流做法，详见 ch17。

## 8.6 FlashAttention 融合 kernel

本章在 `main.cpp` 中原有的 3-kernel 朴素注意力之外，新增了一个 **FlashAttention 融合 kernel** 演示。对应的着色器是 `shaders/flash_attn.comp`。

### 8.6.1 动机

3-kernel 朴素注意力需要将 N×N 的分数矩阵 `S` 和概率矩阵 `P` 完整写入显存，再读回来。当序列长度 N 较大时，这导致严重的显存带宽瓶颈。FlashAttention 的核心思想是**不物化 N×N 的中间矩阵**，而是在片上（寄存器）边算分数边做 softmax 边累加输出，最终只写回 O(N×d) 的结果。

### 8.6.2 算法：在线 Softmax（Online Softmax）

标准 softmax 需要知道所有分数后才能归一化。在线 softmax 可以在不预先知道所有分数的情况下逐步更新：

```
初始化: m = -inf, d_sum = 0, O = 0

对于每个 key 位置 k:
  score = Q[q] · K[k] / √d
  m_new = max(m, score)
  rescale = exp(m - m_new)        // 重缩放因子 (≤1)
  p = exp(score - m_new)
  d_sum = d_sum * rescale + p
  O = O * rescale + p * V[k]
  m = m_new

结束后: O = O / d_sum
```

关键性质：
1. **数值稳定**：始终减去当前最大值，`exp` 永不溢出
2. **数学等价**：最终结果与标准 softmax 完全一致
3. **逐步更新**：可以在不存储所有分数的情况下完成 softmax

### 8.6.3 着色器设计

`flash_attn.comp` 使用以下工作组结构：
- `local_size_x = 32`：每个工作组一个 subgroup，32 个线程
- 每个工作组处理 1 个 query 行（Br = 1）
- 每个线程负责 d/32 个输出维度的累加
- K/V tile 大小 Bc = 32

**状态变量**（每个线程独有，寄存器中）：
- `q_reg[elemsPerThread]`：Q 的本地元素
- `m`：运行最大值（running max）
- `d_sum`：运行 exp 和（running denominator sum）
- `o_acc[elemsPerThread]`：输出累加器

**点积归约**使用 `subgroupAdd` 在 32 线程间一步完成，需要 `GL_KHR_shader_subgroup_arithmetic` 扩展。

### 8.6.4 与 3-kernel 对比

新代码在 `main.cpp` 中：
1. 保持原有的 3-kernel 朴素注意力（用于教学理解）
2. 新增一个 FlashAttention 演示，使用 GPU 时间戳精确计时
3. 验证 FlashAttention 的输出与朴素注意力一致
4. 打印两者的耗时对比

FlashAttention 的优势在于：
- **显存**：不物化 N×N 的中间矩阵，显存占用 O(N) 而非 O(N²)
- **带宽**：减少全局显存读写，算术强度更高
- **延迟**：单 kernel 替代三 kernel，减少 kernel launch 开销

下一章（ch09）我们正式进入**性能分析**：用 GPU 硬件时间戳精确测时，并套用 Roofline 模型判断瓶颈在带宽还是算力。
