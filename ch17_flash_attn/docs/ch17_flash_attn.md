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

### 17.5.1 ncnn 的 SDPA 家族（2026-11 现状）

着色器都在 `src/layer/vulkan/shader/`（单数目录），已从「单个 `sdpa_fa.comp`」扩展为 6 个：

| Shader | 职责 |
|--------|------|
| `sdpa_fa.comp` / `sdpa_fa_cm.comp` | 标量 / coopmat 版 FlashAttention（2×2 unroll + local memory tile） |
| `sdpa_cross.comp` / `sdpa_cross_cm.comp` | 交叉注意力（encoder-decoder / 多模态） |
| `sdpa_kvcache_append.comp` / `sdpa_kvcache_copy.comp` | 增量解码时的 KV cache 追加与搬运 |

`src/layer/vulkan/sdpa_vulkan.cpp` 负责 GQA、mask 广播、KV cache 与 fp32 累加路径选择；2026-11 的 #7247 把 Reduction 的共享累加器从 half 改为 **fp32**，正是本章「在线 softmax 数值稳定」在工业代码里的对应物。

### 17.5.2 ggml (llama.cpp) 的 `flash_attn*` 家族

`ggml/src/ggml-vulkan/vulkan-shaders/` 下不再是「一个文件三路径」，而是拆成 8 个 `.comp` 变体 + 3 个 `.glsl` 公共片段：

- `flash_attn.comp` + `flash_attn_base.glsl` / `flash_attn_dequant.glsl` / `flash_attn_mmq_funcs.glsl`：标量基线与公共片段
- `flash_attn_cm1.comp` / `flash_attn_cm2.comp`：KHR coopmat / NVIDIA CM2
- `flash_attn_decode_phase_1.comp` / `flash_attn_decode_phase_2.comp`：**decode 两阶段**拆分（prefill 与逐 token 解码走不同 kernel）
- `flash_attn_split_k_reduce.comp` / `mul_mat_split_k_reduce.comp`：Split-K 归约
- `flash_attn_sparse_compact.comp` / `flash_attn_mask_opt.comp`：稀疏压缩与 mask 优化
- `dequant_funcs.glsl` / `dequant_funcs_cm2.glsl`：量化 K/V 在 FA 内部就地解包

### 17.5.3 对比总结

| 特性 | 本章实现 | ncnn SDPA 家族 | ggml flash_attn 家族 |
|------|---------|---------------|---------------------|
| Subgroup 归约 | ✓ (subgroupAdd) | ✓ | ✓ |
| 共享内存 K/V tile | ✗（直接读全局） | ✓ (local mem) | ✓ |
| Online softmax | ✓ | ✓（fp32 累加） | ✓ |
| CM 加速 | ✗ | ✓ (`sdpa_fa_cm.comp`) | ✓ (CM1 + CM2) |
| Split-K | ✗ | ✗ | ✓ (`split_k_reduce.comp`) |
| KV cache 支持 | ✗ | ✓ (`kvcache_append/copy.comp`) | ✓（decode 两阶段） |
| 交叉注意力 | ✗ | ✓ (`sdpa_cross*.comp`) | ✓（mtmd 路径） |
| 量化 K/V 就地解包 | ✗ | ✗ | ✓ (`flash_attn_dequant.glsl`) |

> 进阶练习 #3「实现 KV cache 版本」**已有可运行参考实现，见 17.8**；对照 `sdpa_kvcache_append.comp` 与 `flash_attn_decode_phase_*.comp` 可以看到工业级做法：把 prefill 与 decode 分成不同 kernel，而不是让同一个融合 kernel 兼顾两者。

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

---

## 17.8 进阶实现：KV cache 增量解码与 decode 两阶段

`ch17_kvcache`（`main_kvcache.cpp` + 3 个新 shader）是「实现 KV cache 版本」这道动手题的可运行参考实现，也是本次同步四个参考仓库后新增的代码。它是独立可执行目标，不改动 `main.cpp`，因此不影响 ch00~ch20 的既有验证结论。

主程序的 `flash_attn.comp` 是 **prefill 思路**：每个 workgroup 处理一个 query 行，串行扫完整个序列。真实 LLM 解码时每步只有 1 个新 query，却要读不断增长的 KV cache，因此工业引擎把 decode 拆成不同 kernel：

| 阶段 | Shader | 作用 | 工业对应 |
|------|--------|------|----------|
| append | `kvcache_append.comp` | 把新 token 的 K/V 写到 cache 末尾，无需重排 | ncnn `sdpa_kvcache_append.comp` / `sdpa_kvcache_copy.comp` |
| split | `flash_attn_decode_phase1.comp` | cache 按 tile=32 分给多个 workgroup，各自算局部 online softmax 状态 `(m_i, l_i, o_i)` | ggml `flash_attn_decode_phase_1.comp` |
| reduce | `flash_attn_decode_phase2.comp` | 全局归约得到最终输出 | ggml `flash_attn_decode_phase_2.comp` / `flash_attn_split_k_reduce.comp` |

### 关键公式

phase1 输出的局部状态**未归一化**：`o_i = Σ exp(score − m_i)·V`、`l_i = Σ exp(score − m_i)`。phase2 用与 online softmax 同源的重缩放合并：

```
m = max_i m_i
O = Σ_i exp(m_i − m) · o_i
L = Σ_i exp(m_i − m) · l_i
final = O / L
```

这就是 FlashAttention 的 split-K 归约：把「一个 workgroup 扫全部」换成「多个 workgroup 分块 + 一次归约」，长上下文下的并行度从 1 提升到 `ceil(seqLen / tileSize)`。

### 验证

demo 取 `d=64`、prefill 长度 256、4 个 decode step（seqLen 257→260），每步同时跑「两阶段」与「单 kernel 基线」（复用本章 `flash_attn.spv`，只 dispatch 1 个 query 行），做 4 组检查：cache append 内容、两阶段 vs CPU 参考、单 kernel vs CPU 参考、两阶段 vs 单 kernel 交叉验证。4 步全部 **0 错误**。

> 本 demo **不宣称加速比**：短序列下两阶段多一次 dispatch 与全局同步，实测可能更慢；收益来自长上下文的并行度。这也解释了 ncnn 把 KV cache 做成独立 shader、ggml 把 decode 拆成 `phase_1` / `phase_2` 的做法。

### 动手

1. 把 `PROMPT_LEN` 改成 1024，观察 `tiles` 从 9 增长到 33，并用 GPU 时间戳对比单 kernel 与两阶段
2. 把 phase2 改成两级树形归约（先按 group 归约再全局归约），适应更长上下文
3. 给 append 加上写满后的 ring buffer / 滑动窗口逻辑

下一章（ch18）我们将探索 **量化体系深入**——从 int8 到 Q4_0/Q8_0/K-quant 再到 MXFP4 的 GPU 端反量化与融合推理。