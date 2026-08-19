# 第 19 章 · MoE 路由（Mixture of Experts）

> 目标：理解 **Mixture of Experts（MoE）** 架构的核心路由机制——Router（top-k 选择）+ Expert dispatch + Combine，并在 GPU 上实现完整的 MoE 前向传播流程。这是现代大语言模型（如 Mixtral 8x7B、DeepSeek 系列）实现稀疏激活的关键技术。

---

## 19.1 什么是 MoE 架构

### 动机

传统 Transformer 的 FFN（前馈网络）层对所有 token 使用**相同的参数**。随着模型规模扩大，计算量随参数数量线性增长，但并非每个 token 都需要全部参数——某些 token 可能只需要部分"专家"来处理。

**MoE（Mixture of Experts）** 的核心思想是：维护一组 **专家（expert）**，每个专家是一个独立的 FFN（或 MLP），通过一个 **路由器（router）** 为每个 token 动态选择最合适的 top-k 个专家，只激活这些专家进行计算，然后将结果加权求和。

### 架构图

```
输入 Token: [t0, t1, t2, ..., tn]
                │
                ▼
         ┌──────────────┐
         │   Router     │  ← Softmax over num_experts
         │  (top-k 选择) │
         └──────┬───────┘
                │ 输出: 每个 token 选 top-k 个专家
                ▼
    ┌───────┬───────┬───────┬───────┐
    │ Exp 0 │ Exp 1 │ Exp 2 │ Exp 3 │  ...  ← 专家并行计算
    └───┬───┴───┬───┴───┬───┴───┬───┘
        │       │       │       │
        └───────┴───┬───┴───────┘
                    │
         ┌──────────▼──────────┐
         │      Combine        │  ← 加权求和
         └──────────┬──────────┘
                    │
              输出 Token
```

### 关键公式

对于每个 token $x$，MoE 层的输出为：

$$y = \sum_{i=1}^{k} w_i \cdot E_i(x)$$

其中：
- $E_i$ 是第 $i$ 个被选中的专家
- $w_i$ 是路由器输出的权重（softmax 值）
- $k$ 是 top-k 参数（通常为 1 或 2）

路由器计算：

$$w_i = \text{Softmax}(\text{top-k}(\text{logits}))_i$$

$$\text{logits} = x \cdot W_{\text{router}}$$

---

## 19.2 Router：Top-k 路由

### Softmax 归一化

路由器首先计算每个 token 对应每个专家的 logits，然后沿 expert 维度做 softmax：

$$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{E} e^{x_j - \max(x)}}$$

在我们的 GPU 实现中，使用 **subgroupAdd** 进行 warp 级归约，避免了共享内存的开销：

```glsl
// 每个线程持有自己专家的 logit 值
float v = logits[tid * num_experts + lid];

// Step 1: 行最大值（数值稳定）
float row_max = subgroupMax(v);

// Step 2: exp(x - max)
float e = exp(v - row_max);

// Step 3: 求和
float sum = subgroupAdd(e);

// Step 4: 归一化
float softmax_val = e / sum;
```

### Top-k 选择

Softmax 之后，选择权重最大的 k 个专家。在我们的实现中，使用共享内存 + 迭代最大值查找：

```
for k = 0 to top_k-1:
    在线程 0 中遍历所有专家，找到未选中的最大权重
    记录专家索引和权重
    将该专家标记为已选（权重设为 -1）
    同步（barrier）
```

> 注：对于小规模专家（8-64 个），这种简单的 O(k × E) 算法足够高效。对于更大规模，可以使用并行选择算法。

---

## 19.3 Expert Dispatch：分发 Token 到专家

### 标准做法

在实际系统中，expert dispatch 涉及：

1. **Gather**：根据路由器输出，收集每个专家被分配的 token
2. **计算**：对每个专家，在其分配的 token 上运行 FFN
3. **Scatter**：将结果写回原始 token 位置

### 本实现

本章使用简化方案：
- 每个专家对所有 token 执行 GEMM（但只有被分配的 token 在 combine 阶段被使用）
- 专家权重矩阵存储在 `[num_experts, K, N]` 的连续 buffer 中
- 通过 push constant 中的 `expert_id` 索引到对应专家的权重

专家 GEMM 使用标准的 2D tiled 分块矩阵乘（与第 4 章相同），每个工作组处理输出矩阵的 16×16 小块。

---

## 19.4 Combine：加权求和

Combine 阶段将各专家的输出按路由权重加权求和：

```glsl
for each token t, each hidden dim d:
    final_out[t, d] = Σ_{k=0}^{top_k-1} weights[t, k] × exp_out[indices[t, k], t, d]
```

每个线程处理一个 `(token, hidden_dim)` 元素，循环累加 top_k 个专家的贡献。

---

## 19.5 负载均衡

### 为什么负载均衡重要

如果路由器倾向于将 token 分配给少数专家，会导致：
- **热点专家**：计算瓶颈，其他专家空闲
- **显存碎片**：每个专家需要预分配最大 token 数
- **批量效率低**：GPU 的 SIMD 特性要求工作负载均匀

### 负载均衡策略

| 策略 | 描述 | 代价 |
|------|------|------|
| 辅助损失（Auxiliary Loss） | 在训练时添加负载均衡正则项 | 增加训练复杂度 |
| 专家容量（Expert Capacity） | 限制每个专家最多处理 C 个 token | 可能导致 token 溢出 |
| 随机路由 | 在 top-k 中引入随机性 | 降低路由质量 |
| 动态调度 | 运行时动态调整专家分配 | 实现复杂 |

### 本实验的负载分析

本章的 main.cpp 会输出每个专家分配的 token 数、负载标准差和变异系数，帮助直观理解负载分布。

---

## 19.6 与 ncnn-MoE-Runtime 的 ExpertDispatcher 对比

**ncnn-MoE-Runtime** 的 `ExpertDispatcher` 组件实现了类似的 MoE 路由机制：

| 特性 | 本章实现 | ncnn ExpertDispatcher |
|------|---------|----------------------|
| 路由算法 | Softmax + top-k | Softmax + top-k |
| 专家 GEMM | 分块 tiled GEMM | 分块 GEMM + 融合激活 |
| Token 分发 | 全量计算 + combine 筛选 | Gather → 计算 → Scatter |
| 负载均衡 | 分析工具 | 专家容量控制 |
| Push constant | 传递 expert_id | 传递 expert 元数据 |

ncnn 的实现更注重实际部署效率，使用了 gather/scatter 模式来避免不必要的计算。本章的实现更侧重于教学——清晰地分离 Router → Expert → Combine 三个阶段。

---

## 19.7 与 ggml 的 MUL_MAT_ID 对比

**ggml** 的 `MUL_MAT_ID` 操作专门用于 MoE 的稀疏矩阵乘法：

| 特性 | 本章实现 | ggml MUL_MAT_ID |
|------|---------|-----------------|
| 路由方式 | 显式 Router shader | 通过 id 矩阵隐式选择 |
| 专家权重存储 | 连续 buffer，expert_id 索引 | 独立的权重矩阵列表 |
| 计算模式 | 每个专家独立 dispatch | 单个操作内处理所有专家 |
| 稀疏性 | 始终计算所有专家 | 只计算被选中的专家 |
| 适用场景 | 教学演示 | 生产推理引擎 |

`MUL_MAT_ID` 的核心思路是将"为每个 token 选择不同的专家权重矩阵"建模为 GPU 友好的索引操作，避免了显式的 gather/scatter 步骤。

---

## 19.8 代码结构

```
ch19_moe/
├── CMakeLists.txt
├── main.cpp                    # 主程序：Router → Expert → Combine
├── shaders/
│   ├── router.comp             # 路由着色器：Softmax + top-k
│   ├── expert_gemm.comp        # 专家 GEMM 着色器
│   └── combine.comp            # 组合着色器：加权求和
└── docs/
    ├── ch19_moe.md             # 中文文档
    └── ch19_moe.en.md          # 英文文档
```

---

## 19.9 运行

```bash
# 构建
cd build
cmake ..
make ch19_moe

# 运行
./ch19_moe

# 英文模式
./ch19_moe --lang en
```

### 预期输出

```
===== 第 19 章：MoE 路由 (Mixture of Experts) =====
  Token 数: 32
  Expert 数: 8
  top_k: 2
  隐藏维度: 64

--- 参考路由结果（前 8 个 token）---
  Token 0: expert3(0.42) expert7(0.31)
  Token 1: expert1(0.38) expert5(0.29)
  ...

--- Step 1: Router (Softmax + Top-2) ---
  Router indices 错误数=0，权重 RMSE=0.00012

--- Step 2: Expert GEMM (8 个专家) ---
  专家 0 已计算 (负载=7)
  专家 1 已计算 (负载=9)
  ...

--- Step 3: Combine ---
  Combine 错误数=0，最大误差=0.00023，RMSE=0.00008

--- 专家负载分布 ---
  总分配次数: 64 (=32 × 2)
  理想平均负载: 8
  专家 0: 7 个 token (87.5% 均衡)
  专家 1: 9 个 token (112.5% 均衡)
  ...
  负载标准差: 1.25
  负载变异系数: 15.6%
```

---

## 19.10 练习

1. **修改 top_k**：将 `TOP_K` 改为 1 或 3，观察路由结果和负载分布的变化
2. **增加专家数量**：将 `NUM_EXPERTS` 改为 16 或 32，观察 Router 性能变化
3. **实现 Gather/Scatter**：在 main.cpp 中实现真正的 expert dispatch（gather → 计算 → scatter），减少不必要的计算
4. **添加辅助损失**：在 CPU 参考中实现负载均衡辅助损失计算
5. **融合 Router**：将 Router 的 logits 计算（`x · W_router`）合并到 Router 着色器中
6. **对比不同 top-k 选择算法**：实现基于排序的 top-k 选择，与本章的迭代最大值选择对比性能
7. **专家并行**：修改 main.cpp 使用多个命令缓冲并行执行多个专家 GEMM