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

## 19.6 与 ncnn-MoE-Runtime 现行架构对比（2026-11 同步）

> 首版课程引用的 `ExpertDispatcher` 已在 v0.1.0 的架构清理中**移除**。`63745f1`（2026-11-07）把路由、专家取用、组合拆成 `graph / engine / backends / storage` 四层，并把「模型包」编译成**不可变执行图**，Expert 权重外置于 dense working set。

| 特性 | 本章实现 | ncnn-MoE-Runtime（`63745f1`） |
|------|---------|------------------------------|
| 执行图 | 三个 shader 顺序提交 | 模型包 → 不可变执行图；多 Session 共享权重 |
| 路由 | `router.comp`：softmax + top-k | `src/graph/router.h::forward_router()` → `ExpertDispatchPlan`；Vulkan 侧 `backends/ncnn/router_vulkan.h`；shader `router_select.comp` / `router_select_raw.comp` |
| 专家计算 | 8 个专家全量 GEMM | `src/engine/expert.h::prepare_experts()/request_experts()/forward_moe()`；`expert_gather.comp` 只取被选中的 token；`expertbank_{gate_up,down,combine}.comp` 承担专家投影 |
| 组合 | `combine.comp` 加权求和 | `backends/ncnn/moecombine_vulkan.h::MoeCombine_vulkan`；`moe_combine.comp` / `moe_combine_single_row.comp`；另有 `route_aggregation` |
| 缓存 / 均衡 | 打印负载分布 | ARC + `ExpertVictimCache` + `ExpertResidencyCoordinator`（`src/storage/`）；原「专家容量控制」改为 **cache lease 限额** + GPU resident bank / 异步 IO |
| 调度 | 单 Session、单设备 | `BatchScheduler`（多 Session 微批 / DSpark 投机解码）、`HybridMode = Auto / CPU / Hybrid`（`include/ncnn/moe/option.h`） |
| 权重格式 | FP32 教学权重 | BF16 / FP8 E4M3 / INT4 / INT8 / MXFP4 / Q2_K~Q8_K / Q4_0 / IQ4；Safetensors 目录或 GGUF 文件 |
| 模型 | 随机权重 | Qwen3 dense、Qwen3.5/3.6 dense+MoE、Qwen3.8-Flash-Next、GPT-OSS、DeepSeek-V4、Phi-mini-MoE、Ling-3.0-tiny（8 类） |

`src/kernels/vulkan/` 现在共 **60 个 `.comp`**（其中没有 `gemm*.comp`，专家投影改用 `projection_matvec`、`bfloat16_cooperative_projection`、`qnk_projection`、`float8_projection`），MXFP4 有五件专用 kernel（`gate_up / indexed / projection / route_aggregation / row_tile`）。本章「全量计算 + combine 筛选」仍是教学版；生产路径是 `router_select` → `expert_gather` → ExpertBank → `moe_combine` 的稀疏链路，靠缓存 lease 而非静态专家容量做均衡。

> 做本章练习 #3「实现 Gather/Scatter」时，可直接对照 `expert_gather.comp` 与 `moe_combine.comp`。

---

## 19.7 与 ggml 的 MoE 内核对比

`MUL_MAT_ID` 仍是 ggml 的 MoE 算子，但 Vulkan 后端入口现在明确为 `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_mul_mat_id()`（声明在 `ggml-vulkan-common.h`），配套 `mul_mm_id_funcs.glsl`；路由侧新增 `topk_moe.comp` 与 `count_experts.comp`，不再只靠一张 id 矩阵。

| 特性 | 本章实现 | ggml-vulkan（当前） |
|------|---------|---------------------|
| 路由方式 | 显式 Router shader | `topk_moe.comp` 选专家 + `count_experts.comp` 统计每专家 token 数，再进 `MUL_MAT_ID` |
| 专家权重存储 | 连续 buffer，expert_id 索引 | 按 id 索引的专家权重视图，配合 `mul_mm_id_funcs.glsl` 解包 |
| 计算模式 | 每个专家独立 dispatch | 单个 `ggml_vk_mul_mat_id()` 内按专家分批（`ggml_vk_mul_mat_id_q_f16` / `max_tiles` 预估显存） |
| 稀疏性 | 始终计算所有专家 | 只计算被选中专家的 token 行 |
| 相邻能力 | 无 | `flash_attn_decode_phase_*`、`split_k_reduce`、`sparse_compact` 与 FA 融合；新增 WebGPU 后端 |

核心思路未变：把「每个 token 选不同专家权重」建模成 GPU 友好的索引/分批操作；变化是**路由与计数已融合成独立 kernel**，并与显存预估、decode 路径协同。

---

## 19.8 代码结构

```
ch19_moe/
├── CMakeLists.txt              # 两个可执行目标：ch19_moe / ch19_moe_sparse
├── main.cpp                    # 主程序：dense Router → Expert → Combine
├── main_sparse.cpp             # 进阶 demo：稀疏 Gather / Scatter（见 19.11）
├── shaders/
│   ├── router.comp             # 路由着色器：Softmax + top-k（两个目标共用）
│   ├── expert_gemm.comp        # dense 专家 GEMM 着色器
│   ├── combine.comp            # dense 组合着色器：加权求和
│   ├── count_experts.comp      # 稀疏：专家计数 + prefix sum
│   ├── expert_gather.comp      # 稀疏：scatter 到紧凑槽位 + 反查表
│   ├── expert_gemm_sparse.comp # 稀疏：只算被选中的 (token, expert) 对
│   └── moe_combine_sparse.comp # 稀疏：按反查表 gather 加权和
└── docs/
    ├── ch19_moe.md             # 中文文档
    └── ch19_moe.en.md          # 英文文档
```

---

## 19.9 运行

```bash
# 在仓库根目录
cmake -B build -G Ninja
cmake --build build                                 # 全量构建（含 24 个可执行目标）

# 主程序（dense 教学版）
./build/ch19_moe/ch19_moe
./build/ch19_moe/ch19_moe --lang en

# 进阶 demo（稀疏 Gather / Scatter，见 19.11）
./build/ch19_moe/ch19_moe_sparse
./build/ch19_moe/ch19_moe_sparse --lang en
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
3. **实现 Gather/Scatter**：实现真正的 expert dispatch（gather → 计算 → scatter），减少不必要的计算 —— **已有可运行参考实现，见 19.11 与 `main_sparse.cpp`**
4. **添加辅助损失**：在 CPU 参考中实现负载均衡辅助损失计算
5. **融合 Router**：将 Router 的 logits 计算（`x · W_router`）合并到 Router 着色器中
6. **对比不同 top-k 选择算法**：实现基于排序的 top-k 选择，与本章的迭代最大值选择对比性能

---

## 19.11 进阶实现：稀疏 Gather / Scatter MoE

`ch19_moe_sparse` 是练习 #3 的可运行参考实现，也是**本次同步四个参考仓库后新增的代码**：它把主程序「所有专家 × 所有 token 全量计算、combine 时再筛掉 3/4」换成现行工业引擎实际使用的稀疏链路。它是独立可执行目标，不改动 `main.cpp`，因此不影响 ch00~ch20 的既有验证结论。

```
router.spv → count_experts.spv → expert_gather.spv → expert_gemm_sparse.spv → moe_combine_sparse.spv
```

### 数据流

| 阶段 | Shader | 输出 | 对应工业实现 |
|------|--------|------|--------------|
| 路由 | `router.comp`（复用主程序） | `indices[E]` / `weights[E]`，E = `num_tokens × top_k` = 64 | `router_select.comp`、`topk_moe.comp` |
| 计数 + prefix sum | `count_experts.comp` | `counts[e]`、`offsets[e]`、初值游标 `cursor[e]` | `count_experts.comp`、`ExpertDispatchPlan` |
| Scatter | `expert_gather.comp` | `gather_token[slot]`、`gather_expert[slot]`、反查表 `gather_slot[entry]` | `expert_gather.comp` |
| 稀疏 GEMM | `expert_gemm_sparse.comp` | `partial[slot, d]`，只算被选中的 64 行 | `expertbank_gate_up.comp`、`ggml_vk_mul_mat_id()` |
| 反查组合 | `moe_combine_sparse.comp` | `final_out[t, d]` | `moe_combine.comp` / `moe_combine_single_row.comp` |

- `count_experts.comp` 在**单个 workgroup** 内用 shared atomic 计数，由 lane 0 做串行 prefix sum（专家数只有 8），避免跨 workgroup 的全局同步。专家 e 占用连续槽位 `[offsets[e], offsets[e] + counts[e])`。
- `expert_gather.comp` 用 `atomicAdd(cursor[e], 1)` 把每个路由条目放进该专家区间，同时写反查表。槽位分配顺序由硬件决定，但**不影响正确性**：combine 只依赖 `gather_slot`，不依赖顺序。
- `expert_gemm_sparse.comp` 把 token 激活行放进 shared memory 复用。如果每个线程各自从全局内存重读整行，稀疏版反而会被访存拖慢（实测加速比接近 1×）——这正是工业实现在 gather 之后必须接 tiled / coopmat GEMM 的原因。
- `moe_combine_sparse.comp` 用反查表取结果，不必物化 `[num_experts][num_tokens][dim]` 的稠密专家输出。

### 验证与实测

程序使用与主程序相同的随机种子与维度，`dense` CPU 参考作为 golden，做三组检查：

1. `counts` / `offsets` / gather 结束后的 `cursor`（应等于 `offsets[e] + counts[e]`）与 CPU prefix sum 一致，槽位总数 = 64；
2. 每个 `(token, k)` 恰好占一个槽位：槽位唯一、落在对应专家区间、`gather_token` / `gather_expert` 与路由一致；
3. 最终输出对比 dense CPU golden：`错误数=0，最大误差≈9.5e-7，RMSE≈1.8e-7`。

| 指标 | dense 主程序 | sparse 进阶 demo |
|------|--------------|------------------|
| 专家 GEMM 行数 | `8 × 32 = 256` | `32 × 2 = 64` |
| MAC | 1,048,576 | 262,144 |
| 理论节省 | — | **4×**（= `num_experts / top_k`） |
| GTX 1080 Ti 实测 | 8 次 dispatch | 1 次 dispatch，约 **1.9~3.2×** |

实测低于理论上界是**正常且值得理解**的：教学规模（32 token × 64 维）太小，workgroup 调度、权重访存和 `count` / `gather` 两次额外 dispatch 都在摊薄收益；测量还会随 GPU 频率与缓存状态波动。把 `NUM_EXPERTS` 调大、`TOP_K` 调小、放大 `HIDDEN_DIM`，或把稀疏 GEMM 换成 ch04 的 16×16 tiling / ch15 的 coopmat，才会逐步逼近上界。这也解释了 ncnn-MoE-Runtime 为何用 ExpertBank + 缓存 lease、ggml 为何用 `max_tiles` 预估显存来**批量化**专家计算：稀疏化的收益要靠「把每个专家的 token 攒够」才能兑现。

### 动手

1. 把 `TOP_K` 改成 1，理论上界变为 8×，观察实测加速比是否同步变化
2. 把 `NUM_EXPERTS` 改成 16（注意 `router.comp` 的 64 线程上限与 `count_experts.comp` 的 `s_counts[64]` 上限），观察负载分布与加速比
3. 把稀疏 GEMM 换成 ch04 的 tiled 版本或 ch15 的 coopmat 版本，对比加速比
4. 把 prefix sum 换成多 workgroup 的 GPU scan（本 demo 因专家数只有 8，用单 workgroup 串行 scan）
7. **专家并行**：修改 main.cpp 使用多个命令缓冲并行执行多个专家 GEMM