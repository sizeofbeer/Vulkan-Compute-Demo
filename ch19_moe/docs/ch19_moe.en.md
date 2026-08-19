# Chapter 19 · MoE Routing (Mixture of Experts)

> Goal: Understand the core routing mechanism of **Mixture of Experts (MoE)** architecture—Router (top-k selection) + Expert dispatch + Combine—and implement a complete MoE forward pass on the GPU. This is the key technology behind modern large language models (e.g., Mixtral 8x7B, DeepSeek series) for achieving sparse activation.

---

## 19.1 What is MoE Architecture

### Motivation

In traditional Transformers, the FFN (Feed-Forward Network) layer applies the **same parameters** to every token. As models scale up, computational cost grows linearly with parameter count—but not every token needs all parameters. Some tokens may only need a subset of "experts."

**MoE (Mixture of Experts)** core idea: maintain a pool of **experts** (each is an independent FFN or MLP), use a **router** to dynamically select the top-k most suitable experts for each token, compute only those experts, and combine the results via weighted sum.

### Architecture Diagram

```
Input Tokens: [t0, t1, t2, ..., tn]
                    │
                    ▼
            ┌──────────────┐
            │   Router     │  ← Softmax over num_experts
            │  (top-k sel) │
            └──────┬───────┘
                    │  Output: top-k expert indices per token
                    ▼
    ┌───────┬───────┬───────┬───────┐
    │ Exp 0 │ Exp 1 │ Exp 2 │ Exp 3 │  ...  ← Parallel expert compute
    └───┬───┴───┬───┴───┬───┴───┬───┘
        │       │       │       │
        └───────┴───┬───┴───────┘
                    │
            ┌───────▼──────────┐
            │    Combine       │  ← Weighted sum
            └───────┬──────────┘
                    │
              Output Tokens
```

### Key Formula

For each token $x$, the MoE layer output is:

$$y = \sum_{i=1}^{k} w_i \cdot E_i(x)$$

Where:
- $E_i$ is the $i$-th selected expert
- $w_i$ is the routing weight from the router (softmax value)
- $k$ is the top-k parameter (typically 1 or 2)

Router computation:

$$w_i = \text{Softmax}(\text{top-k}(\text{logits}))_i$$

$$\text{logits} = x \cdot W_{\text{router}}$$

---

## 19.2 Router: Top-k Routing

### Softmax Normalization

The router first computes logits for each token-expert pair, then applies softmax along the expert dimension:

$$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{E} e^{x_j - \max(x)}}$$

In our GPU implementation, we use **subgroupAdd** for warp-level reduction, avoiding shared memory overhead:

```glsl
// Each thread holds its expert's logit value
float v = logits[tid * num_experts + lid];

// Step 1: Row maximum (numerical stability)
float row_max = subgroupMax(v);

// Step 2: exp(x - max)
float e = exp(v - row_max);

// Step 3: Sum
float sum = subgroupAdd(e);

// Step 4: Normalize
float softmax_val = e / sum;
```

### Top-k Selection

After softmax, select the k experts with the highest weights. In our implementation, we use shared memory with iterative maximum finding:

```
for k = 0 to top_k-1:
    Thread 0 scans all experts, finds the unselected maximum
    Record expert index and weight
    Mark that expert as selected (set weight to -1)
    Synchronize (barrier)
```

> Note: For small expert counts (8-64), this simple O(k × E) algorithm is efficient enough. For larger scales, parallel selection algorithms can be used.

---

## 19.3 Expert Dispatch: Routing Tokens to Experts

### Standard Approach

In production systems, expert dispatch involves:

1. **Gather**: Collect the tokens assigned to each expert based on router output
2. **Compute**: Run each expert's FFN on its assigned tokens
3. **Scatter**: Write results back to the original token positions

### This Implementation

This chapter uses a simplified approach:
- Each expert runs GEMM on ALL tokens (only assigned tokens contribute in the combine phase)
- Expert weight matrices are stored in a contiguous `[num_experts, K, N]` buffer
- The `expert_id` push constant indexes into the correct expert's weights

Expert GEMM uses standard 2D tiled matrix multiplication (same as Chapter 4), with each workgroup handling a 16×16 tile of the output matrix.

---

## 19.4 Combine: Weighted Sum

The combine phase sums each expert's output weighted by the routing weights:

```glsl
for each token t, each hidden dim d:
    final_out[t, d] = Σ_{k=0}^{top_k-1} weights[t, k] × exp_out[indices[t, k], t, d]
```

Each thread handles one `(token, hidden_dim)` element, looping over the top_k expert contributions.

---

## 19.5 Load Balancing

### Why Load Balancing Matters

If the router tends to assign tokens to a few experts, it causes:
- **Hot experts**: Compute bottleneck while other experts idle
- **Memory fragmentation**: Each expert must pre-allocate for max tokens
- **Poor batch efficiency**: GPU SIMD characteristics require uniform workloads

### Load Balancing Strategies

| Strategy | Description | Cost |
|----------|-------------|------|
| Auxiliary Loss | Add load balancing regularization during training | Increases training complexity |
| Expert Capacity | Limit each expert to at most C tokens | May cause token overflow |
| Random Routing | Introduce randomness in top-k selection | Reduces routing quality |
| Dynamic Scheduling | Adjust expert assignment at runtime | Complex to implement |

### Load Analysis in This Experiment

The main.cpp outputs per-expert token counts, load standard deviation, and coefficient of variation, providing an intuitive understanding of load distribution.

---

## 19.6 Comparison with ncnn-MoE-Runtime's ExpertDispatcher

**ncnn-MoE-Runtime**'s `ExpertDispatcher` component implements similar MoE routing:

| Feature | This Implementation | ncnn ExpertDispatcher |
|---------|-------------------|----------------------|
| Routing | Softmax + top-k | Softmax + top-k |
| Expert GEMM | Tiled GEMM | Tiled GEMM + fused activation |
| Token dispatch | Full compute + combine filter | Gather → Compute → Scatter |
| Load balancing | Analysis tools | Expert capacity control |
| Push constant | expert_id | Expert metadata |

ncnn's implementation prioritizes deployment efficiency using gather/scatter patterns to avoid unnecessary computation. This chapter's implementation focuses on pedagogical clarity—cleanly separating the Router → Expert → Combine phases.

---

## 19.7 Comparison with ggml's MUL_MAT_ID

**ggml**'s `MUL_MAT_ID` operation is specifically designed for MoE sparse matrix multiplication:

| Feature | This Implementation | ggml MUL_MAT_ID |
|---------|-------------------|-----------------|
| Routing | Explicit Router shader | Implicit selection via id matrix |
| Expert weights | Contiguous buffer, expert_id index | Separate weight matrix list |
| Compute mode | Per-expert dispatch | All experts in a single operation |
| Sparsity | Always computes all experts | Only computes selected experts |
| Use case | Teaching demo | Production inference engine |

`MUL_MAT_ID`'s core idea is to model "selecting different expert weight matrices per token" as a GPU-friendly index operation, avoiding explicit gather/scatter steps.

---

## 19.8 Code Structure

```
ch19_moe/
├── CMakeLists.txt
├── main.cpp                    # Main: Router → Expert → Combine
├── shaders/
│   ├── router.comp             # Router shader: Softmax + top-k
│   ├── expert_gemm.comp        # Expert GEMM shader
│   └── combine.comp            # Combine shader: weighted sum
└── docs/
    ├── ch19_moe.md             # Chinese documentation
    └── ch19_moe.en.md          # English documentation
```

---

## 19.9 Running

```bash
# Build
cd build
cmake ..
make ch19_moe

# Run
./ch19_moe

# English mode
./ch19_moe --lang en
```

### Expected Output

```
===== Chapter 19: MoE Routing (Mixture of Experts) =====
  Tokens: 32
  Experts: 8
  top_k: 2
  Hidden dim: 64

--- Reference Routing (first 8 tokens) ---
  Token 0: expert3(0.42) expert7(0.31)
  Token 1: expert1(0.38) expert5(0.29)
  ...

--- Step 1: Router (Softmax + Top-2) ---
  Router indices errors=0, weights RMSE=0.00012

--- Step 2: Expert GEMM (8 experts) ---
  Expert 0 computed (load=7)
  Expert 1 computed (load=9)
  ...

--- Step 3: Combine ---
  Combine errors=0, max error=0.00023, RMSE=0.00008

--- Expert Load Distribution ---
  Total assignments: 64 (=32 × 2)
  Ideal average load: 8
  Expert 0: 7 tokens (87.5% balanced)
  Expert 1: 9 tokens (112.5% balanced)
  ...
  Load stddev: 1.25
  Load CV: 15.6%
```

---

## 19.10 Exercises

1. **Modify top_k**: Change `TOP_K` to 1 or 3 and observe routing results and load distribution
2. **Increase expert count**: Change `NUM_EXPERTS` to 16 or 32 and observe Router performance
3. **Implement Gather/Scatter**: Implement true expert dispatch (gather → compute → scatter) in main.cpp to reduce unnecessary computation
4. **Add auxiliary loss**: Implement load balancing auxiliary loss computation in the CPU reference
5. **Fuse Router**: Merge the Router logits computation (`x · W_router`) into the Router shader
6. **Compare top-k algorithms**: Implement a sort-based top-k selection and compare performance with the iterative maximum approach
7. **Parallel experts**: Modify main.cpp to use multiple command buffers for parallel expert GEMM execution