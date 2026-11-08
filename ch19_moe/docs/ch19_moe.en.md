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

## 19.6 Comparison with the Current ncnn-MoE-Runtime (synced 2026-11)

> The `ExpertDispatcher` component cited in the first release was **removed** during the v0.1.0 architecture cleanup. At `63745f1` (2026-11-07) routing, expert acquisition and combination are split across `graph / engine / backends / storage`, and a model package is compiled into an **immutable execution graph** with expert weights kept outside the dense working set.

| Feature | This implementation | ncnn-MoE-Runtime (`63745f1`) |
|---------|--------------------|------------------------------|
| Execution graph | Three shaders submitted in order | Model package → immutable graph; multiple Sessions share weights |
| Routing | `router.comp`: softmax + top-k | `src/graph/router.h::forward_router()` → `ExpertDispatchPlan`; Vulkan side `backends/ncnn/router_vulkan.h`; shaders `router_select.comp` / `router_select_raw.comp` |
| Expert compute | All 8 experts run a full GEMM | `src/engine/expert.h::prepare_experts()/request_experts()/forward_moe()`; `expert_gather.comp` fetches only selected tokens; `expertbank_{gate_up,down,combine}.comp` performs the projections |
| Combine | `combine.comp` weighted sum | `backends/ncnn/moecombine_vulkan.h::MoeCombine_vulkan`; `moe_combine.comp` / `moe_combine_single_row.comp` + `route_aggregation` |
| Cache / balancing | Prints load distribution | ARC + `ExpertVictimCache` + `ExpertResidencyCoordinator` (`src/storage/`); "expert capacity" replaced by **cache lease limits** plus a GPU resident bank / async IO |
| Scheduling | Single session, single device | `BatchScheduler` (multi-session microbatching / DSpark speculative decoding), `HybridMode = Auto / CPU / Hybrid` |
| Weight formats | FP32 teaching weights | BF16 / FP8 E4M3 / INT4 / INT8 / MXFP4 / Q2_K~Q8_K / Q4_0 / IQ4; Safetensors directory or GGUF file |
| Models | Random weights | Qwen3 dense, Qwen3.5/3.6 dense+MoE, Qwen3.8-Flash-Next, GPT-OSS, DeepSeek-V4, Phi-mini-MoE, Ling-3.0-tiny (8 families) |

`src/kernels/vulkan/` now holds **60 `.comp` shaders** (no `gemm*.comp`; expert projections use `projection_matvec`, `bfloat16_cooperative_projection`, `qnk_projection`, `float8_projection`). MXFP4 has five dedicated kernels (`gate_up / indexed / projection / route_aggregation / row_tile`). The production path is the sparse chain `router_select` → `expert_gather` → ExpertBank → `moe_combine`, balanced by cache leases rather than a static expert capacity.

> For exercise #3 ("implement Gather/Scatter"), compare directly with `expert_gather.comp` and `moe_combine.comp`.

---

## 19.7 Comparison with ggml's MoE Kernels

`MUL_MAT_ID` is still ggml's MoE op, but the Vulkan entry point is now `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_mul_mat_id()` (declared in `ggml-vulkan-common.h`) with `mul_mm_id_funcs.glsl`. Routing gained dedicated `topk_moe.comp` and `count_experts.comp` kernels instead of relying on a single id matrix.

| Feature | This implementation | ggml-vulkan (current) |
|---------|--------------------|-----------------------|
| Routing | Explicit router shader | `topk_moe.comp` selects experts + `count_experts.comp` counts tokens per expert, then `MUL_MAT_ID` |
| Expert weights | Contiguous buffer, `expert_id` index | Expert weight views indexed by id, unpacked via `mul_mm_id_funcs.glsl` |
| Compute mode | One dispatch per expert | One `ggml_vk_mul_mat_id()` batches by expert (`ggml_vk_mul_mat_id_q_f16`, `max_tiles` estimates memory) |
| Sparsity | Always computes all experts | Only the selected experts' token rows |
| Adjacent capabilities | None | `flash_attn_decode_phase_*`, `split_k_reduce`, `sparse_compact` fused with FA; new WebGPU backend |

The core idea is unchanged — model "a different expert weight matrix per token" as a GPU-friendly indexing/batching operation — but routing and counting are now separate fused kernels, coordinated with memory estimation and the decode path.

---

## 19.8 Code Structure

```
ch19_moe/
├── CMakeLists.txt              # Two targets: ch19_moe / ch19_moe_sparse
├── main.cpp                    # Main: dense Router → Expert → Combine
├── main_sparse.cpp             # Advanced demo: sparse Gather / Scatter (see 19.11)
├── shaders/
│   ├── router.comp             # Router: Softmax + top-k (shared by both targets)
│   ├── expert_gemm.comp        # dense expert GEMM
│   ├── combine.comp            # dense combine: weighted sum
│   ├── count_experts.comp      # sparse: expert counts + prefix sum
│   ├── expert_gather.comp      # sparse: scatter into compact slots + reverse index
│   ├── expert_gemm_sparse.comp # sparse: selected (token, expert) pairs only
│   └── moe_combine_sparse.comp # sparse: gather-weighted sum via reverse index
└── docs/
    ├── ch19_moe.md             # Chinese documentation
    └── ch19_moe.en.md          # English documentation
```

---

## 19.9 Running

```bash
# From the repository root
cmake -B build -G Ninja
cmake --build build                                 # full build (24 executable targets)

# Main program (dense teaching version)
./build/ch19_moe/ch19_moe
./build/ch19_moe/ch19_moe --lang en

# Advanced demo (sparse Gather / Scatter, see 19.11)
./build/ch19_moe/ch19_moe_sparse
./build/ch19_moe/ch19_moe_sparse --lang en
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
3. **Implement Gather/Scatter**: Implement true expert dispatch (gather → compute → scatter) to reduce unnecessary computation — **a runnable reference implementation now exists, see 19.11 and `main_sparse.cpp`**
4. **Add auxiliary loss**: Implement load balancing auxiliary loss computation in the CPU reference
5. **Fuse Router**: Merge the Router logits computation (`x · W_router`) into the Router shader
6. **Compare top-k algorithms**: Implement a sort-based top-k selection and compare performance with the iterative maximum approach
7. **Parallel experts**: Modify main.cpp to use multiple command buffers for parallel expert GEMM execution

---

## 19.11 Advanced: Sparse Gather / Scatter MoE

`ch19_moe_sparse` is the runnable reference implementation of exercise #3, and the **code added after syncing the four reference repositories**. It replaces the main program's "compute every expert for every token, then discard 3/4 of the results during combine" with the sparse chain used by current inference engines. It is a separate executable target and does not touch `main.cpp`, so the existing ch00–ch20 verification results are unaffected.

```
router.spv → count_experts.spv → expert_gather.spv → expert_gemm_sparse.spv → moe_combine_sparse.spv
```

### Data flow

| Stage | Shader | Output | Industrial counterpart |
|-------|--------|--------|------------------------|
| Routing | `router.comp` (reused) | `indices[E]` / `weights[E]`, E = `num_tokens × top_k` = 64 | `router_select.comp`, `topk_moe.comp` |
| Count + prefix sum | `count_experts.comp` | `counts[e]`, `offsets[e]`, initial `cursor[e]` | `count_experts.comp`, `ExpertDispatchPlan` |
| Scatter | `expert_gather.comp` | `gather_token[slot]`, `gather_expert[slot]`, reverse index `gather_slot[entry]` | `expert_gather.comp` |
| Sparse GEMM | `expert_gemm_sparse.comp` | `partial[slot, d]`, only the 64 selected rows | `expertbank_gate_up.comp`, `ggml_vk_mul_mat_id()` |
| Reverse-index combine | `moe_combine_sparse.comp` | `final_out[t, d]` | `moe_combine.comp` / `moe_combine_single_row.comp` |

- `count_experts.comp` counts with shared atomics inside **one workgroup**; lane 0 performs a serial prefix sum (there are only 8 experts), avoiding cross-workgroup synchronization. Expert e owns the contiguous slot range `[offsets[e], offsets[e] + counts[e])`.
- `expert_gather.comp` places each routing entry into that range with `atomicAdd(cursor[e], 1)` and writes the reverse index. Hardware decides the slot order, but **correctness does not depend on it**: combine only uses `gather_slot`.
- `expert_gemm_sparse.comp` stages the token activation row in shared memory for reuse. If every thread re-reads the whole row from global memory, the sparse version is actually slower (measured speedup near 1×) — exactly why production engines follow gather with a tiled / coopmat GEMM.
- `moe_combine_sparse.comp` uses the reverse index, so the dense `[num_experts][num_tokens][dim]` output never has to be materialized.

### Verification and measurements

The demo uses the same RNG seed and dimensions as the main program, with the dense CPU reference as golden, and runs three checks:

1. `counts` / `offsets` / post-gather `cursor` (which should equal `offsets[e] + counts[e]`) match the CPU prefix sum; total slots = 64.
2. Each `(token, k)` owns exactly one slot: slots are unique, fall inside the right expert range, and `gather_token` / `gather_expert` match the routing decision.
3. Final output vs dense CPU golden: `errors=0, max error≈9.5e-7, RMSE≈1.8e-7`.

| Metric | dense main program | sparse advanced demo |
|--------|--------------------|----------------------|
| Expert GEMM rows | `8 × 32 = 256` | `32 × 2 = 64` |
| MAC | 1,048,576 | 262,144 |
| Theoretical reduction | — | **4×** (= `num_experts / top_k`) |
| GTX 1080 Ti measured | 8 dispatches | 1 dispatch, about **1.9–3.2×** |

A measured speedup below the theoretical bound is **expected and worth understanding**: at this teaching scale (32 tokens × 64 dims), workgroup scheduling, weight traffic and the two extra `count` / `gather` dispatches dilute the gain, and the numbers also vary with GPU clocks and cache state. Increasing `NUM_EXPERTS`, lowering `TOP_K`, enlarging `HIDDEN_DIM`, or replacing the sparse GEMM with ch04's 16×16 tiling / ch15's coopmat kernel moves it closer to the bound. This is also why ncnn-MoE-Runtime uses an ExpertBank with cache leases and ggml estimates memory with `max_tiles`: sparse MoE only pays off when **each expert gets enough tokens to batch**.

### Try it

1. Set `TOP_K` to 1 — the theoretical bound becomes 8×; check whether the measured speedup follows
2. Set `NUM_EXPERTS` to 16 (mind the 64-thread limit in `router.comp` and `s_counts[64]` in `count_experts.comp`) and observe load distribution and speedup
3. Replace the sparse GEMM with ch04's tiled version or ch15's coopmat version and compare
4. Replace the prefix sum with a multi-workgroup GPU scan (this demo uses a single-workgroup serial scan because there are only 8 experts)