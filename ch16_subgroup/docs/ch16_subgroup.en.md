# Chapter 16 · Subgroup Programming (Warp-Level Parallelism)

> Goal: Understand the lowest-level hardware execution unit on GPUs—**subgroups (warps/wavefronts)**—and learn to use subgroup operations to replace shared memory for more efficient reduction, broadcast, and shuffle communication patterns. This is a key optimization technique used by compilers and frameworks (ncnn, ggml).

---

## 16.1 What is a Subgroup

### Hardware Perspective

GPU "threads" are not fully independent. Hardware actually executes instructions in fixed-size **warps (NVIDIA)** or **wavefronts (AMD)** —threads within the same warp execute the same instruction in the same clock cycle (SIMD-style). Vulkan abstracts this hardware execution unit as a **subgroup**.

- **NVIDIA GPUs**: warp = 32 threads, subgroupSize = 32
- **AMD GPUs**: wavefront = 64 threads (GCN) or 32 threads (RDNA), subgroupSize = 32 or 64
- **Intel GPUs**: subgroupSize = 8 or 16 or 32

> Key insight: **The subgroup is the hardware-native concurrency granularity. Threads within a subgroup can exchange data at zero cost** (via registers), without going through shared memory.

### Workgroup vs Subgroup

```
┌─────────────────────────────────────┐
│          Workgroup                   │
│  ┌──────┬──────┬──────┬──────┐      │
│  │ SG 0 │ SG 1 │ SG 2 │ SG 3 │ ...  │  ← Subgroups (hardware-determined size)
│  └──────┴──────┴──────┴──────┘      │
│   Shared memory / barrier for       │
│   cross-subgroup communication      │
│   Subgroup ops: register-level      │
│   communication, zero latency       │
└─────────────────────────────────────┘
```

- **Workgroup**: programmer-specified logical grouping, variable size (e.g., 64, 128, 256), uses shared memory + barrier for intra-group collaboration.
- **Subgroup**: hardware-fixed physical execution unit, size determined by the device. Subgroup operations enable register-level communication within a subgroup, no shared memory, no barrier needed.

### Advantages of Subgroup Operations

| Feature | Shared Memory | Subgroup Operations |
|---------|---------------|---------------------|
| Latency | Tens of cycles (SRAM access) | 1-2 cycles (register bypass) |
| Synchronization | Requires barrier | Implicit (completed within same instruction) |
| Occupancy | Occupies shared memory, reduces active threads | No shared memory used, no occupancy impact |
| Code size | Multiple lines (shmem declaration + tree reduction + barrier) | One line |

---

## 16.2 Subgroup Operation Categories

### 16.2.1 Basic (Info Queries)

| GLSL Built-in | Meaning |
|--------------|---------|
| `gl_SubgroupSize` | Number of threads in current subgroup (device constant) |
| `gl_SubgroupInvocationID` | Index of current thread within subgroup (0 ~ size-1) |
| `gl_NumSubgroups` | Number of subgroups in current workgroup |
| `gl_SubgroupID` | Index of current subgroup within workgroup |

**Use cases**: Query hardware properties, dynamically adapt algorithms.

### 16.2.2 Vote

| Function | Meaning |
|----------|---------|
| `subgroupAll(cond)` | Is cond true for all threads in this subgroup? |
| `subgroupAny(cond)` | Is cond true for any thread in this subgroup? |
| `subgroupBallot(cond)` | Returns a uint bitmask; bit i = cond of lane i |
| `subgroupAllEqual(value)` | Is value the same for all threads? |

**Use cases**: Detect convergence (e.g., all threads exiting a loop), conditional branch statistics.

### 16.2.3 Arithmetic (Reductions)

| Function | Meaning |
|----------|---------|
| `subgroupAdd(value)` | Sum across subgroup |
| `subgroupMin(value)` | Minimum across subgroup |
| `subgroupMax(value)` | Maximum across subgroup |
| `subgroupMul(value)` | Product across subgroup |
| `subgroupAnd(value)` | Bitwise AND |
| `subgroupOr(value)` | Bitwise OR |
| `subgroupXor(value)` | Bitwise XOR |
| `subgroupInclusiveAdd(value)` | Prefix sum (including current thread) |
| `subgroupExclusiveAdd(value)` | Prefix sum (excluding current thread) |

**Use cases**: Replace shared memory tree reduction with a single instruction.

### 16.2.4 Ballot (Bitmask Operations)

| Function | Meaning |
|----------|---------|
| `subgroupBallot(cond)` | Returns 32/64-bit mask |
| `subgroupBallotBitCount(mask)` | Count 1-bits in mask |
| `subgroupBallotFindLSB(mask)` | Find least significant set bit |
| `subgroupBallotFindMSB(mask)` | Find most significant set bit |
| `bitCount(mask)` | GLSL built-in, standard bit count |

**Use cases**: Efficiently count threads satisfying a condition, dynamic warp management.

### 16.2.5 Shuffle (Register Data Exchange)

| Function | Meaning |
|----------|---------|
| `subgroupBroadcast(value, id)` | Broadcast value from lane `id` to all threads |
| `subgroupShuffle(value, id)` | Get value from lane `id` |
| `subgroupShuffleXor(value, mask)` | XOR-addressed butterfly exchange |
| `subgroupShuffleUp(value, delta)` | Get value from `id - delta` |
| `subgroupShuffleDown(value, delta)` | Get value from `id + delta` |

**Use cases**: Implement arbitrary data exchange patterns without shared memory. Foundation for many advanced algorithms.

---

## 16.3 Subgroup Broadcast: One Thread Shares, All Benefit

`subgroupBroadcast` is one of the most commonly used shuffle operations: it lets one thread in the subgroup share its value with all others.

```glsl
// Lane 0 holds the reduction result, broadcast to all threads
float result = subgroupBroadcast(partial_sum, 0);
```

**Typical scenario**: After tree reduction, `lane 0` holds the final result. Broadcast gives every thread access for subsequent processing. With shared memory, this requires: write smem + barrier + read smem — at least 3 instructions with multi-cycle latency. Subgroup broadcast is a single instruction.

---

## 16.4 Subgroup Shuffle: Register-to-Register Data Exchange

The `subgroupShuffle` family of functions allows threads to directly read another thread's register value, **without going through shared memory**.

```glsl
// Get value from the next thread (circular)
float next_val = subgroupShuffle(my_val, (gl_SubgroupInvocationID + 1) % gl_SubgroupSize);

// Get value from the previous thread
float prev_val = subgroupShuffleUp(my_val, 1);

// XOR shuffle: butterfly network exchange
float partner = subgroupShuffleXor(my_val, 1u << stage);
```

**Why it's fast**: Modern GPUs have a unified register file. Shuffle instructions route data through the register bypass network, with only 1-2 cycles of latency, far less than shared memory (~20-30 cycles + barrier synchronization overhead).

---

## 16.5 Subgroup Arithmetic Reductions

Traditional shared memory reduction (ch07) requires:
1. Declare `shared float smem[256]`
2. Each thread writes `smem[tid] = value`
3. `barrier()`
4. Tree loop: `for (s=128; s>0; s>>=1) { if (tid < s) smem[tid] += smem[tid+s]; barrier(); }`

Subgroup reduction is one line:
```glsl
float sum = subgroupAdd(value);
```

**Performance difference**: The subgroup version has no shared memory access latency, no barrier synchronization overhead, and doesn't consume shared memory resources (meaning higher occupancy).

### Example: Warp-Level Reduction

```glsl
// Each thread holds its own value
float val = in[gl_GlobalInvocationID.x];

// One line replaces the entire shared memory tree reduction
float sum = subgroupAdd(val);
float max_val = subgroupMax(val);
float min_val = subgroupMin(val);
```

---

## 16.6 Softmax Optimization: Subgroup Replaces Shared Memory

In ch08, softmax uses shared memory for two reductions (find max, compute sum). With subgroup operations, shared memory can be eliminated entirely:

```glsl
// Numerically stable warp-level softmax
float v = in[idx];                  // Each thread holds one element
float m = subgroupMax(v);           // One-step row max
float e = exp(v - m);               // Exponentiate
float sum = subgroupAdd(e);         // One-step sum
out[idx] = e / sum;                 // Normalize
```

**4 lines replace ~20 lines** of the shared memory version, eliminating barriers and shared memory access.

### Performance Comparison

Running `ch16_subgroup`'s benchmark shows the subgroup version is typically **2-5x faster** than the shared memory version (depending on GPU architecture):

```
Subgroup softmax:   0.15 μs/iter
Shared mem softmax: 0.45 μs/iter
Subgroup version is 3.0x faster
```

> Note: This benchmark measures single kernel launches. In real inference, the difference is amortized by larger kernels. But on hot paths (like attention score computation), these microsecond savings accumulate into significant speedups.

---

## 16.7 Subgroup Size Control Extension

`VK_EXT_subgroup_size_control` allows shaders to specify a preferred subgroup size, important for cross-platform optimization:

- **NVIDIA**: Fixed subgroupSize = 32
- **AMD RDNA**: Supports 32 or 64, selectable via extension
- **Intel**: Supports 8, 16, 32

```cpp
// C++ side check
if (ctx.features.subgroupSizeControl) {
    // Can request the compiler to use a specific subgroup size
    // Combined with VK_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT
}
```

Shader side:
```glsl
#extension GL_EXT_subgroup_size_control : require
layout(local_size_x = 32) in;
layout(subgroup_size = 32) in;  // Request size 32
```

---

## 16.8 Industry Comparison

### Subgroup Usage in ncnn

ncnn's Vulkan backend extensively uses subgroup operations for acceleration:

- **Packing conversion**: `subgroupShuffle` for efficient data rearrangement (e.g., AoS ↔ SoA)
- **Winograd convolution**: Subgroup accelerates data movement within tiles
- **Reduction operators**: `subgroupAdd` replaces shared memory reductions
- **Softmax**: Warp-level softmax for normalization within attention heads

### Subgroup Usage in ggml (llama.cpp)

ggml heavily uses subgroup operations in hot paths like `flash_attn` and `ggml_mul_mat`:

- **Flash Attention**: Uses `subgroupShuffle` and `subgroupAdd` for online softmax partial reductions within warps, avoiding shared memory bottlenecks
- **Matmul**: `subgroupBroadcast` to broadcast A matrix elements within warps, `subgroupShuffleXor` for butterfly reductions
- **RMSNorm**: `subgroupAdd` computes sum of squares in one step, replacing shared memory reduction

Key insight: **In production inference engines, every operator involving "intra-group collaboration" will preferentially use subgroup operations over shared memory when subgroup support is available**.

---

## 16.9 Exercises

1. **Check subgroup size**: What is your device's subgroupSize? If `subgroupSizeControl` is supported, try forcing different subgroup sizes.
2. **Implement warp-level prefix sum**: Use `subgroupInclusiveAdd` or `subgroupExclusiveAdd` to implement a parallel prefix sum (scan).
3. **Butterfly reduction**: Implement a manual reduction using `subgroupShuffleXor` (without `subgroupAdd`) to appreciate the flexibility of shuffle.
4. **Extend softmax**: When N > subgroupSize, how would you extend the warp-level softmax? (Hint: each thread processes multiple elements, or multiple subgroups cooperate.)
5. **Read ncnn source**: Examine ncnn's Vulkan backend `softmax` implementation to see if it uses subgroup operations.

---

## 16.10 Summary

- The subgroup is the GPU's native hardware execution unit (warp/wavefront). Threads within a subgroup share the program counter and execute the same instruction.
- Subgroup operations provide register-level communication primitives with much lower latency than shared memory, and they don't consume shared memory resources.
- Five operation categories: **Basic** (info queries), **Vote** (condition testing), **Arithmetic** (reductions), **Ballot** (bitmasks), **Shuffle** (data exchange).
- **Warp-level softmax** accomplishes reduction + normalization in just 4 lines—a classic example of subgroup optimization.
- Production inference engines (ncnn, ggml) extensively use subgroup operations to replace shared memory on hot paths.

Next chapter (ch17) explores **Cooperative Matrix**—Vulkan's approximation of Tensor Core matrix multiplication acceleration hardware.