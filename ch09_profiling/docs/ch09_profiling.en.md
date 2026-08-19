# Chapter 9 · Performance Analysis Methodology

> Goal: learn to **measure for real** instead of **guessing**. Use GPU hardware timestamps for accurate timing, use the Roofline model to judge whether the bottleneck is bandwidth or compute, and know which vendor tool to use for deep profiling.

---

## 9.1 The First Iron Rule: CPU timing without waiting on the fence is fake

The most common beginner mistake: recording "elapsed time" with `std::chrono` right after `submit`. But Vulkan is **asynchronous** — `submit` only "sends out" the commands; the GPU may not have started computing yet while the CPU has already reached the timing point. Real elapsed time is only complete after `waitForFences` returns.

This chapter's `main.cpp` measures two numbers:
- **CPU wall clock (including submission)**: from `submit` to `waitForFences` returning — includes queue submission, driver, and synchronization overhead, usually **inflated**, and does not represent pure computation.
- **GPU timestamp**: using `VK_QUERY_TYPE_TIMESTAMP`, insert two `writeTimestamp` calls (compute shader stage) into the command stream; the GPU itself marks "actually started computing" and "finished computing". This is the **real computation time**.

> The gap is often several times. So the first step of performance optimization: **use the right timing method**, otherwise you may be optimizing "submission overhead" instead of the "kernel" after a long time.

## 9.2 How to use timestamp queries

1. Create pool: `createQueryPool({..., eTimestamp, 2, 0})` (2 slots: start/end).
2. Record: `cmd.writeTimestamp(eComputeShader, pool, 0)` → your dispatch → `writeTimestamp(..., pool, 1)`.
3. Submit and wait for fence.
4. Read results: `getQueryPoolResults(..., eWait | e64)` to get two `uint64_t` ticks.
5. Convert: `ns = (t1 - t0) * device.limits.timestampPeriod` (the tick→nanosecond factor, visible in the ch00 report).

## 9.3 Roofline Model: bottleneck at bandwidth or compute

```
Performance ceiling = min( Peak compute, Peak bandwidth × Arithmetic intensity )
Arithmetic intensity = Compute amount (FLOPs) / Memory traffic (Bytes)
```

- **High arithmetic intensity** (e.g. GEMM, ~tens of FLOP/byte) → hits the "compute roof", optimization direction is to **improve utilization** (tiling, coopmat, FP16).
- **Low arithmetic intensity** (e.g. ch06 element-wise, ~0.25) → hits the "bandwidth slope", optimization direction is to **reduce data movement** (operator fusion, packing, quantization).

This chapter prints the FLOPs, memory traffic, arithmetic intensity, and measured GFLOP/s for a 512³ GEMM, so you can directly see "theoretical peak vs how much of it you actually achieved" — that loss is the optimization headroom.

## 9.4 Vendor Deep Profiling Tools

| Tool | Vendor | What to look at |
|------|------|--------|
| **Nsight Compute** | NVIDIA | per-kernel time, occupancy, memory throughput, bottleneck attribution |
| **Radeon GPU Profiler (RGP)** | AMD | command stream, barriers, ALU/memory utilization |
| **Intel GPA** | Intel | pipeline, bottleneck |
| **RenderDoc** | Cross-platform | frame capture, inspect resources/state per dispatch (debug correctness) |
| **Validation Layers** | Khronos | correctness: missing barriers, out-of-bounds, wrong state (must be on throughout the course for debugging) |

> For LLM/large matrices, Nsight Compute's "Speed-of-Light" page will directly tell you: whether this kernel is memory-bound or compute-bound, and what the L2 hit rate is — more convenient than hand-computing the Roofline.

## 9.5 Hands-on Experiments

1. Run this chapter, and compare the gap between GPU timestamp and CPU wall clock.
2. Shrink `M/N/K` (e.g. 64) and re-measure: under small matrices the "submission overhead" portion of the CPU wall clock rises significantly — meaning small operators should not be submitted frequently on their own (preview of batched submission in ch12).
3. Make the matrix "thin and long" (e.g. M=1024, N=1024, K=64), and see arithmetic intensity drop and become more bandwidth-bound.
4. Capture this chapter's GEMM once with Nsight Compute, compare with its Speed-of-Light conclusion, and verify your hand-computed Roofline.

## 9.6 Compare with Industrial Grade

- **ncnn**: the `benchmark` tool measures per-layer latency with a similar timestamp approach; its `PipelineCache` also counts the overhead of "first compilation vs reuse".
- **llama.cpp / ggml-vulkan**: the `ggml_time` system records the time of each stage; v1.17 introduced **SPIR-V binary cache** precisely to eliminate the wall-clock overhead of "recompiling shaders on every startup" (you'll also see the slow first compilation in this chapter).

The next chapter (ch10) returns to video memory: how to replace scattered allocations with a **memory pool / blob allocator**, keep weights resident and buffers reused, to further reduce latency.
