# Chapter 12 · Graph Execution and Scheduling

> Goal: organize scattered operators into **a single compute graph** — record them into the same command buffer, connect data dependencies with barriers, and submit once. This is the key step from "it runs" to "it runs fast", and the same idea behind ncnn's layer scheduling and ggml's compute graph.

---

## 12.1 Why "Graph-ify"

In ch01–ch11, every operator does its own `vkSubmit` + waits on a fence. The problem: **every submit has CPU-side queue overhead**. When one network layer has dozens of operators (matmul, bias, activation, norm…), submitting them one by one eats into a noticeable amount of latency — especially for small operators (the small matrices mentioned in ch09).

Solution: **pre-record** the entire chain into a single command buffer, and submit only once at the end.

## 12.2 Recording + Barriers for Dependencies

`common/vk_base.cpp`'s `Graph` does three things:

```cpp
Graph g; g.begin();                              // open a command buffer
g.node(pMM1,  &pc, ..., prevOut=nullptr);        // operator 1: write T1
g.node(pBias, &pc, ..., prevOut=T1.buf);         // operator 2: read T1 (insert write-after-read barrier) write T2
g.node(pRelu, &pc, ..., prevOut=T2.buf);         // operator 3: read T2 write T3
g.node(pMM2,  &pc, ..., prevOut=T3.buf);         // operator 4: read T3 write OUT
g.end();                                         // one-time submit + wait on fence
```

Before recording each operator, `node()` inserts a `pipelineBarrier` (compute→compute, write-after-read) if it has a `prevOut` (the previous node's output it needs to read). This guarantees: when the GPU executes operator 2, operator 1's write to T1 is already visible — the same thing as the hand-written barrier in ch02, only now inserted automatically by the graph.

## 12.3 Why Barriers Are Indispensable

Omit a barrier and operator 2 may read **a half-finished or stale value** of T1 — this is the classic "occasionally wrong, hard to reproduce" GPU bug. Validation Layers have dedicated checks for this kind of hazard (they should be on throughout the course).

## 12.4 Further: Deferred Barriers and Multithreading

Real engines do even more aggressive scheduling optimizations:

- **Deferred barrier (execution dependency)**: only insert a barrier at genuine cross-operator dependencies; intra-operator synchronization is left to `barrier()` (within a workgroup).
- **Multi-queue parallelism**: distribute independent operators across different queues (even compute + transfer in parallel for weight搬运); llama.cpp distinguishes a compute queue from a transfer queue.
- **Static scheduling / constant folding**: in the graph, mergeable expressions like `scale * x + bias` are fused at compile time.

## 12.5 Hands-on Experiments

1. Change `Graph` to "one `vkSubmit` per node" (remove barriers, submit separately), and compare end-to-end latency of the two approaches using the ch09 timestamps — the gap is especially pronounced for small operators.
2. Add an **independent** operator to the graph (e.g. compute a separate statistic on T1 at the same time), and think about whether it can run in parallel with the matmul (needs dual queues; advanced).
3. Remove the graph's barriers and run again with Validation Layers on, watch it report a "write-after-read hazard" — experience first-hand why barriers are necessary.

## 12.6 Compared with Industrial Implementations

- **ncnn**: the model is expanded into a `layer` linked list, and during forward pass `VkCompute::submit_and_wait` is called layer by layer; later versions introduced "record commands, batch submit" to reduce overhead, and automatically insert barriers between layers.
- **llama.cpp / ggml-vulkan**: `ggml_graph_compute` first records the entire graph into a command buffer (including automatic dependency barriers), then submits in one shot; it also distinguishes compute / transfer queues so weight transfer overlaps with computation.

The next chapter (ch13) is a **capstone project**: use all the building blocks so far (resident weights + matmul + activation + graph execution) to run forward inference of a small model, putting the course goal into practice.
