# 第 9 章 · 性能分析方法论

> 目标：学会**真测量**而非**拍脑袋**。用 GPU 硬件时间戳精确测时，用 Roofline 模型判断瓶颈在带宽还是算力，并知道该用哪款厂商工具做深度剖析。

---

## 9.1 第一铁律：不等 fence 的 CPU 计时是假的

新手最常犯的错：在 `submit` 之后立刻用 `std::chrono` 记录「耗时」。但 Vulkan 是**异步**的——`submit` 只是把命令「寄出去」，GPU 可能还没开始算，CPU 已经跑到计时点了。真实耗时要等 `waitForFences` 之后才算完。

本章 `main.cpp` 同时测了两个数：
- **CPU 墙钟（含提交）**：从 `submit` 到 `waitForFences` 返回——包含队列提交、驱动、同步开销，通常**偏大**，不能代表纯计算。
- **GPU 时间戳**：用 `VK_QUERY_TYPE_TIMESTAMP` 在命令流里插两个 `writeTimestamp`（计算着色器阶段），GPU 自己在「真正开始算」和「算完」打点。这才是**真实计算耗时**。

> 差距往往有几倍。所以性能优化第一步：**用对计时方法**，否则你优化半天可能在优化「提交开销」而非「kernel」。

## 9.2 时间戳查询怎么用

1. 建池：`createQueryPool({..., eTimestamp, 2, 0})`（2 个 slot：起点/终点）。
2. 录制：`cmd.writeTimestamp(eComputeShader, pool, 0)` → 你的 dispatch → `writeTimestamp(..., pool, 1)`。
3. 提交+等 fence。
4. 取数：`getQueryPoolResults(..., eWait | e64)` 拿到两个 `uint64_t` tick。
5. 换算：`ns = (t1 - t0) * device.limits.timestampPeriod`（tick→纳秒的因子，ch00 报告里能看到）。

## 9.3 Roofline 模型：瓶颈在带宽还是算力

```
性能上限 = min( 算力峰值, 带宽峰值 × 算术强度 )
算术强度 = 计算量(FLOPs) / 访存量(Bytes)
```

- **算术强度高**（如 GEMM，~几十 FLOP/byte）→ 撞「算力屋顶」，优化方向是**提高利用率**（ tiling、coopmat、FP16）。
- **算术强度低**（如 ch06 逐元素，~0.25）→ 撞「带宽斜线」，优化方向是**减少搬运**（算子融合、打包、量化）。

本章对一个 512³ GEMM 打印了它的 FLOPs、访存量、算术强度与实测 GFLOP/s，让你直观看到「理论峰值 vs 实测打了几折」——这折损就是优化空间。

## 9.4 厂商深度剖析工具

| 工具 | 厂商 | 看什么 |
|------|------|--------|
| **Nsight Compute** | NVIDIA | per-kernel 耗时、占用率、内存吞吐、瓶颈归因 |
| **Radeon GPU Profiler (RGP)** | AMD | 命令流、屏障、ALU/访存利用率 |
| **Intel GPA** | Intel | 流水线、瓶颈 |
| **RenderDoc** | 跨平台 | 抓帧、看每次 dispatch 的资源/状态（调试正确性） |
| **Validation Layers** | Khronos | 正确性：漏 barrier、越界、状态错（全课程调试必开） |

> 对 LLM/大矩阵，Nsight Compute 的「Speed-of-Light」页会直接告诉你：这个 kernel 是 memory-bound 还是 compute-bound、L2 命中率多少——比手算 Roofline 还省事。

## 9.5 动手实验

1. 跑本章，对比 GPU 时间戳与 CPU 墙钟的差距。
2. 把 `M/N/K` 改小（如 64），重测：小矩阵下 CPU 墙钟里「提交开销」占比会显著上升——说明小算子不该频繁单独提交（预告 ch12 的批提交）。
3. 把矩阵改成「又瘦又长」（如 M=1024,N=1024,K=64），看算术强度下降、更偏带宽受限。
4. 用 Nsight Compute 抓一次本章的 GEMM，对照它的 Speed-of-Light 结论，验证你手算的 Roofline。

## 9.6 对照工业级

- **ncnn**：`benchmark` 工具用类似时间戳方式测每层延迟；它的 `PipelineCache` 还统计「首次编译 vs 复用」的开销。
- **llama.cpp / ggml-vulkan**：`ggml_time` 体系记录各阶段耗时；v1.17 引入 **SPIR-V 二进制缓存**正是为了消除「每次启动重编译 shader」的墙钟开销（你在本章也会看到首次编译慢）。

下一章（ch10）回到显存：如何把零散分配换成**内存池 / blob 分配器**，把权重常驻、buffer 复用，进一步压低延迟。
