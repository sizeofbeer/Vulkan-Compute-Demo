# 第 7 章 · 归一化（RMSNorm / LayerNorm）

> 目标：学会在 GPU 上做**跨元素协作**——用「工作组内共享内存归约（reduction）」算出一行特征的统计量，再逐元素归一化。这是 Transformer 里 RMSNorm/LayerNorm 的核心。

---

## 7.1 为什么归一化需要「协作」

逐元素算子（ch06）各算各的，互不依赖。但归一化要先知道**整行**的统计量（均值/方差/RMS），再拿它去归一化每个元素——这要求同一行的线程先「汇总」再「广播」。GPU 上靠**共享内存 + barrier** 实现这种组内协作。

RMSNorm 公式（LLaMA 等模型用）：
```
rms = sqrt( mean_i(x_i^2) )
y_i = x_i / rms * gamma_i
```
（LayerNorm 多一步减均值：`y_i = (x_i - mean)/std * gamma + beta`，归约时多算一个均值即可。）

## 7.2 共享内存归约（reduction）

`rmsnorm.comp` 的思路，每个工作组负责**一行**：

1. 256 个线程，每人算自己那个特征的平方，写进共享数组 `ssum[tid]`；`barrier()`。
2. **树形归约**：步长 `s` 从 128 减半到 1，每轮 `tid < s` 的线程把 `ssum[tid] += ssum[tid+s]`，然后 `barrier()`。32 → 16 → … → 1 轮后，`ssum[0]` 就是全部平方和。
3. 所有线程读到同一个 `rms = sqrt(ssum[0]/F)`，各自归一化并乘 gamma。

> 为什么用「树形归约 + barrier」而不让线程 0 一个人累加？因为 256 个线程同时写 `ssum[0]` 会数据竞争；归约让每轮只有一半线程写，且写前 `barrier()` 保证上一轮结果可见。这是 GPU reduction 的标准写法（类似 ch02 的 barrier，只是发生在工作组内部）。

## 7.3 局限与扩展

本教学版假设 `F ≤ 256`（一个工作组线程数）。真实模型 `F` 可能是 4096、8192：

- **多趟归约**：每个线程先处理多个元素做局部和，再走树形归约；
- 或用 **warp/subgroup 原语**（`subgroupAdd`）进一步加速——llama.cpp 在支持 `subgroup` 的设备上就用它做 reduction，比共享内存更快。

这些在 docs 里点到为止，核心是理解「先汇总、再广播」的模式。

## 7.4 动手实验

1. 把 `F` 改成 128，重跑——`tid < F` 的分支让多余线程空转，结果仍正确。
2. （进阶）改成 LayerNorm：在归约时同时算 `sum(x)` 得到均值，归一化处换成 `(x - mean)/std * gamma + beta`，CPU 参考同步改。
3. （进阶）把 `F` 扩到 1024：每个线程循环处理 4 个元素做局部和（`for(uint j=tid;j<F;j+=256)`），再走树形归约。

## 7.5 对照工业级

- **ncnn**：`Normalize`、`InstanceNorm` 等通过两次 kernel 完成——先 reduction 求均值/方差（工作组内），再归一化；或用 `pack` 把通道打包加速。
- **llama.cpp / ggml-vulkan**：`rmsnorm` / `layernorm` kernel 用共享内存归约求统计量；在支持 `VK_KHR_shader_subgroup_*` 的设备上，用 `subgroupAdd` 把归约收敛到一次 warp 操作，显著快于纯共享内存。权重 `gamma` 从量化模型反量化后传入。

下一章（ch08）我们上注意力机制：数值稳定的 softmax（又是一次归约），以及把 QKᵀ、softmax、PV 串成一次简化注意力。
