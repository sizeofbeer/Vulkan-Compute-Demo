# 第 2 章 · 同步模型：fence 与 pipeline barrier

> 目标：理解 GPU 为什么「异步」，以及两条同步原语——**fence（CPU↔GPU）** 和 **barrier（GPU 内部算子间）**——分别解决什么问题。

---

## 2.1 GPU 是异步的

调用 `vkQueueSubmit` 返回时，**算子大概率还没开始算**。命令只是被塞进了队列，GPU 在另一个时间轴上慢慢执行。这带来一个反直觉的事实：

- 在 `submit` 前后用 CPU 时钟计时，测到的只是「提交耗时」（几十微秒），不是「计算耗时」。
- 在 `submit` 之后立刻去读结果 buffer，读到的可能是上一帧的垃圾数据。

所以必须有同步。

## 2.2 两种同步原语

| 原语 | 作用域 | 解决什么问题 |
|------|--------|--------------|
| **fence** | CPU ↔ GPU | 让 CPU 等到「整批命令执行完」再继续（如读结果、释放资源） |
| **semaphore** | 队列 ↔ 队列 | 跨队列/跨提交的顺序（本课程暂不涉及） |
| **pipeline barrier** | GPU 内部 | 同一条命令缓冲里，前一个算子写完、后一个算子才能读（避免冒险） |

本章两个 kernel 串在同一条命令缓冲里：先 `b=a+1`，再 `c=b*2`。第二个 kernel 要读第一个写过的 `b`，这就是典型的 **写后读（Write-After-Read 的反向：Read-After-Write）冒险**。必须在中间插 `pipelineBarrier`，告诉驱动：「第二个 dispatch 开始前，b 的所有写入必须全局可见」。

## 2.3 代码里发生了什么

`main.cpp` 的关键段：

```cpp
cmd.bindPipeline(..., p1); cmd.pushConstants(...); cmd.dispatch(...);   // 写 b
// ↓ 插入 barrier
vk::BufferMemoryBarrier bmb{eShaderWrite, eShaderRead, 0, 0, bufB.buf, 0, VK_WHOLE_SIZE};
cmd.pipelineBarrier(eComputeShader, eComputeShader, {}, 0,nullptr, 1,&bmb, 0,nullptr);
cmd.bindPipeline(..., p2); cmd.pushConstants(...); cmd.dispatch(...);   // 读 b，写 c
```

`pipelineBarrier` 的参数：
- `srcStageMask = eComputeShader`：前面的「生产者」阶段是 compute shader。
- `dstStageMask = eComputeShader`：后面的「消费者」阶段也是 compute shader。
- `bmb`：`b` 的访问从 `ShaderWrite` 变成 `ShaderRead`，驱动据此插入正确的等待。

最后 `submit` + `waitForFences`：这是 fence 在起作用——保证我们 `vkDownload` 读 `c` 时，GPU 已经真的算完了。

## 2.4 动手实验（必做）

1. **验证 barrier 的必要性**：把 `cmd.pipelineBarrier(...)` 那一行注释掉，重新编译运行。在部分 GPU / 驱动上，你会看到「错误数 > 0」或结果随机——这就是没同步的后果。重新加回 barrier 错误数归零。
2. **验证 fence 的必要性**：在 `waitForFences` 之前插入 `vkDownload` 并立刻打印前几个 `c` 值，对比等待前/后——等待前可能是未初始化数据。
3. **思考**：如果把两个 dispatch 拆成两次 `submit`（中间各加一个 fence），还需要 barrier 吗？（提示：跨 submit 的 fence 是否隐含 barrier？答案是 Vulkan 中同一队列的 submit 之间**有隐式 barrier**，但显式写 barrier 更清晰、且能在一次提交内省掉额外 submit 开销。）

## 2.5 对照工业级

- **ncnn** 的 `VkCompute` 内部就维护着 `pending_dispatch_total` 和 `delayed_records`，它会**自动在你录制下一个算子前插入正确的 barrier**——调用者从不需要手写 `pipelineBarrier`。本章程的「手写 barrier」正是为了让你理解 ncnn 替你做了什么。
- **llama.cpp** 在 AMD 设备上曾专门补过 `buffer ownership transfer` 的 barrier（因为队列/内存所有权切换需要额外同步）。同步是 Vulkan 推理最容易踩坑的地方，务必在这里吃透。

下一章（ch03）讲 **显存管理**：为什么 `vkUpload` 在 device-local 时要走 staging 中转？`HOST_VISIBLE / DEVICE_LOCAL / COHERENT` 到底差在哪？以及 FP16 怎么存进 buffer。
