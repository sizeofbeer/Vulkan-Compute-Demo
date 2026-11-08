# 第 12 章 · 图执行与调度

> 目标：把零散算子组织成**一张计算图**——录进同一条命令缓冲、用 barrier 衔接数据依赖、一次性提交。这是把「能跑」变成「跑得快」的关键一步，也是 ncnn 图层调度、ggml 计算图的同款思想。

---

## 12.1 为什么要「图化」

ch01–ch11 每个算子都单独 `vkSubmit` + 等 fence。问题是：**每次 submit 都有 CPU 端队列开销**。当一层网络有几十个算子（matmul、bias、激活、norm…），逐个提交的开销会吃掉可观延迟——尤其小算子（ch09 提到的小矩阵）更是如此。

解决：把整条链路**预先录制**成一条命令缓冲，只在最后 submit 一次。

## 12.2 录制 + barrier 衔接依赖

`common/vk_base.cpp` 的 `Graph` 做三件事：

```cpp
Graph g; g.begin();                              // 开一条命令缓冲
g.node(pMM1,  &pc, ..., prevOut=nullptr);        // 算子1：写 T1
g.node(pBias, &pc, ..., prevOut=T1.buf);         // 算子2：读 T1（插写后读 barrier）写 T2
g.node(pRelu, &pc, ..., prevOut=T2.buf);         // 算子3：读 T2 写 T3
g.node(pMM2,  &pc, ..., prevOut=T3.buf);         // 算子4：读 T3 写 OUT
g.end();                                         // 一次性 submit + 等 fence
```

`node()` 在录制每个算子**之前**，如果它有 `prevOut`（要读的上一节点输出），就插一条 `pipelineBarrier`（compute→compute，写后读）。这保证：GPU 执行算子2 时，算子1 对 T1 的写入已经可见——和 ch02 手写的 barrier 是同一回事，只是现在由图自动插入。

## 12.3 为什么 barrier 不能少

漏掉 barrier，算子2 可能读到 T1 的**半成品或旧值**——这就是 GPU 结果「偶尔错、难以复现」的经典 bug。Validation Layers 对这类 hazard 有专门检查（全课程都该开着）。

## 12.4 进一步：延迟 barrier 与多线程

真实引擎还会做更激进的调度优化：

- **延迟 barrier（execution dependency）**：只在真正跨算子依赖处插 barrier，同一算子的内部同步交给 `barrier()`（工作组内）。
- **多队列并行**：把无依赖的算子分到不同队列（甚至 compute + transfer 并行搬运权重）；llama.cpp 就区分 compute 队列与 transfer 队列。
- **静态调度 / 常量折叠**：图里 `scale * x + bias` 这种可提前合并的，编译期融掉。

## 12.5 动手实验

1. 把 `Graph` 改成「每节点一次 `vkSubmit`」（删掉 barrier，分开提交），用 ch09 时间戳对比两种方式的端到端延迟——小算子差距尤其明显。
2. 在图里加一个**无依赖**的算子（如同时对 T1 另算一个统计量），思考它能不能和 matmul 并行（需要双队列，进阶）。
3. 把本图的 barrier 删掉再跑，开着 Validation Layers，观察它报出「写后读 hazard」——亲手体验为何 barrier 必要。

## 12.6 对照工业级（2026-11 同步）

- **ncnn**：`VkCompute::record_pipeline()` 把 pipeline + 描述符 + push constant + dispatch 录进 `delayed_records`，`submit_and_wait()` 一次性提交并按依赖插入 barrier；`pending_dispatch_total()` 暴露待提交 dispatch 总数。**不是「逐层 submit」**——现行版本从一开始就是「延迟录制 + 批提交」。
- **llama.cpp / ggml-vulkan**：入口已改名为 `ggml_backend_vk_graph_compute()`（`ggml-vulkan.cpp`，注册在 backend 的 `.graph_compute` 函数指针），整图录进命令缓冲、按依赖自动插 barrier、区分 compute / transfer 队列。设备与管线结构收敛到 `ggml-vulkan-types.h` 的 `vk_device_struct` / `vk_pipeline_struct`；旧文档中的 `vk-device.cpp`、`pipeline.cpp`、`ggml_vk::graph_compute` 均已不存在。
- **ncnn-compiler（本次新增参考）**：把「图执行」推到编译期——AOT 生成模型专用 `.so`，运行时不再解析 plan、也不解释 kernel schedule（见 12.7）。

## 12.7 进阶：从「运行时录图」到「AOT 固定图」

`ncnn-compiler`（`867c6ab`）的 `--backend=vulkan` 产品可以当作本章 `Graph` 的工业级终点：它把 ncnn `.param/.bin` 经 ncnn 方言 → TOSA → Linalg → GPU dialect 降级为 SPIR-V，再生成**模型专用 fixed-run 宿主 C++**（`tools/VulkanLifecycleCodegen.cpp`、`lib/Support/VulkanHostProgram.cpp`）。

| 维度 | 本章 `Graph` | ncnn-compiler fixed-run |
|------|--------------|-------------------------|
| 图来源 | 运行时由 C++ 代码逐个 `node()` 搭建 | 编译期静态 FP32 图，运行时不解析 plan |
| `create` | 上传权重、建管线 | 生成的代码完成权重上传 / flush / transfer barrier / copy / fence |
| `run` | 每次录制命令缓冲 | 每次 run **重置并重录**：输入 staging → barrier → pipeline/descriptor/push constant/dispatch → 输出回读 / invalidate / fence |
| 同步 | 手写「写后读」barrier | 静态三段屏障：host→transfer→compute、compute 写→读/写、compute→transfer→host |
| 形状 | 固定 | 固定；**明确拒绝动态 shape** |
| 产物 | 可执行文件 | `<model>_vulkan.h` + `.json` + `lib<model>_vulkan.so` + `.plan.json`；SPIR-V/权重内嵌，无 LLVM/plan 运行时依赖 |
| 验收 | 手动跑一遍 | 四 profile 门禁：release（双 ICD）/ sanitize（ASan+UBSan）/ offline（无设备）/ software（仅 llvmpipe）；`run_gate.py` 另有 `hardware` 变体 |

> 注意它**不调用** `ncnn::VkCompute` / `VkAllocator` / `PipelineCache`，而是用原生 Vulkan API 生成宿主代码（`lib/VulkanDeviceHelper/VulkanDeviceHelper.cpp`）。这条路线与本章互补：本章教你「图为什么能一次提交」，AOT 告诉你「当图完全静态时，连调度器本身都可以删掉」。读 `docs/ncnn-vulkan-backend-contract.md` 可与本章 12.2~12.4 逐条对照。

下一章（ch13）是**综合项目**：用前面所有积木（常驻权重 + matmul + 激活 + 图执行）跑通一个小模型的前向推理，把课程目标真正落地。
