# 第 3 章 · 显存管理入门

> 目标：搞懂 GPU 显存不是「一块内存」那么简单——它有不同**类型**和不同**可见性**。这一章讲清三件事：**内存类型**、**staging 中转上传**、**FP16 存储 buffer（省一半带宽）**。这是后面所有性能优化的地基。

---

## 3.1 为什么显存管理是「第一课之后最重要的事」

在 ch01 里我们无脑用 `eDeviceLocal`，在 ch02 里没关心数据怎么上去的。但真实引擎 90% 的延迟和 bug 来自显存：

- **数据怎么传到 GPU？** CPU 不能直接写 device-local 显存，必须中转。
- **权重要不要常驻？** 每帧重新上传权重是巨大的浪费（ch10 解决）。
- **能不能省内存？** 模型动辄几百 MB，FP16 / 量化能把 footprint 砍半甚至砍到 1/4（ch11）。

## 3.2 三种内存类型（重点）

Vulkan 把显存分成若干「类型」，由 property flags 描述。我们关心最常见的两/三类：

| 类型 | flags | CPU 能直接写？ | 速度 | 用途 |
|------|-------|----------------|------|------|
| device-local | `eDeviceLocal` | ❌ | 最快 | 计算 buffer、权重、激活值 |
| host-visible + coherent | `eHostVisible`+`eHostCoherent` | ✅ 直接 map | 慢 | staging 中转、偶尔读回 |
| host-visible + cached | `eHostVisible`+`eHostCached` | ✅ 但需 flush | 读快写慢 | 上传大块数据（CPU 侧有缓存） |

`ch00` 报告的「内存类型」列表就是这些。`eHostCoherent` 表示 map 后写的内容**立即可被 GPU 看到**，不用手动 `flushMappedMemoryRanges`；`eHostCached` 表示 CPU 读这块内存走缓存（快），但写后通常需要 flush。

## 3.3 staging 中转：把数据搬进「快但不可直接写」的显存

device-local 显存 CPU 碰不到，所以上传要分两步：

```
CPU 内存 --map 写--> host-visible staging buffer --GPU copyBuffer--> device-local buffer
```

`common/vk_base.cpp` 的 `vkUpload` 已经把这步封装了：如果目标是 device-local，它就内部建一块 staging、写进去、再 `copyBuffer` + 等 fence。**ch01/ch02 你一直在用它，只是没察觉**。本章用计时对比让你看见这笔开销：

- `host-visible` 直接映射：只是一次 `memcpy`，微秒级。
- `device-local` 走 staging：多了一次 GPU 拷贝 + 一次 CPU/GPU 同步，**慢得多**。

> 启示：**device-local 计算虽快，但「上传」本身是额外成本**。所以权重应当一次性传好常驻（ch10），而不是每帧重传。这正是 ncnn `VkWeightAllocator`（权重常驻）和 llama.cpp 把模型权重映射进 device-local 的动机。

## 3.4 FP16 存储 buffer：带宽与内存都省一半

神经网络对精度不敏感，用 **half（16 位浮点）** 代替 float32 存储，能：

- 显存 footprint 减半；
- 带宽需求减半（对「带宽受限」算子收益巨大，ch06 会量化）；
- 部分硬件的 FP16 算力是 FP32 的 2~16 倍。

本章 `fp16_roundtrip.comp` 演示：输入 buffer 用 `float16_t[]` 声明（需设备开启 `storageBuffer16BitAccess`），主机侧把 float 编码成 half（`float_to_half`）再上传；GPU 读出后 `float(a[i])` 转回 float32 计算。读回校验 `c[i]=2*i`，容差按 half 精度设定。报告会显示输入 buffer 只占一半字节。

> 关键点：FP16 是**存储格式**，计算时通常转回 FP32（或累加用 FP32），避免误差累积。ch04 的 GEMM 累加器就用 FP32。

## 3.5 动手实验

1. 跑本章，看内存类型列表与上传耗时对比。把 `N` 调大到 `1<<22`（4M），重测两者差距。
2. 把 `bufHost` 改成 `eHostVisible | eHostCached`（去掉 coherent），观察 `vkUpload` 在 cached 路径会多一次 `flushMappedMemoryRanges`（看 common 源码第 89~98 行）。
3. 修改 `fp16_roundtrip.comp`：改成 `c[i] = a[i] + a[i]*a[i]`（多项式），重跑，确认 half 精度仍足够。
4. 思考：若你的设备 `shaderFloat16` 不支持，如何改？—— 用 `uint16_t` 存 half，着色器里手写 half→float 位运算（ch00 的 `half_to_float` 同理）。这正是 llama.cpp 在老硬件上的回退路径。

## 3.6 对照工业级

- **ncnn**：显式区分三类分配器——`VkBlobAllocator`（16MB 池，放激活值，device-local）、`VkWeightAllocator`（权重常驻 device-local）、`VkStagingAllocator`（host-visible 中转）。权重只在模型加载时上传一次。
- **llama.cpp / ggml-vulkan**：加载模型时把权重映射到 device-local；推理时输入/输出走 staging；并通过 `dev->device_vk` 记录的内存属性选择最优上传路径。

下一章（ch04）我们终于进入神经网络的核心运算——**矩阵乘 GEMM**，并第一次认真用上「共享内存」和「合并访存」。
