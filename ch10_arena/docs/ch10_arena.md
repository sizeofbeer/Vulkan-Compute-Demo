# 第 10 章 · 显存管理进阶（内存池 / Blob 分配器）

> 目标：从「每次需要就 `vkCreateBuffer`」升级到「一个大池子里切切切」。这一章引入 **bump 分配器（显存池）**，是 ncnn `VkBlobAllocator` 与 ggml arena 的同款思路。

---

## 10.1 为什么不能直接每次分配

每次 `vkAllocateMemory` 都有 CPU 端开销，且频繁分配/释放会产生**显存碎片**。神经网络推理每帧要产生几十上百个临时张量（激活值），如果每个都单独分配：

- 分配/释放调用本身吃掉延迟；
- 碎片让大块连续显存越来越难申请；
- 权重每次重新上传（ch03 提过，最忌讳）。

解决：**预分配一块大显存当池子，用「指针偏移」切分**。

## 10.2 bump 分配器（最基础的池）

`common/vk_base.cpp` 的 `DeviceArena` 就是最简单的 bump 分配器：

```cpp
DeviceArena arena; arena.init(ctx, 4MB);     // 一次性要一大块 device-local
vk::DeviceSize off = arena.alloc(1024*4);    // 从当前偏移切 4KB，返回偏移
arena.reset();                               // 一帧结束：偏移归零，全部子区域作废
```

`alloc` 只是「当前偏移 += 对齐后的尺寸」——O(1)，零系统调用。所有子区域共享同一块 `Buffer`，靠**字节偏移**区分。ch10 的 `main.cpp` 演示了：从池里切 A/B/C 三个区，GPU 上算 `C=A+B`，校验正确；`reset()` 后重新切，偏移原样复用。

> 这就是 ncnn 的 `VkBlobAllocator`（默认 16MB 一块，存激活值）与 ggml 的 `ggml_arena`（一帧的中间结果全放里，`ggml_free` 只是把 `alloc_ptr` 指回开头）的核心机制。

## 10.3 进阶：不同生命周期用不同池

真实引擎通常分池：

| 池 | 内容 | 生命周期 | 对应 |
|----|------|----------|------|
| **权重池** | 模型权重 | 常驻（加载一次） | ncnn `VkWeightAllocator` |
| **激活池** | 每帧中间结果 | 每帧 reset 复用 | ncnn `VkBlobAllocator` |
| **中转池** | staging 上传/读回 | 短命 | ncnn `VkStagingAllocator` |

权重池**绝不 reset**——这正是 ch03 强调「权重一次性上传常驻」的工程落地。

## 10.4 对齐与越界

`alloc` 做了 256 字节对齐（Vulkan 对偏移/大小常要求对齐）。子区域靠 `base + i` 索引，越界会踩到邻居——所以用池时**必须**保证每个算子只访问自己那一段（本章 kernel 都带 `n` 边界保护）。Validation Layers 能抓到跨区越界。

## 10.5 动手实验

1. 把池子缩到刚好够（如 `N*4*3 + 小余量`），再 `alloc` 第四次看是否抛「容量不足」异常（确认边界检查生效）。
2. 把 `reset()` 去掉再跑第二轮——偏移继续往后涨，证明「不复用就浪费」；也说明为何要 reset。
3. （进阶）实现一个 `free(offset, size)` 的**空闲链表**分配器替代 bump，支持任意顺序释放（ncnn 的 blob 复用更接近这种）。

## 10.6 对照工业级

- **ncnn**：`VkBlobAllocator` 以 `preferred_block_size`（默认 16MB）维护 device-local 池，`fastMalloc()` 按对齐切块并在池内复用；`VkWeightAllocator`（默认 8MB）单独存权重并常驻；`VkStagingAllocator` 负责 host-visible 中转。另有 `PipelineCache::save_cache()/load_cache()` 可把编译好的管线跨进程持久化。
- **llama.cpp / ggml-vulkan**：`ggml_backend_vk_graph_compute()`（见 ch12）为整图临时张量复用大 buffer，结构定义在 `ggml-vulkan-types.h`；权重常驻 device-local，不按帧分配。首版提到的 `ggml_arena` / `ggml_free` / `dev->device_vk` 均已不存在。

下一章（ch11）我们在 matmul 里**就地融合反量化**——权重以 int8 存储省 4 倍显存，GPU 读取时当场还原成 float 再累加，这是 LLM 推理省显存的核心手段。
