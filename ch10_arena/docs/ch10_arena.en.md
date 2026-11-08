# Chapter 10 · Advanced Device Memory Management (Memory Pool / Blob Allocator)

> Goal: upgrade from "call `vkCreateBuffer` whenever needed" to "carve slices out of one big pool". This chapter introduces the **bump allocator (device memory pool)** — the same idea behind ncnn's `VkBlobAllocator` and ggml's arena.

---

## 10.1 Why you can't allocate on every request

Every `vkAllocateMemory` carries CPU-side overhead, and frequent allocate/free produces **device memory fragmentation**. Neural-network inference generates dozens or hundreds of temporary tensors (activations) per frame; if each is allocated separately:

- The allocate/free calls themselves eat into latency;
- Fragmentation makes large contiguous allocations increasingly hard to obtain;
- Weights get re-uploaded every time (mentioned in ch03 — the worst thing to do).

Solution: **pre-allocate one large block of device memory as a pool, and carve it up using pointer offsets**.

## 10.2 The bump allocator (the most basic pool)

`common/vk_base.cpp`'s `DeviceArena` is the simplest bump allocator:

```cpp
DeviceArena arena; arena.init(ctx, 4MB);     // grab one large device-local block at once
vk::DeviceSize off = arena.alloc(1024*4);    // carve 4KB from the current offset, return offset
arena.reset();                               // frame ends: offset back to zero, all sub-regions invalidated
```

`alloc` simply does "current offset += aligned size" — O(1), zero system calls. All sub-regions share one `Buffer`, distinguished by **byte offset**. ch10's `main.cpp` demonstrates: carve three regions A/B/C from the pool, compute `C=A+B` on the GPU, verify correctness; after `reset()`, re-carve and the offsets are reused as-is.

> This is the core mechanism of ncnn's `VkBlobAllocator` (16MB per block by default, stores activations) and ggml's `ggml_arena` (all intermediate results of a frame go in one place; `ggml_free` just points `alloc_ptr` back to the start).

## 10.3 Advanced: separate pools for different lifetimes

Real engines usually split pools:

| Pool | Contents | Lifetime | Corresponds to |
|----|------|----------|------|
| **Weight pool** | model weights | resident (loaded once) | ncnn `VkWeightAllocator` |
| **Activation pool** | per-frame intermediates | reset/reused each frame | ncnn `VkBlobAllocator` |
| **Staging pool** | staging upload/readback | short-lived | ncnn `VkStagingAllocator` |

The weight pool is **never reset** — this is exactly the engineering realization of ch03's emphasis on "upload weights once and keep them resident".

## 10.4 Alignment and out-of-bounds

`alloc` does 256-byte alignment (Vulkan commonly requires alignment on offsets/sizes). Sub-regions are indexed by `base + i`; going out of bounds will stomp a neighbor — so when using a pool you **must** ensure each operator only accesses its own slice (all kernels in this chapter carry an `n` boundary guard). Validation Layers can catch cross-region overruns.

## 10.5 Hands-on experiments

1. Shrink the pool to just barely enough (e.g. `N*4*3 + small margin`), then do a fourth `alloc` and see whether it throws a "capacity insufficient" exception (confirm boundary checks engage).
2. Remove `reset()` and run the second round again — the offset keeps growing, proving "no reuse means waste"; also shows why reset is needed.
3. (Advanced) Implement a **free-list** allocator with `free(offset, size)` to replace bump, supporting release in arbitrary order (ncnn's blob reuse is closer to this).

## 10.6 Comparing with industry-grade

- **ncnn**: `VkBlobAllocator` keeps a device-local pool sized by `preferred_block_size` (16MB default); `fastMalloc()` carves aligned blocks out of it and reuses them. `VkWeightAllocator` (8MB default) keeps weights resident and `VkStagingAllocator` handles host-visible transfers. `PipelineCache::save_cache()/load_cache()` can persist compiled pipelines across processes.
- **llama.cpp / ggml-vulkan**: `ggml_backend_vk_graph_compute()` (see ch12) reuses large buffers for the whole graph's temporary tensors, with structures in `ggml-vulkan-types.h`; weights stay resident in device-local and are not allocated per frame. The `ggml_arena` / `ggml_free` / `dev->device_vk` names from the first release no longer exist.

Next chapter (ch11) we **fuse dequantization in place** inside matmul — weights are stored as int8 (4× memory saved), and the GPU restores them to float on read before accumulating. This is the core technique for saving memory in LLM inference.
