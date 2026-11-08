# Chapter 3 · Memory Management Basics

> Goal: understand that GPU memory is not simply "one block of memory" — it has different **types** and different **visibility**. This chapter clarifies three things: **memory types**, **staging relay uploads**, and **FP16 storage buffers (halving bandwidth)**. This is the foundation for all later performance optimizations.

---

## 3.1 Why memory management is "the most important thing after lesson one"

In ch01 we blindly used `eDeviceLocal`, and in ch02 we did not worry about how data got onto the GPU. Yet in real engines, 90% of latency and bugs come from memory:

- **How does data reach the GPU?** The CPU cannot write device-local memory directly; a relay is required.
- **Should weights stay resident?** Re-uploading weights every frame is a huge waste (solved in ch10).
- **Can memory be saved?** Models easily reach hundreds of MB; FP16 / quantization can cut the footprint by half or even down to 1/4 (ch11).

## 3.2 Three memory types (key)

Vulkan divides memory into several "types", described by property flags. The two or three most common ones we care about:

| Type | flags | CPU can write directly? | Speed | Use |
|------|-------|----------------|------|------|
| device-local | `eDeviceLocal` | ❌ | fastest | compute buffers, weights, activations |
| host-visible + coherent | `eHostVisible`+`eHostCoherent` | ✅ map directly | slow | staging relay, occasional readback |
| host-visible + cached | `eHostVisible`+`eHostCached` | ✅ but needs flush | fast read, slow write | uploading large blocks (CPU-side cache) |

The "memory types" list reported by `ch00` is exactly these. `eHostCoherent` means that after mapping, writes are **immediately visible to the GPU** without manually calling `flushMappedMemoryRanges`; `eHostCached` means the CPU reads this memory through cache (fast), but after writing it usually needs a flush.

## 3.3 staging relay: moving data into "fast but not directly writable" memory

The CPU cannot touch device-local memory, so uploading takes two steps:

```
CPU memory --map & write--> host-visible staging buffer --GPU copyBuffer--> device-local buffer
```

`common/vk_base.cpp`'s `vkUpload` already wraps this step: if the target is device-local, it internally allocates a staging buffer, writes into it, then does `copyBuffer` + waits on a fence. **You have been using it since ch01/ch02 without noticing.** This chapter uses timing comparisons to make this cost visible:

- `host-visible` direct mapping: just one `memcpy`, at microsecond scale.
- `device-local` via staging: an extra GPU copy + an extra CPU/GPU sync, **much slower**.

> Takeaway: **device-local compute is fast, but the "upload" itself is an extra cost.** Weights should therefore be uploaded once and kept resident (ch10), not re-uploaded every frame. This is precisely the motivation behind ncnn's `VkWeightAllocator` (resident weights) and llama.cpp mapping model weights into device-local memory.

## 3.4 FP16 storage buffer: halving both bandwidth and memory

Neural networks are not sensitive to precision. Using **half (16-bit float)** instead of float32 for storage yields:

- half the memory footprint;
- half the bandwidth demand (huge benefit for "bandwidth-bound" operators, quantified in ch06);
- on some hardware, FP16 throughput is 2~16× that of FP32.

This chapter's `fp16_roundtrip.comp` demonstrates: the input buffer is declared with `float16_t[]` (requires the device to enable `storageBuffer16BitAccess`); the host encodes floats into halfs (`float_to_half`) before upload; the GPU reads them and converts back to float32 via `float(a[i])` for computation. On readback we verify `c[i]=2*i`, with the tolerance set according to half precision. The report will show the input buffer occupies only half the bytes.

> Key point: FP16 is a **storage format**; during computation it is usually converted back to FP32 (or FP32 is used for accumulation) to avoid error buildup. The ch04 GEMM accumulator uses FP32.

## 3.5 Hands-on experiments

1. Run this chapter and look at the memory-type list and the upload-time comparison. Increase `N` to `1<<22` (4M) and re-measure the gap between the two.
2. Change `bufHost` to `eHostVisible | eHostCached` (drop coherent) and observe that `vkUpload` does one extra `flushMappedMemoryRanges` on the cached path (see common source lines 89~98).
3. Modify `fp16_roundtrip.comp`: change it to `c[i] = a[i] + a[i]*a[i]` (a polynomial), rerun, and confirm half precision is still sufficient.
4. Think about it: if your device does not support `shaderFloat16`, how would you change it? — Store the half as `uint16_t` and hand-write the half→float bit operations in the shader (same as `half_to_float` from ch00). This is exactly llama.cpp's fallback path on older hardware.

## 3.6 Industrial-grade comparison

- **ncnn**: three allocators in `src/allocator.h` — `VkBlobAllocator` (`preferred_block_size` defaults to 16MB, activations), `VkWeightAllocator` (8MB, resident weights), `VkStagingAllocator` (host-visible relay); all allocate via `fastMalloc()`. Weights are uploaded once at model load time.
- **llama.cpp / ggml-vulkan**: weights are mapped into device-local at load; inputs/outputs use staging at inference; memory properties live in `vk_device_struct` (`ggml-vulkan-types.h`). The old `dev->device_vk` field no longer exists.

Next chapter (ch04) we finally enter the core operation of neural networks — **matrix multiplication GEMM**, and for the first time seriously use "shared memory" and "coalesced access".
