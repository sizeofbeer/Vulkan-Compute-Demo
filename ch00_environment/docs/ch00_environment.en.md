# Chapter 0 · Environment Setup and Device Enumeration

> Goal: Before you write a single line of compute code, confirm that "the toolchain is installed, the GPU can be discovered, and what it supports." This chapter is a **pure probing tool** — it does not run any operators.

---

## 0.1 Why This Chapter Exists

Vulkan is an "explicit" API: it will not judge device capabilities for you, nor will it gently warn you when you use it wrong. In neural-network computation you will repeatedly hit these prerequisites:

- How many threads can one workgroup hold on my GPU at most? (decides the `local_size_x` upper bound)
- What is the maximum shared-memory size in bytes? (decides the tile size of GEMM, ch04)
- Does it support FP16 storage? (decides whether half precision can halve the bandwidth, ch03/ch11)
- What is the timestamp precision in ns/tick? (decides the credibility of performance measurement, ch09)

These answers all live in "device capabilities (device properties / features / limits / memory)." This chapter prints that report, so you can come back to it later when problems arise.

## 0.2 Installation and Build Environment

| Component | Purpose | Check Command |
|------|------|----------|
| **Vulkan SDK ≥ 1.2** | Provides headers, `glslc` (shader compiler), Validation Layers | `glslc --version` |
| **Driver** | The GPU vendor driver ships with the Vulkan runtime | Running this chapter's program and seeing the device name means success |
| **CMake ≥ 3.16** | Build system | `cmake --version` |
| **C++17 compiler** | Compiles the course code | MSVC / GCC / Clang all work |

> Windows users: After installing the SDK, the `VULKAN_SDK` environment variable is set automatically, so that `find_package(Vulkan)` and `find_program(GLSLC)` can locate them.

## 0.3 What This Chapter Does (Code Walkthrough)

`main.cpp` manually walks through the lowest-level discovery chain (**without using common/vk_base**, deliberately, so you can see the skeleton clearly):

1. **Create the Instance**: enable `VK_LAYER_KHRONOS_validation` if possible. Validation Layers are Vulkan's "compiler warnings" — they report errors directly when you pass wrong parameters, miss a barrier, or access out of bounds — **the whole course recommends keeping them on while debugging**.
2. **Enumerate physical devices**: `inst.enumeratePhysicalDevices()` returns all available GPUs. Multi-GPU machines will see several.
3. **Query capabilities**:
   - `getProperties()` → name / type / limits (`maxComputeWorkGroupSize`, `maxStorageBufferRange`, `timestampPeriod`…)
   - `getFeatures2()` + `VkPhysicalDeviceVulkan12Features` → `shaderFloat16`, `storageBuffer16BitAccess`
   - `getMemoryProperties()` → memory heaps (VRAM size) and memory types (device-local / host-visible / coherent / cached)
   - `getQueueFamilyProperties()` → which queue families support `COMPUTE`
4. **Print the report**: output all the above fields per device.

## 0.4 How to Read This Report (Key Points)

- **`maxComputeWorkGroupInvocations`**: commonly 256 or 1024. Your `local_size_x*y*z` must not exceed it, nor exceed each component of `maxComputeWorkGroupSize`.
- **`maxComputeSharedMemorySize`**: commonly 32KB or 64KB. The shared-memory tile for the ch04 GEMM must fit inside.
- **`shaderFloat16` / `storageBuffer16BitAccess`**: both must be "supported" to use a half-precision storage buffer (ch03). Apple / desktop discrete GPUs generally support it; a few old mobile GPUs do not.
- **`timestampPeriod`**: used for performance timing. AMD is commonly 1ns, some NVIDIA cards are different; ch09 will use it to convert timestamp ticks into nanoseconds.
- **Memory types**: you will see `device-local` (VRAM, fast but CPU cannot access directly) and `host-visible` (CPU-mappable, slow). The transfer between them is the core of ch03.

## 0.5 Hands-on Experiments

1. Build and run: `cmake -B build && cmake --build build --target ch00_environment`, and observe your device report.
2. If you have a dual-GPU laptop (integrated + discrete), check whether two devices appear in the report. Think: the course selects device #0 by default — could it pick the slow integrated GPU? (You can prefer `eDiscreteGpu` in `common/vk_base.cpp`.)
3. Confirm that `shaderFloat16` is "supported" — if not, the half-precision parts of ch03/ch11 need to fall back to manual `uint16` packing, and the course will note the alternative.

## 0.6 Comparison with Industrial Implementations

- **ncnn**: `create_gpu_instance()` (`src/gpu.cpp`) creates the instance and `get_gpu_device()->info` (`GpuInfo`, `src/gpu.h`) probes fp16 / int8 / bf16 / subgroup / cooperative_matrix / cooperative_vector / float8 once; extension probing is `VulkanDevice::init_device_extension()`. These capability bits decide which kernel variants load later. This chapter is its "mini probing version."
- **llama.cpp / ggml-vulkan**: device state now lives in `vk_device_struct` (`ggml-vulkan-types.h`, initialized in `ggml-vulkan.cpp`), recording fp16, subgroup size, workgroup limits, timestamp period, etc. The `vk-device.cpp` / `dev->device_vk` names from the first release no longer exist.

In the next chapter (ch01) we will officially run our first operator — vector addition — and establish the main pipeline of "create buffer → upload → create pipeline → dispatch → read back."
