# 第 0 章 · 环境搭建与设备枚举

> 目标：在你写第一行计算代码之前，先确认「工具链装好了、GPU 能被发现、它支持什么」。这一章是一个**纯探测工具**，不跑任何算子。

---

## 0.1 为什么要有这一章

Vulkan 是「显式」的 API：它不会替你判断设备能力，也不会在你用错时温柔提示。神经网络计算里你会反复碰到这些前提条件：

- 我的 GPU 一个工作组最多多少线程？（决定 `local_size_x` 上限）
- 共享内存上限多少字节？（决定 GEMM 的 tile 尺寸，ch04）
- 支持 FP16 存储吗？（决定能否用半精度省一半带宽，ch03/ch11）
- 时间戳精度多少 ns/tick？（决定性能测量的可信度，ch09）

这些答案都在「设备能力（device properties / features / limits / memory）」里。本章把这份报告打印出来，后面遇事就来翻。

## 0.2 安装与构建环境

| 组件 | 作用 | 检查命令 |
|------|------|----------|
| **Vulkan SDK ≥ 1.2** | 提供头文件、`glslc`（着色器编译器）、Validation Layers | `glslc --version` |
| **驱动** | 显卡厂商驱动自带 Vulkan 运行时 | 运行本章程序能看到设备名即成功 |
| **CMake ≥ 3.16** | 构建系统 | `cmake --version` |
| **C++17 编译器** | 编译课程代码 | MSVC / GCC / Clang 均可 |

> Windows 用户：装完 SDK 后，`VULKAN_SDK` 环境变量会被自动设置，`find_package(Vulkan)` 与 `find_program(GLSLC)` 才能找到它们。

## 0.3 本章做了什么（代码导读）

`main.cpp` 手动走了一遍最底层的发现链路（**没有用 common/vk_base**，刻意如此，让你看清骨架）：

1. **创建实例（Instance）**：尽量开启 `VK_LAYER_KHRONOS_validation`。Validation Layers 是 Vulkan 的「编译器 warning」，能在你传错参数、漏 barrier、越界访问时直接报错——**整个课程都建议开着它调试**。
2. **枚举物理设备**：`inst.enumeratePhysicalDevices()` 返回所有可用 GPU。多卡机器会看到多块。
3. **查询能力**：
   - `getProperties()` → 名称/类型/limits（`maxComputeWorkGroupSize`、`maxStorageBufferRange`、`timestampPeriod`…）
   - `getFeatures2()` + `VkPhysicalDeviceVulkan12Features` → `shaderFloat16`、`storageBuffer16BitAccess`
   - `getMemoryProperties()` → 内存堆（显存大小）与内存类型（device-local / host-visible / coherent / cached）
   - `getQueueFamilyProperties()` → 哪些队列族支持 `COMPUTE`
4. **打印报告**：逐设备输出上面的所有字段。

## 0.4 怎么读这份报告（重点关注）

- **`maxComputeWorkGroupInvocations`**：常见 256 或 1024。你的 `local_size_x*y*z` 不能超过它，也不能超过 `maxComputeWorkGroupSize` 各分量。
- **`maxComputeSharedMemorySize`**：常见 32KB 或 64KB。ch04 GEMM 的共享内存 tile 必须塞得下。
- **`shaderFloat16` / `storageBuffer16BitAccess`**：两个都「支持」才能用半精度 storage buffer（ch03）。苹果/桌面独显普遍支持；极少数老移动 GPU 不支持。
- **`timestampPeriod`**：性能计时用。AMD 常见 1ns，部分 NVIDIA 是别的值；ch09 会用它把 timestamp tick 换算成纳秒。
- **内存类型**：你会看到 `device-local`（显存，快但 CPU 不能直接访问）和 `host-visible`（CPU 可映射，慢）。它们之间的搬运就是 ch03 的核心。

## 0.5 动手实验

1. 构建并运行：`cmake -B build && cmake --build build --target ch00_environment`，观察你的设备报告。
2. 如果你是双显卡笔记本（核显 + 独显），看报告里是否出现两块设备。思考：课程默认选第 0 块，会不会选到慢的核显？（可在 `common/vk_base.cpp` 里优先选 `eDiscreteGpu`。）
3. 确认 `shaderFloat16` 是「支持」——若不支持，ch03/ch11 的半精度部分需要改用 `uint16` 手工打包，课程会注明替代方案。

## 0.6 对照工业级

- **ncnn**：`create_gpu_instance()`（`src/gpu.cpp`）建实例，`get_gpu_device()->info`（`GpuInfo`，`src/gpu.h`）一次性探测 fp16 / int8 / bf16 / subgroup / cooperative_matrix / cooperative_vector / float8 等能力；扩展探测为 `VulkanDevice::init_device_extension()`，能力位决定后续加载哪套 kernel 变体。本章就是它的「迷你探测版」。
- **llama.cpp / ggml-vulkan**：设备状态收敛到 `ggml-vulkan-types.h` 的 `vk_device_struct`（在 `ggml-vulkan.cpp` 初始化），记录 fp16、subgroup size、workgroup 上限、timestamp period 等。首版提到的 `vk-device.cpp` / `dev->device_vk` 已不存在。

下一章（ch01）我们正式跑通第一条算子——向量加法，建立「建 buffer → 上传 → 建管线 → dispatch → 读回」的主链路。
