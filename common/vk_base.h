#pragma once
// vk_base.h —— 课程共享的极简 Vulkan 封装
// 目的：把「实例/设备/队列/内存/上传下载/一键提交」等样板收敛到这里，
//       让每一章的 main.cpp 只聚焦「要讲的概念」。
// 注意：这是教学用最小封装，不是工业级引擎；越往后的章节会逐步增强它。

#include <vulkan/vulkan.hpp>
#include <vector>
#include <cstdint>
#include <string>
#include <functional>
#include <unordered_map>
#include <array>
#include <optional>

// ============================================================
// 1. 基础数据结构
// ============================================================

// 一个 GPU buffer 及其绑定的显存
struct Buffer {
    vk::Buffer      buf{};
    vk::DeviceMemory mem{};
    vk::DeviceSize  size = 0;
    vk::MemoryPropertyFlags flags{};
};

// 设备特性探测（用于 ch15+ 的扩展检测）
struct DeviceFeatures {
    // --- Subgroup ---
    bool subgroup                    = false;
    bool subgroupBasic               = false;
    bool subgroupVote                = false;
    bool subgroupArithmetic          = false;
    bool subgroupBallot              = false;
    bool subgroupShuffle             = false;
    bool subgroupShuffleRelative     = false;
    bool subgroupClustered           = false;
    bool subgroupQuad                = false;
    uint32_t subgroupSize            = 32;
    bool subgroupSizeControl         = false;
    // --- Cooperative Matrix ---
    bool cooperativeMatrixKHR        = false;
    bool cooperativeMatrixNV         = false;
    bool cooperativeMatrixMaintenance1 = false;
    // --- Subgroup Size Control ---
    bool subgroupSizeControlFullGroups = false;
    uint32_t minSubgroupSize         = 1;
    uint32_t maxSubgroupSize         = 64;
    // --- Float16 / Int8 ---
    bool shaderFloat16               = false;
    bool shaderInt8                  = false;
    bool storageBuffer16BitAccess    = false;
    // --- Integer Dot Product ---
    bool integerDotProduct           = false;
    // --- Robustness ---
    bool robustness2                 = false;
    // --- Microscaling ---
    bool shaderOCPMicroscalingTypes  = false;
    // --- Pipeline Cache ---
    bool pipelineCacheControl        = false;
};

// 整个课程的 Vulkan 上下文（一台设备 + 一个计算队列 + 命令池 + 一个复用 fence）
struct VulkanContext {
    vk::Instance             instance{};
    vk::PhysicalDevice       physical{};
    vk::Device               device{};
    vk::Queue                queue{};
    uint32_t                 queueFamily = 0;
    vk::PhysicalDeviceProperties  props{};
    vk::PhysicalDeviceSubgroupProperties subgroupProps{};
    vk::CommandPool          cmdPool{};
    vk::Fence                fence{};
    DeviceFeatures           features{};
    // 保存设备创建时的扩展名列表，供 pipeline cache 等使用
    std::vector<const char*> enabledExtensions{};
};

// ============================================================
// 2. 初始化与销毁
// ============================================================

// 初始化：创建实例、探测特性、选第一块支持 compute 的 GPU、建队列/命令池/fence
VulkanContext vkInit();

// 可指定额外扩展名列表初始化
VulkanContext vkInitWithExtensions(const std::vector<const char*>& extraExtensions);

// 释放所有资源
void vkDestroyContext(VulkanContext& ctx);

// ============================================================
// 3. Buffer 操作
// ============================================================

// 创建 buffer 并分配显存。usage=缓冲用途，memFlags=期望的内存属性（见 ch03）
Buffer vkCreateBuffer(const VulkanContext& ctx, vk::DeviceSize size,
                      vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memFlags);

void vkDestroyBuffer(const VulkanContext& ctx, Buffer& b);

// 上传：host-visible 直接 map 写；device-local 则经 staging 中转（见 ch03）
void vkUpload(const VulkanContext& ctx, Buffer& dst, const void* data, vk::DeviceSize size);

// 下载：返回 size 字节。host-visible 直接 map 读；device-local 经 staging 中转
std::vector<uint8_t> vkDownload(const VulkanContext& ctx, const Buffer& src, vk::DeviceSize size);

// 读取 .spv 文件并创建 shader module
vk::ShaderModule vkLoadShader(const VulkanContext& ctx, const std::string& spvPath);

// 一键提交：录制命令 -> 提交 -> 等 fence（ch01 用；ch02 会手写同样的流程以示教学）
void vkSubmit(const VulkanContext& ctx, const std::function<void(vk::CommandBuffer)>& recorder);

// ============================================================
// 4. Pipeline Cache（持久化管线缓存）
// ============================================================

// 创建或加载 pipeline cache
// 返回一个功能性 VkPipelineCache。如果 diskPath 非空，尝试从磁盘加载已有缓存。
// 在 destroy 阶段调用 vkSavePipelineCache 写入磁盘。
vk::PipelineCache vkCreatePipelineCache(const VulkanContext& ctx, const std::string& diskPath = "");

// 将 pipeline cache 保存到磁盘
void vkSavePipelineCache(const VulkanContext& ctx, vk::PipelineCache cache, const std::string& diskPath);

// ============================================================
// 5. Specialization Constant 辅助（ch15+ 使用）
// ============================================================

// 辅助函数：构建 specialization constant 的 map entries + 数据块
// 用法：
//   std::array<uint32_t, 3> specData = {M, N, K};
//   auto specInfo = vkMakeSpecializationInfo(specData);
//   然后传给 createComputePipeline 的 SpecializationInfo
struct SpecializationInfo {
    std::vector<vk::SpecializationMapEntry> entries;
    std::vector<uint8_t> data;
    vk::SpecializationInfo info;
};

// 从一个类型匹配的数组构建 SpecializationInfo
// T 必须是一个平凡的 POD 类型（如 std::array<uint32_t, N> 或 std::vector<uint32_t>）
template<typename T>
SpecializationInfo vkMakeSpecializationInfo(const T& values, uint32_t startId = 0) {
    SpecializationInfo result;
    size_t elemSize = sizeof(typename T::value_type);
    result.data.resize(values.size() * elemSize);
    std::memcpy(result.data.data(), values.data(), result.data.size());
    result.entries.reserve(values.size());
    for (uint32_t i = 0; i < values.size(); ++i) {
        result.entries.push_back({startId + i, i * elemSize, elemSize});
    }
    result.info = vk::SpecializationInfo{
        (uint32_t)result.entries.size(), result.entries.data(),
        (uint32_t)result.data.size(), result.data.data()
    };
    return result;
}

// ============================================================
// 6. ComputePass —— 迷你「VkCompute」封装
// ============================================================

// 把「建管线 + 建描述符集 + 绑 buffer + push constant + dispatch」收敛成一个对象，
// 让后面的算子章节只关心「喂哪些 buffer、压什么常量、dispatch 几组」。
// 这直接对应 ncnn 的 VkCompute / llama.cpp 的 vk_pipeline_wrapper 思路。
struct ComputePass {
    VulkanContext* ctx = nullptr;
    vk::ShaderModule        module{};
    vk::DescriptorSetLayout dsl{};
    vk::PipelineLayout      layout{};
    vk::Pipeline            pipeline{};
    vk::DescriptorPool      pool{};
    vk::DescriptorSet       set{};
    vk::PipelineCache       pipelineCache{};  // 可选，ch17+ 使用

    // 准备阶段：加载 .spv、建描述符布局/管线布局/管线、分配描述符集。
    // bindings：每个 storage buffer 的 (binding 号, 阶段)；pcs：push constant 范围。
    void prepare(VulkanContext& c, const std::string& spvPath,
                 const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                 const std::vector<vk::PushConstantRange>& pcs);

    // 带 specialization constant 的 prepare
    void prepareWithSpec(VulkanContext& c, const std::string& spvPath,
                         const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                         const std::vector<vk::PushConstantRange>& pcs,
                         const vk::SpecializationInfo& specInfo,
                         vk::PipelineCache cache = nullptr);

    // 把真实 buffer 绑到描述符集（全部按 VK_WHOLE_SIZE 绑定）
    void bindBuffers(const std::vector<std::pair<uint32_t, vk::Buffer>>& binds);

    // 录制一次 dispatch（调用前需自行 begin 命令缓冲；这里只录计算相关命令）
    void dispatch(vk::CommandBuffer cmd, const void* pc, size_t pcSize,
                  uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);

    void destroy();
};

// ============================================================
// 7. DeviceArena —— 显存池（bump 分配器）
// ============================================================

// 从一大块 device-local buffer 里「切」出多个 tensor，避免每次 tiny 分配都调 Vulkan。
// 对应 ncnn 的 VkBlobAllocator / ggml 的 arena。alloc 返回字节偏移；reset 一次性回收全部。
struct DeviceArena {
    VulkanContext* ctx = nullptr;
    Buffer          buf{};
    vk::DeviceSize  offset = 0;
    vk::DeviceSize  capacity = 0;
    void init(VulkanContext& c, vk::DeviceSize bytes);
    vk::DeviceSize alloc(vk::DeviceSize size);   // 256 字节对齐后切分
    void reset();                                // bump 归零（所有已分配作废）
    void destroy();
};

// ============================================================
// 8. Graph —— 计算图（录制到同一条命令缓冲，一次性提交）
// ============================================================

// 把多个算子录进一个 command buffer，依赖处插 pipeline barrier（写后读），最后只 submit 一次。
// 对应 ggml 的计算图录制 / ncnn 的 layer 串行提交。避免每算子一次 submit 的队列开销（ch09 提过）。
struct Graph {
    VulkanContext* ctx = nullptr;
    vk::CommandBuffer cmd{};
    void begin();                                              // 分配并 begin 命令缓冲
    // 录制一个算子；prevOut=上一个算子的输出 buffer（需要写后读 barrier 保证可见）
    void node(ComputePass& pass, const void* pc, size_t pcSize,
              uint32_t gx, uint32_t gy, uint32_t gz, vk::Buffer prevOut);
    void end();                                                // end + submit + 等 fence + 释放
};

// ============================================================
// 9. Timing 辅助（ch09 增强）
// ============================================================

// 创建 GPU timestamp query pool
vk::QueryPool vkCreateTimestampPool(const VulkanContext& ctx, uint32_t count = 2);

// 读取 timestamp 结果（纳秒）
uint64_t vkReadTimestamp(const VulkanContext& ctx, vk::QueryPool pool, uint32_t index);

// ============================================================
// 10. 扩展辅助宏
// ============================================================

// 在着色器中启用扩展的 GLSL preamble
// 使用方式：在 computePass 的 dispatch 前，把 GLSL 中通过扩展名识别
// 这些宏在着色器编译时已经确定，这里只是 C++ 端的辅助信息

// 检查设备是否支持 Cooperative Matrix
inline bool hasCooperativeMatrix(const VulkanContext& ctx) {
    return ctx.features.cooperativeMatrixKHR || ctx.features.cooperativeMatrixNV;
}

// 检查设备是否支持 Subgroup 操作
inline bool hasSubgroupOps(const VulkanContext& ctx) {
    return ctx.features.subgroup;
}

// 检查设备是否支持 Integer Dot Product
inline bool hasIntegerDotProduct(const VulkanContext& ctx) {
    return ctx.features.integerDotProduct;
}
