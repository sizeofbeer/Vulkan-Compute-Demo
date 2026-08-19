// vk_base.cpp —— 共享 Vulkan 封装的实现
#include "vk_base.h"
#include "i18n.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <unordered_set>

// ============================================================
// 内部辅助
// ============================================================

static uint32_t findMemoryType(const VulkanContext& ctx, uint32_t typeFilter,
                               vk::MemoryPropertyFlags props) {
    auto memProps = ctx.physical.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error(i18n::tr("找不到满足条件的内存类型", "no suitable memory type found"));
}

// ============================================================
// 扩展探测
// ============================================================

// 检查物理设备上是否启用了某个扩展
static bool hasExtension(const vk::PhysicalDevice& phys, const std::string& name) {
    auto exts = phys.enumerateDeviceExtensionProperties();
    for (auto& e : exts) {
        if (e.extensionName == name) return true;
    }
    return false;
}

// 探测设备所有特性
static DeviceFeatures probeFeatures(const vk::PhysicalDevice& phys) {
    DeviceFeatures f;

    // --- Subgroup properties ---
    vk::PhysicalDeviceSubgroupProperties sgProps;
    vk::PhysicalDeviceProperties2 props2;
    props2.pNext = &sgProps;
    phys.getProperties2(&props2);

    f.subgroup = (sgProps.subgroupSize > 0);
    f.subgroupSize = sgProps.subgroupSize;
    f.subgroupBasic = !!(sgProps.supportedOperations & vk::SubgroupFeatureFlagBits::eBasic);
    f.subgroupVote = !!(sgProps.supportedOperations & vk::SubgroupFeatureFlagBits::eVote);
    f.subgroupArithmetic = !!(sgProps.supportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic);
    f.subgroupBallot = !!(sgProps.supportedOperations & vk::SubgroupFeatureFlagBits::eBallot);
    f.subgroupShuffle = !!(sgProps.supportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);
    f.subgroupShuffleRelative = !!(sgProps.supportedOperations & vk::SubgroupFeatureFlagBits::eShuffleRelative);
    f.subgroupClustered = !!(sgProps.supportedOperations & vk::SubgroupFeatureFlagBits::eClustered);
    f.subgroupQuad = !!(sgProps.supportedOperations & vk::SubgroupFeatureFlagBits::eQuad);

    // --- Cooperative Matrix ---
    f.cooperativeMatrixKHR = hasExtension(phys, "VK_KHR_cooperative_matrix");
    f.cooperativeMatrixNV = hasExtension(phys, "VK_NV_cooperative_matrix");
    f.cooperativeMatrixMaintenance1 = hasExtension(phys, "VK_EXT_cooperative_matrix_maintenance1");

    // --- Subgroup Size Control ---
    f.subgroupSizeControl = hasExtension(phys, "VK_EXT_subgroup_size_control");
    if (f.subgroupSizeControl) {
        vk::PhysicalDeviceSubgroupSizeControlProperties sgscProps;
        vk::PhysicalDeviceProperties2 p2;
        p2.pNext = &sgscProps;
        phys.getProperties2(&p2);
        f.subgroupSizeControlFullGroups = sgscProps.maxSubgroupSize > 0;
        f.minSubgroupSize = sgscProps.minSubgroupSize;
        f.maxSubgroupSize = sgscProps.maxSubgroupSize;
    }

    // --- Float16 / Int8 (Vulkan 1.2 features) ---
    vk::PhysicalDeviceVulkan12Features feat12;
    vk::PhysicalDeviceFeatures2 feat2;
    feat2.pNext = &feat12;
    phys.getFeatures2(&feat2);
    f.shaderFloat16 = feat12.shaderFloat16;
    f.shaderInt8 = feat12.shaderInt8;

    // --- 16-bit storage ---
    vk::PhysicalDevice16BitStorageFeatures feat16;
    if (feat12.pNext) { /* already chained */ }
    // Re-query with 16-bit storage
    feat2.pNext = &feat12;
    feat12.pNext = &feat16;
    phys.getFeatures2(&feat2);
    f.storageBuffer16BitAccess = feat16.storageBuffer16BitAccess;

    // --- Integer Dot Product ---
    f.integerDotProduct = hasExtension(phys, "VK_KHR_shader_integer_dot_product");

    // --- Robustness 2 ---
    f.robustness2 = hasExtension(phys, "VK_EXT_robustness2");

    // --- OCP Microscaling ---
    f.shaderOCPMicroscalingTypes = hasExtension(phys, "VK_EXT_shader_ocp_microscaling_types");

    // --- Pipeline Cache ---
    f.pipelineCacheControl = true; // core feature

    return f;
}

// ============================================================
// 公共函数实现
// ============================================================

VulkanContext vkInit() {
    return vkInitWithExtensions({});
}

VulkanContext vkInitWithExtensions(const std::vector<const char*>& extraExtensions) {
    VulkanContext ctx;

    // 1. 创建实例
    vk::ApplicationInfo appInfo{"vulkan_nn_course", 1, nullptr, 0, VK_API_VERSION_1_3};
    std::vector<const char*> layers;
#ifdef _DEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif
    vk::InstanceCreateInfo instCi{{}, &appInfo, (uint32_t)layers.size(), layers.data()};
    ctx.instance = vk::createInstance(instCi);

    // 2. 选择物理设备
    auto phys = ctx.instance.enumeratePhysicalDevices();
    if (phys.empty()) throw std::runtime_error(i18n::tr("没有可用的 Vulkan 设备", "no available Vulkan device"));

    // 设备选择策略（保持默认行为的同时，避免选到集成核显）：
    //   1) 设置环境变量 VULKAN_NN_DEVICE=<名称子串> 时，强制选名称包含该子串的设备；
    //   2) 否则优先选「独立 GPU」（如 NVIDIA/AMD 独显，跳过 Intel 核显）；
    //   3) 都没有则退回第一块设备（与原始行为一致）。
    const char* wantEnv = std::getenv("VULKAN_NN_DEVICE");
    std::string want = wantEnv ? wantEnv : "";
    ctx.physical = phys[0];
    if (!want.empty()) {
        bool found = false;
        for (auto& p : phys) {
            auto props = p.getProperties();
            if (std::string(props.deviceName).find(want) != std::string::npos) {
                ctx.physical = p;
                found = true;
                break;
            }
        }
        if (!found)
            std::cout << i18n::tr("[vk] 警告: 未找到名称含 '", "[vk] warn: no device name contains '")
                      << want << i18n::tr("' 的设备，使用默认设备\n", "', using default\n");
    } else {
        // 默认偏好独立 GPU（跳过集成显卡）
        for (auto& p : phys) {
            auto props = p.getProperties();
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                ctx.physical = p;
                break;
            }
        }
    }
    ctx.props = ctx.physical.getProperties();
    std::cout << i18n::tr("[vk] 选中设备: ", "[vk] selected device: ") << ctx.props.deviceName << "\n";

    // 3. 探测特性
    ctx.features = probeFeatures(ctx.physical);
    // 保存 subgroup 属性供着色器使用
    vk::PhysicalDeviceSubgroupProperties sgProps;
    vk::PhysicalDeviceProperties2 p2;
    p2.pNext = &sgProps;
    ctx.physical.getProperties2(&p2);
    ctx.subgroupProps = sgProps;

    // 打印特性摘要
    std::cout << i18n::tr("  Subgroup: ", "  Subgroup: ") << ctx.features.subgroupSize << " lanes, "
              << "Arith=" << ctx.features.subgroupArithmetic
              << " Shuffle=" << ctx.features.subgroupShuffle
              << " Ballot=" << ctx.features.subgroupBallot << "\n";
    std::cout << i18n::tr("  CooperativeMatrix: ", "  CooperativeMatrix: ")
              << (ctx.features.cooperativeMatrixKHR ? "KHR " : "")
              << (ctx.features.cooperativeMatrixNV ? "NV" : "")
              << (ctx.features.cooperativeMatrixMaintenance1 ? " +Maintenance1" : "")
              << (!ctx.features.cooperativeMatrixKHR && !ctx.features.cooperativeMatrixNV ? "none" : "") << "\n";
    std::cout << i18n::tr("  Float16: ", "  Float16: ") << ctx.features.shaderFloat16
              << " Int8: " << ctx.features.shaderInt8
              << " IntDot: " << ctx.features.integerDotProduct << "\n";

    // 4. 选择队列族
    auto qf = ctx.physical.getQueueFamilyProperties();
    ctx.queueFamily = 0;
    for (uint32_t i = 0; i < qf.size(); ++i) {
        if (qf[i].queueFlags & vk::QueueFlagBits::eCompute) { ctx.queueFamily = i; break; }
    }

    // 5. 收集所需的扩展
    std::vector<const char*> extensions;
    // 基础扩展
    extensions.push_back("VK_KHR_16bit_storage");
    // 可选扩展
    if (ctx.features.cooperativeMatrixKHR)
        extensions.push_back("VK_KHR_cooperative_matrix");
    if (ctx.features.cooperativeMatrixNV)
        extensions.push_back("VK_NV_cooperative_matrix");
    if (ctx.features.cooperativeMatrixMaintenance1)
        extensions.push_back("VK_EXT_cooperative_matrix_maintenance1");
    if (ctx.features.subgroupSizeControl)
        extensions.push_back("VK_EXT_subgroup_size_control");
    if (ctx.features.integerDotProduct)
        extensions.push_back("VK_KHR_shader_integer_dot_product");
    if (ctx.features.shaderOCPMicroscalingTypes)
        extensions.push_back("VK_EXT_shader_ocp_microscaling_types");
    // 用户指定的额外扩展
    for (auto& ext : extraExtensions) {
        extensions.push_back(ext);
    }
    ctx.enabledExtensions = extensions;

    // 6. 创建设备
    float priority = 1.0f;
    vk::DeviceQueueCreateInfo dqCi{{}, ctx.queueFamily, 1, &priority};

    vk::PhysicalDeviceVulkan12Features features12;
    vk::PhysicalDevice16BitStorageFeatures features16;
    vk::PhysicalDeviceSubgroupSizeControlFeatures sgscFeat;
    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR cmFeatKHR;
    vk::PhysicalDeviceCooperativeMatrixFeaturesNV cmFeatNV;

    features12.shaderFloat16 = ctx.features.shaderFloat16;
    features12.shaderInt8    = ctx.features.shaderInt8;
    features12.pNext = &features16;
    features16.storageBuffer16BitAccess = ctx.features.storageBuffer16BitAccess;

    // 链入可选特性
    void** tail = &features16.pNext;
    if (ctx.features.subgroupSizeControl) {
        sgscFeat.subgroupSizeControl = VK_TRUE;
        sgscFeat.computeFullSubgroups = VK_TRUE;
        *tail = &sgscFeat;
        tail = &sgscFeat.pNext;
    }
    if (ctx.features.cooperativeMatrixKHR) {
        cmFeatKHR.cooperativeMatrix = VK_TRUE;
        *tail = &cmFeatKHR;
        tail = &cmFeatKHR.pNext;
    }
    if (ctx.features.cooperativeMatrixNV) {
        cmFeatNV.cooperativeMatrix = VK_TRUE;
        *tail = &cmFeatNV;
        tail = &cmFeatNV.pNext;
    }

    vk::DeviceCreateInfo devCi{{}, 1, &dqCi};
    devCi.pNext = &features12;
    devCi.enabledExtensionCount = (uint32_t)extensions.size();
    devCi.ppEnabledExtensionNames = extensions.data();

    ctx.device = ctx.physical.createDevice(devCi);
    ctx.queue = ctx.device.getQueue(ctx.queueFamily, 0);
    ctx.cmdPool = ctx.device.createCommandPool({vk::CommandPoolCreateFlags{}, ctx.queueFamily});
    ctx.fence  = ctx.device.createFence({});

    return ctx;
}

void vkDestroyContext(VulkanContext& ctx) {
    if (ctx.fence)    ctx.device.destroyFence(ctx.fence);
    if (ctx.cmdPool)  ctx.device.destroyCommandPool(ctx.cmdPool);
    if (ctx.device)   ctx.device.destroy();
    if (ctx.instance) ctx.instance.destroy();
}

// ---- Buffer 操作 ----

Buffer vkCreateBuffer(const VulkanContext& ctx, vk::DeviceSize size,
                      vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memFlags) {
    Buffer b;
    b.size = size;
    b.flags = memFlags;
    b.buf = ctx.device.createBuffer({{}, size, usage});
    auto req = ctx.device.getBufferMemoryRequirements(b.buf);
    vk::MemoryAllocateInfo allocCi{req.size, findMemoryType(ctx, req.memoryTypeBits, memFlags)};
    b.mem = ctx.device.allocateMemory(allocCi);
    ctx.device.bindBufferMemory(b.buf, b.mem, 0);
    return b;
}

void vkDestroyBuffer(const VulkanContext& ctx, Buffer& b) {
    if (b.buf) ctx.device.destroyBuffer(b.buf);
    if (b.mem) ctx.device.freeMemory(b.mem);
    b = Buffer{};
}

void vkUpload(const VulkanContext& ctx, Buffer& dst, const void* data, vk::DeviceSize size) {
    if (dst.flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        void* p = ctx.device.mapMemory(dst.mem, 0, size);
        std::memcpy(p, data, (size_t)size);
        ctx.device.unmapMemory(dst.mem);
        if (!(dst.flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            vk::MappedMemoryRange range{dst.mem, 0, size};
            ctx.device.flushMappedMemoryRanges(1, &range);
        }
    } else {
        Buffer staging = vkCreateBuffer(ctx, size,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        void* p = ctx.device.mapMemory(staging.mem, 0, size);
        std::memcpy(p, data, (size_t)size);
        ctx.device.unmapMemory(staging.mem);

        vk::CommandBuffer cmd =
            ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
        cmd.begin(vk::CommandBufferBeginInfo{});
        vk::BufferCopy copy{0, 0, size};
        cmd.copyBuffer(staging.buf, dst.buf, 1, &copy);
        cmd.end();
        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(1, &si, ctx.fence);
        ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
        ctx.device.resetFences(1, &ctx.fence);
        ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
        vkDestroyBuffer(ctx, staging);
    }
}

std::vector<uint8_t> vkDownload(const VulkanContext& ctx, const Buffer& src, vk::DeviceSize size) {
    std::vector<uint8_t> out((size_t)size);
    if (src.flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        void* p = ctx.device.mapMemory(src.mem, 0, size);
        std::memcpy(out.data(), p, (size_t)size);
        ctx.device.unmapMemory(src.mem);
    } else {
        Buffer staging = vkCreateBuffer(ctx, size,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        vk::CommandBuffer cmd =
            ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
        cmd.begin(vk::CommandBufferBeginInfo{});
        vk::BufferCopy copy{0, 0, size};
        cmd.copyBuffer(src.buf, staging.buf, 1, &copy);
        cmd.end();
        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        ctx.queue.submit(1, &si, ctx.fence);
        ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
        ctx.device.resetFences(1, &ctx.fence);
        void* p = ctx.device.mapMemory(staging.mem, 0, size);
        std::memcpy(out.data(), p, (size_t)size);
        ctx.device.unmapMemory(staging.mem);
        ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
        vkDestroyBuffer(ctx, staging);
    }
    return out;
}

vk::ShaderModule vkLoadShader(const VulkanContext& ctx, const std::string& spvPath) {
    std::ifstream f(spvPath, std::ios::binary);
    if (!f) throw std::runtime_error(i18n::tr("无法打开着色器文件: ", "cannot open shader file: ") + spvPath);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.empty()) throw std::runtime_error(i18n::tr("着色器文件为空: ", "shader file is empty: ") + spvPath);
    if (bytes.size() % 4 != 0) throw std::runtime_error(i18n::tr("着色器文件大小不是 4 字节对齐: ", "shader file size is not 4-byte aligned: ") + spvPath);
    std::vector<uint32_t> code(bytes.size() / 4);
    std::memcpy(code.data(), bytes.data(), bytes.size());
    return ctx.device.createShaderModule({{}, code});
}

void vkSubmit(const VulkanContext& ctx, const std::function<void(vk::CommandBuffer)>& recorder) {
    vk::CommandBuffer cmd =
        ctx.device.allocateCommandBuffers({ctx.cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
    cmd.begin(vk::CommandBufferBeginInfo{});
    recorder(cmd);
    cmd.end();
    vk::SubmitInfo si;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    ctx.queue.submit(1, &si, ctx.fence);
    ctx.device.waitForFences(1, &ctx.fence, true, ~0ULL);
    ctx.device.resetFences(1, &ctx.fence);
    ctx.device.freeCommandBuffers(ctx.cmdPool, 1, &cmd);
}

// ---- Pipeline Cache ----

vk::PipelineCache vkCreatePipelineCache(const VulkanContext& ctx, const std::string& diskPath) {
    vk::PipelineCacheCreateInfo ci;
    if (!diskPath.empty()) {
        std::ifstream f(diskPath, std::ios::binary | std::ios::ate);
        if (f) {
            size_t sz = (size_t)f.tellg();
            f.seekg(0);
            std::vector<uint8_t> data(sz);
            f.read((char*)data.data(), sz);
            ci = vk::PipelineCacheCreateInfo{{}, sz, data.data()};
            std::cout << i18n::tr("[pipeline-cache] 从磁盘加载 ", "[pipeline-cache] loaded from disk ")
                      << sz << " bytes\n";
        }
    }
    return ctx.device.createPipelineCache(ci);
}

void vkSavePipelineCache(const VulkanContext& ctx, vk::PipelineCache cache, const std::string& diskPath) {
    if (diskPath.empty() || !cache) return;
    auto data = ctx.device.getPipelineCacheData(cache);
    std::ofstream f(diskPath, std::ios::binary);
    if (f) {
        f.write((const char*)data.data(), data.size());
        std::cout << i18n::tr("[pipeline-cache] 保存到磁盘 ", "[pipeline-cache] saved to disk ")
                  << data.size() << " bytes\n";
    }
}

// ---- ComputePass ----

void ComputePass::prepare(VulkanContext& c, const std::string& spvPath,
                          const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                          const std::vector<vk::PushConstantRange>& pcs) {
    prepareWithSpec(c, spvPath, bindings, pcs, vk::SpecializationInfo{}, nullptr);
}

void ComputePass::prepareWithSpec(VulkanContext& c, const std::string& spvPath,
                                   const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                                   const std::vector<vk::PushConstantRange>& pcs,
                                   const vk::SpecializationInfo& specInfo,
                                   vk::PipelineCache cache) {
    ctx = &c;
    pipelineCache = cache;
    module = vkLoadShader(c, spvPath);
    dsl = c.device.createDescriptorSetLayout(
        {vk::DescriptorSetLayoutCreateFlags{}, (uint32_t)bindings.size(), bindings.data()});
    layout = c.device.createPipelineLayout(
        {vk::PipelineLayoutCreateFlags{}, 1, &dsl, (uint32_t)pcs.size(), pcs.data()});
    vk::PipelineShaderStageCreateInfo stageCi{
        vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, module, "main"};
    // 如果有 specialization info，附加到 stage
    if (specInfo.mapEntryCount > 0) {
        stageCi.pSpecializationInfo = &specInfo;
    }
    vk::ComputePipelineCreateInfo pipeCi{{}, stageCi, layout};
    pipeline = c.device.createComputePipeline(cache ? cache : vk::PipelineCache{}, pipeCi).value;

    uint32_t nStorage = 0;
    for (auto& b : bindings) if (b.descriptorType == vk::DescriptorType::eStorageBuffer) ++nStorage;
    vk::DescriptorPoolSize ps{vk::DescriptorType::eStorageBuffer, nStorage};
    pool = c.device.createDescriptorPool(
        {vk::DescriptorPoolCreateFlags{}, 1, 1, &ps});
    set = c.device.allocateDescriptorSets({pool, 1, &dsl})[0];
}

void ComputePass::bindBuffers(const std::vector<std::pair<uint32_t, vk::Buffer>>& binds) {
    std::vector<vk::DescriptorBufferInfo> infos;
    std::vector<vk::WriteDescriptorSet> writes;
    infos.reserve(binds.size());
    writes.reserve(binds.size());
    for (auto& [binding, buf] : binds) {
        infos.push_back({buf, 0, VK_WHOLE_SIZE});
        writes.push_back({set, binding, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos.back()});
    }
    ctx->device.updateDescriptorSets((uint32_t)writes.size(), writes.data(), 0, nullptr);
}

void ComputePass::dispatch(vk::CommandBuffer cmd, const void* pc, size_t pcSize,
                           uint32_t gx, uint32_t gy, uint32_t gz) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, 1, &set, 0, nullptr);
    if (pc && pcSize) cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, (uint32_t)pcSize, pc);
    cmd.dispatch(gx, gy, gz);
}

void ComputePass::destroy() {
    if (!ctx) return;
    auto& d = ctx->device;
    if (pipeline) d.destroyPipeline(pipeline);
    if (layout)   d.destroyPipelineLayout(layout);
    if (dsl)      d.destroyDescriptorSetLayout(dsl);
    if (pool)     d.destroyDescriptorPool(pool);
    if (module)   d.destroyShaderModule(module);
    *this = ComputePass{};
}

// ---- DeviceArena ----

void DeviceArena::init(VulkanContext& c, vk::DeviceSize bytes) {
    ctx = &c;
    capacity = bytes;
    offset = 0;
    buf = vkCreateBuffer(c, bytes,
        vk::BufferUsageFlagBits::eStorageBuffer
      | vk::BufferUsageFlagBits::eTransferSrc
      | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
}

vk::DeviceSize DeviceArena::alloc(vk::DeviceSize size) {
    vk::DeviceSize aligned = (size + 255) & ~vk::DeviceSize(255);
    if (offset + aligned > capacity)
        throw std::runtime_error(i18n::tr("DeviceArena 容量不足", "DeviceArena capacity exceeded"));
    vk::DeviceSize at = offset;
    offset += aligned;
    return at;
}

void DeviceArena::reset() { offset = 0; }

void DeviceArena::destroy() {
    if (ctx) vkDestroyBuffer(*ctx, buf);
    *this = DeviceArena{};
}

// ---- Graph ----

void Graph::begin() {
    cmd = ctx->device.allocateCommandBuffers(
        {ctx->cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
    cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
}

void Graph::node(ComputePass& pass, const void* pc, size_t pcSize,
                 uint32_t gx, uint32_t gy, uint32_t gz, vk::Buffer prevOut) {
    if (prevOut) {
        vk::BufferMemoryBarrier bmb{
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
            0, 0, prevOut, 0, VK_WHOLE_SIZE};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            vk::DependencyFlags{}, 0, nullptr, 1, &bmb, 0, nullptr);
    }
    pass.dispatch(cmd, pc, pcSize, gx, gy, gz);
}

void Graph::end() {
    cmd.end();
    vk::SubmitInfo si; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    ctx->queue.submit(1, &si, ctx->fence);
    ctx->device.waitForFences(1, &ctx->fence, true, ~0ULL);
    ctx->device.resetFences(1, &ctx->fence);
    ctx->device.freeCommandBuffers(ctx->cmdPool, 1, &cmd);
}

// ---- Timing ----

vk::QueryPool vkCreateTimestampPool(const VulkanContext& ctx, uint32_t count) {
    vk::QueryPoolCreateInfo qpci;
    qpci.queryType = vk::QueryType::eTimestamp;
    qpci.queryCount = count;
    return ctx.device.createQueryPool(qpci);
}

uint64_t vkReadTimestamp(const VulkanContext& ctx, vk::QueryPool pool, uint32_t index) {
    uint64_t ts = 0;
    ctx.device.getQueryPoolResults(pool, index, 1, sizeof(ts), &ts, sizeof(ts),
                                   vk::QueryResultFlagBits::e64);
    // 转换为纳秒：timestamp * timestampPeriod
    uint64_t ns = uint64_t(double(ts) * ctx.props.limits.timestampPeriod);
    return ns;
}