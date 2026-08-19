// ch00_environment/main.cpp
// 环境搭建与设备枚举：在写任何算子之前，先弄清楚「你手上这块 GPU 长什么样」。
// 这一章不跑计算，只打印一份设备能力报告（队列/内存/关键限制/特性）。
// 后续每一章用到的分组上限、FP16 支持、时间戳精度，都能在这里提前看到。
//
// 注意：本章刻意不用 common/vk_base，而是手动创建 instance，目的是让你看清
//       「实例 → 物理设备枚举 → 能力查询」这条最底层的链路。

#include <vulkan/vulkan.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include "i18n.h"

static std::string deviceTypeStr(vk::PhysicalDeviceType t) {
    switch (t) {
        case vk::PhysicalDeviceType::eDiscreteGpu:    return i18n::tr("独立显卡", "discrete GPU");
        case vk::PhysicalDeviceType::eIntegratedGpu:  return i18n::tr("集成显卡", "integrated GPU");
        case vk::PhysicalDeviceType::eVirtualGpu:     return i18n::tr("虚拟显卡", "virtual GPU");
        case vk::PhysicalDeviceType::eCpu:            return i18n::tr("CPU 软件实现", "CPU software impl");
        default:                                      return i18n::tr("其他", "other");
    }
}

int main(int argc, char** argv) {
    i18n::init(argc, argv);
    try {
        // ---- 1. 创建实例（尽量开启 Validation Layers，没有也不强求）----
        vk::ApplicationInfo appInfo{"vulkan_nn_course_ch00", 1, nullptr, 0, VK_API_VERSION_1_2};
        std::vector<const char*> layers;
        auto availLayers = vk::enumerateInstanceLayerProperties();
        bool hasValidation = false;
        for (auto& l : availLayers)
            if (std::string(l.layerName) == "VK_LAYER_KHRONOS_validation") { hasValidation = true; break; }
        if (hasValidation) layers.push_back("VK_LAYER_KHRONOS_validation");

        vk::InstanceCreateInfo instCi{{}, &appInfo, (uint32_t)layers.size(), layers.data()};
        vk::Instance inst = vk::createInstance(instCi);
        std::cout << "Validation Layers: " << (hasValidation ? i18n::tr("已开启", "enabled") : i18n::tr("未安装/未开启", "not installed/disabled")) << "\n";

        // ---- 2. 枚举物理设备 ----
        auto phys = inst.enumeratePhysicalDevices();
        if (phys.empty()) { std::cerr << i18n::tr("没有可用的 Vulkan 设备", "no available Vulkan device") << "\n"; return 1; }
        std::cout << i18n::tr("发现 ", "found ") << phys.size() << i18n::tr(" 块 Vulkan 设备：", " Vulkan device(s):") << "\n\n";

        for (size_t idx = 0; idx < phys.size(); ++idx) {
            auto& p = phys[idx];
            auto props = p.getProperties();
            auto feat2 = vk::PhysicalDeviceFeatures2{};
            vk::PhysicalDeviceVulkan12Features f12{};
            vk::PhysicalDevice16BitStorageFeatures f16{};
            feat2.pNext = &f12;
            f12.pNext = &f16;
            p.getFeatures2(&feat2);
            auto mem = p.getMemoryProperties();
            auto qf = p.getQueueFamilyProperties();

            std::cout << "==== 设备 #" << idx << " ====\n";
            std::cout << "  名称           : " << props.deviceName << "\n";
            std::cout << "  类型           : " << deviceTypeStr(props.deviceType) << "\n";
            std::cout << "  厂商ID         : 0x" << std::hex << props.vendorID << std::dec << "\n";
            std::cout << "  驱动版本       : " << props.driverVersion << "\n";
            std::cout << i18n::tr("  支持的 Vulkan  : ", "  Vulkan supported: ")
                      << VK_VERSION_MAJOR(props.apiVersion) << "."
                      << VK_VERSION_MINOR(props.apiVersion) << "."
                      << VK_VERSION_PATCH(props.apiVersion) << "\n";

            std::cout << i18n::tr("  --- 对 NN 计算关键的 limits ---", "  --- limits critical for NN compute ---") << "\n";
            std::cout << "  maxComputeWorkGroupInvocations : "
                      << props.limits.maxComputeWorkGroupInvocations << "\n";
            std::cout << "  maxComputeWorkGroupSize[xyz]   : "
                      << props.limits.maxComputeWorkGroupSize[0] << " x "
                      << props.limits.maxComputeWorkGroupSize[1] << " x "
                      << props.limits.maxComputeWorkGroupSize[2] << "\n";
            std::cout << "  maxComputeSharedMemorySize     : "
                      << props.limits.maxComputeSharedMemorySize << i18n::tr(" 字节", " bytes") << "\n";
            std::cout << "  maxStorageBufferRange          : "
                      << props.limits.maxStorageBufferRange << i18n::tr(" 字节", " bytes") << "\n";
            std::cout << "  timestampPeriod                : "
                      << props.limits.timestampPeriod << i18n::tr(" ns / tick（性能计时精度）", " ns / tick (perf timing precision)") << "\n";

            std::cout << i18n::tr("  --- 关键特性开关 ---", "  --- key feature switches ---") << "\n";
            std::cout << "  shaderFloat16                 : " << (f12.shaderFloat16 ? i18n::tr("支持", "supported") : i18n::tr("不支持", "not supported")) << "\n";
            std::cout << "  storageBuffer16BitAccess      : " << (f16.storageBuffer16BitAccess ? i18n::tr("支持", "supported") : i18n::tr("不支持", "not supported")) << "\n";

            std::cout << i18n::tr("  --- 内存堆/类型（显存管理基础）---", "  --- memory heaps/types (VRAM mgmt basis) ---") << "\n";
            for (uint32_t h = 0; h < mem.memoryHeapCount; ++h) {
                std::cout << "  堆#" << h << " : "
                          << (mem.memoryHeaps[h].size / (1024 * 1024)) << " MB"
                          << ((mem.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal) ? " [device-local]" : "")
                          << "\n";
            }
            for (uint32_t t = 0; t < mem.memoryTypeCount; ++t) {
                auto f = mem.memoryTypes[t].propertyFlags;
                std::cout << "    类型#" << t << " 堆" << mem.memoryTypes[t].heapIndex
                          << " ["
                          << ((f & vk::MemoryPropertyFlagBits::eDeviceLocal)      ? "device-local " : "")
                          << ((f & vk::MemoryPropertyFlagBits::eHostVisible)      ? "host-visible " : "")
                          << ((f & vk::MemoryPropertyFlagBits::eHostCoherent)     ? "host-coherent " : "")
                          << ((f & vk::MemoryPropertyFlagBits::eHostCached)       ? "host-cached " : "")
                          << "]\n";
            }

            std::cout << i18n::tr("  --- 队列族（找支持 compute 的）---", "  --- queue families (find compute-capable) ---") << "\n";
            for (uint32_t q = 0; q < qf.size(); ++q) {
                bool c = static_cast<bool>(qf[q].queueFlags & vk::QueueFlagBits::eCompute);
                std::cout << "    族#" << q << " 数量=" << qf[q].queueCount
                          << (c ? " [支持 COMPUTE]" : "") << "\n";
            }
            std::cout << "\n";
        }

        inst.destroy();
        std::cout << i18n::tr("提示：后续章节默认选第 0 块「支持 compute 的」设备。若你有多卡，可在 common/vk_base.cpp 里改选择逻辑。", "hint: later chapters pick device #0 that supports compute by default. If you have multiple GPUs, change the selection logic in common/vk_base.cpp.") << "\n";
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
