// common/test_vectors.h
// 测试向量支持 —— 头文件仅实现，零外部依赖，通过编译期开关隔离。
//
// 用法（以 ch04_gemm 为例）：
//
//   #include "test_vectors.h"
//
//   int main(int argc, char** argv) {
//       TestVectorConfig tv;           // 默认 disabled
//       tv.parse(argc, argv);          // 解析 --test-vectors / --output
//
//       // 生成输入数据（测试向量模式从文件加载）
//       std::vector<float> h_A(M * K), h_B(K * N);
//       if (tv.enabled()) {
//           tv.load("A", {M, K}, h_A.data());
//           tv.load("B", {K, N}, h_B.data());
//       } else {
//           // 原有随机生成逻辑
//       }
//
//       // ... 原有 compute 逻辑不变 ...
//
//       // 写回输出（仅测试向量模式）
//       tv.save("C", {M, N}, h_C.data());
//
//       return 0;
//   }
//
// 编译期隔离：定义 TEST_VECTORS_DISABLED 可彻底移除所有测试向量代码。
// 运行时隔离：不传 --test-vectors 时与原始行为 100% 一致。
//
// 自适应维度（推荐）：当测试向量的形状与章节硬编码尺寸不一致时，可先
// 用 queryShape() 读取真实形状，再按需分配 buffer / 设置 push constant：
//
//   uint32_t M = 256, N = 256, K = 256;          // 默认（原始行为）
//   if (tv.enabled()) {
//       std::vector<uint32_t> sA, sB;
//       tv.queryShape("A", sA);                   // sA = {M, K}
//       tv.queryShape("B", sB);                   // sB = {K, N}
//       M = sA[0]; K = sA[1]; N = sB[1];
//   }

#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

#ifndef TEST_VECTORS_DISABLED

// ============================================================
// 二进制张量格式
// 文件 = [ndims:u32] [dim0:u32] [dim1:u32] ... [data:f32...]
// ============================================================

class TestVectorConfig {
public:
    TestVectorConfig() : enabled_(false) {}

    // 解析 argc/argv，查找 --test-vectors <dir> 和 --output <dir>
    // 返回 true 表示测试向量模式开启
    bool parse(int argc, char** argv) {
        for (int i = 1; i < argc - 1; ++i) {
            std::string arg(argv[i]);
            if (arg == "--test-vectors" && i + 1 < argc) {
                input_dir_ = argv[++i];
                enabled_ = true;
            } else if (arg == "--output" && i + 1 < argc) {
                output_dir_ = argv[++i];
            }
        }
        return enabled_;
    }

    bool enabled() const { return enabled_; }
    const std::string& input_dir() const { return input_dir_; }
    const std::string& output_dir() const { return output_dir_; }

    // 读取 <input_dir>/name.bin 的形状（不读数据），用于让章节适配自身维度。
    // 成功返回 true，并把形状写入 out_shape；文件不存在/读失败返回 false。
    bool queryShape(const std::string& name, std::vector<uint32_t>& out_shape) const {
        if (!enabled_) return false;
        std::string path = input_dir_ + "/" + name + ".bin";
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "[test_vectors] 无法打开: " << path << "\n";
            return false;
        }
        uint32_t ndims = 0;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
        if (!file || ndims > 16) {
            std::cerr << "[test_vectors] " << name << " 读取维度数失败\n";
            return false;
        }
        out_shape.resize(ndims);
        file.read(reinterpret_cast<char*>(out_shape.data()), ndims * sizeof(uint32_t));
        return !!file;
    }

    // 从 input_dir/name.bin 加载张量
    // shape: 期望的形状，用于校验
    // data: 输出缓冲区，必须已分配足够空间
    bool load(const std::string& name, const std::vector<uint32_t>& expected_shape,
              float* data) const {
        if (!enabled_) return false;

        std::string path = input_dir_ + "/" + name + ".bin";
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "[test_vectors] 无法打开: " << path << "\n";
            return false;
        }

        // 读维度数
        uint32_t ndims = 0;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));

        // 读各维度
        std::vector<uint32_t> shape(ndims);
        file.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(uint32_t));

        // 校验形状
        if (shape.size() != expected_shape.size()) {
            std::cerr << "[test_vectors] " << name << " 维度数不匹配: "
                      << "期望 " << expected_shape.size() << ", 实际 " << shape.size() << "\n";
            return false;
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != expected_shape[i]) {
                std::cerr << "[test_vectors] " << name << " 维度 " << i << " 不匹配: "
                          << "期望 " << expected_shape[i] << ", 实际 " << shape[i] << "\n";
                return false;
            }
        }

        // 计算元素数
        size_t total = 1;
        for (auto d : shape) total *= d;

        // 读数据
        file.read(reinterpret_cast<char*>(data), total * sizeof(float));

        if (!file) {
            std::cerr << "[test_vectors] " << name << " 读取失败\n";
            return false;
        }

        return true;
    }

    // 便捷版：加载到 std::vector<float>，自动分配。
    // 成功返回 true 并填充 out；失败返回 false 且 out 被清空。
    bool loadVector(const std::string& name, std::vector<float>& out) const {
        out.clear();
        std::vector<uint32_t> shape;
        if (!queryShape(name, shape)) return false;
        size_t total = 1;
        for (auto d : shape) total *= d;
        out.resize(total);
        std::vector<uint32_t> empty;  // 不校验形状（直接用读到的形状）
        // 复用 load 的读数据逻辑：构造匹配的形状向量
        if (!load(name, shape, out.data())) { out.clear(); return false; }
        return true;
    }

    // 将张量写入 output_dir/name.bin
    bool save(const std::string& name, const std::vector<uint32_t>& shape,
              const float* data) const {
        if (output_dir_.empty()) return false;

        std::string path = output_dir_ + "/" + name + ".bin";
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "[test_vectors] 无法写入: " << path << "\n";
            return false;
        }

        // 写维度数
        uint32_t ndims = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));

        // 写各维度
        file.write(reinterpret_cast<const char*>(shape.data()),
                   ndims * sizeof(uint32_t));

        // 计算元素数
        size_t total = 1;
        for (auto d : shape) total *= d;

        // 写数据
        file.write(reinterpret_cast<const char*>(data), total * sizeof(float));

        return true;
    }

    // 便捷版：将 std::vector<float> 写入 output_dir/name.bin
    bool saveVector(const std::string& name, const std::vector<float>& data,
                    const std::vector<uint32_t>& shape) const {
        return save(name, shape, data.data());
    }

    // 校验 output_dir 是否存在（非空即认为需要写文件）
    bool hasOutput() const { return !output_dir_.empty(); }

private:
    bool enabled_;
    std::string input_dir_;
    std::string output_dir_;
};

#else  // TEST_VECTORS_DISABLED

// 编译期禁用：所有方法为空操作，编译器可完全优化掉
class TestVectorConfig {
public:
    TestVectorConfig() {}
    bool parse(int, char**) { return false; }
    bool enabled() const { return false; }
    const std::string& input_dir() const { static std::string s; return s; }
    const std::string& output_dir() const { static std::string s; return s; }
    bool queryShape(const std::string&, std::vector<uint32_t>& out) const { out.clear(); return false; }
    bool load(const std::string&, const std::vector<uint32_t>&, float*) const { return false; }
    bool loadVector(const std::string&, std::vector<float>& out) const { out.clear(); return false; }
    bool save(const std::string&, const std::vector<uint32_t>&, const float*) const { return false; }
    bool saveVector(const std::string&, const std::vector<float>&, const std::vector<uint32_t>&) const { return false; }
    bool hasOutput() const { return false; }
};

#endif  // TEST_VECTORS_DISABLED
