#pragma once
// model_loader.h —— 标准 ncnn 模型加载器（ch13 路径 A 用）
//
// 这个加载器对接你导出的 Qwen3-ASR ncnn 子图。经实测，这些 .bin 是
// 「无 blob 头的纯 fp32 权重流」：每个权重 blob 直接是 float32 数据连续排列，
// 顺序与 param 中权重 blob 出现顺序一致，每个 blob 的大小由 param 里层的
// weight_data_size（元素数）决定。所以 loader 不需要解析 blob 维度头，
// 只需按 param 声明的元素数顺序切分即可。
//
// 本文件 header-only，被 main.cpp 直接 include，无需改动 CMake。
// 目前只实现「解析 param + 读权重」这一公共底座；推理部分（路径 A 是单 InnerProduct）
// 在 main.cpp 里完成。后续路径 B 要支持的 Gemm/RMSNorm/Swish 等，可在此追加解析。

#include "vk_base.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <stdexcept>

// 一层 ncnn 层：类型/名字/输入输出 blob，以及整型与浮点参数（0=.. 1=.. 2=.. 形式）
struct NcnnLayer {
    std::string type, name;
    std::vector<std::string> inBlobs, outBlobs;
    std::map<int, long long> iParams;   // 整型参数，如 InnerProduct 的 0=num_output 2=weight_data_size
    std::map<int, double>    fParams;   // 浮点参数，如 RMSNorm 的 1=1e-6
};

struct NcnnModel {
    std::vector<NcnnLayer> layers;

    // 权重需求：按层顺序，InnerProduct 贡献 weight（可选 bias）
    struct WeightReq { std::string layer; bool isBias; long long elems; };
    std::vector<WeightReq> weights;

    // 解析标准 ncnn param（首行 magic 7767517，次行 layer_count blob_count，随后逐层定义）
    static NcnnModel load(const std::string& paramPath) {
        std::ifstream f(paramPath);
        if (!f) throw std::runtime_error("无法打开 ncnn param: " + paramPath);
        NcnnModel m;
        std::string line;
        std::getline(f, line);                 // 第 1 行：magic 7767517（此处忽略校验）
        std::getline(f, line);                 // 第 2 行：layer_count blob_count（忽略）
        while (std::getline(f, line)) {
            size_t a = line.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) continue;       // 跳过空行
            std::istringstream iss(line.substr(a));
            NcnnLayer L;
            iss >> L.type >> L.name;
            int inC = 0, outC = 0;
            iss >> inC >> outC;
            std::string s;
            for (int i = 0; i < inC;  ++i) { iss >> s; L.inBlobs.push_back(s); }
            for (int i = 0; i < outC; ++i) { iss >> s; L.outBlobs.push_back(s); }
            // 剩余 token 形如 0=151936 / 1=0 / 2=1.000000e-6
            std::string tok;
            while (iss >> tok) {
                auto eq = tok.find('=');
                if (eq == std::string::npos) continue;
                int key = std::stoi(tok.substr(0, eq));
                std::string val = tok.substr(eq + 1);
                // 含 '.' 或 'e'/'E' 视为浮点，否则整数
                if (val.find_first_of(".eE") != std::string::npos)
                    L.fParams[key] = std::stod(val);
                else
                    L.iParams[key] = std::stoll(val);
            }
            // 收集 InnerProduct 的权重需求
            if (L.type == "InnerProduct") {
                long long numOut   = L.iParams[0];
                long long biasTerm = L.iParams[1];
                long long wElems   = L.iParams[2];   // 本导出：weight 元素数
                m.weights.push_back({L.name, false, wElems});
                if (biasTerm == 1) m.weights.push_back({L.name, true, numOut});
            }
            m.layers.push_back(L);
        }
        return m;
    }
};

// 读取 ncnn bin（无 blob 头的 fp32 权重流），按 weights 顺序填充。
// 返回 key = "层名.weight" / "层名.bias" -> float 向量。
inline std::map<std::string, std::vector<float>> readNcnnBin(
        const std::string& binPath, const NcnnModel& m) {
    std::ifstream f(binPath, std::ios::binary);
    if (!f) throw std::runtime_error("无法打开 ncnn bin: " + binPath);
    std::map<std::string, std::vector<float>> out;
    for (const auto& w : m.weights) {
        std::vector<float> buf(static_cast<size_t>(w.elems));
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(w.elems * sizeof(float)));
        if (!f)
            throw std::runtime_error("ncnn bin 读取越界，权重布局假设可能不对: " + w.layer);
        out[w.layer + (w.isBias ? ".bias" : ".weight")] = std::move(buf);
    }
    return out;
}
