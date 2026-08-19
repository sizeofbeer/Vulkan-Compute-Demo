// llm_model.h
// Qwen3-ASR 0.6B 模型结构与权重容器定义。
//
// 设计说明（与用户的 ncnn C++ decoder 严格对齐）：
//  - 权重来自 ncnn 导出的 .param / .bin（自定义层：Embed / RMSNorm / Gemm / InnerProduct / Swish / BinaryOp / Split / Reshape / Permute）。
//  - 注意力核心（RoPE + GQA + causal softmax + KV cache）不在 param 文件里，
//    而是由引擎在 Q/K/V 投影之后、O 投影之前实施。这正是用户参考实现的做法。
//  - Gemm / InnerProduct 权重布局与 ncnn 一致：weight 为 [N_out, K_in] 行主序，
//    计算 Y = X · Wᵀ（即标准 Linear，无 bias）。
//  - .bin 每个 blob 以 4 字节 tag 开头：0x01306B47 表示 fp16（随后 w 个 half，4 字节对齐），
//    否则这 4 字节就是第一个 fp32 数值（raw fp32）。本模型混合精度：
//    除 lm_head / text_embed 外的投影权重均为 fp16，lm_head 为 fp32。

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <cmath>

// 单个权重张量：row-major，rows = 输出维度(N)，cols = 输入维度(K)。
// 对 RMSNorm gamma：rows = dim，cols = 1。
struct Weight {
    int rows = 0;   // N (out)
    int cols = 0;   // K (in)
    std::vector<float> data;  // 长度 = rows*cols，统一转成 fp32 存放
};

// 单个 decoder 层所需的全部权重
struct DecoderLayerWeights {
    Weight rmsn_gamma;      // [hidden]
    Weight wq;              // [n_q*hidden_per_q=2048, hidden]
    Weight wk;              // [n_kv*hidden_per_head=1024, hidden]
    Weight wv;              // [1024, hidden]
    Weight wo;              // [hidden, 2048]
    Weight q_norm;          // [head_dim=128] Qwen3 QK-Norm（逐头 RMSNorm，RoPE 前；缺省为空）
    Weight k_norm;          // [head_dim=128]
    Weight mlp_rmsn_gamma;  // [hidden]
    Weight wgate;           // [inter=3072, hidden]
    Weight wup;             // [3072, hidden]
    Weight wdown;           // [hidden, 3072]
};

struct QwenConfig {
    int vocab = 151936;
    int hidden = 1024;
    int n_layers = 28;
    int n_q_heads = 16;
    int n_kv_heads = 8;
    int head_dim = 128;
    int inter_size = 3072;
    float rope_theta = 1e6f;
    float rms_norm_eps = 1e-6f;

    int group_size() const { return n_q_heads / n_kv_heads; }      // 2
    float attn_scale() const { return 1.0f / std::sqrt(float(head_dim)); }
};

// 完整模型（CPU 参考实现）。所有权重载入后统一 fp32。
struct QwenModel {
    QwenConfig cfg;
    Weight embed;                       // [vocab, hidden]
    std::vector<DecoderLayerWeights> layers;
    Weight final_norm_gamma;            // [hidden]
    Weight lm_head;                     // [vocab, hidden]  (InnerProduct, 无 bias)

    // 从模型目录加载（解析 85 个 decoder_*.param/bin + text_embed + decoder_norm + lm_head）
    bool load(const std::string& model_dir);
};

// fp16 -> fp32 转换
float half_to_float(uint16_t h);
