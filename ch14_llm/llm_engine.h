// llm_engine.h
// Qwen3-ASR 0.6B CPU 推理引擎（参考实现，逐算子与后续 Vulkan 内核一一对应）。
//
// 数据流（每层）：
//   residual = x
//   x = RMSNorm(x, gamma_pre)
//   q = x · Wqᵀ  -> [1,16,128];  k = x · Wkᵀ -> [1,8,128];  v = x · Wvᵀ -> [1,8,128]
//   q,k 施加 RoPE(theta=1e6, rotate_half) +（Qwen3）QK-Norm（RoPE 前逐头 RMSNorm）
//   attn = GQA(q, k_cache, v_cache, causal) -> [1,16,128] -> [1,2048]
//   attn = attn · Woᵀ -> [1,hidden];  x = residual + attn
//   residual = x
//   x = RMSNorm(x, gamma_mlp)
//   h = SiLU(x · Wgateᵀ) ⊙ (x · Wupᵀ);  x = residual + h · Wdownᵀ
// 末层后：x = RMSNorm(x, gamma_final);  logits = x · lm_headᵀ
//
// Decode 流程（与 F:\Qwen-ASR\Qwen-ASR-ncnn 的 AsrModel::process_one/step/generate 严格对齐）：
//   prefill(prompt) : 逐 token 调用 process_one 把 prompt 全部写入 KV cache，返回最后 prompt token 的 logits
//   decode_step(tok): 调用 process_one 处理「单个新 token」（追加到 KV cache 当前位置），返回其 logits
//   step(tok)       : embed → process_one → final norm → lm_head → argmax（贪心采样下一个 token）
//   generate(...)   : reset → prefill → 循环 step（带重复惩罚 + EOS 停止）
// process_one 永远只处理「一个 token」，位置 = cache_len_，处理完 cache_len_++。KV cache 跨 decode 步复用。

#pragma once

#include "llm_model.h"
#include <vector>

class QwenEngine {
public:
    explicit QwenEngine(const QwenModel& model) : m(model) {
        // 预分配每层 KV cache 容器（内层为空，process_one 时按需 append）
        kv_k_.assign(m.cfg.n_layers, std::vector<float>());
        kv_v_.assign(m.cfg.n_layers, std::vector<float>());
    }

    // 复位 KV cache（开始新一轮 prefill 前调用）
    void reset_cache();

    // —— 显式 prefill / decode（与参考 AsrModel 对齐）——
    // prefill：逐 token 处理整个 prompt，写入 KV cache，返回最后一个 prompt token 的 logits
    //          （即「prompt 之后第一个 token」的预测分布）。
    std::vector<float> prefill(const std::vector<int>& prompt_tokens);

    // decode 一步：处理单个新 token（追加到 KV cache 当前位置），返回其 logits（不直接采样）。
    std::vector<float> decode_step_logits(int token);

    // step：处理单个 token（first==true 时先复位 cache），返回贪心采样的下一个 token。
    //       与参考 AsrModel::step 语义一致（embed → process_one → final norm → lm_head → argmax）。
    int step(int token, bool first = false);

    // 贪心自回归生成（prefill + decode 循环 + 重复惩罚 + EOS 停止）。
    // 返回 prompt + 生成 token 的完整序列。
    std::vector<int> generate(const std::vector<int>& prompt_tokens,
                              int max_new_tokens,
                              int eos_id = -1);

    // 兼容便捷入口：一次性前向（= reset + 处理所有 token + 返回最后 token 的 logits），用于 --debug 查看 top-5。
    std::vector<float> forward_last_logits(const std::vector<int>& tokens);

    // 工具：直接用 embed + 单层做 Q/K 投影并施加 RoPE，用于对齐验证。
    void project_and_rope(const std::vector<float>& hidden_seq,  // [seq,hidden]
                          int seq, std::vector<float>& q,        // out [seq,16,128]
                          std::vector<float>& k);                // out [seq,8,128]

    // 调试：prefill 后返回最后一个 prompt token 的最终 hidden（过完 28 层、未 final norm）；
    //       可选返回该 token 在 layer0 旋转后的 q（[n_q*head_dim]），用于与参考实现 dump 对拍。
    void debug_last_token(const std::vector<int>& prompt,
                          std::vector<float>* hidden_out,
                          std::vector<float>* qroped_out = nullptr);

    bool apply_qk_norm_ = false;  // Qwen3 在 RoPE 前对 Q/K 做逐头 RMSNorm；param 无该层时默认关

private:
    const QwenModel& m;

    // KV cache：每层缓存所有已处理位置的 k/v。维度 [n_kv, head_dim]，按位置连续追加。
    std::vector<std::vector<float>> kv_k_; // [layer][pos*n_kv*hdim ...]
    std::vector<std::vector<float>> kv_v_;
    int cache_len_ = 0;  // 已处理（已写入 cache）的 token 数 = 下一个 token 的绝对位置

    // 取单个 token 的 embedding 到 out[H]
    void embed_token(int tok, float* out);
    // 处理单个 token（位置 = cache_len_），更新 KV cache，跑完所有层，返回最终 hidden（未做 final norm）。
    std::vector<float> process_one(const float* x_in);
    // final RMSNorm + lm_head：hidden[H] -> logits[V]
    std::vector<float> lm_head(const float* x);
};

// 基础算子（行主序）
void matmul(const float* X, const Weight& W, float* Y, int M, int K); // Y[M,N] = X[M,K]·W[N,K]ᵀ
void rmsnorm(const float* x, const float* gamma, float* out, int n, float eps);
void rope_apply(float* vec, const float* cos128, const float* sin128, int head_dim); // 单头 rotate_half
void build_rope_cache(std::vector<float>& cos, std::vector<float>& sin,
                      int seqlen, int head_dim, float theta, int position_id = 0); // [seqlen,128]
float silu(float x);
