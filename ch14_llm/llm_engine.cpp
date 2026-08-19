// llm_engine.cpp
#include "llm_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {

inline float rms_scalar(const float* x, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    return 1.f / std::sqrt(ss / n + eps);
}

inline float rms_of(const float* x, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += double(x[i]) * x[i];
    return float(std::sqrt(ss / n));
}

} // namespace

static bool g_stats() {
    static int v = -1;
    if (v < 0) v = (std::getenv("CH14_STATS") != nullptr) ? 1 : 0;
    return v != 0;
}

void matmul(const float* X, const Weight& W, float* Y, int M, int K) {
    const int N = W.rows;
    for (int m = 0; m < M; m++) {
        const float* xr = X + (size_t)m * K;
        for (int n = 0; n < N; n++) {
            const float* wr = W.data.data() + (size_t)n * K;
            float s = 0.f;
            for (int k = 0; k < K; k++) s += xr[k] * wr[k];
            Y[(size_t)m * N + n] = s;
        }
    }
}

void rmsnorm(const float* x, const float* gamma, float* out, int n, float eps) {
    float inv = rms_scalar(x, n, eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * gamma[i];
}

// 逐头 RMSNorm（Q/K norm）。gamma 为 nullptr 时仅做归一化；否则乘以 learned gamma（Qwen3 QK-Norm）。
static void rmsnorm_heads_gamma(float* x, int seq, int n_heads, int hd,
                                const float* gamma, float eps) {
    for (int i = 0; i < seq; i++) {
        float* base = x + (size_t)i * n_heads * hd;
        for (int h = 0; h < n_heads; h++) {
            float* p = base + h * hd;
            float inv = rms_scalar(p, hd, eps);
            for (int d = 0; d < hd; d++)
                p[d] = p[d] * inv * (gamma ? gamma[d] : 1.f);
        }
    }
}

void rope_apply(float* vec, const float* cos128, const float* sin128, int head_dim) {
    const int half = head_dim / 2;
    for (int j = 0; j < half; j++) {
        float x0 = vec[j];
        float x1 = vec[j + half];
        float c = cos128[j];
        float s = sin128[j];
        vec[j] = x0 * c - x1 * s;
        vec[j + half] = x0 * s + x1 * c;
    }
}

void build_rope_cache(std::vector<float>& cos, std::vector<float>& sin,
                      int seqlen, int head_dim, float theta, int position_id) {
    const int half = head_dim / 2;
    cos.assign((size_t)seqlen * head_dim, 0.f);
    sin.assign((size_t)seqlen * head_dim, 0.f);
    std::vector<float> inv(half);
    for (int j = 0; j < half; j++) inv[j] = std::pow(theta, -2.f * j / head_dim);
    for (int i = 0; i < seqlen; i++) {
        int p = position_id + i;
        float* cr = cos.data() + (size_t)i * head_dim;
        float* sr = sin.data() + (size_t)i * head_dim;
        for (int j = 0; j < half; j++) {
            float t = p * inv[j];
            float c = std::cos(t), s = std::sin(t);
            cr[j] = c; cr[j + half] = c;
            sr[j] = s; sr[j + half] = s;
        }
    }
}

float silu(float x) {
    return x / (1.f + std::exp(-x));
}

// ============================================================
// 核心：单 token 前向（与参考 AsrModel::process_one 对齐）
// ============================================================
void QwenEngine::reset_cache() {
    for (int L = 0; L < m.cfg.n_layers; ++L) { kv_k_[L].clear(); kv_v_[L].clear(); }
    cache_len_ = 0;
}

void QwenEngine::embed_token(int tok, float* out) {
    const int H = m.cfg.hidden;
    if (tok < 0 || tok >= m.embed.rows) tok = 0;
    std::memcpy(out, m.embed.data.data() + (size_t)tok * H, H * sizeof(float));
}

std::vector<float> QwenEngine::process_one(const float* x_in) {
    const QwenConfig& c = m.cfg;
    const int H = c.hidden;
    const int n_q = c.n_q_heads, n_kv = c.n_kv_heads, hd = c.head_dim, grp = c.group_size();
    const float scale = c.attn_scale();

    std::vector<float> x(x_in, x_in + H);

    // 当前 token 的绝对位置（= 处理前已写入 cache 的个数）
    const int pos = cache_len_;

    std::vector<float> cosc, sinc;
    build_rope_cache(cosc, sinc, 1, hd, c.rope_theta, pos);

    std::vector<float> tmp(H);
    std::vector<float> qb((size_t)n_q * hd);
    std::vector<float> kb((size_t)n_kv * hd);
    std::vector<float> vb((size_t)n_kv * hd);

    for (int L = 0; L < c.n_layers; L++) {
        const DecoderLayerWeights& w = m.layers[L];

        // RMSNorm(pre) + Q/K/V 投影
        rmsnorm(x.data(), w.rmsn_gamma.data.data(), tmp.data(), H, c.rms_norm_eps);
        matmul(tmp.data(), w.wq, qb.data(), 1, H);
        matmul(tmp.data(), w.wk, kb.data(), 1, H);
        matmul(tmp.data(), w.wv, vb.data(), 1, H);

        // Qwen3 QK-Norm：投影后、RoPE 前对每个头做 RMSNorm（可学习 gamma）。
        // 模型带 q_norm/k_norm 权重时自动启用；--qknorm 可强制对无权重模型做纯归一化。
        if (!w.q_norm.data.empty() || apply_qk_norm_) {
            const float* qg = w.q_norm.data.empty() ? nullptr : w.q_norm.data.data();
            const float* kg = w.k_norm.data.empty() ? nullptr : w.k_norm.data.data();
            rmsnorm_heads_gamma(qb.data(), 1, n_q, hd, qg, c.rms_norm_eps);
            rmsnorm_heads_gamma(kb.data(), 1, n_kv, hd, kg, c.rms_norm_eps);
        }

        // RoPE（rotate_half，当前 token 唯一位置）
        const float* cr = cosc.data();
        const float* sr = sinc.data();
        for (int h = 0; h < n_q; h++) rope_apply(qb.data() + h * hd, cr, sr, hd);
        for (int h = 0; h < n_kv; h++) rope_apply(kb.data() + h * hd, cr, sr, hd);

        // 追加 KV cache（位置 pos）
        size_t off = (size_t)pos * n_kv * hd;
        size_t need = off + (size_t)n_kv * hd;
        if (kv_k_[L].size() < need) { kv_k_[L].resize(need); kv_v_[L].resize(need); }
        std::memcpy(kv_k_[L].data() + off, kb.data(), (size_t)n_kv * hd * sizeof(float));
        std::memcpy(kv_v_[L].data() + off, vb.data(), (size_t)n_kv * hd * sizeof(float));

        // GQA 因果注意力（query 当前 token，key/value 覆盖 0..pos）
        std::vector<float> attn((size_t)n_q * hd);
        std::vector<float> scores(pos + 1);
        for (int h = 0; h < n_q; h++) {
            int g = h / grp;
            const float* qh = qb.data() + h * hd;
            const float* Kc = kv_k_[L].data() + (size_t)g * hd;  // 基址：kv-head g
            const float* Vc = kv_v_[L].data() + (size_t)g * hd;
            for (int j = 0; j <= pos; j++) {
                const float* kj = Kc + (size_t)j * n_kv * hd;
                float dot = 0.f;
                for (int d = 0; d < hd; d++) dot += qh[d] * kj[d];
                scores[j] = dot * scale;
            }
            float mx = scores[0];
            for (int j = 1; j <= pos; j++) if (scores[j] > mx) mx = scores[j];
            float sum = 0.f;
            for (int j = 0; j <= pos; j++) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
            float inv = 1.f / sum;
            float* ah = attn.data() + h * hd;
            for (int d = 0; d < hd; d++) ah[d] = 0.f;
            for (int j = 0; j <= pos; j++) {
                const float* vj = Vc + (size_t)j * n_kv * hd;
                float wj = scores[j] * inv;
                for (int d = 0; d < hd; d++) ah[d] += wj * vj[d];
            }
        }

        // O 投影 + 残差
        std::vector<float> ao(H);
        matmul(attn.data(), w.wo, ao.data(), 1, n_q * hd);
        for (int i = 0; i < H; i++) x[i] += ao[i];

        // MLP (SwiGLU) + 残差
        rmsnorm(x.data(), w.mlp_rmsn_gamma.data.data(), tmp.data(), H, c.rms_norm_eps);
        std::vector<float> gate(c.inter_size), up(c.inter_size);
        matmul(tmp.data(), w.wgate, gate.data(), 1, H);
        matmul(tmp.data(), w.wup, up.data(), 1, H);
        for (size_t i = 0; i < gate.size(); i++) gate[i] = silu(gate[i]) * up[i];
        std::vector<float> down(H);
        matmul(gate.data(), w.wdown, down.data(), 1, c.inter_size);
        for (int i = 0; i < H; i++) x[i] += down[i];

        if (g_stats()) {
            if (L == 0)
                std::cout << "[stats] L0 qb RMS=" << rms_of(qb.data(), (int)qb.size())
                          << " attn RMS=" << rms_of(attn.data(), (int)attn.size())
                          << " down RMS=" << rms_of(down.data(), (int)down.size()) << "\n";
            if (L == 0 || L == c.n_layers - 1)
                std::cout << "[stats] layer " << L << " hidden RMS=" << rms_of(x.data(), H) << "\n";
        }
    }

    cache_len_++;  // 本 token 已写入 cache
    return x;
}

std::vector<float> QwenEngine::lm_head(const float* x) {
    const QwenConfig& c = m.cfg;
    const int H = c.hidden;
    std::vector<float> xlast(H);
    rmsnorm(x, m.final_norm_gamma.data.data(), xlast.data(), H, c.rms_norm_eps);
    std::vector<float> logits(m.lm_head.rows);
    matmul(xlast.data(), m.lm_head, logits.data(), 1, H);
    if (g_stats()) {
        float mx = 0.f;
        for (float v : logits) if (v > mx) mx = v;
        std::cout << "[stats] lm_head.W RMS=" << rms_of(m.lm_head.data.data(), (int)m.lm_head.data.size())
                  << "  logits RMS=" << rms_of(logits.data(), (int)logits.size())
                  << "  logits max=" << mx << "\n";
    }
    return logits;
}

// ============================================================
// 显式 prefill / decode（与参考 AsrModel 对齐）
// ============================================================
std::vector<float> QwenEngine::prefill(const std::vector<int>& prompt_tokens) {
    reset_cache();
    std::vector<float> last_logits;
    const int n = (int)prompt_tokens.size();
    for (int i = 0; i < n; i++) {
        float x[m.cfg.hidden];
        embed_token(prompt_tokens[i], x);
        std::vector<float> h = process_one(x);
        if (i == n - 1) last_logits = lm_head(h.data());  // 最后 prompt token 的预测分布
    }
    return last_logits;
}

std::vector<float> QwenEngine::decode_step_logits(int token) {
    float x[m.cfg.hidden];
    embed_token(token, x);
    std::vector<float> h = process_one(x);
    return lm_head(h.data());
}

int QwenEngine::step(int token, bool first) {
    if (first) reset_cache();
    float x[m.cfg.hidden];
    embed_token(token, x);
    std::vector<float> h = process_one(x);
    std::vector<float> lg = lm_head(h.data());
    return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
}

std::vector<int> QwenEngine::generate(const std::vector<int>& prompt_tokens,
                                      int max_new_tokens, int eos_id) {
    std::vector<int> tokens = prompt_tokens;
    std::vector<float> logits = prefill(prompt_tokens);
    float rep_penalty = 1.1f;
    for (int step = 0; step < max_new_tokens; step++) {
        // 重复惩罚（与参考 decoder 一致）：已生成的 token 概率压低
        for (int t : tokens) {
            if (t >= 0 && t < (int)logits.size()) {
                if (logits[t] < 0.f) logits[t] *= rep_penalty;
                else logits[t] /= rep_penalty;
            }
        }
        int next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (eos_id >= 0 && next == eos_id) break;
        if (next < 0 || next >= m.cfg.vocab) break;
        tokens.push_back(next);
        logits = decode_step_logits(next);
    }
    return tokens;
}

// 兼容便捷入口：= reset + 处理所有 token + 返回最后 token 的 logits。
std::vector<float> QwenEngine::forward_last_logits(const std::vector<int>& tokens) {
    reset_cache();
    const int n = (int)tokens.size();
    std::vector<float> h;
    for (int i = 0; i < n; i++) {
        float x[m.cfg.hidden];
        embed_token(tokens[i], x);
        h = process_one(x);
    }
    return lm_head(h.data());
}

void QwenEngine::project_and_rope(const std::vector<float>& hidden_seq,
                                  int seq, std::vector<float>& q, std::vector<float>& k) {
    const QwenConfig& c = m.cfg;
    const int H = c.hidden, n_q = c.n_q_heads, n_kv = c.n_kv_heads, hd = c.head_dim;
    std::vector<float> tmp((size_t)seq * H);
    for (int i = 0; i < seq; i++)
        rmsnorm(hidden_seq.data() + (size_t)i * H, m.layers[0].rmsn_gamma.data.data(),
                tmp.data() + (size_t)i * H, H, c.rms_norm_eps);
    q.assign((size_t)seq * n_q * hd, 0.f);
    k.assign((size_t)seq * n_kv * hd, 0.f);
    matmul(tmp.data(), m.layers[0].wq, q.data(), seq, H);
    matmul(tmp.data(), m.layers[0].wk, k.data(), seq, H);
    if (!m.layers[0].q_norm.data.empty() || apply_qk_norm_) {
        const float* qg = m.layers[0].q_norm.data.empty() ? nullptr : m.layers[0].q_norm.data.data();
        const float* kg = m.layers[0].k_norm.data.empty() ? nullptr : m.layers[0].k_norm.data.data();
        rmsnorm_heads_gamma(q.data(), seq, n_q, hd, qg, c.rms_norm_eps);
        rmsnorm_heads_gamma(k.data(), seq, n_kv, hd, kg, c.rms_norm_eps);
    }
    std::vector<float> cosc, sinc;
    build_rope_cache(cosc, sinc, seq, hd, c.rope_theta, 0);
    for (int i = 0; i < seq; i++) {
        const float* cr = cosc.data() + (size_t)i * hd;
        const float* sr = sinc.data() + (size_t)i * hd;
        float* qh = q.data() + (size_t)i * n_q * hd;
        for (int h = 0; h < n_q; h++) rope_apply(qh + h * hd, cr, sr, hd);
        float* kh = k.data() + (size_t)i * n_kv * hd;
        for (int h = 0; h < n_kv; h++) rope_apply(kh + h * hd, cr, sr, hd);
    }
}

void QwenEngine::debug_last_token(const std::vector<int>& prompt,
                                  std::vector<float>* hidden_out,
                                  std::vector<float>* qroped_out) {
    reset_cache();
    const int n = (int)prompt.size();
    const int H = m.cfg.hidden;
    std::vector<float> h;
    std::vector<float> xlast((size_t)H);
    for (int i = 0; i < n; i++) {
        float x[H];
        embed_token(prompt[i], x);
        if (i == n - 1) std::memcpy(xlast.data(), x, (size_t)H * sizeof(float));
        h = process_one(x);
    }
    if (hidden_out) *hidden_out = h;

    if (qroped_out) {
        // 复现参考 debug_dump_q_roped：最后一个 token 在 layer0 旋转后的 q。
        // 输入是「最后 token 的 embedding」(xlast)，位置 = n-1，与参考 build_rope(npos_) 一致。
        const int n_q = m.cfg.n_q_heads, hd = m.cfg.head_dim;
        const DecoderLayerWeights& w = m.layers[0];
        std::vector<float> tmp(H);
        rmsnorm(xlast.data(), w.rmsn_gamma.data.data(), tmp.data(), H, m.cfg.rms_norm_eps);
        std::vector<float> q((size_t)n_q * hd);
        matmul(tmp.data(), w.wq, q.data(), 1, H);
        const float* qg = w.q_norm.data.empty() ? nullptr : w.q_norm.data.data();
        rmsnorm_heads_gamma(q.data(), 1, n_q, hd, qg, m.cfg.rms_norm_eps);
        std::vector<float> cosc, sinc;
        build_rope_cache(cosc, sinc, 1, hd, m.cfg.rope_theta, n - 1);
        for (int hh = 0; hh < n_q; hh++) rope_apply(q.data() + hh * hd, cosc.data(), sinc.data(), hd);
        *qroped_out = q;
    }
}
