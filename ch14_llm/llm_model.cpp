// llm_model.cpp
// ncnn param/bin 加载器：解析自定义层，按 blob tag 判定 fp16/fp32，统一转 fp32。

#include "llm_model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t r;
    if (exp == 0) {
        if (mant == 0) {
            r = sign << 31;
        } else {
            // 非规约数
            uint32_t e = 0;
            while (!(mant & 0x400)) { mant <<= 1; e++; }
            mant &= 0x3ff; mant <<= 1;
            exp = 127 - 15 - e;
            r = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        r = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        exp = exp - 15 + 127;
        r = (sign << 31) | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &r, sizeof(r));
    return f;
}

namespace {

// 简易 ncnn .bin 读取器：顺序读取 blob，每个 blob 以 4 字节 tag 开头。
struct BinReader {
    std::ifstream f;
    bool open(const std::string& path) {
        f.open(path, std::ios::binary);
        return f.good();
    }
    // 读取 count 个数值（内部处理 fp16/fp32 tag），返回 fp32 向量（长度 count）
    std::vector<float> read_weight(int count) {
        uint32_t tag = 0;
        f.read(reinterpret_cast<char*>(&tag), 4);
        std::vector<float> out(count);
        if (tag == 0x01306B47u) {
            // fp16：随后 count 个 half，4 字节对齐
            size_t padded = (size_t(count) * 2 + 3) & ~size_t(3);
            std::vector<uint16_t> buf(padded / 2);
            f.read(reinterpret_cast<char*>(buf.data()), padded);
            for (int i = 0; i < count; i++) out[i] = half_to_float(buf[i]);
        } else {
            // raw fp32：刚才读的 4 字节就是第一个 float
            std::memcpy(&out[0], &tag, 4);
            if (count > 1) {
                f.read(reinterpret_cast<char*>(&out[1]), (count - 1) * sizeof(float));
            }
        }
        return out;
    }
};

// 解析单层 param 行里的 key=value 参数
void parse_params(const std::vector<std::string>& toks, size_t start,
                  std::map<int, long long>& pm) {
    for (size_t i = start; i < toks.size(); i++) {
        const std::string& t = toks[i];
        size_t eq = t.find('=');
        if (eq == std::string::npos) continue;
        int key = std::stoi(t.substr(0, eq));
        long long val = std::stoll(t.substr(eq + 1));
        pm[key] = val;
    }
}

// 载入单个 .param/.bin 对，返回 name -> Weight
bool load_param_file(const std::string& dir, const std::string& base,
                     std::map<std::string, Weight>& out) {
    std::string ppath = dir + "/" + base + ".ncnn.param";
    std::string bpath = dir + "/" + base + ".ncnn.bin";
    std::ifstream pf(ppath);
    if (!pf.good()) { std::fprintf(stderr, "无法打开 param: %s\n", ppath.c_str()); return false; }

    std::string line;
    std::getline(pf, line); // magic 7767517
    std::getline(pf, line); // layer_count blob_count
    int layer_count = 0, blob_count = 0;
    {
        std::istringstream ss(line);
        ss >> layer_count >> blob_count;
    }

    BinReader br;
    if (!br.open(bpath)) { std::fprintf(stderr, "无法打开 bin: %s\n", bpath.c_str()); return false; }

    for (int li = 0; li < layer_count; li++) {
        std::getline(pf, line);
        std::istringstream ss(line);
        std::vector<std::string> toks;
        std::string tk;
        while (ss >> tk) toks.push_back(tk);
        if (toks.size() < 4) continue;
        const std::string& type = toks[0];
        const std::string& name = toks[1];
        int in_count = std::stoi(toks[2]);
        int out_count = std::stoi(toks[3]);
        size_t blob_end = 4 + in_count + out_count; // blob 名结束位置
        std::map<int, long long> pm;
        parse_params(toks, blob_end, pm);

        Weight w;
        bool has_weight = false;
        if (type == "Embed") {
            int dim = (int)pm[0];
            int vocab = (int)pm[1];
            long long total = pm.count(3) ? pm[3] : (long long)dim * vocab;
            w.rows = vocab; w.cols = dim;
            w.data = br.read_weight((int)total);
            has_weight = true;
        } else if (type == "RMSNorm") {
            int dim = (int)pm[0];
            w.rows = dim; w.cols = 1;
            w.data = br.read_weight(dim);
            has_weight = true;
        } else if (type == "Gemm") {
            int N = (int)pm[8];
            int K = (int)pm[9];
            w.rows = N; w.cols = K;
            w.data = br.read_weight(N * K);
            has_weight = true;
        } else if (type == "InnerProduct") {
            int N = (int)pm[0];
            long long total = pm[2]; // weight_data_size
            w.rows = N; w.cols = (int)(total / N);
            w.data = br.read_weight((int)total);
            has_weight = true;
        }
        // Swish / BinaryOp / Split / Reshape / Permute 无权重，跳过
        if (has_weight) out[name] = std::move(w);
    }
    return true;
}

// 读取纯 fp32 二进制文件（无 ncnn tag），用于 asr 导出里分离的 q_norm/k_norm。
static bool read_raw_fp32(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), sz);
    return !out.empty();
}

} // namespace

bool QwenModel::load(const std::string& model_dir) {
    // text_embed
    {
        std::map<std::string, Weight> m;
        if (!load_param_file(model_dir, "text_embed", m)) return false;
        if (m.find("embed_0") == m.end()) { std::fprintf(stderr, "缺少 embed_0\n"); return false; }
        embed = std::move(m["embed_0"]);
    }
    cfg.vocab = embed.rows;
    cfg.hidden = embed.cols;

    layers.resize(cfg.n_layers);
    char buf[64];
    for (int i = 0; i < cfg.n_layers; i++) {
        std::snprintf(buf, sizeof(buf), "decoder_%02d_pre", i);
        std::map<std::string, Weight> pre;
        if (!load_param_file(model_dir, buf, pre)) return false;
        std::snprintf(buf, sizeof(buf), "decoder_%02d_o", i);
        std::map<std::string, Weight> o;
        if (!load_param_file(model_dir, buf, o)) return false;
        std::snprintf(buf, sizeof(buf), "decoder_%02d_mlp", i);
        std::map<std::string, Weight> mlp;
        if (!load_param_file(model_dir, buf, mlp)) return false;

        auto& L = layers[i];
        L.rmsn_gamma = std::move(pre["rmsn_3"]);
        L.wq = std::move(pre["gemm_0"]);
        L.wk = std::move(pre["gemm_1"]);
        L.wv = std::move(pre["gemm_2"]);
        L.wo = std::move(o["gemm_0"]);
        // Qwen3 QK-Norm：逐头 RMSNorm（RoPE 之前）。asr 导出缺省，按可选处理。
        if (pre.count("q_norm")) L.q_norm = std::move(pre["q_norm"]);
        if (pre.count("k_norm")) L.k_norm = std::move(pre["k_norm"]);
        L.mlp_rmsn_gamma = std::move(mlp["rmsn_3"]);
        L.wgate = std::move(mlp["gemm_0"]);
        L.wup = std::move(mlp["gemm_1"]);
        L.wdown = std::move(mlp["gemm_2"]);
    }

    // 兼容 asr 导出：q_norm/k_norm 可能以分离文件 decoder_%02d_qnorm.bin / knorm.bin 提供
    // （参考实现 AsrModel::load 即从这些文件读取并应用）。若文件存在则用其覆盖（base 模型已内嵌在
    // pre.bin 且无独立文件，不受影响）。
    char b2[96];
    for (int i = 0; i < cfg.n_layers; i++) {
        std::snprintf(b2, sizeof(b2), "%s/decoder_%02d_qnorm.bin", model_dir.c_str(), i);
        std::vector<float> qn;
        if (read_raw_fp32(b2, qn) && qn.size() == (size_t)cfg.head_dim) {
            layers[i].q_norm.rows = cfg.head_dim; layers[i].q_norm.cols = 1;
            layers[i].q_norm.data = std::move(qn);
        }
        std::snprintf(b2, sizeof(b2), "%s/decoder_%02d_knorm.bin", model_dir.c_str(), i);
        std::vector<float> kn;
        if (read_raw_fp32(b2, kn) && kn.size() == (size_t)cfg.head_dim) {
            layers[i].k_norm.rows = cfg.head_dim; layers[i].k_norm.cols = 1;
            layers[i].k_norm.data = std::move(kn);
        }
    }

    {
        std::map<std::string, Weight> fn;
        if (!load_param_file(model_dir, "decoder_norm", fn)) return false;
        final_norm_gamma = std::move(fn["rmsn_0"]);
    }
    {
        // Qwen3 系列 tie_word_embeddings=true：lm_head 与 embed 共享权重。
        // 参考实现 AsrModel 也直接用 embed_w_（来自 embed.bin）做 lm_head，从不读取独立的
        // lm_head.ncnn.bin——该文件在某些导出里首 4 字节恰好等于 ncnn 的 fp16 魔数 0x01306B47，
        // 会被误判为 fp16 而整体读坏（RMS 由 0.027 暴涨到 0.8，logits 量级错乱）。
        // 因此这里直接 tied 到 embed，既与参考对齐又规避该歧义。
        lm_head = embed;
    }
    return true;
}
