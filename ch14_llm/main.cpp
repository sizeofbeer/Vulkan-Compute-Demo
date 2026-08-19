// ch14_llm/main.cpp
// 完整 LLM 推理引擎主程序（阶段二：加载真实权重并自回归生成英文）。
//
// 用法：
//   ch14_llm --prompt "Hello, my name is" --max_new 32
//   ch14_llm --selftest            # 仅做 tokenizer 往返 + 模型加载烟雾测试
//
// 模型与 tokenizer 同目录（TOKENIZER_DIR 指向 qwen3_asr_0.6b），内含 .param/.bin
// 与 vocab.txt/merges.txt。

#include "tokenizer/bpe_tokenizer.h"
#include "llm_model.h"
#include "llm_engine.h"
#include "i18n.h"
#include "test_vectors.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#ifndef TOKENIZER_DIR
#define TOKENIZER_DIR "F:/vulkan/model/qwen3_0.6b_base/"
#endif

// 跨平台：模型目录优先取环境变量 VULKAN_NN_MODEL_DIR，
// 其次才是编译期 TOKENIZER_DIR（CMake 构建时会指向 ${CMAKE_SOURCE_DIR}/../model，跨平台可用；
// 手动 g++ 编译时回退到上面的 Windows 专属路径）。
static std::string resolve_model_dir() {
    if (const char* e = std::getenv("VULKAN_NN_MODEL_DIR")) {
        std::string d = e;
        if (!d.empty() && d.back() != '/') d += '/';
        return d;
    }
    return std::string(TOKENIZER_DIR);
}
static std::string model_dir = resolve_model_dir();

static void run_selftest(BPETokenizer& tok) {
    std::cout << i18n::tr("[ch14] tokenizer 加载完成, vocab=", "[ch14] tokenizer loaded, vocab=") << tok.vocabSize()
              << " bos=" << tok.bosId() << " eos=" << tok.eosId() << "\n";
    std::vector<std::string> samples = {
        "Hello world!",
        "My name is Zigou.",
        "1 + 1 = 2, the answer is forty-two.",
    };
    for (auto& text : samples) {
        auto ids = tok.encode(text);
        std::string back = tok.decode(ids);
        std::cout << i18n::tr("\n[ch14] 原文 : ", "\n[ch14] original : ") << text << "\n";
        std::cout << i18n::tr("[ch14] ids  :", "[ch14] ids  :");
        for (int id : ids) std::cout << " " << id;
        std::cout << "  (" << ids.size() << " tokens)\n";
        std::cout << i18n::tr("[ch14] 解码 : ", "[ch14] decoded : ") << back << "\n";
    }
}

int main(int argc, char** argv) {
    i18n::init(argc, argv);
    std::string prompt = "Hello, my name is";
    int max_new = 32;
    bool selftest = false;
    bool qknorm = false;
    bool debug = false;
    std::string tokens_arg;   // 直接喂原始 token id（绕过 BPE），用于与参考实现逐位对拍
    int eos_override = -1;
    std::string dump_hidden, dump_qroped;  // 导出最后 token 的 hidden / layer0 q_roped 用于对拍

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--max_new" && i + 1 < argc) max_new = std::atoi(argv[++i]);
        else if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--qknorm") qknorm = true;
        else if (a == "--debug") debug = true;
        else if (a == "--selftest") selftest = true;
        else if (a == "--tokens" && i + 1 < argc) tokens_arg = argv[++i];
        else if (a == "--eos" && i + 1 < argc) eos_override = std::atoi(argv[++i]);
        else if (a == "--dump_hidden" && i + 1 < argc) dump_hidden = argv[++i];
        else if (a == "--dump_qroped" && i + 1 < argc) dump_qroped = argv[++i];
        // 消费 --test-vectors / --output（本章无匹配测试向量，仅保证参数不报错）
        else if (a == "--test-vectors" && i + 1 < argc) ++i;
        else if (a == "--output" && i + 1 < argc) ++i;
    }

    try {
        BPETokenizer tok;
        tok.load(model_dir);

        if (selftest) {
            run_selftest(tok);
            std::cout << i18n::tr("\n[ch14] 加载模型（权重较大，请稍候）...\n", "\n[ch14] loading model (weights are large, please wait)...\n");
            QwenModel model;
            if (!model.load(model_dir)) { std::cerr << i18n::tr("模型加载失败\n", "model load failed\n"); return 1; }
            std::cout << i18n::tr("[ch14] 模型加载成功: vocab=", "[ch14] model loaded: vocab=") << model.cfg.vocab
                      << " layers=" << model.cfg.n_layers
                      << " hidden=" << model.cfg.hidden << "\n";
            // 烟雾测试：跑一个 4 token 的前向，确认不崩溃
            QwenEngine eng(model);
            std::vector<int> pt = {9707, 1879, 363, 1019};
            auto lg = eng.forward_last_logits(pt);
            std::cout << i18n::tr("[ch14] 烟雾前向完成，logits 长度=", "[ch14] smoke forward done, logits length=") << lg.size()
                      << i18n::tr(" 有限=", " finite=") << (std::isfinite(lg[0]) ? "是" : "否") << "\n";
            return 0;
        }

        std::cout << i18n::tr("[ch14] 加载模型（权重较大，请稍候）...\n", "[ch14] loading model (weights are large, please wait)...\n");
        auto t0 = std::chrono::steady_clock::now();
        QwenModel model;
        if (!model.load(model_dir)) { std::cerr << i18n::tr("模型加载失败\n", "model load failed\n"); return 1; }
        auto t1 = std::chrono::steady_clock::now();
        std::cout << i18n::tr("[ch14] 模型加载成功（", "[ch14] model loaded (")
                  << std::chrono::duration<double>(t1 - t0).count() << i18n::tr("s），开始推理。\n", "s), starting inference.\n");

        auto ids = tok.encode(prompt);
        if (!tokens_arg.empty()) {
            ids.clear();
            std::istringstream iss(tokens_arg);
            int v;
            while (iss >> v) ids.push_back(v);
        }
        std::cout << "[ch14] prompt tokens (" << ids.size() << "):";
        for (int id : ids) std::cout << " " << id;
        std::cout << "\n";

        // 调试导出：把最后 prompt token 的 hidden / layer0 q_roped 写成二进制，便于与参考实现 dump 对拍
        if (!dump_hidden.empty() || !dump_qroped.empty()) {
            QwenEngine eng(model);
            eng.apply_qk_norm_ = qknorm;
            std::vector<float> hh, qr;
            eng.debug_last_token(ids, dump_hidden.empty() ? nullptr : &hh,
                                 dump_qroped.empty() ? nullptr : &qr);
            auto write_bin = [](const std::string& p, const std::vector<float>& v) {
                std::ofstream f(p, std::ios::binary);
                f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
            };
            if (!dump_hidden.empty()) { write_bin(dump_hidden, hh);
                std::cout << i18n::tr("[ch14] 写出 hidden -> ", "[ch14] wrote hidden -> ") << dump_hidden << " (" << hh.size() << " floats)\n"; }
            if (!dump_qroped.empty()) { write_bin(dump_qroped, qr);
                std::cout << i18n::tr("[ch14] 写出 q_roped -> ", "[ch14] wrote q_roped -> ") << dump_qroped << " (" << qr.size() << " floats)\n"; }
            return 0;
        }

        int eos = (eos_override >= 0) ? eos_override : tok.eosId();

        QwenEngine eng(model);
        eng.apply_qk_norm_ = qknorm;
        auto t2 = std::chrono::steady_clock::now();
        auto out = eng.generate(ids, max_new, eos);
        auto t3 = std::chrono::steady_clock::now();

        if (debug) {
            QwenEngine eng2(model); eng2.apply_qk_norm_ = qknorm;
            auto lg = eng2.forward_last_logits(ids);
            std::vector<std::pair<float,int>> scored;
            for (size_t i = 0; i < lg.size(); i++) scored.push_back({lg[i], (int)i});
            std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                              [](auto&a,auto&b){return a.first>b.first;});
            std::cout << i18n::tr("[ch14] prompt 之后 top-5 候选 token:\n", "[ch14] top-5 candidate tokens after prompt:\n");
            for (int k = 0; k < 5; k++)
                std::cout << "   " << scored[k].second << "  " << tok.decode({scored[k].second})
                          << "  (logit=" << scored[k].first << ")\n";
        }

        std::string generated = tok.decode(out);
        std::cout << i18n::tr("\n[ch14] ===== 生成结果 =====\n", "\n[ch14] ===== generation result =====\n");
        std::cout << generated << "\n";
        std::cout << i18n::tr("[ch14] 生成 token id（", "[ch14] generated token ids (") << out.size() << i18n::tr("）:", "):");
        for (int id : out) std::cout << " " << id;
        std::cout << "\n";
        std::cout << "[ch14] =====================\n";
        std::cout << i18n::tr("[ch14] 生成 token 数=", "[ch14] generated token count=") << (out.size() - ids.size())
                  << i18n::tr(" 耗时=", " time=") << std::chrono::duration<double>(t3 - t2).count() << "s\n";
    } catch (const std::exception& e) {
        std::cerr << i18n::tr("错误: ", "error: ") << e.what() << "\n";
        return 1;
    }
    return 0;
}
