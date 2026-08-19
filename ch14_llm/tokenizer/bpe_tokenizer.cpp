// bpe_tokenizer.cpp —— BPETokenizer 实现
#include "bpe_tokenizer.h"
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <climits>

static void trimCr(std::string& line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
}

void BPETokenizer::load(const std::string& dir) {
    // ---- vocab.txt：行号 = id ----
    {
        std::ifstream f(dir + "/vocab.txt");
        if (!f) throw std::runtime_error("无法打开 vocab.txt: " + dir);
        std::string line;
        int id = 0;
        while (std::getline(f, line)) {
            trimCr(line);
            if (id2tok_.size() <= (size_t)id) id2tok_.resize(id + 1);
            id2tok_[id] = line;
            if (tok2id_.find(line) == tok2id_.end()) tok2id_[line] = id; // 首次出现优先
            ++id;
        }
    }

    // ---- merges.txt：首行 #version，其后 "a b" -> 优先级 ----
    {
        std::ifstream f(dir + "/merges.txt");
        std::string line;
        int rank = 0;
        bool first = true;
        while (std::getline(f, line)) {
            trimCr(line);
            if (first) {
                first = false;
                if (line.rfind("#version", 0) == 0) continue; // 跳过版本行
            }
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string a = line.substr(0, sp);
            std::string b = line.substr(sp + 1);
            mergeRank_[a + " " + b] = rank++;
        }
    }

    // ---- 推断辅助信息 ----
    hasByteTokens_ = (tok2id_.find("<0x00>") != tok2id_.end());
    auto findTok = [&](const std::string& t) {
        auto it = tok2id_.find(t);
        return it == tok2id_.end() ? -1 : it->second;
    };
    unkId_ = findTok("<unk>");
    if (unkId_ < 0) unkId_ = 0;
    // ASR 模型 vocab 里通常没有标准 chat 特殊 token，保持 -1，由外部 setSpecialIds 覆盖
    eosId_ = findTok("<|endoftext|>");
    bosId_ = findTok("<|begin_of_sentence|>");
    if (bosId_ < 0) bosId_ = findTok("<|im_start|>");
}

std::vector<std::string> BPETokenizer::utf8Chars(const std::string& s) const {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        int n = 1;
        if      (c < 0x80)       n = 1;
        else if ((c >> 5) == 0x6) n = 2;
        else if ((c >> 4) == 0xe) n = 3;
        else if ((c >> 3) == 0x1e) n = 4;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

int BPETokenizer::symbolToId(const std::string& sym) const {
    auto it = tok2id_.find(sym);
    if (it != tok2id_.end()) return it->second;
    if (hasByteTokens_) {
        // byte-fallback：把一个码点按 UTF-8 字节拆成多个 <0xXX> token
        for (unsigned char b : sym) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto it2 = tok2id_.find(buf);
            if (it2 != tok2id_.end()) return it2->second; // 仅首字节命中（粗粒度），其余走 unk
        }
    }
    return unkId_;
}

std::vector<std::string> BPETokenizer::bpeMerge(const std::vector<std::string>& inSym) const {
    std::vector<std::string> symbols = inSym;
    while (symbols.size() >= 2) {
        int bestRank = INT_MAX;
        size_t bestIdx = 0;
        bool found = false;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = mergeRank_.find(symbols[i] + " " + symbols[i + 1]);
            if (it != mergeRank_.end() && it->second < bestRank) {
                bestRank = it->second;
                bestIdx = i;
                found = true;
            }
        }
        if (!found) break;
        symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
        symbols.erase(symbols.begin() + bestIdx + 1);
    }
    return symbols;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    bool firstWord = true;
    std::string word;

    auto flush = [&]() {
        if (word.empty()) return;
        // 词首空格用 'Ġ' 表示（首个词不加）
        std::string w = firstWord ? word : ("Ġ" + word);
        firstWord = false;
        std::vector<std::string> symbols = utf8Chars(w);
        std::vector<std::string> merged = bpeMerge(symbols);
        for (auto& s : merged) ids.push_back(symbolToId(s));
        word.clear();
    };

    for (char c : text) {
        if (c == ' ') { flush(); /* 下一个非空词首会加 Ġ */ }
        else word += c;
    }
    flush();
    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= (int)id2tok_.size()) continue;
        const std::string& t = id2tok_[id];
        size_t i = 0;
        while (i < t.size()) {
            unsigned char c = (unsigned char)t[i];
            int n = 1;
            if      (c < 0x80)       n = 1;
            else if ((c >> 5) == 0x6) n = 2;
            else if ((c >> 4) == 0xe) n = 3;
            else if ((c >> 3) == 0x1e) n = 4;
            std::string cp = t.substr(i, n);
            i += n;
            if      (cp == "Ġ") out += ' ';   // 空格占位符
            else if (cp == "Ċ") out += '\n';  // 换行占位符
            else                 out += cp;
        }
    }
    return out;
}
