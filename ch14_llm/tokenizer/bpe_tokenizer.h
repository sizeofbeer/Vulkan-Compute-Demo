#pragma once
// bpe_tokenizer.h —— 轻量 Qwen / SentencePiece 风格 BPE 分词器（课程 ch14 LLM 用）
//
// 设计目标：不依赖任何第三方库，纯 STL，能在无 GPU 环境下独立验证「文字 <-> token」。
// 加载目录下的：
//   vocab.txt   —— 每行一个 token，行号即 token id（第 0 行 = id 0）
//   merges.txt  —— 首行 #version，其后每行 "a b" 为一个 BPE 合并规则，行序即合并优先级
//   special_tokens.txt —— 可选（本模型为空，特殊 token 由外部 setSpecialIds 指定）
//
// 能力：常规文本 BPE 编码、UTF-8 字符拆分、byte-fallback（vocab 含 <0xXX> 时）、
//       decode（Ġ -> 空格，Ċ -> 换行）。

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class BPETokenizer {
public:
    // 加载模型目录下的 vocab.txt / merges.txt
    void load(const std::string& modelDir);

    // 文字 -> token ids（含空格归一：词间空格用 'Ġ' 表示）
    std::vector<int> encode(const std::string& text) const;

    // token ids -> 文字（拼接；Ġ->空格, Ċ->换行）
    std::string decode(const std::vector<int>& ids) const;

    size_t vocabSize() const { return id2tok_.size(); }
    int    unkId()     const { return unkId_; }
    int    bosId()     const { return bosId_; }   // -1 表示未识别
    int    eosId()     const { return eosId_; }   // -1 表示未识别

    // 外部（对齐你的 C++ decoder）显式指定 BOS/EOS id
    void setSpecialIds(int bos, int eos) { bosId_ = bos; eosId_ = eos; }

private:
    std::vector<std::string>          id2tok_;                 // id -> token 字符串
    std::unordered_map<std::string,int> tok2id_;                // token 字符串 -> id（首次出现优先）
    std::unordered_map<std::string,int> mergeRank_;             // "a b" -> 合并优先级（越小越优先）
    bool                              hasByteTokens_ = false;   // vocab 是否含 <0xXX>
    int                               unkId_  = 0;
    int                               bosId_  = -1;
    int                               eosId_  = -1;

    // 把 UTF-8 字符串拆成 Unicode 码点（每个码点一个 std::string）
    std::vector<std::string> utf8Chars(const std::string& s) const;

    // 对一组字符符号做 BPE 贪婪合并，返回合并后的符号序列
    std::vector<std::string> bpeMerge(const std::vector<std::string>& symbols) const;

    // 单个符号 -> id（含 byte-fallback / unk）
    int symbolToId(const std::string& sym) const;
};
