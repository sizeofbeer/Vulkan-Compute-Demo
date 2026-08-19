# ch14 · 完整 LLM 推理引擎（CPU 参考实现，数值已验证）

> 前置：ch00~ch13 已覆盖 Vulkan 基础与全部需要的算子内核（GEMM / RMSNorm / Swish / Softmax+Attention）。
> 本章把课程成果组装成一个**真正能「输入文字、输出文字」的 LLM 推理引擎**，并实现为
> 后续 Vulkan 算子内核的一一对齐基准。

---

## 1. 它做了什么

端到端 CPU 自回归推理引擎（纯 STL，无需 GPU）：

```
文字 ──tokenizer──► ids ──embed──► 28×Decoder ──final RMSNorm──► lm_head ──贪心/重复惩罚──► logits ──解码──► 文字
                                    │
                                    └─ 每层：RMSNorm → QKV(Gemm) → QK-Norm → RoPE → GQA causal attn → O(Gemm)
                                        → 残差 → RMSNorm → SwiGLU(Gemm×3) → 残差
```

- 模型：`Qwen3-0.6B`（28 层 decoder-only，hidden=1024，16/8 GQA heads，head_dim=128，SwiGLU，RoPE θ=1e6，vocab=151936）。
- 权重格式：与 `F:\plugin-VL` 同源的 **ncnn 自定义层**（`Embed / RMSNorm / Gemm / InnerProduct`），blob 以 4 字节 tag 区分 fp16/fp32。
- 注意力 core（RoPE + GQA + causal softmax + KV cache）不在 `.param` 里，由本引擎在 QKV 投影之后、O 投影之前实施——正是你 ncnn C++ decoder 手写的部分。

---

## 2. 两个可用模型

| 模型目录 | 来源 | 文本生成表现 | 用途 |
|----------|------|--------------|------|
| `qwen3_0.6b_base` | 真实 `Qwen/Qwen3-0.6B-Base`（HF → 转 ncnn） | **连贯、事实正确** ✅ | 本课 demo 默认 |
| `qwen3_asr_0.6b` | 你原有的 ASR 模型文本塔 | 无音频条件下退化（重复/乱码） | 语音场景（需音频条件） |

> 默认 `TOKENIZER_DIR` 已指向 `qwen3_0.6b_base`（CMake 构建指向 `CMAKE_SOURCE_DIR/../model/`），直接 `./ch14_llm --prompt "..."` 即可跑通。
> 建议设环境变量 `VULKAN_NN_MODEL_DIR` 指向模型根目录（Windows/Linux 通用），模型目录会优先取它。
> 用 `--model "$VULKAN_NN_MODEL_DIR/qwen3_asr_0.6b"` 可切回原模型（引擎会自动跳过 QK-Norm，因为该导出不含它）。

### 验证样例（base 模型）
```
The capital of France is  →  Paris. The capital of Germany is Berlin. ...
Once upon a time, there was a small robot who  →  lived in the world of robots. He had many friends ...
```

---

## 3. 关键工程结论（都是踩坑踩出来的）

1. **ncnn blob 格式**：每个 blob 以 4 字节 tag 开头。`0x01306B47` ⇒ fp16（随后 `count` 个 half，4 字节对齐）；
   否则这 4 字节就是第一个 fp32 数值。本模型 `text_embed`/`decoder_*` 投影为 fp16，`lm_head` 为 fp32（权重绑定 embed）。
   **已用「逐 blob 走查 + 文件体积精确吻合」校验**，不是猜的。

2. **RoPE**：θ=1e6，`inv_freq[j]=θ^(-2j/128)`，缓存为**重复 [128]**（前 64=后 64），`rotate_half`（非交错）：
   `out0=x0·c−x1·s; out1=x0·s+x1·c`。已对拍你给的 `_hf_cos/_hf_sin.npy`（max err ~1e-6）。

3. **Qwen3 QK-Norm（最重要的一坑）**：真实 Qwen3 在 RoPE **之前**对每个注意力头做一层可学习的逐头 RMSNorm
   （`q_norm`/`k_norm`，各 [128]）。**缺它，真实模型会输出扁平分布**（"capital of France" 预测不出 Paris）。
   - `qwen3_asr_0.6b` 的导出里 **没有** qk-norm ⇒ 引擎按「可选」处理（权重存在才启用），对其自动跳过。
   - `qwen3_0.6b_base` 的转出**包含** qk-norm，引擎自动启用。
   - `--qknorm` 仍可强制对无权重模型做「纯归一化」做实验。

4. **数值已对拍**：
   - 引擎 vs 独立 numpy 参考实现（同一套权重、按 Qwen3 规范重写）：logits 误差 ~1e-3（asr 模型上完全一致）。
   - 引擎 vs HF 原始 bf16 权重（全 fp32 参考）："Paris" 同为 top-1（logit 21.04 vs 21.04）。
   - 因此「asr 模型输出垃圾」是**模型本身**在无音频条件下退化，不是引擎 bug。

---

## 4. 文件清单

| 文件 | 作用 |
|------|------|
| `llm_model.h/.cpp` | ncnn param/bin 加载器（fp16/fp32 tag、可选 qk-norm、lm_head 绑定） |
| `llm_engine.h/.cpp` | CPU 前向：embed → 28 层 → final norm → lm_head；GQA causal attn + KV cache；贪心+重复惩罚 |
| `tokenizer/bpe_tokenizer.*` | 纯 STL 的 Qwen BPE 分词器（UTF-8 拆分、byte-fallback、`Ġ/Ċ` 还原） |
| `main.cpp` | CLI：`--prompt / --max_new / --model / --selftest / --qknorm / --debug` |
| `convert.py` | HF `Qwen3-0.6B-Base`(safetensors,bf16) → 本引擎 ncnn 自定义格式 |
| `ref.py`   | 独立 numpy 参考实现 + 全 fp32 原始权重验证（无需 GPU，用于 A/B 对拍） |
| `walk.py`  | bin blob 走查小工具（验证 blob 布局/文件体积） |

---

## 5. 如何运行

```bash
# 编译（纯 CPU，无需 Vulkan SDK；或走顶层 cmake）。
# 输出名：Windows 上为 ch14_llm.exe，Linux/macOS 上为 ch14_llm（无后缀）。
cd <本项目>/ch14_llm
g++ -std=c++17 -O2 -I. main.cpp tokenizer/bpe_tokenizer.cpp llm_model.cpp llm_engine.cpp -o ch14_llm

# 默认跑真实 Qwen3-0.6B-Base（英文 demo）。
# 模型目录：建议设环境变量 VULKAN_NN_MODEL_DIR 指向模型根目录（Windows/Linux 通用）：
export VULKAN_NN_MODEL_DIR=/path/to/model          # Linux 示例；Windows 下用 set VULKAN_NN_MODEL_DIR=...
./ch14_llm --prompt "The capital of France is" --max_new 24
./ch14_llm --prompt "Once upon a time, there was a small robot who" --max_new 30

# 切回原 ASR 模型（无音频→退化，仅演示加载/分词）
./ch14_llm --model "$VULKAN_NN_MODEL_DIR/qwen3_asr_0.6b" --prompt "hello"

# 调试：打印 prompt 之后 top-5 候选 token
./ch14_llm --prompt "The capital of France is" --debug
```

> 调试辅助：设环境变量 `CH14_STATS=1` 可打印各算子 RMS（embed/RMSNorm/Gemm/attn/ao/down、逐层 hidden），
> 用于定位数值异常。

---

## 6. 如何把任意 HF Qwen3 模型接入本课

```bash
# 1) 下载 safetensors（以 0.6B-Base 为例）
curl -L https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors -o qwen3_0.6b_base_model.safetensors
# 2) 转成本引擎 ncnn 格式（自动含 qk-norm，并复用同源 vocab.txt/merges.txt）。
#    convert.py 默认用环境变量 VULKAN_NN_MODEL_DIR 作模型根目录（Windows/Linux 通用）。
export VULKAN_NN_MODEL_DIR=/path/to/model
python convert.py "$VULKAN_NN_MODEL_DIR/qwen3_0.6b_base_model.safetensors" "$VULKAN_NN_MODEL_DIR/qwen3_0.6b_base"
# 3) 运行
./ch14_llm --model "$VULKAN_NN_MODEL_DIR/qwen3_0.6b_base" --prompt "你的提示"
```
转换器做了：bf16→fp32→fp16、按 `Embed/RMSNorm/Gemm/InnerProduct` 自定义层重写 param、blob 按 fp16 tag 落盘、
`tie_word_embeddings=true` 时 lm_head 复用 embed。

---

## 7. 下一步（课程主线）：把 CPU 算子搬到 Vulkan 计算着色器

引擎的每个算子（Embed 查表、RMSNorm、Gemm/InnerProduct、SwiGLU、softmax+GQA attention、RoPE）
都要实现为 **Vulkan compute shader**，并复用本课 ch04/06/07/08 的内核封装。移植顺序建议：

1. 先在 GPU 上实现单个算子并与 CPU 版本对拍（A/B 误差 < 1e-2）。
2. 再串起整层、整网络；KV cache 用 `vkBuffer` 持久化（参考 ch10 arena 显存管理）。
3. 最后核对：Vulkan 路径 logits 与 CPU 路径一致，且 `ch14_llm` 可以 `--vulkan` 开关切换后端。

---

## 8. 已知限制

1. 仅英文 demo（Qwen tokenizer 对 CJK 走 byte-fallback，本课权重未含中文 token，中文会回退 `<unk>`）。
2. 解码用贪心 + 重复惩罚（1.1），未做 top-p/temperature 采样（参考 `ncnn_text_runtime.cpp` 可加）。
3. 当前为 CPU 参考实现；Vulkan 后端为课程下一步，尚未实现。
