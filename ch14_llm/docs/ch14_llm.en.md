# ch14 · Complete LLM Inference Engine (CPU reference implementation, numerically verified)

> Prerequisite: ch00~ch13 already covered Vulkan basics and all required operator kernels (GEMM / RMSNorm / Swish / Softmax+Attention).
> This chapter assembles the course results into a **real LLM inference engine that takes text in and produces text out**, and serves as
> the one-to-one alignment baseline for the subsequent Vulkan operator kernels.

---

## 1. What it does

End-to-end CPU autoregressive inference engine (pure STL, no GPU required):

```
text ──tokenizer──► ids ──embed──► 28×Decoder ──final RMSNorm──► lm_head ──greedy/repeat-penalty──► logits ──decode──► text
                                    │
                                    └─ per layer: RMSNorm → QKV(Gemm) → QK-Norm → RoPE → GQA causal attn → O(Gemm)
                                        → residual → RMSNorm → SwiGLU(Gemm×3) → residual
```

- Model: `Qwen3-0.6B` (28-layer decoder-only, hidden=1024, 16/8 GQA heads, head_dim=128, SwiGLU, RoPE θ=1e6, vocab=151936).
- Weight format: **ncnn custom layers** (`Embed / RMSNorm / Gemm / InnerProduct`) sourced from the same lineage as `F:\plugin-VL`; blobs are distinguished as fp16/fp32 by a 4-byte tag.
- The attention core (RoPE + GQA + causal softmax + KV cache) is not in the `.param`; this engine implements it after the QKV projection and before the O projection — exactly the part you hand-wrote in your ncnn C++ decoder.

---

## 2. Two available models

| Model directory | Source | Text generation quality | Purpose |
|-----------------|--------|-------------------------|---------|
| `qwen3_0.6b_base` | Real `Qwen/Qwen3-0.6B-Base` (HF → converted to ncnn) | **Coherent, factually correct** ✅ | Default for this chapter's demo |
| `qwen3_asr_0.6b` | Your existing ASR model's text tower | Degrades without audio (repetition/garbage) | Speech scenario (requires audio condition) |

> The default `TOKENIZER_DIR` already points to `qwen3_0.6b_base` (CMake build points to `CMAKE_SOURCE_DIR/../model/`), so just run `./ch14_llm --prompt "..."` to make it work.
> Set the `VULKAN_NN_MODEL_DIR` env var to the model root (works on Windows & Linux); the model dir takes priority over it.
> Use `--model "$VULKAN_NN_MODEL_DIR/qwen3_asr_0.6b"` to switch back to the original model (the engine auto-skips QK-Norm, because that export does not contain it).

### Validation samples (base model)
```
The capital of France is  →  Paris. The capital of Germany is Berlin. ...
Once upon a time, there was a small robot who  →  lived in the world of robots. He had many friends ...
```

---

## 3. Key engineering findings (learned the hard way)

1. **ncnn blob format**: each blob starts with a 4-byte tag. `0x01306B47` ⇒ fp16 (followed by `count` halfs, 4-byte aligned);
   otherwise these 4 bytes are the first fp32 value. In this model `text_embed`/`decoder_*` projections are fp16, `lm_head` is fp32 (weights tied to embed).
   **Verified by "per-blob walk-through + exact file-size match"**, not guessed.

2. **RoPE**: θ=1e6, `inv_freq[j]=θ^(-2j/128)`, cached as a **repeated [128]** (first 64 = last 64), `rotate_half` (non-interleaved):
   `out0=x0·c−x1·s; out1=x0·s+x1·c`. Aligned against your provided `_hf_cos/_hf_sin.npy` (max err ~1e-6).

3. **Qwen3 QK-Norm (the single most important pitfall)**: real Qwen3 applies a learnable per-head RMSNorm to every attention head **before** RoPE
   (`q_norm`/`k_norm`, each [128]). **Without it, the real model outputs a flat distribution** (cannot predict "Paris" from "capital of France").
   - The `qwen3_asr_0.6b` export **does not** contain qk-norm ⇒ the engine treats it as "optional" (enabled only if weights exist) and auto-skips it.
   - The `qwen3_0.6b_base` export **includes** qk-norm, and the engine auto-enables it.
   - `--qknorm` can still force a "pure normalization" experiment on a weightless model.

4. **Numerics already aligned**:
   - Engine vs independent numpy reference (same weights, rewritten per Qwen3 spec): logits error ~1e-3 (exactly equal on the asr model).
   - Engine vs HF original bf16 weights (full fp32 reference): "Paris" is top-1 in both (logit 21.04 vs 21.04).
   - Therefore "asr model outputs garbage" is the **model itself** degrading without audio, not an engine bug.

---

## 4. File list

| File | Role |
|------|------|
| `llm_model.h/.cpp` | ncnn param/bin loader (fp16/fp32 tag, optional qk-norm, lm_head tying) |
| `llm_engine.h/.cpp` | CPU forward pass: embed → 28 layers → final norm → lm_head; GQA causal attn + KV cache; greedy + repeat penalty |
| `tokenizer/bpe_tokenizer.*` | Pure-STL Qwen BPE tokenizer (UTF-8 split, byte-fallback, `Ġ/Ċ` restoration) |
| `main.cpp` | CLI: `--prompt / --max_new / --model / --selftest / --qknorm / --debug` |
| `convert.py` | HF `Qwen3-0.6B-Base`(safetensors,bf16) → this engine's ncnn custom format |
| `ref.py`   | Independent numpy reference + full fp32 original-weight verification (no GPU needed, for A/B alignment) |
| `walk.py`  | Small bin blob walk-through tool (verifies blob layout / file size) |

---

## 5. How to run

```bash
# Build (pure CPU, no Vulkan SDK needed; or use the top-level cmake).
# Output name: ch14_llm.exe on Windows, ch14_llm (no suffix) on Linux/macOS.
cd <project>/ch14_llm
g++ -std=c++17 -O2 -I. main.cpp tokenizer/bpe_tokenizer.cpp llm_model.cpp llm_engine.cpp -o ch14_llm

# Run the real Qwen3-0.6B-Base by default (English demo).
# Model dir: set the VULKAN_NN_MODEL_DIR env var to the model root (works on Windows & Linux):
export VULKAN_NN_MODEL_DIR=/path/to/model          # Linux; on Windows use: set VULKAN_NN_MODEL_DIR=...
./ch14_llm --prompt "The capital of France is" --max_new 24
./ch14_llm --prompt "Once upon a time, there was a small robot who" --max_new 30

# Switch back to the original ASR model (no audio → degrades, demo loading/tokenizing only)
./ch14_llm --model "$VULKAN_NN_MODEL_DIR/qwen3_asr_0.6b" --prompt "hello"

# Debug: print top-5 candidate tokens after the prompt
./ch14_llm --prompt "The capital of France is" --debug
```

> Debug aid: set the environment variable `CH14_STATS=1` to print per-operator RMS (embed/RMSNorm/Gemm/attn/ao/down, per-layer hidden),
> useful for locating numerical anomalies.

---

## 6. How to plug any HF Qwen3 model into this chapter

```bash
# 1) Download safetensors (using 0.6B-Base as example)
curl -L https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors -o qwen3_0.6b_base_model.safetensors
# 2) Convert to this engine's ncnn format (auto-includes qk-norm, reuses the same-origin vocab.txt/merges.txt).
#    convert.py uses the VULKAN_NN_MODEL_DIR env var as the model root (Windows & Linux).
export VULKAN_NN_MODEL_DIR=/path/to/model
python convert.py "$VULKAN_NN_MODEL_DIR/qwen3_0.6b_base_model.safetensors" "$VULKAN_NN_MODEL_DIR/qwen3_0.6b_base"
# 3) Run
./ch14_llm --model "$VULKAN_NN_MODEL_DIR/qwen3_0.6b_base" --prompt "your prompt"
```
The converter does: bf16→fp32→fp16, rewrites param with `Embed/RMSNorm/Gemm/InnerProduct` custom layers, writes blobs with fp16 tag,
and reuses embed for lm_head when `tie_word_embeddings=true`.

---

## 7. Next step (course main line): port the CPU operators to Vulkan compute shaders

Each engine operator (Embed lookup, RMSNorm, Gemm/InnerProduct, SwiGLU, softmax+GQA attention, RoPE)
must be implemented as a **Vulkan compute shader**, reusing the kernel wrappers from ch04/06/07/08. Suggested porting order:

1. First implement a single operator on the GPU and align it with the CPU version (A/B error < 1e-2).
2. Then chain up a whole layer and the whole network; persist the KV cache with `vkBuffer` (see ch10 arena memory management).
3. Final check: the Vulkan path logits match the CPU path, and `ch14_llm` can switch backends via a `--vulkan` flag.

---

## 8. Known limitations

1. English demo only (the Qwen tokenizer uses byte-fallback for CJK; this chapter's weights do not include Chinese tokens, so Chinese falls back to `<unk>`).
2. Decoding uses greedy + repeat penalty (1.1), no top-p/temperature sampling (can be added by referring to `ncnn_text_runtime.cpp`).
3. Currently a CPU reference implementation; the Vulkan backend is the next course step and is not yet implemented.
