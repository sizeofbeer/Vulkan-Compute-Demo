# Chapter 13 · Capstone: Running a Small Model on Your Own Engine

> Goal: assemble the building blocks from the previous 12 chapters — **resident weights + tiled GEMM + bias add + ReLU + compute graph** — to run the forward inference of a two-layer MLP. This is the course's "graduation project".

---

## 13.1 What We Will Run

A lightweight MNIST-style MLP:

```
h = ReLU(W1 · x + b1)      # 784 -> 256
y = W2 · h + b2            # 256 -> 10
```

`x` is the input (784 dims), `W1/b1/W2/b2` are the model parameters. The course uses **randomly initialized** weights (no real trained model file); the point is to **prove the entire forward path runs correctly on your engine**, not classification accuracy.

## 13.2 How the Building Blocks Are Used

| Chapter | Contribution | Reflected in ch13 |
|---------|--------------|-------------------|
| ch03 | Memory types / staging / resident weights | Weights `bW1/bW2/bb1/bb2` uploaded once, device-local resident |
| ch04 | Tiled GEMM | Both fully-connected layers use `gemm_tiled.comp` |
| ch06 | Element-wise ops | ReLU node (`elemwise.comp` mode=0) |
| ch10 | Memory pool | Intermediate tensors could go in an arena (here we use standalone buffers for clarity; idea is the same) |
| ch11 | (Optional) Quantization | If W1/W2 are swapped for int8 + fused dequant, memory drops another 4x |
| ch12 | Compute graph | 5 ops recorded into one command buffer, connected by barriers, submitted once |

> Note in `main.cpp`: **weights are `vkUpload`ed only once at startup**; each sample only re-uploads input `x`, runs the graph, and reads back output `y`. This is exactly the memory behavior of a real inference engine — model resident, data fed per sample.

## 13.3 Inference Loop

```cpp
for (each sample s) {
    vkUpload(bX, x);                 // only upload input
    Graph g; g.begin();              // record: matmul -> bias -> relu -> matmul -> bias
    g.end();                         // single submit
    y = vkDownload(bOUT);            // read back 10-way logits
    pred = argmax(y);                // predicted class
}
```

The five ops reuse the same set of `ComputePass` (prepare once, bind once); each sample only swaps input, re-records commands, and re-reads output — extremely low overhead.

## 13.4 Verification and "Prediction"

Each sample runs a CPU reference forward pass, comparing the GPU result element-by-element (tolerance 1e-3, error from floating-point accumulation order). Since weights are random, the classification itself is meaningless, but `argmax` gives a formally "predicted class" — if you swap in a real trained MNIST weight (784×256×10 is about 2MB of floats), it will actually recognize handwritten digits.

## 13.5 Hands-On Experiments (Capstone Extensions)

1. **Load a real model**: Replace random weights with trained MNIST MLP weights (can be exported from PyTorch as a plain float binary) and run a set of paired samples to see classification accuracy.
2. **Apply quantization**: Change `W1/W2` to the int8 + fused dequant from ch11, saving another 4x memory and bandwidth, and compare output differences.
3. **Add batching**: Expand input `x` to `[batch, 784]` and run multiple samples in one graph pass; use ch09 timing to see throughput gains.
4. **Go Transformer**: Replace the GEMM chain with ch08 attention + ch07 RMSNorm to make a mini GPT block — you already have all the parts.

## 13.6 Comparison with Industry-Grade (Course Endpoint)

- **ncnn**: Your MLP path ≈ ncnn's `InnerProduct` (fully-connected) + `ReLU` + `InnerProduct`; the only difference is that ncnn also does `elempack` packing, convolution substitution, and op fusion. Swap this chapter's `gemm_tiled` for ncnn's packed kernel and you get the industrial version.
- **llama.cpp / ggml-vulkan**: Your "resident weights + compute graph + single submit" is exactly the skeleton of its inference loop; the difference is it handles a Transformer (attention + RMSNorm + quantized weights + coopmat acceleration). You already have all the prerequisite knowledge to understand its source code.

---

## 13.7 Practice: Loading a Real ncnn Model (Path A)

Sections 13.1~13.5 used **randomly initialized** weights, to prove the forward path is correct. Now we upgrade ch13 to **load the real Qwen3-ASR ncnn subgraph you exported** — this is the course's endpoint of "running a real model on your own engine".

### 13.7.1 Where the Model Is and What Format

The real model is located at `F:\vulkan\model\qwen3_asr_0.6b` (Windows default; on Linux pass it via command-line arguments, or set the `VULKAN_NN_MODEL_DIR` env var to the model root), and contains a **set of ncnn-exported subgraphs for Qwen3-ASR 0.6B** (`audio_XX_*` audio encoder, `decoder_XX_*` Transformer decoder, `lm_head` language-model head, plus `_golden_*`/`_ref_*`/`cpp_*`/`torch_*` per-layer consistency verification fixtures).

The format is the **standard ncnn `param` + `bin`**:
- `.param`: first line magic `7767517`, second line `layer_count blob_count`, then per layer `Type name in_c out_c in_blobs... out_blobs... key=value...`.
- `.bin`: measured in practice to be a **plain fp32 weight stream with no blob header**: each weight blob is directly contiguous float32 data, in the order the weights appear in the param, with size determined by `weight_data_size` (element count) in the param.

> Measured correction: the `InnerProduct` param `2=155582464` for `lm_head` is the **element count**, 155582464 = 151936 × **1024**, i.e. vocab=151936, hidden=1024 (consistent with your Qwen3-ASR hidden=1024), with weight layout `[numOut][numIn]` row-major.

### 13.7.2 Why Path A Picks `lm_head`

The subgraph heavily uses native layers ncnn doesn't have (`Gemm`/`RMSNorm`/`Swish`/`MemoryData`/`torch.var`), which are custom ops from the Qwen3-ASR port. The current ch13 engine only understands `InnerProduct + no bias`, so **the only subgraph that runs directly is `lm_head`** (a single fully-connected layer, the vocab projection head). It is enough to close the minimal loop:

"standard ncnn parsing → fp32 weight upload resident → GPU GEMM → diff against CPU reference".

The new `model_loader.h` (header-only) is this parsing foundation; `main.cpp` is changed to load `lm_head` and run `gemm_tiled` inference.

### 13.7.3 Running

```bash
# Loads lm_head by default (needs a GPU with >=1GB VRAM, 622MB weights resident)
./build/ch13_mlp/ch13_mlp
# You can also specify another pure InnerProduct subgraph
./build/ch13_mlp/ch13_mlp some.ncnn.param some.ncnn.bin
```

Expected output: load 622MB weights, 3 inference passes, each `maxAbsDiff` on the order of 1e-3, `argmax` matches the CPU reference, total error count 0.

### 13.7.4 Next Steps (Path B / C)

- **Path B**: Extend the engine to support `Gemm`/`RMSNorm`/`Swish`/`BinaryOp`/`Split`, then load SwiGLU blocks like `decoder_XX_mlp` and do per-layer diffing with `_golden_*` (review ch04 GEMM / ch06 element-wise / ch07 RMSNorm).
- **Path C**: Add a LayerNorm micro-op cluster (`MemoryData`/`Reduction`/`torch.var`/`UnaryOp`/`Reshape`/`Concat`) to run the audio encoder.

## 🎓 Course Summary

From enumerating devices in `ch00` to running model forward in `ch13`, you have completed the full path of "zero foundation → building your own inference engine":

- **Basics**: environment, minimal compute loop, synchronization, memory types and FP16;
- **Operators**: GEMM, convolution (im2col), element-wise, normalization, softmax/attention;
- **Engineering**: performance analysis (timestamps + Roofline), memory pool, quantization fusion, compute-graph scheduling;
- **Integration**: inferring a small model with your own engine.

Directions to go deeper: automatic operator-fusion generation, cooperative matrix (Tensor-Core-like), multi-queue pipelining, dynamic shape, comparison with a CUDA backend — all built on concepts you have already mastered. May you write your own "mini ncnn / mini ggml-vulkan".
