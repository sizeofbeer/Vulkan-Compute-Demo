# 第 11 章 · 量化与融合反量化

> 目标：搞懂 LLM 推理省显存的头号手段——**权重量化**，以及为什么要把「反量化」**融合**进 matmul 而不是单独做。本章从 int8 起步，再用 Q4_0 和 Q8_0 两种 group-wise 格式深入 group-wise 量化与 GPU 融合反量化 GEMM。

---

## 11.1 为什么要量化

模型权重常常占显存的 90%+。FP32 权重每个 4 字节；换成 **int8** 只要 1 字节——**显存和带宽同时省 4 倍**。而神经网络对权重的精度不敏感，int8 的误差通常可忽略（配合后续层的冗余）。这也是为什么 7B 的 LLaMA 用 Q4（每权重 4 bit）能塞进 4GB 显卡。

## 11.2 对称 int8 量化（per-tensor）

最简单的形式：

```
scale = max|W| / 127
w_q  = round( w / scale )          # 钳制到 [-127, 127]
反量化：w ≈ w_q * scale
```

`ch11_quant/main.cpp` 里把 `K*N` 个 float 权重压成 int8，上传到显存只占 1/4 字节。`q8_gemm.comp` 读取 `int8_t` 权重，**当场** `float(w) * scale` 还原成 float，再参与乘加——反量化发生在 GEMM 最内层循环，**没有任何中间 float 权重 buffer**。

## 11.3 关键：「融合」反量化 vs 分离反量化

- ❌ 分离：先把 int8 权重解压成一个 float 临时 buffer，再跑普通 GEMM。→ 多吃一份显存 + 多一次全量搬运，**量化省下的带宽又还回去了**。
- ✅ 融合（本章做法）：反量化直接写进 matmul 的累加循环。→ 权重始终以 int8 待在显存，带宽省满。

这正是 llama.cpp `dequant_funcs.glsl` 的思路：每种量化格式（q4_0、q8_0、iq4_nl…）对应一套 GPU 上的「解包+反量化」函数，被 `mul_mm` 在累加时就地调用。

## 11.4 Group-wise 量化：Q4_0 与 Q8_0 格式

Per-tensor int8 用一个 scale 覆盖整个矩阵，对于 outlier 不友好的情况，误差会放大。**Group-wise 量化**将权重矩阵按列分组，每 32 个元素共享一个 scale，精度更细。

### Q4_0 格式

Q4_0 是 ggml/llama.cpp 中最基础的 4-bit 量化格式：

```
Q4_0 Block（32 元素，20 字节 = 5 × uint32）：
  ├── d: fp16 scale（2 字节）          ← 该 block 32 个元素共享
  ├── qs: 4-bit 有符号 int 打包（16 字节）← 2 元素/字节，低 4 位 = 元素 i，高 4 位 = 元素 i+1
  └── padding: 2 字节填充到 20 字节对齐
```

- 每权重存储：4 bit（理论压缩比 8×）
- 实际压缩比：20 字节 / 32 元素 = 5 bit/weight，约 6.4×（含 scale 和 padding 开销）
- 有符号 int4 范围：[-8, 7]
- 反量化：`w = float(q) * scale`

### Q8_0 格式

Q8_0 是 ggml 中的 8-bit 量化格式，与 per-tensor int8 的区别在于它是 group-wise 的：

```
Q8_0 Block（32 元素，36 字节 = 9 × uint32）：
  ├── d: fp16 scale（2 字节）          ← 该 block 32 个元素共享
  ├── qs: int8 数据（32 字节）
  └── padding: 2 字节填充到 36 字节对齐
```

- 每权重存储：8 bit（理论压缩比 4×）
- 实际压缩比：36 字节 / 32 元素 = 9 bit/weight，约 3.56×（含 scale 开销）
- 有符号 int8 范围：[-127, 127]（避免 -128）

### 与 ggml 格式的兼容性

本章的 Q4_0/Q8_0 格式与 ggml 的定义完全兼容：
- 相同的 block 大小（32 元素）
- 相同的 fp16 scale 存储
- 相同的 4-bit/8-bit 数据排布
- 唯一的区别是 GPU 端做了 20/36 字节对齐（uint32 对齐），方便着色器用 `uint` 读取

## 11.5 融合反量化在 Q4_0/Q8_0 中的实现

对于 int8，反量化只是一个 `float(w) * scale`。对于 Q4_0，反量化更复杂：

```
1. 读 block 头部的 fp16 scale → 转 float
2. 定位元素在 block 中的偏移
3. 从 packed 字节中提取 nibble（位运算 & 0xF / >> 4）
4. 有符号扩展（if q >= 8: q -= 16）
5. w = float(q) * scale
```

Q8_0 类似但更简单（不需要 nibble 解包）：

```
1. 读 block 头部的 fp16 scale → 转 float
2. 读 int8 值
3. 有符号扩展（if q >= 128: q -= 256）
4. w = float(q) * scale
```

这些操作被**融合**在 `q4_0_gemm.comp` 和 `q8_0_gemm.comp` 的共享内存分块循环中：每个线程在加载权重 tile 时，就地解包反量化，然后存入共享内存供后续乘加使用。与 int8 版本相比，Q4_0/Q8_0 版本使用了 **16×16 的共享内存 tiling**，进一步减少全局内存访问。

## 11.6 误差从哪来、怎么控

误差只来自「量化阶梯」：`w_q * scale` 与真值 `w` 最多差 `scale/2`。累加 K 项后，误差被平均，通常比单点更小。

Group-wise 量化（Q4_0/Q8_0）相比 per-tensor（int8）的优势：
- 每个 block 独立 scale，能更精细地表示局部数值范围
- 对 outlier 列不敏感——outlier 只影响其所在的 block，不会污染整个矩阵
- 实践中 Q8_0 的精度接近 FP32，Q4_0 在 5-10% 相对误差内可接受

本章校验：
- int8 / Q8_0：5% 相对容差
- Q4_0：10% 相对容差（int4 精度更低，容差更宽）

## 11.7 动手实验

1. 把量化粒度从 per-tensor 改成 **per-channel**（每个输出列 col 一个 scale），重测最大误差是否明显下降。
2. 把 int8 换成「伪 int4」：把两个 int4 打包进一个字节（位运算解包），体验 Q4 的 8 倍压缩（注意 GLSL 里用 `& 0xF` / `>> 4` 解包）。
3. 用 ch09 的计时法对比「int8 权重 + 融合反量化」与「纯 float32 权重」的带宽占用与耗时——直观看到量化如何缓解带宽瓶颈（Roofline 左移）。
4. 修改 Q4_0 block 大小（比如 64 元素/block），观察压缩比与精度的 trade-off。

## 11.8 对照工业级

- **ncnn**：int8 卷积/全连接仍是主力；`src/layer/vulkan/shader/` 下新增 `convolution_*_int8.comp`、`gemm_int8.comp`、`gemm_quantize_{A,B}_int8.comp`、`gemm_quantize_B_{absmax,descale}_int8.comp`，并有 int8 版 coopmat / Winograd 变换（见 ch15）。权重以 int8 存储，kernel 内融合「乘 scale + 加 bias」。
- **llama.cpp / ggml-vulkan**（2026-11 现状）：反量化不再是「一个 `dequant_funcs` 文件」，而是三层：
  - `ggml/src/ggml-quants.c`：CPU 侧 block 结构与量化算法定义（旧的 `iq-quants.c` 已并入此文件）；
  - `vulkan-shaders/dequant_funcs.glsl` / `dequant_funcs_cm2.glsl` / `dequant_head.glsl`：GLSL 公共解包片段；
  - 26 个 `vulkan-shaders/dequant_<format>.comp`：`q4_0/q8_0/q*_k/iq*/mxfp4/nvfp4/q1_0/q2_0/tq1_0/tq2_0` 各自的独立反量化 kernel。
  - SPIR-V **在构建期**由 `vulkan-shaders-gen` + glslc 从 155 个 `.comp` 模板生成，产物以 `.comp.cpp` 数组形式编进二进制，仓库里没有预编译 `.spv`；因此首版文档里「v1.17 起缓存编译好的 SPIR-V 以加速启动」的说法已不适用。
- **量化类型全景**：`ggml/include/ggml.h` 现有 `GGML_TYPE_COUNT = 44`，除 Q4_0/Q8_0、K-quant、I-quant 外还有 Q1_0/Q2_0、TQ1_0/TQ2_0、MXFP4/NVFP4、F8_E4M3（详见 ch20）。
- **Q4_0 vs Q8_0 的选择**：Q8_0 精度高、解包开销小，适合精度敏感层（attention 的 Q/K/V 投影）；Q4_0 压缩比高，适合显存瓶颈严重的场景。ggml 支持逐层混合精度（`llama-quant.cpp`）。

## 11.9 本章代码结构

```
ch11_quant/
├── main.cpp               # 主程序：int8/Q4_0/Q8_0 量化、GPU 融合 GEMM、验证、基准测试
├── CMakeLists.txt          # 构建配置（自动编译 .comp → .spv）
├── shaders/
│   ├── q8_gemm.comp        # int8 融合反量化 GEMM（原版，无共享内存 tiling）
│   ├── q4_0_gemm.comp      # Q4_0 融合反量化 GEMM（共享内存 tiling 16×16）
│   ├── q8_0_gemm.comp      # Q8_0 融合反量化 GEMM（共享内存 tiling 16×16）
│   └── gemm_fp32.comp      # FP32 基准 GEMM（共享内存 tiling 16×16）
└── docs/
    ├── ch11_quant.md        # 中文文档
    └── ch11_quant.en.md     # 英文文档
```

下一章（ch12）我们把多个算子串成**一张计算图**：录进同一个命令缓冲、用 barrier 衔接依赖、一次性提交——把零碎算子变成高效管线。