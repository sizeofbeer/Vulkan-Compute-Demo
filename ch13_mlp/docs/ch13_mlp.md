# 第 13 章 · 综合项目：用自研引擎跑通一个小模型

> 目标：把前 12 章的积木拼起来——**常驻权重 + 分块 GEMM + 加偏置 + ReLU + 计算图**——跑通一个两层 MLP 的前向推理。这是课程的「毕业设计」。

---

## 13.1 我们要跑什么

一个轻量版 MNIST 风格 MLP：

```
h = ReLU(W1 · x + b1)      # 784 -> 256
y = W2 · h + b2            # 256 -> 10
```

`x` 是输入（784 维），`W1/b1/W2/b2` 是模型参数。课程用**随机初始化**的权重（没有真实训练好的模型文件），重点在于**证明整条前向链路在你的引擎上正确运行**，而不是分类精度。

## 13.2 各章积木如何被用上

| 章节 | 贡献 | 在 ch13 的体现 |
|------|------|----------------|
| ch03 | 显存类型 / staging / 权重常驻 | 权重 `bW1/bW2/bb1/bb2` 上传一次、device-local 常驻 |
| ch04 | 分块 GEMM | 两个全连接层用 `gemm_tiled.comp` |
| ch06 | 逐元素算子 | ReLU 节点（`elemwise.comp` mode=0） |
| ch10 | 显存池 | 中间张量本可放 arena（此处为清晰用独立 buffer，思路一致） |
| ch11 | （可选）量化 | 若把 W1/W2 换成 int8 + 融合反量化，显存再省 4 倍 |
| ch12 | 计算图 | 5 个算子录进一条命令缓冲、barrier 衔接、一次提交 |

> 注意 `main.cpp` 里：**权重只在启动时 `vkUpload` 一次**，每个样本只重传输入 `x`、跑图、读回输出 `y`。这正是真实推理引擎的内存行为——模型常驻，逐样本喂数据。

## 13.3 推理循环

```cpp
for (每个样本 s) {
    vkUpload(bX, x);                 // 只传输入
    Graph g; g.begin();              // 录制：matmul -> bias -> relu -> matmul -> bias
    g.end();                         // 一次提交
    y = vkDownload(bOUT);            // 读回 10 路 logits
    pred = argmax(y);                // 预测类别
}
```

五个算子复用同一组 `ComputePass`（prepare 一次、bind 一次），每个样本只是换输入、重录命令、重读输出——开销极低。

## 13.4 校验与「预测」

每个样本都做 CPU 参考前向，逐元素比对 GPU 结果（容差 1e-3，误差来自浮点累加次序）。由于权重随机，分类本身没意义，但 `argmax` 给出了一个形式上「预测类别」——你若换成真实训练好的 MNIST 权重（784×256×10 大约 2MB 浮点），它就真的能识别手写数字。

## 13.5 动手实验（毕业拓展）

1. **接真实模型**：把随机权重换成训练好的 MNIST MLP 权重（可从 PyTorch 导出为纯 float 二进制），跑一组成对样本看分类准确率。
2. **上量化**：把 `W1/W2` 改成 ch11 的 int8 + 融合反量化，显存与带宽再省 4 倍，对比输出差异。
3. **加批处理（batch）**：把输入 `x` 扩成 `[batch, 784]`，一次图推理多个样本，用 ch09 计时看吞吐提升。
4. **上Transformer**：把 GEMM 链换成 ch08 的注意力 + ch07 的 RMSNorm，做一个迷你 GPT 块——你已经具备全部零件。

## 13.6 对照工业级（课程终点）

- **ncnn**：你的 MLP 链路 ≈ ncnn 的 `InnerProduct`（全连接）+ `ReLU` + `InnerProduct`，区别只在 ncnn 还做了 `elempack` 打包、卷积替代、算子融合。把本章 `gemm_tiled` 换成 ncnn 的 packed kernel，就是工业版。
- **llama.cpp / ggml-vulkan**：你的「常驻权重 + 计算图 + 一次提交」正是它推理 loop 的骨架；区别在它处理的是 Transformer（注意力 + RMSNorm + 量化权重 + coopmat 加速）。你已拥有理解它源码的全部前置知识。

---

## 13.7 实战：加载真实 ncnn 模型（路径 A）

前面 13.1~13.5 用的是**随机初始化**权重，目的是证明前向链路正确。现在把 ch13 升级为**加载你导出的真实 Qwen3-ASR ncnn 子图**——这正是课程「能自研引擎跑真实模型」的终点。

### 13.7.1 模型在哪、是什么格式

真实模型位于 `F:\vulkan\model\qwen3_asr_0.6b`（Windows 默认；Linux 上请通过命令行参数指定，或设环境变量 `VULKAN_NN_MODEL_DIR` 指向模型根目录），里面是 **Qwen3-ASR 0.6B 的 ncnn 导出子图集合**（`audio_XX_*` 音频编码器、`decoder_XX_*` Transformer 解码器、`lm_head` 语言模型头，外加 `_golden_*`/`_ref_*`/`cpp_*`/`torch_*` 逐层一致性验证夹具）。

格式就是**标准 ncnn `param` + `bin`**：
- `.param` 首行 magic `7767517`，次行 `layer_count blob_count`，其后逐层 `Type name in_c out_c in_blobs... out_blobs... key=value...`。
- `.bin` 经实测是**无 blob 头的纯 fp32 权重流**：每个权重 blob 直接是 float32 数据连续排列，顺序对应 param 中权重出现顺序，大小由 param 里 `weight_data_size`（元素数）决定。

> 实测修正：`lm_head` 的 `InnerProduct` 参数 `2=155582464` 是**元素数**，155582464 = 151936 × **1024**，即 vocab=151936、hidden=1024（与你 Qwen3-ASR 的 hidden=1024 吻合），权重布局 `[numOut][numIn]` 行主。

### 13.7.2 为什么路径 A 选 `lm_head`

子图里大量使用 ncnn 没有的原生层（`Gemm`/`RMSNorm`/`Swish`/`MemoryData`/`torch.var`），属于 Qwen3-ASR 移植的自定义算子。ch13 当前引擎只认 `InnerProduct + 无 bias`，**唯一能直接跑的通子图就是 `lm_head`**（单全连接层、vocab 投影头）。它足以打通：
「标准 ncnn 解析 → fp32 权重上传常驻 → GPU GEMM → 与 CPU 参考对拍」这条最小闭环。

新增的 `model_loader.h`（header-only）就是这套解析底座；`main.cpp` 改为加载 `lm_head` 并跑 `gemm_tiled` 推理。

### 13.7.3 运行

```bash
# 默认加载 lm_head（需一块 >=1GB 显存的 GPU，权重 622MB 常驻）
./build/ch13_mlp/ch13_mlp
# 也可指定其它纯 InnerProduct 子图
./build/ch13_mlp/ch13_mlp some.ncnn.param some.ncnn.bin
```

预期输出：加载 622MB 权重、3 次推理、每次 `maxAbsDiff` 在 1e-3 量级、`argmax` 与 CPU 参考一致、总错误数 0。

### 13.7.4 下一步（路径 B / C）

- **路径 B**：扩引擎支持 `Gemm`/`RMSNorm`/`Swish`/`BinaryOp`/`Split`，即可加载 `decoder_XX_mlp` 等 SwiGLU 块，并用 `_golden_*` 做逐层对拍（复习 ch04 GEMM / ch06 逐元素 / ch07 RMSNorm）。
- **路径 C**：再补 LayerNorm 微操作簇（`MemoryData`/`Reduction`/`torch.var`/`UnaryOp`/`Reshape`/`Concat`），跑通音频编码器。

## 🎓 课程总结

从 `ch00` 枚举设备，到 `ch13` 跑通模型前向，你走完了「零基础 → 能自研推理引擎」的完整路径：

- **基础**：环境、最小计算闭环、同步、显存类型与 FP16；
- **算子**：GEMM、卷积（im2col）、逐元素、归一化、softmax/注意力；
- **工程化**：性能分析（时间戳 + Roofline）、显存池、量化融合、计算图调度；
- **综合**：用自研引擎推理小模型。

继续往深走的方向：算子融合自动生成、cooperative matrix（类 Tensor Core）、多队列流水线、动态 shape、与 CUDA 后端对比——这些都建立在你已经掌握的概念之上。祝你写出自己的「迷你 ncnn / 迷你 ggml-vulkan」。
