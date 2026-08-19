# 第 1 章 · 第一个计算着色器：向量加法

> 目标：跑通「一条 GPU 算子」的完整闭环。**不要追求理解每一行 Vulkan 调用**，先把「建 buffer → 上传 → 建管线 → dispatch → 读回」这条主链路刻进脑子里。

---

## 1.1 为什么从向量加法开始

神经网络里几乎所有运算最终都是「对一大坨数做并行计算」：矩阵乘是嵌套的并行，卷积是带偏移的并行，激活函数是对每个元素并行。向量加法（逐元素并行）是其中最简单的一种，足以让你看清 GPU 计算的骨架，而不会被算子数学细节干扰。

## 1.2 一条算子在 Vulkan 里的生命周期

```
CPU 数据 ──上传──▶ GPU buffer（storage buffer）
                        │
            绑定管线 + 描述符 + push constant
                        │
                 vkCmdDispatch（启动 N 个线程）
                        │
            GPU 并行执行 compute shader（每个线程算一个元素）
                        │
            结果留在 GPU buffer ──读回──▶ CPU 校验
```

对应到 `main.cpp` 的几个步骤：

1. **建 buffer**：`vkCreateBuffer` 三个 buffer（a、b、c），用途都带 `eStorageBuffer`（供 shader 读写）和 `eTransferSrc/Dst`（供上传/读回）。内存选 `eDeviceLocal`（设备本地显存，最快）。
2. **上传**：`vkUpload` 把 CPU 的 `a`、`b` 拷进 GPU。device-local 无法直接被 CPU 写，common 库内部会自动用一块 host-visible 的 staging buffer 中转（这个中转机制会在 **ch03** 详细讲）。
3. **建管线**：`vkLoadShader` 读 `.spv` → `createShaderModule` → `createComputePipeline`。管线里包含「描述符布局」（说明 shader 要几个 buffer）和「push constant 范围」（说明能压入哪些常量）。
4. **绑描述符**：`updateDescriptorSets` 把真实 buffer 绑到 shader 的 `binding 0/1/2`。这一步相当于「告诉 shader：a 在 0 号、b 在 1 号、c 在 2 号」。
5. **dispatch**：工作组数 = `ceil(N/64)`，因为一个工作组 64 个线程（`vec_add.comp` 里 `local_size_x = 64`）。
6. **读回 + 校验**：`vkDownload` 把结果拷回 CPU，逐个比对 `c[i] == a[i]+b[i]`。

## 1.3 两个关键概念（现在先记住，后面深挖）

- **工作组（workgroup）**：`dispatch(groups,1,1)` 启动 `groups` 个工作组，每个工作组 64 线程。全局线程号 `gl_GlobalInvocationID.x` 在 0..N-1 之间，天然就是元素下标。GPU 按「工作组」调度，同一工作组内的线程能共享共享内存（ch04 才用得上）。
- **push constant vs 描述符**：形状参数（这里是一个 `n`）用 push constant 更轻量；真正的大块数据（buffer）才用描述符。ncnn / llama.cpp 都大量用 push constant 传形状。

## 1.4 动手实验

1. 打开 Validation Layers 编译运行，确认「错误数=0」。
2. 把 `N` 改成不是 64 整数倍的数（如 1000），验证 `if (i >= n) return;` 的边界保护是必需的——去掉它会发生什么？（会写到越界地址，Validation Layers 可能报、也可能静默算错）
3. 把 `local_size_x` 改成 32 或 256，相应调整 `groups` 公式 `(N+local-1)/local`，重跑。感受「工作组尺寸」是可调的。
4. 试着把加法改成乘法或 `a[i]*2 + 1`，只需要改 `.comp` 再重新编译，不需要动 C++（这就是声明式着色器的好处）。

## 1.5 对照工业级

- ncnn 的向量加法对应它的「逐元素算子」（`Elementwise` 系列 GPU kernel），但 ncnn 会顺手做 **elempack=4 打包**（一次处理 4 个元素）并融合进相邻算子。
- llama.cpp 里没有独立「向量加法」，类似的逐元素运算被融进 `mul_f32` / `add_f32` 等 kernel，并通过 `vk_op_binary_push_constants` 传形状。

下一章（ch02）我们不急着加新算子，而是**把同步（fence / barrier）这件事讲透**——它是 GPU 计算里最容易被忽略、也最容易出 bug 的地方。
