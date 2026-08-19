# 测试向量系统 实施计划 v2

## 一、问题

当前测试框架仅依赖每个章节内部的「自验证」：
1. `main.cpp` 生成随机数据
2. 运行 CPU 参考实现
3. 运行 GPU kernel
4. 对比输出，打印 `errors=0`

**缺陷**：没有独立于 C++ 实现的测试向量，无法验证 CPU 参考实现本身是否正确，也无法在不运行 GPU 的情况下验证算子逻辑。

## 二、设计方案

### 2.1 核心思路：PyTorch + 生产库优先

**不再手写 NumPy 参考实现**，而是用 PyTorch 作为基础，并优先使用生产库中的实现：

| 算子 | 参考来源（优先级链） | chXX |
|------|---------------------|------|
| GEMM (矩阵乘) | `torch.matmul` | ch04, ch15 |
| 卷积 | `torch.nn.functional.conv2d` | ch05 |
| 激活函数 | `torch.relu` / `F.gelu` / `F.silu` | ch06 |
| LayerNorm | `torch.nn.functional.layer_norm` | ch07 |
| RMSNorm | `transformers.LlamaRMSNorm` > `vLLM.RMSNorm` > 手工 | ch07 |
| Softmax | `torch.nn.functional.softmax` | ch08 |
| 注意力 | `torch.nn.functional.scaled_dot_product_attention` | ch08, ch17 |
| Q4_0/Q8_0 量化 | ggml 特有格式，始终手工实现 | ch11, ch18 |
| MoE 路由 | `transformers.MixtralSparseMoeBlock` > 手工 | ch19 |
| Subgroup 操作 | GPU 原语，始终手工模拟 | ch16 |
| K-quant/MXFP4 | ggml 特有格式，始终手工实现 | ch20 |

**用 PyTorch + 生产库的好处**：
- 零额外编写参考实现工作量（PyTorch 本身就是参考）
- 生产库（transformers/vLLM）中的实现经过大规模验证，比手工实现更可靠
- 生产库不可用时自动降级到手工实现，无需用户干预
- PyTorch CUDA 可作为第二 GPU 基线，与 Vulkan 结果交叉验证
- 测试向量可动态生成（无需手写 JSON）
- 数值容差评估更直接（PyTorch FP32 vs Vulkan FP32）

### 2.2 架构

```
tests/
  ├── plan.md                   ← 本文件
  ├── test_framework.py         ← 已有，运行 C++ 可执行文件
  ├── vector_verify.py          ← 向量验证器（新增）
  ├── gen_vectors.py            ← 用 PyTorch 生成测试向量（新增）
  ├── vectors/                  ← 生成的测试向量（.pt 或 .npz 文件）
  └── ref_impl.py               ← 少量需手工实现的参考（RMSNorm、量化、MoE）
```

### 2.3 测试向量生成（gen_vectors.py）

不再手写 JSON，而是用 PyTorch 生成测试向量并保存：

```python
# gen_vectors.py 核心逻辑
def gen_gemm_vectors():
    M, N, K = 4, 4, 4
    A = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], dtype=torch.float32)
    B = torch.tensor([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]], dtype=torch.float32)
    C_ref = torch.matmul(A, B)  # PyTorch 参考
    save_vector("gemm/small_4x4", A=A, B=B, C_ref=C_ref)
```

**生成多组向量**，覆盖边界情况：

| 类型 | 说明 |
|------|------|
| 小整数 | 方便人工验算 |
| 随机浮点 | 覆盖典型范围 |
| 零/单位矩阵 | 边界测试 |
| 极端值 | 大数、小数、NaN 处理 |
| 不规则维度 | 非平方矩阵 |

### 2.4 向量验证器工作流程（vector_verify.py）

```
vector_verify.py
  │
  ├─ 模式 1: --gen-only         ← 生成/更新测试向量
  │   调用 gen_vectors.py 生成所有向量
  │   保存到 tests/vectors/*.pt
  │
  ├─ 模式 2: --ref-only         ← 只验证 PyTorch 参考（无 GPU 依赖，CI 可用）
  │   加载 tests/vectors/*.pt
  │   → 对每个向量，重新用 PyTorch 计算
  │   → 与保存的 C_ref 对比
  │   → 输出 PASS/FAIL（回归测试）
  │
  ├─ 模式 3: --gpu-only         ← 对比 Vulkan 输出 vs PyTorch 参考
  │   加载 tests/vectors/*.pt
  │   → 将输入写入临时文件
  │   → 调用 chXX 可执行文件（带 --test-vectors 参数）
  │   → 读取 Vulkan 输出
  │   → 与 PyTorch C_ref 对比
  │   → 输出 PASS/FAIL 及 max_diff
  │
  ├─ 模式 4: --cuda-baseline    ← 对比 Vulkan vs PyTorch CUDA（可选）
  │   与模式 3 相同，但增加 PyTorch CUDA 基线
  │   帮助区分「Vulkan 实现错误」vs「GPU 浮点舍入差异」
  │
  └─ 模式 5: (默认) mode 1 + 2 + 3 依次执行
```

### 2.5 需要修改 C++ 章节

为了让 GPU 接受外部测试向量，每个章节需新增：

```cpp
// 新增命令行参数
--test-vectors <input.raw>   // 读取 PyTorch 导出的二进制输入
--output <output.raw>        // 导出 GPU 计算结果供 PyTorch 读取
```

**格式约定**：简化的二进制格式（不依赖 JSON，避免解析开销）：
- 前 4 字节：维度数 (uint32)
- 后续 4×N 字节：各维度大小 (uint32 × N)
- 后续：float32 数据

**但这是一个较大的改动（21 章节都要改，约 30 行/章节）。**

**替代方案（推荐）**：先做模式 1 + 2，实现「无 GPU 的参考验证」。模式 3 作为后续扩展，暂不修改 C++ 代码。

### 2.6 验证流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. gen_vectors.py         生成测试向量                     │
│     └─ PyTorch 计算参考     →  tests/vectors/*.pt          │
│                                                             │
│  2. vector_verify.py --ref-only  验证 PyTorch 参考一致性    │
│     └─ 重新加载向量 → 重新计算 → 对比 → 回归测试           │
│                                                             │
│  3. cmake --build         构建 C++ 章节                     │
│                                                             │
│  4. test_framework.py     运行 C++ 可执行文件（自验证）     │
│     └─ 加载测试向量 → 运行 GPU → 检查 errors=0             │
│                                                             │
│  5. (可选) vector_verify.py --gpu-only  对比 Vulkan 输出    │
│     └─ 加载向量 → 调用 chXX --test-vectors → 对比 PyTorch  │
└─────────────────────────────────────────────────────────────┘
```

### 2.7 优先级

| 优先级 | 算子 | 向量数 | PyTorch 参考实现工作量 |
|--------|------|--------|----------------------|
| **P0** | GEMM | 5 | 0 行（`torch.matmul`） |
| **P0** | Softmax | 3 | 0 行（`F.softmax`） |
| **P0** | RMSNorm | 3 | ~10 行手工实现 |
| **P0** | 注意力 | 3 | 0 行（`F.scaled_dot_product_attention`） |
| **P0** | Q4_0/Q8_0 量化 | 4 | ~30 行手工实现 |
| **P1** | 卷积 (Direct+Winograd) | 3 | 0 行（`F.conv2d`） |
| **P1** | 激活函数 (ReLU/GELU/SiLU) | 2 | 0 行（`F.relu`/`F.gelu`/`F.silu`） |
| **P1** | LayerNorm | 2 | 0 行（`F.layer_norm`） |
| **P1** | MoE 路由 | 3 | ~20 行手工实现 |
| **P2** | Subgroup 操作 | 2 | ~20 行手工实现 |
| **P2** | Cooperative Matrix | 2 | 0 行（`torch.matmul`） |
| **P2** | K-quant/MXFP4 | 3 | ~40 行手工实现 |

## 三、实施步骤

### Step 1: 创建目录结构

```
tests/vectors/         ← 测试向量存储目录
```

### Step 2: 实现 gen_vectors.py

用 PyTorch 生成所有测试向量，保存为 `.pt` 文件。

### Step 3: 实现 ref_impl.py

参考实现，生产库优先 + 手工降级：
- RMSNorm：尝试 `transformers.LlamaRMSNorm` → `vLLM.RMSNorm` → 手工（~10 行）
- MoE 路由：尝试 `transformers.MixtralSparseMoeBlock` → 手工（~20 行）
- Q4_0/Q8_0 量化/反量化（~30 行）— ggml 特有，始终手工
- K-quant/MXFP4（~40 行）— ggml 特有，始终手工
- Subgroup 模拟（~20 行）— GPU 原语，始终手工模拟

### Step 4: 实现 vector_verify.py

验证器主入口，支持 `--gen-only`、`--ref-only`、`--gpu-only` 模式。

### Step 5: 集成到 test_framework.py

新增 `--verify` 模式，调用 `vector_verify.py`。

## 四、工作量估算

| 步骤 | 文件 | 代码行数（估算） | 依赖 |
|------|------|-----------------|------|
| gen_vectors.py | 1 | 150 | PyTorch |
| ref_impl.py | 1 | 120 | PyTorch + NumPy |
| vector_verify.py | 1 | 200 | 上述两个模块 |
| 测试向量数据 | 自动生成 | 0 | 无 |
| 集成测试框架 | 修改 1 处 | 50 | 无 |
| **合计** | 3 个新文件 + 1 处修改 | ~520 行 | 仅 PyTorch |

## 五、C++ 端后续扩展（可选）

如果后续需要 GPU 直接加载外部测试向量，每章修改 ~30 行：
- 解析 `--test-vectors` / `--output` 参数
- 从二进制文件读取输入
- 写入 GPU buffer
- 运行 dispatch
- 读取 GPU 输出到文件
- 与 PyTorch 参考对比

**建议先做 Python 端验证，C++ 端按需扩展。**