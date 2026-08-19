#!/usr/bin/env python3
"""
用 PyTorch 生成测试向量。

每个测试向量是一个 .pt 文件，包含字典：
  {
    "name": str,
    "description": str,
    "chapters": [str, ...],     # 适用的章节
    "params": dict,              # 算子参数
    "inputs": dict,              # 输入张量
    "expected": dict,            # 期望输出（PyTorch 参考结果）
    "tolerance": float,          # 最大允许误差
    "tolerance_rmse": float,     # 最大允许 RMSE
  }

用法:
  python tests/gen_vectors.py                    # 生成所有向量
  python tests/gen_vectors.py --filter gemm      # 只生成 GEMM 向量
  python tests/gen_vectors.py --list             # 列出所有向量
"""

import torch
import torch.nn.functional as F
import os
import sys
import json
from pathlib import Path

# 确保能导入 ref_impl
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_impl import (
    rms_norm, q4_0_quant, q4_0_dequant, q8_0_quant, q8_0_dequant,
    dequant_q2_k, dequant_q6_k, mxfp4_e2m1_quantize_value, mxfp4_dequantize_value,
    moe_router, moe_combine,
    subgroup_shuffle_sim, subgroup_broadcast_sim,
    subgroup_reduce_sum_sim, subgroup_reduce_max_sim,
    subgroup_ballot_sim, subgroup_softmax_sim,
)

VECTORS_DIR = Path(__file__).resolve().parent / "vectors"


def save_vector(subdir: str, name: str, **data):
    """保存测试向量到 .pt 文件"""
    dir_path = VECTORS_DIR / subdir
    os.makedirs(dir_path, exist_ok=True)
    filepath = dir_path / f"{name}.pt"
    torch.save(data, filepath)
    print(f"  ✓ {subdir}/{name}.pt")
    return filepath


# ============================================================
# GEMM 测试向量 (ch04, ch15)
# ============================================================

def gen_gemm_vectors():
    print("\n[GEMM]")

    # 1. 小整数 4x4
    A = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=torch.float32)
    B = torch.tensor([
        [16, 15, 14, 13],
        [12, 11, 10, 9],
        [8, 7, 6, 5],
        [4, 3, 2, 1],
    ], dtype=torch.float32)
    C_ref = torch.matmul(A, B)
    save_vector("gemm", "small_4x4",
                name="gemm_small_4x4",
                description="4x4 整数矩阵乘法",
                chapters=["ch04", "ch15"],
                params={"M": 4, "N": 4, "K": 4},
                inputs={"A": A, "B": B},
                expected={"C": C_ref},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 2. 单位矩阵
    A = torch.eye(8, dtype=torch.float32)
    B = torch.randn(8, 8, dtype=torch.float32)
    C_ref = torch.matmul(A, B)
    save_vector("gemm", "unit_eye",
                name="gemm_unit_eye",
                description="单位矩阵 × 随机矩阵，验证 A=I 时 C=B",
                chapters=["ch04", "ch15"],
                params={"M": 8, "N": 8, "K": 8},
                inputs={"A": A, "B": B},
                expected={"C": C_ref},
                tolerance=1e-5, tolerance_rmse=1e-6)

    # 3. 零矩阵
    A = torch.zeros(4, 6, dtype=torch.float32)
    B = torch.randn(6, 5, dtype=torch.float32)
    C_ref = torch.matmul(A, B)
    save_vector("gemm", "zero_input",
                name="gemm_zero_input",
                description="零矩阵乘法，验证输出全零",
                chapters=["ch04", "ch15"],
                params={"M": 4, "N": 5, "K": 6},
                inputs={"A": A, "B": B},
                expected={"C": C_ref},
                tolerance=1e-6, tolerance_rmse=1e-6)

    # 4. 不规则维度 2x8x3
    A = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                      [9, 10, 11, 12, 13, 14, 15, 16]], dtype=torch.float32)
    B = torch.tensor([[8, 7, 6],
                      [5, 4, 3],
                      [2, 1, 0],
                      [-1, -2, -3],
                      [-4, -5, -6],
                      [-7, -8, -9],
                      [10, 11, 12],
                      [13, 14, 15]], dtype=torch.float32)
    C_ref = torch.matmul(A, B)
    save_vector("gemm", "irregular_2x8x3",
                name="gemm_irregular_2x8x3",
                description="非整数倍 workgroup 尺寸的 GEMM",
                chapters=["ch04", "ch15"],
                params={"M": 2, "N": 3, "K": 8},
                inputs={"A": A, "B": B},
                expected={"C": C_ref},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 5. 大随机矩阵 (分段测试)
    torch.manual_seed(42)
    A = torch.randn(64, 128, dtype=torch.float32)
    B = torch.randn(128, 32, dtype=torch.float32)
    C_ref = torch.matmul(A, B)
    save_vector("gemm", "random_64x128x32",
                name="gemm_random_64x128x32",
                description="较大随机矩阵，覆盖典型推理场景",
                chapters=["ch04", "ch15"],
                params={"M": 64, "N": 32, "K": 128},
                inputs={"A": A, "B": B},
                expected={"C": C_ref},
                tolerance=1e-4, tolerance_rmse=1e-5)


# ============================================================
# 卷积测试向量 (ch05)
# ============================================================

def gen_conv_vectors():
    print("\n[卷积]")

    torch.manual_seed(7)

    # 1. 简单 3x3 卷积, 1x1 输入输出
    x = torch.tensor([[[[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20],
                         [21, 22, 23, 24, 25]]]], dtype=torch.float32)
    w = torch.tensor([[[[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]]]], dtype=torch.float32)
    ref = F.conv2d(x, w, padding=1)
    save_vector("conv", "sobel_3x3",
                name="conv_sobel_3x3",
                description="Sobel 边缘检测 3x3 卷积核",
                chapters=["ch05"],
                params={"N": 1, "C": 1, "H": 5, "W": 5, "OC": 1, "KH": 3, "KW": 3, "padding": 1},
                inputs={"x": x, "w": w},
                expected={"out": ref},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 2. 多通道 3x3 卷积
    x = torch.randn(1, 3, 8, 8, dtype=torch.float32)
    w = torch.randn(4, 3, 3, 3, dtype=torch.float32)
    ref = F.conv2d(x, w, padding=1)
    save_vector("conv", "multi_channel_3x3",
                name="conv_multi_channel_3x3",
                description="多输入多输出通道的 3x3 卷积",
                chapters=["ch05"],
                params={"N": 1, "C": 3, "H": 8, "W": 8, "OC": 4, "KH": 3, "KW": 3, "padding": 1},
                inputs={"x": x, "w": w},
                expected={"out": ref},
                tolerance=1e-4, tolerance_rmse=1e-5)

    # 3. 1x1 卷积 (等同于矩阵乘)
    x = torch.randn(1, 16, 4, 4, dtype=torch.float32)
    w = torch.randn(8, 16, 1, 1, dtype=torch.float32)
    ref = F.conv2d(x, w)
    save_vector("conv", "pointwise_1x1",
                name="conv_pointwise_1x1",
                description="1x1 逐点卷积，等价于 GEMM",
                chapters=["ch05"],
                params={"N": 1, "C": 16, "H": 4, "W": 4, "OC": 8, "KH": 1, "KW": 1},
                inputs={"x": x, "w": w},
                expected={"out": ref},
                tolerance=1e-4, tolerance_rmse=1e-5)


# ============================================================
# Softmax 测试向量 (ch08)
# ============================================================

def gen_softmax_vectors():
    print("\n[Softmax]")

    # 1. 简单 4 元素
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ref = F.softmax(x, dim=0)
    save_vector("softmax", "simple_4",
                name="softmax_simple_4",
                description="简单 4 元素 softmax",
                chapters=["ch08", "ch17"],
                params={"N": 4, "dim": 0},
                inputs={"x": x},
                expected={"out": ref},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 2. 数值稳定性测试: 大值
    x = torch.tensor([1000.0, 1010.0, 1020.0, 1030.0])
    ref = F.softmax(x, dim=0)
    save_vector("softmax", "large_values",
                name="softmax_large_values",
                description="大数值 softmax，验证数值稳定性（减最大值）",
                chapters=["ch08", "ch17"],
                params={"N": 4, "dim": 0},
                inputs={"x": x},
                expected={"out": ref},
                tolerance=1e-3, tolerance_rmse=1e-3)

    # 3. 2D 矩阵 softmax (dim=1)
    x = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=torch.float32)
    ref = F.softmax(x, dim=1)
    save_vector("softmax", "matrix_3x3",
                name="softmax_matrix_3x3",
                description="3x3 矩阵行 softmax",
                chapters=["ch08", "ch17"],
                params={"M": 3, "N": 3, "dim": 1},
                inputs={"x": x},
                expected={"out": ref},
                tolerance=1e-5, tolerance_rmse=1e-5)


# ============================================================
# 归一化测试向量 (ch07)
# ============================================================

def gen_norm_vectors():
    print("\n[归一化]")

    torch.manual_seed(17)

    # 1. LayerNorm: 简单 2D
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0]])
    weight = torch.ones(4, dtype=torch.float32)
    bias = torch.zeros(4, dtype=torch.float32)
    ref = F.layer_norm(x, [4], weight, bias, 1e-5)
    save_vector("norm", "layernorm_2x4",
                name="norm_layernorm_2x4",
                description="LayerNorm 2x4 简单输入",
                chapters=["ch07"],
                params={"shape": [2, 4], "normalized_shape": [4], "eps": 1e-5},
                inputs={"x": x, "weight": weight, "bias": bias},
                expected={"out": ref},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 2. RMSNorm: 2x4
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0]])
    weight = torch.ones(4, dtype=torch.float32)
    ref = rms_norm(x, weight, 1e-5)
    save_vector("norm", "rmsnorm_2x4",
                name="norm_rmsnorm_2x4",
                description="RMSNorm 2x4 简单输入",
                chapters=["ch07"],
                params={"shape": [2, 4], "eps": 1e-5},
                inputs={"x": x, "weight": weight},
                expected={"out": ref},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 3. RMSNorm: 随机大向量
    x = torch.randn(4, 128, dtype=torch.float32)
    weight = torch.randn(128, dtype=torch.float32)
    ref = rms_norm(x, weight, 1e-5)
    save_vector("norm", "rmsnorm_4x128",
                name="norm_rmsnorm_4x128",
                description="RMSNorm 4x128 随机输入",
                chapters=["ch07"],
                params={"shape": [4, 128], "eps": 1e-5},
                inputs={"x": x, "weight": weight},
                expected={"out": ref},
                tolerance=1e-4, tolerance_rmse=1e-5)


# ============================================================
# 注意力测试向量 (ch08, ch17)
# ============================================================

def gen_attention_vectors():
    print("\n[注意力]")

    torch.manual_seed(42)

    # 1. 简单注意力: N=4, D=8
    N, D = 4, 8
    Q = torch.randn(N, D, dtype=torch.float32)
    K = torch.randn(N, D, dtype=torch.float32)
    V = torch.randn(N, D, dtype=torch.float32)
    ref = F.scaled_dot_product_attention(Q, K, V, scale=1.0 / (D ** 0.5))
    save_vector("attention", "simple_4x8",
                name="attn_simple_4x8",
                description="4 token, 8 维的简单注意力",
                chapters=["ch08", "ch17"],
                params={"N": N, "D": D},
                inputs={"Q": Q, "K": K, "V": V},
                expected={"out": ref},
                tolerance=1e-4, tolerance_rmse=1e-5)

    # 2. 因果注意力
    N, D = 8, 16
    Q = torch.randn(N, D, dtype=torch.float32)
    K = torch.randn(N, D, dtype=torch.float32)
    V = torch.randn(N, D, dtype=torch.float32)
    is_causal = True
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, scale=1.0 / (D ** 0.5))
    save_vector("attention", "causal_8x16",
                name="attn_causal_8x16",
                description="因果注意力（LLM Decode 场景）",
                chapters=["ch08", "ch17"],
                params={"N": N, "D": D, "is_causal": True},
                inputs={"Q": Q, "K": K, "V": V},
                expected={"out": ref},
                tolerance=1e-4, tolerance_rmse=1e-5)

    # 3. FlashAttention 场景: 较长序列, 小维度
    N, D = 32, 8
    Q = torch.randn(N, D, dtype=torch.float32)
    K = torch.randn(N, D, dtype=torch.float32)
    V = torch.randn(N, D, dtype=torch.float32)
    ref = F.scaled_dot_product_attention(Q, K, V, scale=1.0 / (D ** 0.5))
    save_vector("attention", "flash_32x8",
                name="attn_flash_32x8",
                description="较长序列，适合 FlashAttention 分块策略",
                chapters=["ch08", "ch17"],
                params={"N": N, "D": D},
                inputs={"Q": Q, "K": K, "V": V},
                expected={"out": ref},
                tolerance=1e-4, tolerance_rmse=1e-5)


# ============================================================
# 量化测试向量 (ch11, ch18)
# ============================================================

def gen_quant_vectors():
    print("\n[量化]")

    torch.manual_seed(99)

    # 1. int8 量化: 简单整数
    x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=torch.float32)
    save_vector("quant", "simple_int8",
                name="quant_simple_int8",
                description="简单 int8 量化/反量化",
                chapters=["ch11"],
                params={"block_size": 8},
                inputs={"x": x},
                expected={"x": x},  # 反量化后应恢复原值
                tolerance=1.0,  # int8 量化有精度损失
                tolerance_rmse=0.5)

    # 2. Q4_0 量化: 32 元素
    x = torch.linspace(-7, 7, 32, dtype=torch.float32)
    qdata, scales, block_size = q4_0_quant(x)
    x_deq = q4_0_dequant(qdata, scales, 32, block_size)
    save_vector("quant", "q4_0_linspace",
                name="quant_q4_0_linspace",
                description="Q4_0 量化/反量化 linspace(-7,7,32)",
                chapters=["ch11", "ch18"],
                params={"block_size": 32, "n": 32},
                inputs={"x": x},
                expected={"x_deq": x_deq},
                tolerance=5.0,  # int4 精度损失较大
                tolerance_rmse=2.0)

    # 3. Q8_0 量化: 32 元素
    x = torch.linspace(-127, 127, 64, dtype=torch.float32)
    qdata, scales, block_size = q8_0_quant(x)
    x_deq = q8_0_dequant(qdata, scales, 64, block_size)
    save_vector("quant", "q8_0_linspace",
                name="quant_q8_0_linspace",
                description="Q8_0 量化/反量化 linspace(-127,127,64)",
                chapters=["ch11", "ch18"],
                params={"block_size": 32, "n": 64},
                inputs={"x": x},
                expected={"x_deq": x_deq},
                tolerance=1.0,  # int8 精度较好
                tolerance_rmse=0.5)

    # 4. 随机浮点量化: Q4_0 + Q8_0
    x = torch.randn(128, dtype=torch.float32) * 10.0
    q4_data, q4_scales, _ = q4_0_quant(x)
    x_q4_deq = q4_0_dequant(q4_data, q4_scales, 128)
    q8_data, q8_scales, _ = q8_0_quant(x)
    x_q8_deq = q8_0_dequant(q8_data, q8_scales, 128)
    save_vector("quant", "random_128",
                name="quant_random_128",
                description="128 个随机浮点数 Q4_0 + Q8_0 量化对比",
                chapters=["ch11", "ch18"],
                params={"n": 128, "block_size": 32},
                inputs={"x": x},
                expected={"x_q4_deq": x_q4_deq, "x_q8_deq": x_q8_deq},
                tolerance=20.0,
                tolerance_rmse=5.0)


# ============================================================
# 深层量化测试向量 (ch20)
# ============================================================

def gen_quant_deep_vectors():
    print("\n[深层量化]")

    torch.manual_seed(42)

    # 1. MXFP4 编解码
    test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -2.0, -4.0]
    codes = [mxfp4_e2m1_quantize_value(v) for v in test_values]
    decoded = [mxfp4_dequantize_value(c) for c in codes]
    save_vector("quant_deep", "mxfp4_encoding",
                name="quant_deep_mxfp4_encoding",
                description="MXFP4 E2M1 编码/解码测试",
                chapters=["ch20"],
                params={"n_values": len(test_values)},
                inputs={"test_values": torch.tensor(test_values)},
                expected={"decoded": torch.tensor(decoded)},
                tolerance=1e-5,
                tolerance_rmse=1e-5)

    # 2. K-quant 反量化（简化版 Q2_K）
    # 构造一个简单的 256 元素向量，Q2_K 格式
    x = torch.linspace(-3, 3, 256, dtype=torch.float32)
    # 模拟 Q2_K 量化（简化）
    n_blocks = 1
    d = torch.tensor([0.1], dtype=torch.float16)
    scales = torch.ones(16, dtype=torch.uint8) * 4  # 4/4 = 1.0
    qs = torch.zeros(64, dtype=torch.uint8)
    for i in range(64):
        # 每 4 个元素打包到 1 个 byte
        for j in range(4):
            idx = i * 4 + j
            if idx < 256:
                val = max(-2, min(1, int(round(x[idx] / 0.1))))  # int2
                val = val & 0x03
                qs[i] = qs[i] | (val << (j * 2))
    qdata = {"d": d, "scales": scales.unsqueeze(0), "qs": qs.unsqueeze(0)}
    ref = dequant_q2_k(qdata, 256)
    save_vector("quant_deep", "q2_k_256",
                name="quant_deep_q2_k_256",
                description="Q2_K 反量化 256 元素",
                chapters=["ch20"],
                params={"n": 256, "format": "Q2_K"},
                inputs={"qdata": qdata},
                expected={"out": ref},
                tolerance=1.0,
                tolerance_rmse=0.5)


# ============================================================
# 激活函数测试向量 (ch06)
# ============================================================

def gen_elemwise_vectors():
    print("\n[激活函数]")

    # 1. ReLU
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    ref = F.relu(x)
    save_vector("elemwise", "relu",
                name="elemwise_relu",
                description="ReLU 激活函数",
                chapters=["ch06"],
                params={"n": 7},
                inputs={"x": x},
                expected={"out": ref},
                tolerance=1e-6, tolerance_rmse=1e-6)

    # 2. GELU
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    ref = F.gelu(x)
    save_vector("elemwise", "gelu",
                name="elemwise_gelu",
                description="GELU 激活函数",
                chapters=["ch06"],
                params={"n": 7},
                inputs={"x": x},
                expected={"out": ref},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 3. SiLU
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    ref = F.silu(x)
    save_vector("elemwise", "silu",
                name="elemwise_silu",
                description="SiLU (Swish) 激活函数",
                chapters=["ch06"],
                params={"n": 7},
                inputs={"x": x},
                expected={"out": ref},
                tolerance=1e-5, tolerance_rmse=1e-5)


# ============================================================
# MoE 测试向量 (ch19)
# ============================================================

def gen_moe_vectors():
    print("\n[MoE]")

    torch.manual_seed(42)

    # 1. 简单路由: 4 token, 8 expert, top-2
    n_tokens, n_experts, top_k = 4, 8, 2
    logits = torch.randn(n_tokens, n_experts, dtype=torch.float32)
    indices, weights = moe_router(logits, top_k)
    save_vector("moe", "router_4x8_top2",
                name="moe_router_4x8_top2",
                description="4 token, 8 expert, top-2 路由",
                chapters=["ch19"],
                params={"n_tokens": n_tokens, "n_experts": n_experts, "top_k": top_k},
                inputs={"logits": logits},
                expected={"indices": indices, "weights": weights},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 2. Combine: 加权合并
    d_model = 16
    expert_outputs = torch.randn(n_tokens, n_experts, d_model, dtype=torch.float32)
    combined = moe_combine(expert_outputs, indices, weights, n_experts)
    save_vector("moe", "combine_4x8x16",
                name="moe_combine_4x8x16",
                description="4 token, 8 expert, 16 维的合并",
                chapters=["ch19"],
                params={"n_tokens": n_tokens, "n_experts": n_experts,
                        "d_model": d_model, "top_k": top_k},
                inputs={"expert_outputs": expert_outputs, "indices": indices, "weights": weights},
                expected={"combined": combined},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 3. 完整管线: logits → router → expert → combine
    logits_full = torch.randn(2, 4, dtype=torch.float32)
    idx_full, w_full = moe_router(logits_full, 2)
    expert_out_full = torch.randn(2, 4, 8, dtype=torch.float32)
    combined_full = moe_combine(expert_out_full, idx_full, w_full, 4)
    save_vector("moe", "full_pipeline_2x4x8",
                name="moe_full_pipeline_2x4x8",
                description="完整 MoE 管线: router → expert → combine",
                chapters=["ch19"],
                params={"n_tokens": 2, "n_experts": 4, "d_model": 8, "top_k": 2},
                inputs={"logits": logits_full, "expert_outputs": expert_out_full},
                expected={"indices": idx_full, "weights": w_full, "combined": combined_full},
                tolerance=1e-5, tolerance_rmse=1e-5)


# ============================================================
# Subgroup 测试向量 (ch16)
# ============================================================

def gen_subgroup_vectors():
    print("\n[Subgroup]")

    # 1. Shuffle / Broadcast
    x = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    sg_size = 8
    shuffled = subgroup_shuffle_sim(x, 3, sg_size)  # 所有线程拿 lane 3 的值
    broadcast = subgroup_broadcast_sim(x, 0, sg_size)  # 所有线程拿 lane 0 的值
    save_vector("subgroup", "shuffle_broadcast",
                name="subgroup_shuffle_broadcast",
                description="Subgroup shuffle (src=3) + broadcast (src=0)",
                chapters=["ch16"],
                params={"subgroup_size": sg_size, "src_lane_shuffle": 3, "src_lane_broadcast": 0},
                inputs={"x": x},
                expected={"shuffled": shuffled, "broadcast": broadcast},
                tolerance=1e-6, tolerance_rmse=1e-6)

    # 2. Reduce + Ballot
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    sg_size = 8
    sum_val = subgroup_reduce_sum_sim(x, sg_size)
    max_val = subgroup_reduce_max_sim(x, sg_size)
    predicate = x > 4.0
    ballot = subgroup_ballot_sim(predicate, sg_size)
    softmax_out = subgroup_softmax_sim(x, sg_size)
    save_vector("subgroup", "reduce_ballot_softmax",
                name="subgroup_reduce_ballot_softmax",
                description="Subgroup reduce sum/max + ballot + softmax",
                chapters=["ch16"],
                params={"subgroup_size": sg_size},
                inputs={"x": x},
                expected={"sum_val": sum_val, "max_val": max_val,
                          "ballot": torch.tensor(ballot, dtype=torch.int32),
                          "softmax": softmax_out},
                tolerance=1e-5, tolerance_rmse=1e-5)


# ============================================================
# Cooperative Matrix 测试向量 (ch15)
# ============================================================

def gen_coopmat_vectors():
    print("\n[Cooperative Matrix]")

    # Cooperative Matrix 结果应与 GEMM 相同，所以复用 GEMM 向量
    # 这里只添加 CM 特有的测试

    # 1. 小矩阵，验证 CM 结果与 matmul 一致
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    C_ref = torch.matmul(A, B)
    save_vector("coopmat", "small_2x2",
                name="coopmat_small_2x2",
                description="2x2 矩阵 CM vs matmul 一致性",
                chapters=["ch15"],
                params={"M": 2, "N": 2, "K": 2},
                inputs={"A": A, "B": B},
                expected={"C": C_ref},
                tolerance=1e-5, tolerance_rmse=1e-5)

    # 2. CM 典型尺寸: 16x16
    torch.manual_seed(1)
    A = torch.randn(16, 16, dtype=torch.float32)
    B = torch.randn(16, 16, dtype=torch.float32)
    C_ref = torch.matmul(A, B)
    save_vector("coopmat", "random_16x16",
                name="coopmat_random_16x16",
                description="16x16 随机矩阵 CM 验证",
                chapters=["ch15"],
                params={"M": 16, "N": 16, "K": 16},
                inputs={"A": A, "B": B},
                expected={"C": C_ref},
                tolerance=1e-4, tolerance_rmse=1e-5)


# ============================================================
# 主入口
# ============================================================

def generate_all():
    """生成所有测试向量"""
    print("生成测试向量...")
    print(f"目录: {VECTORS_DIR}")

    gen_gemm_vectors()
    gen_conv_vectors()
    gen_softmax_vectors()
    gen_norm_vectors()
    gen_attention_vectors()
    gen_quant_vectors()
    gen_quant_deep_vectors()
    gen_elemwise_vectors()
    gen_moe_vectors()
    gen_subgroup_vectors()
    gen_coopmat_vectors()

    # 统计
    total = 0
    for root, dirs, files in os.walk(VECTORS_DIR):
        for f in files:
            if f.endswith('.pt'):
                total += 1
    print(f"\n{'=' * 40}")
    print(f"共生成 {total} 个测试向量")
    return total


def list_all():
    """列出所有测试向量"""
    total = 0
    for root, dirs, files in os.walk(VECTORS_DIR):
        if not files:
            continue
        rel = os.path.relpath(root, VECTORS_DIR)
        print(f"\n[{rel}]")
        for f in sorted(files):
            if f.endswith('.pt'):
                filepath = os.path.join(root, f)
                data = torch.load(filepath, weights_only=True)
                print(f"  {f}: {data.get('description', '')}")
                total += 1
    print(f"\n共 {total} 个测试向量")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="生成测试向量")
    parser.add_argument("--filter", type=str, default=None, help="按算子过滤")
    parser.add_argument("--list", action="store_true", help="列出所有向量")
    args = parser.parse_args()

    if args.list:
        list_all()
    else:
        # 清空目录
        import shutil
        if VECTORS_DIR.exists():
            shutil.rmtree(VECTORS_DIR)
        VECTORS_DIR.mkdir(parents=True, exist_ok=True)

        generators = [
            ("gemm", gen_gemm_vectors),
            ("conv", gen_conv_vectors),
            ("softmax", gen_softmax_vectors),
            ("norm", gen_norm_vectors),
            ("attention", gen_attention_vectors),
            ("quant", gen_quant_vectors),
            ("quant_deep", gen_quant_deep_vectors),
            ("elemwise", gen_elemwise_vectors),
            ("moe", gen_moe_vectors),
            ("subgroup", gen_subgroup_vectors),
            ("coopmat", gen_coopmat_vectors),
        ]

        if args.filter:
            generators = [g for g in generators if args.filter in g[0]]
            if not generators:
                print(f"未找到匹配 '{args.filter}' 的生成器")
                sys.exit(1)

        for name, gen in generators:
            gen()

        total = 0
        for root, dirs, files in os.walk(VECTORS_DIR):
            for f in files:
                if f.endswith('.pt'):
                    total += 1
        print(f"\n共 {total} 个测试向量")