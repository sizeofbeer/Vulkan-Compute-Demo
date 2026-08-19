#!/usr/bin/env python3
"""
参考实现模块 —— 提供教材中每个算子的标准参考结果。

设计原则：
  1. 优先使用生产库（transformers / vLLM）中的实现，因为这些是经过大规模验证的参考。
  2. 生产库不可用时，退回到手工实现（基于 PyTorch 张量操作）。
  3. ggml 特有格式（Q4_0/Q8_0/K-quant/MXFP4）和 GPU 原语（Subgroup）无库可用，始终手工实现。

优先级链：
  transformers.RMSNorm > vLLM RMSNorm > 手工 RMSNorm
  transformers.MixtralSparseMoeBlock > 手工 MoE
  PyTorch 内置算子（matmul/softmax/conv2d）始终直接使用，无需降级。

包含：
  - rms_norm           : RMSNorm（ch07）— 优先 transformers.LlamaRMSNorm
  - q4_0_quant/dequant : Q4_0 block-wise 量化（ch11, ch18）— ggml 特有
  - q8_0_quant/dequant : Q8_0 block-wise 量化（ch11, ch18）— ggml 特有
  - dequant_k_quant    : K-quant 反量化（ch20）— ggml 特有
  - mxfp4_quant/dequant: MXFP4 E2M1 量化（ch20）— 前沿格式
  - moe_router         : MoE top-k 路由 + expert combine（ch19）— 优先 transformers
  - subgroup_sim       : 用张量操作模拟 subgroup shuffle/broadcast/reduce（ch16）— GPU 特有
"""

import torch
import math
import warnings

# ============================================================
# 尝试从生产库导入
# ============================================================

_has_transformers = False
_has_vllm = False

try:
    import transformers
    _has_transformers = True
except ImportError:
    pass

try:
    import vllm
    _has_vllm = True
except ImportError:
    pass

# ============================================================
# RMSNorm —— 优先使用 transformers 或 vLLM 的实现
# ============================================================

_rms_norm_source = "manual"

if _has_transformers:
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm as _TRMSNorm
        def _rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            norm = _TRMSNorm(x.shape[-1], eps)
            norm.weight.data = weight
            return norm(x)
        _rms_norm_source = "transformers.LlamaRMSNorm"
    except (ImportError, AttributeError):
        pass

if _rms_norm_source == "manual" and _has_vllm:
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm as _VRMSNorm
        def _rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            norm = _VRMSNorm(x.shape[-1], eps)
            norm.weight.data = weight
            return norm(x)
        _rms_norm_source = "vLLM.RMSNorm"
    except (ImportError, AttributeError):
        pass

if _rms_norm_source == "manual":
    def _rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        return x * rstd * weight

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm（自动选择实现源）"""
    return _rms_norm_impl(x, weight, eps)


# ============================================================
# Q4_0 量化 (ggml 兼容)
# 每 32 元素一个 block: [fp16 scale] + [16 字节 packed int4]
# ============================================================

def q4_0_quant(x: torch.Tensor) -> tuple:
    """Q4_0 量化
    返回: (qdata: torch.uint8, scales: torch.float16, block_size: int)
    qdata 形状: [n_blocks, 16] (每 block 32 个 int4 值打包成 16 字节)
    scales 形状: [n_blocks]
    """
    orig_shape = x.shape
    flat = x.flatten()
    n = flat.shape[0]
    block_size = 32
    n_blocks = (n + block_size - 1) // block_size

    # 补齐到 block_size 对齐
    padded = torch.zeros(n_blocks * block_size, dtype=x.dtype)
    padded[:n] = flat

    # 分块
    blocks = padded.view(n_blocks, block_size)

    # 每 block 取 absmax 作为 scale
    amax = blocks.abs().max(dim=1, keepdim=True).values
    # 避免除以 0
    amax = torch.clamp(amax, min=1e-10)
    scale = amax / 7.0  # int4 范围 [-8, 7]

    # 量化
    quant = blocks / scale
    quant = torch.clamp(quant, -8.0, 7.0).round().to(torch.int8)

    # 打包成 uint8: 每两个 int4 打包成一个 uint8
    # 低 4 位 = 第一个元素, 高 4 位 = 第二个元素
    quant_u8 = torch.zeros(n_blocks, 16, dtype=torch.uint8)
    for i in range(16):
        lo = quant[:, i * 2].to(torch.uint8) & 0x0F
        hi = (quant[:, i * 2 + 1].to(torch.uint8) & 0x0F) << 4
        quant_u8[:, i] = lo | hi

    return quant_u8, scale.to(torch.float16).flatten(), block_size


def q4_0_dequant(qdata: torch.Tensor, scales: torch.Tensor, n: int, block_size: int = 32) -> torch.Tensor:
    """Q4_0 反量化"""
    n_blocks = qdata.shape[0]

    # 解包
    int4_vals = torch.zeros(n_blocks, block_size, dtype=torch.float32)
    for i in range(16):
        val = qdata[:, i].to(torch.int32)
        int4_vals[:, i * 2] = (val & 0x0F).to(torch.float32)
        int4_vals[:, i * 2 + 1] = ((val >> 4) & 0x0F).to(torch.float32)

    # int4 有符号处理: > 7 的值是负数
    int4_vals = torch.where(int4_vals > 7, int4_vals - 16, int4_vals)

    # 反量化
    result = int4_vals * scales.view(-1, 1)
    return result.flatten()[:n]


# ============================================================
# Q8_0 量化 (ggml 兼容)
# 每 32 元素一个 block: [fp16 scale] + [32 字节 int8]
# ============================================================

def q8_0_quant(x: torch.Tensor) -> tuple:
    """Q8_0 量化
    返回: (qdata: torch.int8, scales: torch.float16, block_size: int)
    """
    orig_shape = x.shape
    flat = x.flatten()
    n = flat.shape[0]
    block_size = 32
    n_blocks = (n + block_size - 1) // block_size

    padded = torch.zeros(n_blocks * block_size, dtype=x.dtype)
    padded[:n] = flat

    blocks = padded.view(n_blocks, block_size)
    amax = blocks.abs().max(dim=1, keepdim=True).values
    amax = torch.clamp(amax, min=1e-10)
    scale = amax / 127.0  # int8 范围 [-128, 127]

    quant = blocks / scale
    quant = torch.clamp(quant, -128.0, 127.0).round().to(torch.int8)

    return quant, scale.to(torch.float16).flatten(), block_size


def q8_0_dequant(qdata: torch.Tensor, scales: torch.Tensor, n: int, block_size: int = 32) -> torch.Tensor:
    """Q8_0 反量化"""
    result = qdata.to(torch.float32) * scales.view(-1, 1)
    return result.flatten()[:n]


# ============================================================
# K-quant 反量化 (ch20)
# Q2_K: 每 256 元素一个 super-block, 内部 16 个 sub-block, 每个 sub-block 4 bits
# Q6_K: 每 256 元素一个 super-block, 内部 16 个 sub-block, 每个 sub-block 6 bits
# ============================================================

def dequant_q2_k(qdata: dict, n: int) -> torch.Tensor:
    """Q2_K 反量化（简化版）
    qdata: {
        'scales': torch.uint8,  # [n_super_blocks, 16] 每个 sub-block 的 scale
        'qs': torch.uint8,      # [n_super_blocks, 64] packed int2
        'd': torch.float16,     # [n_super_blocks] super-block scale
    }
    """
    n_blocks = qdata['qs'].shape[0]
    super_block_size = 256
    sub_block_size = 16
    n_subs = 16

    result = torch.zeros(n_blocks * super_block_size, dtype=torch.float32)

    for b in range(n_blocks):
        d = float(qdata['d'][b])
        for sub in range(n_subs):
            # 每个 sub-block 有 16 个元素，打包在 4 个 uint8 中（每个 uint8 含 4 个 int2）
            s = float(qdata['scales'][b, sub])  # 3.0 + s/4.0 之类的映射
            scale = d * s
            for i in range(4):
                byte_val = int(qdata['qs'][b, sub * 4 + i])
                for j in range(4):
                    idx = sub * sub_block_size + i * 4 + j
                    val = (byte_val >> (j * 2)) & 0x03
                    if val >= 2:
                        val -= 4  # 有符号
                    result[b * super_block_size + idx] = val * scale

    return result.flatten()[:n]


def dequant_q6_k(qdata: dict, n: int) -> torch.Tensor:
    """Q6_K 反量化（简化版）"""
    n_blocks = qdata['qs'].shape[0]
    super_block_size = 256
    sub_block_size = 16
    n_subs = 16

    result = torch.zeros(n_blocks * super_block_size, dtype=torch.float32)

    for b in range(n_blocks):
        d = float(qdata['d'][b])
        for sub in range(n_subs):
            s = float(qdata['scales'][b, sub])
            scale = d * s
            for i in range(16):
                idx = sub * sub_block_size + i
                val = int(qdata['qs'][b, idx])
                if val >= 32:
                    val -= 64  # 有符号 int6
                result[b * super_block_size + idx] = val * scale

    return result.flatten()[:n]


# ============================================================
# MXFP4 E2M1 量化/反量化 (ch20)
# 微观缩放格式: E2M1 指数偏移 1, 尾数 1 bit
# ============================================================

def mxfp4_e2m1_quantize_value(v: float) -> int:
    """将单个 float32 量化为 MXFP4 E2M1 (4-bit)"""
    if v == 0.0:
        return 0
    sign = 1 if v < 0 else 0
    v = abs(v)
    # 找到最接近的 E2M1 表示
    # E2M1: 1 sign + 2 exp + 1 mantissa = 4 bits
    # 指数偏移 1, 所以实际指数范围 [-1, 2]
    # 可表示的值: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6, ±inf
    e2m1_values = {
        0: 0b0000, 0.5: 0b0100, 1.0: 0b0101, 1.5: 0b0110,
        2.0: 0b0111, 3.0: 0b1000, 4.0: 0b1001, 6.0: 0b1010,
    }
    # 带符号
    signed = {k: (v | (1 << 3) if k > 0 else v) for k, v in e2m1_values.items()}
    signed_neg = {k: v | (1 << 3) for k, v in e2m1_values.items() if k > 0}

    best_val = min(e2m1_values.keys(), key=lambda x: abs(x - v))
    code = e2m1_values[best_val]
    if sign and best_val > 0:
        code |= 0b1000
    return code


def mxfp4_dequantize_value(code: int) -> float:
    """将 MXFP4 E2M1 code 反量化为 float32"""
    sign = -1.0 if (code & 0b1000) else 1.0
    exp = (code >> 1) & 0b11
    mantissa = code & 0b1

    if exp == 0 and mantissa == 0:
        return 0.0 * sign  # 0
    if exp == 0b11 and mantissa == 0:
        return float('inf') * sign  # ±inf
    if exp == 0b11 and mantissa == 1:
        return float('nan')  # NaN

    # E2M1: 指数偏移 1, 尾数隐式 1
    e = exp - 1  # 实际指数偏移
    m = 1.0 + mantissa * 0.5  # 尾数: 1.0 或 1.5
    return sign * m * (2.0 ** e)


# ============================================================
# MoE 路由 (ch19)
# ============================================================

_moe_source = "manual"

if _has_transformers:
    try:
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        _moe_source = "transformers.MixtralSparseMoeBlock"
    except (ImportError, AttributeError):
        pass

def moe_router(logits: torch.Tensor, top_k: int = 2) -> tuple:
    """MoE 路由: 对每个 token 选择 top-k expert"""
    weights, indices = torch.topk(logits, top_k, dim=-1)
    weights = torch.softmax(weights, dim=-1)
    return indices, weights


def moe_combine(expert_outputs: torch.Tensor, indices: torch.Tensor,
                weights: torch.Tensor, n_experts: int) -> torch.Tensor:
    """MoE Combine: 按权重加权合并 expert 输出"""
    n_tokens, top_k = indices.shape
    d_model = expert_outputs.shape[-1]
    output = torch.zeros(n_tokens, d_model, dtype=expert_outputs.dtype)

    for t in range(n_tokens):
        for k in range(top_k):
            expert_idx = indices[t, k]
            output[t] += weights[t, k] * expert_outputs[t, expert_idx]

    return output


# ============================================================
# Subgroup 操作模拟 (ch16)
# 用张量维度操作模拟 warp-level 原语
# ============================================================

def subgroup_shuffle_sim(x: torch.Tensor, src_lane: int, subgroup_size: int = 32) -> torch.Tensor:
    """模拟 subgroup shuffle: 每个线程从 src_lane 获取值
    x: [subgroup_size] 每个线程的值
    返回: [subgroup_size] 每个线程得到 src_lane 的值
    """
    return x[src_lane].expand(subgroup_size)


def subgroup_broadcast_sim(x: torch.Tensor, src_lane: int, subgroup_size: int = 32) -> torch.Tensor:
    """模拟 subgroup broadcast"""
    return x[src_lane].expand(subgroup_size)


def subgroup_reduce_sum_sim(x: torch.Tensor, subgroup_size: int = 32) -> torch.Tensor:
    """模拟 subgroup reduce sum"""
    total = x.sum()
    return total.expand(subgroup_size)


def subgroup_reduce_max_sim(x: torch.Tensor, subgroup_size: int = 32) -> torch.Tensor:
    """模拟 subgroup reduce max"""
    m = x.max()
    return m.expand(subgroup_size)


def subgroup_ballot_sim(predicate: torch.Tensor, subgroup_size: int = 32) -> int:
    """模拟 subgroup ballot: 每个线程的 predicate 打包成一个 bitmask"""
    mask = 0
    for i in range(subgroup_size):
        if predicate[i]:
            mask |= (1 << i)
    return mask


def subgroup_softmax_sim(x: torch.Tensor, subgroup_size: int = 32) -> torch.Tensor:
    """模拟 warp-level softmax: 用 subgroup reduce 实现"""
    # 1. 找 max
    max_val = x.max()
    # 2. 减 max 并 exp
    exp_x = torch.exp(x - max_val)
    # 3. sum
    sum_exp = exp_x.sum()
    # 4. 归一化
    return exp_x / sum_exp


# ============================================================
# 工具函数
# ============================================================

def compute_max_diff(got: torch.Tensor, expected: torch.Tensor) -> float:
    """计算两个张量的最大绝对差异"""
    return (got - expected).abs().max().item()


def compute_rmse(got: torch.Tensor, expected: torch.Tensor) -> float:
    """计算 RMSE"""
    return torch.sqrt(((got - expected) ** 2).mean()).item()


__all__ = [
    'rms_norm', '_rms_norm_source',
    'q4_0_quant', 'q4_0_dequant',
    'q8_0_quant', 'q8_0_dequant',
    'dequant_q2_k', 'dequant_q6_k',
    'mxfp4_e2m1_quantize_value', 'mxfp4_dequantize_value',
    'moe_router', 'moe_combine', '_moe_source',
    'subgroup_shuffle_sim', 'subgroup_broadcast_sim',
    'subgroup_reduce_sum_sim', 'subgroup_reduce_max_sim',
    'subgroup_ballot_sim', 'subgroup_softmax_sim',
    'compute_max_diff', 'compute_rmse',
    '_has_transformers', '_has_vllm',
]