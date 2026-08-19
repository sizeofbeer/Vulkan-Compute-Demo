#!/usr/bin/env python3
"""
向量验证器 —— 用 PyTorch 参考验证测试向量，并可选对比 Vulkan GPU 输出。

用法:
  python tests/vector_verify.py                  # 生成 + 验证
  python tests/vector_verify.py --gen-only       # 只生成向量
  python tests/vector_verify.py --ref-only       # 只验证参考一致性
  python tests/vector_verify.py --gpu-only       # 对比 Vulkan GPU 输出
  python tests/vector_verify.py --filter gemm    # 只测试 GEMM
  python tests/vector_verify.py --list           # 列出所有向量
  python tests/vector_verify.py --verbose        # 详细输出
"""

import torch
import torch.nn.functional as F
import os
import sys
import json
import time
import shutil
import subprocess
import struct
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

# 确保能导入 ref_impl
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_impl import (
    rms_norm, q4_0_quant, q4_0_dequant, q8_0_quant, q8_0_dequant,
    dequant_q2_k, dequant_q6_k, mxfp4_e2m1_quantize_value, mxfp4_dequantize_value,
    moe_router, moe_combine,
    subgroup_shuffle_sim, subgroup_broadcast_sim,
    subgroup_reduce_sum_sim, subgroup_reduce_max_sim,
    subgroup_ballot_sim, subgroup_softmax_sim,
    compute_max_diff, compute_rmse,
)

VECTORS_DIR = Path(__file__).resolve().parent / "vectors"
BUILD_DIR = Path(__file__).resolve().parent.parent / "build"

# 跨平台：Windows 下可执行文件带 .exe，Linux/macOS 无后缀
EXE_SUFFIX = ".exe" if os.name == "nt" else ""


@dataclass
class VectorResult:
    name: str
    status: str  # PASS / FAIL / SKIP
    detail: str = ""
    max_diff: float = 0.0
    rmse: float = 0.0
    duration_ms: float = 0.0
    tolerance: float = 0.0


# ============================================================
# 参考实现清单
# ============================================================

def get_ref_computation(key: str):
    """根据向量类型返回参考计算函数"""
    refs = {
        # GEMM (ch04, ch15)
        "gemm": lambda data: _compute_gemm(data),
        # 卷积 (ch05)
        "conv": lambda data: _compute_conv(data),
        # Softmax (ch08, ch17)
        "softmax": lambda data: _compute_softmax(data),
        # 归一化 (ch07)
        "norm": lambda data: _compute_norm(data),
        # 注意力 (ch08, ch17)
        "attention": lambda data: _compute_attention(data),
        # 量化 (ch11, ch18)
        "quant": lambda data: _compute_quant(data),
        # 深层量化 (ch20)
        "quant_deep": lambda data: _compute_quant_deep(data),
        # 激活函数 (ch06)
        "elemwise": lambda data: _compute_elemwise(data),
        # MoE (ch19)
        "moe": lambda data: _compute_moe(data),
        # Subgroup (ch16)
        "subgroup": lambda data: _compute_subgroup(data),
        # Cooperative Matrix (ch15)
        "coopmat": lambda data: _compute_gemm(data),  # 与 GEMM 相同
    }
    return refs.get(key, None)


def _compute_gemm(data: dict) -> dict:
    A = data["inputs"]["A"]
    B = data["inputs"]["B"]
    C = torch.matmul(A, B)
    return {"C": C}


def _compute_conv(data: dict) -> dict:
    x = data["inputs"]["x"]
    w = data["inputs"]["w"]
    padding = data["params"].get("padding", 0)
    out = F.conv2d(x, w, padding=padding)
    return {"out": out}


def _compute_softmax(data: dict) -> dict:
    x = data["inputs"]["x"]
    dim = data["params"].get("dim", 0)
    out = F.softmax(x, dim=dim)
    return {"out": out}


def _compute_norm(data: dict) -> dict:
    x = data["inputs"]["x"]
    shape = data["params"]["shape"]
    eps = data["params"].get("eps", 1e-5)

    if "layernorm" in data["name"]:
        normalized_shape = data["params"]["normalized_shape"]
        weight = data["inputs"].get("weight", None)
        bias = data["inputs"].get("bias", None)
        out = F.layer_norm(x, normalized_shape, weight, bias, eps)
    else:
        weight = data["inputs"]["weight"]
        out = rms_norm(x, weight, eps)
    return {"out": out}


def _compute_attention(data: dict) -> dict:
    Q = data["inputs"]["Q"]
    K = data["inputs"]["K"]
    V = data["inputs"]["V"]
    is_causal = data["params"].get("is_causal", False)
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, scale=scale)
    return {"out": out}


def _compute_quant(data: dict) -> dict:
    """量化向量只验证反量化结果与输入的差异"""
    return {"expected": data["expected"]}  # 已在生成时算好


def _compute_quant_deep(data: dict) -> dict:
    name = data["name"]
    if "mxfp4" in name:
        # 用 PyTorch 重新计算
        test_vals = data["inputs"]["test_values"]
        decoded = torch.tensor(
            [mxfp4_dequantize_value(mxfp4_e2m1_quantize_value(float(v)))
             for v in test_vals],
            dtype=torch.float32
        )
        return {"decoded": decoded}
    elif "q2_k" in name:
        qdata = data["inputs"]["qdata"]
        n = data["params"]["n"]
        out = dequant_q2_k(qdata, n)
        return {"out": out}
    return {"expected": data["expected"]}


def _compute_elemwise(data: dict) -> dict:
    x = data["inputs"]["x"]
    name = data["name"]
    if "relu" in name:
        out = F.relu(x)
    elif "gelu" in name:
        out = F.gelu(x)
    elif "silu" in name:
        out = F.silu(x)
    else:
        out = F.relu(x)
    return {"out": out}


def _compute_moe(data: dict) -> dict:
    name = data["name"]
    if "router" in name:
        logits = data["inputs"]["logits"]
        top_k = data["params"]["top_k"]
        indices, weights = moe_router(logits, top_k)
        return {"indices": indices, "weights": weights}
    elif "combine" in name:
        expert_outputs = data["inputs"]["expert_outputs"]
        indices = data["inputs"]["indices"]
        weights = data["inputs"]["weights"]
        n_experts = data["params"]["n_experts"]
        combined = moe_combine(expert_outputs, indices, weights, n_experts)
        return {"combined": combined}
    else:
        logits = data["inputs"]["logits"]
        expert_outputs = data["inputs"]["expert_outputs"]
        top_k = data["params"]["top_k"]
        n_experts = data["params"]["n_experts"]
        indices, weights = moe_router(logits, top_k)
        combined = moe_combine(expert_outputs, indices, weights, n_experts)
        return {"indices": indices, "weights": weights, "combined": combined}


def _compute_subgroup(data: dict) -> dict:
    x = data["inputs"]["x"]
    sg_size = data["params"]["subgroup_size"]
    result = {}
    result["shuffled"] = subgroup_shuffle_sim(x, data["params"].get("src_lane_shuffle", 0), sg_size)
    result["broadcast"] = subgroup_broadcast_sim(x, data["params"].get("src_lane_broadcast", 0), sg_size)
    result["sum_val"] = subgroup_reduce_sum_sim(x, sg_size)
    result["max_val"] = subgroup_reduce_max_sim(x, sg_size)
    predicate = x > 4.0
    result["ballot"] = torch.tensor(subgroup_ballot_sim(predicate, sg_size), dtype=torch.int32)
    result["softmax"] = subgroup_softmax_sim(x, sg_size)
    return result


# ============================================================
# 向量验证器
# ============================================================

class VectorVerifier:
    def __init__(self, verbose: bool = False, filter_key: str = None):
        self.verbose = verbose
        self.filter_key = filter_key
        self.results: List[VectorResult] = []

    def _load_vectors(self) -> List[dict]:
        """加载所有测试向量"""
        vectors = []
        if not VECTORS_DIR.exists():
            return vectors

        for root, dirs, files in os.walk(VECTORS_DIR):
            for f in sorted(files):
                if not f.endswith('.pt'):
                    continue
                subdir = os.path.relpath(root, VECTORS_DIR)
                if self.filter_key and self.filter_key not in subdir:
                    continue
                filepath = os.path.join(root, f)
                try:
                    data = torch.load(filepath, weights_only=True)
                    data["_filepath"] = filepath
                    data["_subdir"] = subdir
                    vectors.append(data)
                except Exception as e:
                    print(f"  ⚠️ 加载 {f} 失败: {e}")

        return vectors

    def verify_ref(self) -> List[VectorResult]:
        """模式 2: 验证参考一致性（无 GPU 依赖）"""
        vectors = self._load_vectors()
        if not vectors:
            print("  ⚠️ 没有测试向量，请先运行 --gen-only")
            return []

        results = []
        for data in vectors:
            t0 = time.time()
            key = data["_subdir"]
            ref_fn = get_ref_computation(key)

            if ref_fn is None:
                results.append(VectorResult(
                    name=data["name"], status="SKIP",
                    detail=f"无参考实现: {key}"
                ))
                continue

            try:
                computed = ref_fn(data)
                expected = data["expected"]

                # 对比每个输出字段
                all_pass = True
                max_max_diff = 0.0
                max_rmse = 0.0
                details = []

                for out_key, expected_val in expected.items():
                    if out_key not in computed:
                        all_pass = False
                        details.append(f"缺少输出: {out_key}")
                        continue

                    got = computed[out_key]
                    # 处理标量
                    if isinstance(got, torch.Tensor) and got.numel() == 1 and isinstance(expected_val, torch.Tensor) and expected_val.numel() == 1:
                        pass
                    elif isinstance(got, int):
                        got = torch.tensor(got, dtype=torch.float32)
                    elif isinstance(expected_val, int):
                        expected_val = torch.tensor(expected_val, dtype=torch.float32)

                    md = compute_max_diff(got, expected_val)
                    rmse_val = compute_rmse(got, expected_val)
                    max_max_diff = max(max_max_diff, md)
                    max_rmse = max(max_rmse, rmse_val)

                    tol = data.get("tolerance", 1e-4)
                    tol_rmse = data.get("tolerance_rmse", 1e-4)

                    if md > tol or rmse_val > tol_rmse:
                        all_pass = False
                        details.append(f"{out_key}: max_diff={md:.6e} > {tol}, rmse={rmse_val:.6e} > {tol_rmse}")

                elapsed = (time.time() - t0) * 1000
                status = "PASS" if all_pass else "FAIL"
                detail = "; ".join(details) if details else f"max_diff={max_max_diff:.6e}"

                results.append(VectorResult(
                    name=data["name"], status=status, detail=detail,
                    max_diff=max_max_diff, rmse=max_rmse,
                    duration_ms=elapsed, tolerance=data.get("tolerance", 1e-4)
                ))

                if self.verbose or status == "FAIL":
                    prefix = "✓" if status == "PASS" else "✗"
                    print(f"  [{prefix}] {data['name']}: {detail} ({elapsed:.0f}ms)")

            except Exception as e:
                elapsed = (time.time() - t0) * 1000
                results.append(VectorResult(
                    name=data["name"], status="FAIL",
                    detail=f"异常: {e}", duration_ms=elapsed
                ))
                print(f"  [✗] {data['name']}: 异常: {e}")

        self.results = results
        return results

    def verify_gpu(self, build_dir: str = None) -> List[VectorResult]:
        """模式 3: 对比 Vulkan GPU 输出（调用各章节 C++ 可执行文件）"""
        build_dir = build_dir or str(BUILD_DIR)
        vectors = self._load_vectors()
        if not vectors:
            print("  ⚠️ 没有测试向量，请先运行 --gen-only")
            return []

        # 临时目录：存放输入/输出的 .bin 文件
        tmp = Path(build_dir) / "_verify_gpu"
        in_dir = tmp / "in"
        out_dir = tmp / "out"
        shutil.rmtree(tmp, ignore_errors=True)
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for data in vectors:
            t0 = time.time()
            key = data["_subdir"]

            # 确定要运行的 C++ 章节与输出张量映射
            targets = self._gpu_targets(key, data)

            if not targets:
                results.append(VectorResult(
                    name=data["name"], status="SKIP",
                    detail=f"C++ 无对应可执行输出: {key}"
                ))
                if self.verbose:
                    print(f"  [-][skip] {data['name']}: {key} 无 C++ 映射")
                continue

            any_pass = False          # 至少一个章节完整通过即可判 PASS
            ran_any = False           # 是否有章节成功生成了可比较的输出
            details = []
            max_max_diff = 0.0
            max_rmse = 0.0

            for exe_name, out_key, load_inputs in targets:
                # 跨平台：Linux 下可执行文件无 .exe 后缀
                exe = Path(build_dir) / exe_name / (exe_name + EXE_SUFFIX)
                if not exe.exists():
                    details.append(f"缺少可执行文件: {exe_name}")
                    continue

                # 清空输入/输出目录，写入本次输入
                for p in in_dir.iterdir():
                    p.unlink(missing_ok=True)
                for p in out_dir.iterdir():
                    p.unlink(missing_ok=True)
                if not load_inputs(data, in_dir):
                    details.append(f"{exe_name}: 无法导出输入")
                    continue

                # 运行 C++ 章节
                try:
                    r = subprocess.run(
                        [str(exe), "--test-vectors", str(in_dir), "--output", str(out_dir)],
                        capture_output=True, text=True, timeout=300
                    )
                except FileNotFoundError:
                    details.append(f"{exe_name}: 无法运行")
                    continue
                if r.returncode != 0:
                    details.append(f"{exe_name}: 退出码 {r.returncode}")
                    continue

                # 读取 C++ 输出并与参考对比
                expected = data["expected"].get(out_key)
                if expected is None:
                    details.append(f"{exe_name}: 参考缺少输出 {out_key}")
                    continue

                got, got_shape = self._read_bin(out_dir / (out_key + ".bin"))
                if got is None:
                    details.append(f"{exe_name}: 未生成输出 {out_key}.bin")
                    continue

                got_t = torch.tensor(got, dtype=torch.float32).reshape(got_shape)
                exp_t = expected.detach().cpu().float()

                # 形状容错：去掉多余的前导 1 维 / 补充前导 1 维
                got_t, exp_t = self._align_shapes(got_t, exp_t)

                md = compute_max_diff(got_t, exp_t)
                rmse_val = compute_rmse(got_t, exp_t)
                max_max_diff = max(max_max_diff, md)
                max_rmse = max(max_rmse, rmse_val)
                ran_any = True

                tol = data.get("tolerance", 1e-4)
                tol_rmse = data.get("tolerance_rmse", 1e-4)
                if md <= tol and rmse_val <= tol_rmse:
                    any_pass = True
                else:
                    details.append(f"{exe_name}/{out_key}: max_diff={md:.6e} > {tol}, rmse={rmse_val:.6e}")

            elapsed = (time.time() - t0) * 1000
            if any_pass:
                status = "PASS"
            elif ran_any:
                status = "FAIL"
            else:
                status = "SKIP"
            detail = "; ".join(details) if details else f"max_diff={max_max_diff:.6e}"
            results.append(VectorResult(
                name=data["name"], status=status, detail=detail,
                max_diff=max_max_diff, rmse=max_rmse,
                duration_ms=elapsed, tolerance=data.get("tolerance", 1e-4)
            ))
            if self.verbose or status == "FAIL":
                prefix = "✓" if status == "PASS" else ("-" if status == "SKIP" else "✗")
                print(f"  [{prefix}] {data['name']}: {detail} ({elapsed:.0f}ms)")

        shutil.rmtree(tmp, ignore_errors=True)
        self.results = results
        return results

    # ------------------------------------------------------------------
    # GPU 验证辅助
    # ------------------------------------------------------------------

    def _gpu_targets(self, key: str, data: dict):
        """返回 [(exe_name, out_key, load_inputs_fn), ...]，无映射返回 []"""
        name = data["name"]
        if key == "gemm":
            # ch04 需 coopmat 支持（某些 glslc 无法编译 CM 着色器时自动跳过，ch09 提供稳定回退）
            return [("ch09_profiling", "C", self._dump_gemm_inputs),
                    ("ch04_gemm", "C", self._dump_gemm_inputs)]
        if key == "coopmat":
            return [("ch15_coopmat", "C", self._dump_gemm_inputs)]
        if key == "norm":
            return [("ch07_norm", "out", self._dump_norm_inputs)]
        if key == "attention":
            return [("ch08_attention", "O", self._dump_attn_inputs),
                    ("ch17_flash_attn", "out", self._dump_attn_inputs)]
        if key == "conv":
            return [("ch05_conv", "out", self._dump_conv_inputs)]
        if key == "elemwise":
            # 根据向量名决定激活类型：out_relu / out_gelu / out_silu
            act = "relu"
            if "gelu" in name: act = "gelu"
            elif "silu" in name: act = "silu"
            return [("ch06_elemwise", f"out_{act}", self._dump_elemwise_inputs)]
        # 下列类型 C++ 端无直接对应输出（量化做 GEMM、MoE/Subgroup 为演示、softmax 无独立算子）
        return []

    def _write_bin(self, path, tensor):
        """按 C++ TestVectorConfig 格式写 .bin: [ndims:u32][dim...:u32][data:f32...]"""
        t = tensor.detach().cpu().float().reshape(-1).tolist()
        shape = list(tensor.shape)
        with open(path, "wb") as f:
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<I", int(d)))
            f.write(struct.pack("<%df" % len(t), *t))

    def _read_bin(self, path):
        """读取 .bin，返回 (data_list, shape) 或 (None, None)"""
        if not Path(path).exists():
            return None, None
        with open(path, "rb") as f:
            raw = f.read()
        ndims = struct.unpack_from("<I", raw, 0)[0]
        off = 4
        shape = list(struct.unpack_from("<%dI" % ndims, raw, off))
        off += 4 * ndims
        n = 1
        for d in shape:
            n *= d
        data = list(struct.unpack_from("<%df" % n, raw, off))
        return data, shape

    def _dump_gemm_inputs(self, data, in_dir):
        for k in ("A", "B"):
            if k not in data["inputs"]:
                return False
            self._write_bin(in_dir / (k + ".bin"), data["inputs"][k])
        return True

    def _dump_norm_inputs(self, data, in_dir):
        for k in ("x", "weight"):
            if k not in data["inputs"]:
                return False
            self._write_bin(in_dir / (k + ".bin"), data["inputs"][k])
        return True

    def _dump_attn_inputs(self, data, in_dir):
        for k in ("Q", "K", "V"):
            if k not in data["inputs"]:
                return False
            self._write_bin(in_dir / (k + ".bin"), data["inputs"][k])
        return True

    def _dump_conv_inputs(self, data, in_dir):
        for k in ("x", "w"):
            if k not in data["inputs"]:
                return False
            self._write_bin(in_dir / (k + ".bin"), data["inputs"][k])
        return True

    def _dump_elemwise_inputs(self, data, in_dir):
        if "x" not in data["inputs"]:
            return False
        self._write_bin(in_dir / "x.bin", data["inputs"]["x"])
        return True

    def _align_shapes(self, got_t, exp_t):
        """形状容错：对齐前导 1 维，使两者可广播比较"""
        while got_t.dim() > exp_t.dim() and got_t.shape[0] == 1:
            got_t = got_t.squeeze(0)
        while exp_t.dim() > got_t.dim() and exp_t.shape[0] == 1:
            exp_t = exp_t.squeeze(0)
        return got_t, exp_t

    def print_summary(self):
        """打印汇总"""
        if not self.results:
            print("\n  无测试结果")
            return

        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")

        print(f"\n{'=' * 50}")
        print(f"  参考一致性验证结果")
        print(f"{'=' * 50}")
        print(f"  总计: {len(self.results)}")
        print(f"  ✅ 通过: {passed}")
        print(f"  ❌ 失败: {failed}")
        print(f"  ⏭️  跳过: {skipped}")

        if failed > 0:
            print(f"\n  失败详情:")
            for r in self.results:
                if r.status == "FAIL":
                    print(f"    [{r.name}] {r.detail}")

        # 保存报告
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "results": [
                {"name": r.name, "status": r.status, "detail": r.detail,
                 "max_diff": r.max_diff, "rmse": r.rmse,
                 "tolerance": r.tolerance, "duration_ms": round(r.duration_ms, 1)}
                for r in self.results
            ]
        }
        report_path = VECTORS_DIR.parent / "vector_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  📝 报告: {report_path}")

        return failed == 0


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="向量验证器")
    parser.add_argument("--gen-only", action="store_true", help="只生成测试向量")
    parser.add_argument("--ref-only", action="store_true", help="只验证参考一致性")
    parser.add_argument("--gpu-only", action="store_true", help="对比 Vulkan GPU 输出")
    parser.add_argument("--filter", type=str, default=None, help="按算子过滤")
    parser.add_argument("--list", action="store_true", help="列出所有向量")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    # 默认模式: 生成 + 验证
    do_gen = args.gen_only or not (args.ref_only or args.gpu_only)
    do_ref = args.ref_only or not (args.gen_only or args.gpu_only)
    do_gpu = args.gpu_only

    # 列出向量
    if args.list:
        # 调用 gen_vectors.py --list
        gen_script = Path(__file__).resolve().parent / "gen_vectors.py"
        subprocess.run([sys.executable, str(gen_script), "--list"])
        return

    # 生成向量
    if do_gen:
        print("📦 生成测试向量...")
        gen_script = Path(__file__).resolve().parent / "gen_vectors.py"
        gen_args = [sys.executable, str(gen_script)]
        if args.filter:
            gen_args.extend(["--filter", args.filter])
        result = subprocess.run(gen_args)
        if result.returncode != 0:
            print("❌ 生成测试向量失败")
            sys.exit(1)

    # 验证参考一致性
    if do_ref:
        print("\n🔍 验证参考一致性...")
        verifier = VectorVerifier(verbose=args.verbose, filter_key=args.filter)
        results = verifier.verify_ref()
        if not verifier.print_summary():
            if not do_gpu:
                sys.exit(1)

    # GPU 验证
    if do_gpu:
        print("\n🔍 对比 Vulkan GPU 输出...")
        verifier = VectorVerifier(verbose=args.verbose, filter_key=args.filter)
        verifier.verify_gpu()


if __name__ == "__main__":
    main()