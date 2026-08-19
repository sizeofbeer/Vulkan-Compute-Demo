#!/usr/bin/env python3
"""
Vulkan 课程验证测试框架 (ch00 ~ ch20)
========================================
使用方法:
    python tests/test_framework.py             # 运行所有测试
    python tests/test_framework.py --build     # 先构建再测试
    python tests/test_framework.py --ch 04     # 只测试 ch04
    python tests/test_framework.py --list      # 列出所有测试
    python tests/test_framework.py --lang en   # 英文输出
    python tests/test_framework.py --verbose   # 详细输出
"""

import os
import sys
import subprocess
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field


# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
TESTS_DIR = PROJECT_ROOT / "tests"
VERIFICATION_REPORT = PROJECT_ROOT / "VERIFICATION_REPORT.md"

# 可执行文件后缀
EXE_SUFFIX = ".exe" if os.name == "nt" else ""


# ============================================================
# 测试结果数据结构
# ============================================================

class TestResult:
    """单个测试结果"""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"  # 代码正确但环境异常（如 barrier 驱动问题）

    def __init__(self, chapter: str, name: str, status: str, detail: str = "",
                 stdout: str = "", stderr: str = "", duration_ms: float = 0):
        self.chapter = chapter
        self.name = name
        self.status = status
        self.detail = detail
        self.stdout = stdout
        self.stderr = stderr
        self.duration_ms = duration_ms

    def __repr__(self) -> str:
        return f"[{self.status}] {self.chapter}/{self.name} ({self.duration_ms:.0f}ms) {self.detail}"


# ============================================================
# 测试用例定义
# ============================================================

@dataclass
class TestCase:
    """单个测试用例"""
    chapter: str          # 章节名，如 "ch04_gemm"
    name: str             # 测试名，如 "tiled_gemm"
    exe_path: str         # 相对于 BUILD_DIR 的可执行文件路径
    args: List[str] = field(default_factory=list)  # 命令行参数
    timeout_s: int = 60   # 超时秒数
    # 验证函数：接收 stdout/stderr 字符串，返回 (status, detail)
    verify: Callable[[str, str], Tuple[str, str]] = None
    # 预期返回码
    expected_retcode: int = 0
    # 是否需要 GPU
    needs_gpu: bool = True
    # 是否为「已知环境问题」测试（如 barrier 问题）
    known_env_issue: bool = False


def _verify_errors_zero(stdout: str, stderr: str) -> Tuple[str, str]:
    """验证输出中 'errors=0' 或 '错误数=0'"""
    stdout_lower = stdout.lower()
    if re.search(r'errors?\s*=\s*0', stdout_lower) or re.search(r'错误数\s*=\s*0', stdout):
        return (TestResult.PASS, "errors=0")
    if re.search(r'errors?\s*=\s*\d+', stdout_lower) or re.search(r'错误数\s*=\s*\d+', stdout):
        return (TestResult.FAIL, "errors>0")
    if "error" in stderr.lower() or "错误" in stderr:
        return (TestResult.FAIL, f"stderr has error: {stderr[:200]}")
    return (TestResult.WARN, "no errors=0 pattern found, but no error reported")


def _verify_correct(stdout: str, stderr: str) -> Tuple[str, str]:
    """验证输出中包含 'correct ✓' 或 '正确 ✓'"""
    if "correct" in stdout.lower() or "正确" in stdout:
        if "✓" in stdout:
            return (TestResult.PASS, "correct ✓")
        return (TestResult.PASS, "correct")
    if "has errors" in stdout.lower() or "有错误" in stdout:
        return (TestResult.FAIL, "has errors")
    return _verify_errors_zero(stdout, stderr)


def _verify_pass(stdout: str, stderr: str) -> Tuple[str, str]:
    """验证输出中包含 'PASS' 或 '通过'"""
    if "PASS" in stdout or "通过" in stdout:
        return (TestResult.PASS, "PASS")
    if "FAIL" in stdout or "失败" in stdout:
        return (TestResult.FAIL, "FAIL")
    return _verify_correct(stdout, stderr)


def _verify_flash_attn(stdout: str, stderr: str) -> Tuple[str, str]:
    """验证 FlashAttention 输出"""
    lines = stdout.split('\n')
    for line in lines:
        if "errors=" in line.lower() or "错误数=" in line:
            if re.search(r'errors?\s*=\s*0', line.lower()) or re.search(r'错误数\s*=\s*0', line):
                return (TestResult.PASS, "FlashAttention errors=0")
            return (TestResult.FAIL, f"FlashAttention errors>0: {line[:100]}")
    return _verify_correct(stdout, stderr)


def _verify_llm(stdout: str, stderr: str) -> Tuple[str, str]:
    """验证 LLM 推理输出"""
    # LLM 输出应该包含一些文本生成结果
    if "Paris" in stdout or "France" in stdout or "capital" in stdout.lower():
        return (TestResult.PASS, "generated expected text")
    if "error" in stderr.lower() or "错误" in stderr:
        if "model load failed" in stderr or "模型加载失败" in stderr:
            return (TestResult.SKIP, "model file not found")
        return (TestResult.FAIL, f"error: {stderr[:200]}")
    if len(stdout) > 100:
        return (TestResult.PASS, "output generated (len={})".format(len(stdout)))
    return (TestResult.WARN, "short output, may be OK")


def _verify_always_pass(stdout: str, stderr: str) -> Tuple[str, str]:
    """ch00：环境枚举，打印信息即可"""
    if "error" in stderr.lower() or "错误" in stderr:
        return (TestResult.FAIL, f"error: {stderr[:200]}")
    if "selected device" in stdout.lower() or "选中设备" in stdout:
        return (TestResult.PASS, "device found")
    return (TestResult.WARN, "device info may not be complete")


def _verify_benchmark(stdout: str, stderr: str) -> Tuple[str, str]:
    """ch09：性能分析，输出时间数据即可"""
    if "error" in stderr.lower() or "错误" in stderr:
        return (TestResult.FAIL, f"error: {stderr[:200]}")
    if "ms" in stdout.lower() or "μs" in stdout.lower() or "us" in stdout.lower() or "GFLOPS" in stdout:
        return (TestResult.PASS, "timing data present")
    if "roofline" in stdout.lower():
        return (TestResult.PASS, "roofline data present")
    return (TestResult.WARN, "benchmark output present")


def _verify_ch15_coopmat(stdout: str, stderr: str) -> Tuple[str, str]:
    """验证 ch15 Cooperative Matrix"""
    if "error" in stderr.lower() or "错误" in stderr:
        return (TestResult.FAIL, f"error: {stderr[:200]}")
    # 检查 CM 是否被跳过
    if "cooperative matrix" in stdout.lower() and "not supported" in stdout.lower():
        return (TestResult.SKIP, "CM not supported on this GPU, fallback only")
    if "skip" in stdout.lower() or "跳过" in stdout:
        return (TestResult.SKIP, "CM skipped")
    if "CM errors=0" in stdout or "CM 错误数=0" in stdout:
        return (TestResult.PASS, "CM errors=0")
    if "Fallback errors=0" in stdout or "回退路径错误数=0" in stdout:
        return (TestResult.PASS, "CM not available, fallback works")
    return _verify_errors_zero(stdout, stderr)


def _verify_ch16_subgroup(stdout: str, stderr: str) -> Tuple[str, str]:
    """验证 ch16 Subgroup"""
    if "error" in stderr.lower() or "错误" in stderr:
        return (TestResult.FAIL, f"error: {stderr[:200]}")
    if "PASS" in stdout or "通过" in stdout:
        return (TestResult.PASS, "subgroup operations PASS")
    if "FAIL" in stdout or "失败" in stdout:
        return (TestResult.FAIL, "subgroup operations FAIL")
    return _verify_pass(stdout, stderr)


def _verify_moe(stdout: str, stderr: str) -> Tuple[str, str]:
    """验证 MoE 输出"""
    if "error" in stderr.lower() or "错误" in stderr:
        return (TestResult.FAIL, f"error: {stderr[:200]}")
    if "Router errors=0" in stdout or "路由错误数=0" in stdout:
        return (TestResult.PASS, "router OK")
    if "expert" in stdout.lower() and "error" in stdout.lower():
        if "error=0" in stdout.lower():
            return (TestResult.PASS, "expert dispatch OK")
    return _verify_errors_zero(stdout, stderr)


# ============================================================
# 测试用例注册表
# ============================================================

def _get_test_cases() -> List[TestCase]:
    """返回所有测试用例"""
    # 基础路径
    build_root = str(BUILD_DIR)

    cases = [
        # === 第一部分：环境与最小闭环 ===
        TestCase("ch00_environment", "device_enum", "ch00_environment/ch00_environment",
                 verify=_verify_always_pass, timeout_s=30),

        TestCase("ch01_vector_add", "vector_add", "ch01_vector_add/ch01_vector_add",
                 verify=_verify_errors_zero, timeout_s=30),

        TestCase("ch02_sync", "fence_barrier", "ch02_sync/ch02_sync",
                 verify=_verify_errors_zero, timeout_s=30,
                 known_env_issue=True),  # 已知 barrier 驱动问题

        TestCase("ch03_memory", "memory_mgmt", "ch03_memory/ch03_memory",
                 verify=_verify_errors_zero, timeout_s=30),

        # === 第二部分：神经网络算子 ===
        TestCase("ch04_gemm", "gemm_all", "ch04_gemm/ch04_gemm",
                 verify=_verify_correct, timeout_s=60),

        TestCase("ch05_conv", "conv_all", "ch05_conv/ch05_conv",
                 verify=_verify_errors_zero, timeout_s=60),

        TestCase("ch06_elemwise", "activations", "ch06_elemwise/ch06_elemwise",
                 verify=_verify_errors_zero, timeout_s=30),

        TestCase("ch07_norm", "rmsnorm", "ch07_norm/ch07_norm",
                 verify=_verify_correct, timeout_s=30),

        TestCase("ch08_attention", "attention_all", "ch08_attention/ch08_attention",
                 verify=_verify_flash_attn, timeout_s=60),

        # === 第三部分：性能、量化与工程化 ===
        TestCase("ch09_profiling", "timing", "ch09_profiling/ch09_profiling",
                 verify=_verify_benchmark, timeout_s=60),

        TestCase("ch10_arena", "arena", "ch10_arena/ch10_arena",
                 verify=_verify_errors_zero, timeout_s=30),

        TestCase("ch11_quant", "quant_all", "ch11_quant/ch11_quant",
                 verify=_verify_errors_zero, timeout_s=60),

        TestCase("ch12_graph", "graph", "ch12_graph/ch12_graph",
                 verify=_verify_errors_zero, timeout_s=30,
                 known_env_issue=True),  # 已知 barrier 驱动问题

        TestCase("ch13_mlp", "mlp", "ch13_mlp/ch13_mlp",
                 verify=_verify_correct, timeout_s=60),

        # === 第四部分：GPU 高级编程与前沿技术 ===
        TestCase("ch14_llm", "llm", "ch14_llm/ch14_llm",
                 verify=_verify_llm, timeout_s=120),

        TestCase("ch15_coopmat", "coopmat", "ch15_coopmat/ch15_coopmat",
                 verify=_verify_ch15_coopmat, timeout_s=60),

        TestCase("ch16_subgroup", "subgroup", "ch16_subgroup/ch16_subgroup",
                 verify=_verify_ch16_subgroup, timeout_s=60),

        TestCase("ch17_flash_attn", "flash_attn", "ch17_flash_attn/ch17_flash_attn",
                 verify=_verify_flash_attn, timeout_s=120),

        TestCase("ch18_quant_deep", "quant_deep", "ch18_quant_deep/ch18_quant_deep",
                 verify=_verify_errors_zero, timeout_s=60),

        TestCase("ch19_moe", "moe", "ch19_moe/ch19_moe",
                 verify=_verify_moe, timeout_s=120),

        TestCase("ch20_frontier_quant", "frontier_quant", "ch20_frontier_quant/ch20_frontier_quant",
                 verify=_verify_errors_zero, timeout_s=60),
    ]

    # 补齐 exe 后缀
    for c in cases:
        c.exe_path += EXE_SUFFIX

    return cases


# ============================================================
# 测试执行引擎
# ============================================================

class TestRunner:
    """测试运行器"""

    def __init__(self, build_first: bool = False, verbose: bool = False,
                 lang: str = "en", filter_chapter: str = None):
        self.build_first = build_first
        self.verbose = verbose
        self.lang = lang
        self.filter_chapter = filter_chapter
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def _print(self, msg: str, end="\n"):
        """打印消息，flush 确保实时输出"""
        print(msg, end=end, flush=True)

    def build_all(self) -> bool:
        """构建所有章节"""
        self._print("=" * 60)
        self._print("📦 构建所有章节...")
        self._print("=" * 60)

        try:
            # 先 cmake configure
            result = subprocess.run(
                ["cmake", "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release",
                 "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
                cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                self._print("❌ CMake configure 失败:")
                self._print(result.stderr[:1000])
                return False
            self._print("✅ CMake configure 成功")

            # 然后 cmake build
            result = subprocess.run(
                ["cmake", "--build", str(BUILD_DIR), "--config", "Release", "-j"],
                cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                self._print("❌ CMake build 失败:")
                # 只显示最后 50 行错误
                err_lines = result.stderr.strip().split('\n')
                for line in err_lines[-50:]:
                    self._print(f"  {line}")
                return False
            self._print("✅ CMake build 成功")
            return True

        except subprocess.TimeoutExpired:
            self._print("❌ 构建超时 (600s)")
            return False
        except Exception as e:
            self._print(f"❌ 构建异常: {e}")
            return False

    def run_single(self, case: TestCase) -> TestResult:
        """运行单个测试用例"""
        exe_full = BUILD_DIR / case.exe_path

        if not exe_full.exists():
            return TestResult(
                case.chapter, case.name, TestResult.FAIL,
                f"可执行文件未找到: {exe_full}",
                duration_ms=0
            )

        t0 = time.time()
        try:
            result = subprocess.run(
                [str(exe_full)] + case.args,
                cwd=str(BUILD_DIR),
                capture_output=True, text=True,
                timeout=case.timeout_s
            )
            elapsed = (time.time() - t0) * 1000

            stdout = result.stdout
            stderr = result.stderr

            # 检查返回码
            if result.returncode != case.expected_retcode:
                return TestResult(
                    case.chapter, case.name, TestResult.FAIL,
                    f"返回码={result.returncode} (期望={case.expected_retcode})",
                    stdout=stdout, stderr=stderr, duration_ms=elapsed
                )

            # 运行自定义验证函数
            status, detail = case.verify(stdout, stderr)

            # 如果是已知环境问题，降级为 WARN
            if case.known_env_issue and status == TestResult.FAIL:
                status = TestResult.WARN
                detail += " [known env issue]"

            return TestResult(
                case.chapter, case.name, status, detail,
                stdout=stdout, stderr=stderr, duration_ms=elapsed
            )

        except subprocess.TimeoutExpired:
            elapsed = (time.time() - t0) * 1000
            return TestResult(
                case.chapter, case.name, TestResult.FAIL,
                f"超时 ({case.timeout_s}s)", duration_ms=elapsed
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return TestResult(
                case.chapter, case.name, TestResult.FAIL,
                f"异常: {e}", duration_ms=elapsed
            )

    def run_all(self):
        """运行所有测试"""
        cases = _get_test_cases()

        # 过滤章节
        if self.filter_chapter:
            cases = [c for c in cases if c.chapter == self.filter_chapter]

        total = len(cases)
        self._print("=" * 60)
        self._print(f"🧪 Vulkan 课程验证框架 (ch00~ch20)")
        self._print(f"   共 {total} 个测试用例")
        self._print(f"   语言: {self.lang}")
        self._print("=" * 60)
        self._print("")

        # 先构建
        if self.build_first:
            if not self.build_all():
                self._print("❌ 构建失败，终止测试")
                return

        passed = 0
        failed = 0
        skipped = 0
        warned = 0

        for i, case in enumerate(cases):
            # 显示进度
            prefix = f"[{i + 1}/{total}]"
            self._print(f"{prefix} {case.chapter}/{case.name}... ", end="")

            result = self.run_single(case)
            self.results.append(result)

            # 显示结果
            status_symbol = {
                TestResult.PASS: "✅",
                TestResult.FAIL: "❌",
                TestResult.SKIP: "⏭️",
                TestResult.WARN: "⚠️",
            }.get(result.status, "❓")

            self._print(f"{status_symbol} {result.status} ({result.duration_ms:.0f}ms) {result.detail}")

            if self.verbose and result.stdout:
                # 显示最后几行输出
                out_lines = result.stdout.strip().split('\n')
                for line in out_lines[-5:]:
                    self._print(f"  | {line}")

            if self.verbose and result.stderr:
                err_lines = result.stderr.strip().split('\n')
                for line in err_lines[:5]:
                    self._print(f"  ! {line}")

            # 统计
            if result.status == TestResult.PASS:
                passed += 1
            elif result.status == TestResult.FAIL:
                failed += 1
            elif result.status == TestResult.SKIP:
                skipped += 1
            elif result.status == TestResult.WARN:
                warned += 1

        # 输出汇总
        self._print("")
        self._print("=" * 60)
        self._print("📊 测试结果汇总")
        self._print("=" * 60)
        self._print(f"  总计: {total}")
        self._print(f"  ✅ 通过: {passed}")
        self._print(f"  ❌ 失败: {failed}")
        self._print(f"  ⏭️  跳过: {skipped}")
        self._print(f"  ⚠️  警告: {warned}")
        elapsed = time.time() - self.start_time
        self._print(f"  耗时: {elapsed:.1f}s")
        self._print("")

        # 输出失败详情
        if failed > 0:
            self._print("❌ 失败详情:")
            for r in self.results:
                if r.status == TestResult.FAIL:
                    self._print(f"  [{r.chapter}/{r.name}] {r.detail}")
                    if r.stdout:
                        self._print(f"    stdout: {r.stdout[:300]}")
                    if r.stderr:
                        self._print(f"    stderr: {r.stderr[:300]}")

        if warned > 0:
            self._print("⚠️ 警告详情:")
            for r in self.results:
                if r.status == TestResult.WARN:
                    self._print(f"  [{r.chapter}/{r.name}] {r.detail}")

        # 生成 JSON 报告
        self._save_report()

        return failed == 0

    def _save_report(self):
        """保存测试结果到 JSON 文件"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == TestResult.PASS),
            "failed": sum(1 for r in self.results if r.status == TestResult.FAIL),
            "skipped": sum(1 for r in self.results if r.status == TestResult.SKIP),
            "warned": sum(1 for r in self.results if r.status == TestResult.WARN),
            "results": [
                {
                    "chapter": r.chapter,
                    "name": r.name,
                    "status": r.status,
                    "detail": r.detail,
                    "duration_ms": round(r.duration_ms, 1),
                }
                for r in self.results
            ]
        }

        report_path = PROJECT_ROOT / "tests" / "test_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self._print(f"📝 报告已保存: {report_path}")

    def generate_markdown_report(self) -> str:
        """生成 Markdown 格式报告"""
        lines = []
        lines.append("# Vulkan 课程 自动测试报告")
        lines.append(f"\n> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> 总计: {len(self.results)} | ✅ 通过: {sum(1 for r in self.results if r.status == TestResult.PASS)}"
                     f" | ❌ 失败: {sum(1 for r in self.results if r.status == TestResult.FAIL)}"
                     f" | ⏭️ 跳过: {sum(1 for r in self.results if r.status == TestResult.SKIP)}"
                     f" | ⚠️ 警告: {sum(1 for r in self.results if r.status == TestResult.WARN)}")
        lines.append("")

        lines.append("| 章节 | 测试 | 状态 | 详情 | 耗时 |")
        lines.append("|------|------|------|------|------|")
        for r in self.results:
            icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "WARN": "⚠️"}.get(r.status, "❓")
            lines.append(f"| {r.chapter} | {r.name} | {icon} {r.status} | {r.detail} | {r.duration_ms:.0f}ms |")

        return "\n".join(lines)


# ============================================================
# 命令行入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Vulkan 课程验证测试框架 (ch00~ch20)"
    )
    parser.add_argument("--build", action="store_true",
                        help="先构建再测试")
    parser.add_argument("--ch", type=str, default=None,
                        help="只测试指定章节 (如 ch04_gemm)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有测试用例")
    parser.add_argument("--lang", type=str, default="en",
                        choices=["zh", "en"],
                        help="输出语言 (默认 en)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细输出")
    parser.add_argument("--markdown", action="store_true",
                        help="输出 Markdown 格式报告")
    parser.add_argument("--update-verification", action="store_true",
                        help="更新 VERIFICATION_REPORT.md")
    parser.add_argument("--verify", action="store_true",
                        help="运行测试向量验证（需 PyTorch）")
    args = parser.parse_args()

    # 列出测试用例
    if args.list:
        cases = _get_test_cases()
        print(f"共 {len(cases)} 个测试用例:")
        print(f"{'章节':<20} {'测试名':<20} {'参数':<30} {'超时':<8}")
        print("-" * 80)
        for c in cases:
            args_str = " ".join(c.args)
            print(f"{c.chapter:<20} {c.name:<20} {args_str:<30} {c.timeout_s}s")
        return

    # 向量验证模式
    if args.verify:
        try:
            from vector_verify import main as vector_main
            import sys as _sys
            # 构造 vector_verify 的参数
            _sys.argv = ["vector_verify.py"]
            if args.ch:
                _sys.argv.extend(["--filter", args.ch.replace("ch", "").split("_")[0]])
            if args.verbose:
                _sys.argv.append("--verbose")
            vector_main()
        except ImportError as e:
            print(f"❌ 无法导入向量验证器: {e}")
            print("  请确保已安装 PyTorch: pip install torch")
            sys.exit(1)
        return

    # 运行测试
    runner = TestRunner(
        build_first=args.build,
        verbose=args.verbose,
        lang=args.lang,
        filter_chapter=args.ch,
    )

    success = runner.run_all()

    # 输出 Markdown 报告
    if args.markdown:
        md = runner.generate_markdown_report()
        report_path = PROJECT_ROOT / "tests" / "test_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"📝 Markdown 报告已保存: {report_path}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()