#!/usr/bin/env python3
"""
pytest 兼容的测试发现入口。
直接运行:  pytest tests/test_discovery.py -v
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"
TEST_SCRIPT = PROJECT_ROOT / "tests" / "test_framework.py"


def test_list_all_cases():
    """列出所有测试用例"""
    result = subprocess.run(
        [sys.executable, str(TEST_SCRIPT), "--list"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    assert result.returncode == 0, f"list failed: {result.stderr}"
    assert "21 个测试用例" in result.stdout or "21 test cases" in result.stdout


def test_import_syntax():
    """验证脚本导入无语法错误"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_framework", str(TEST_SCRIPT))
    assert spec is not None, "无法加载 test_framework.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 验证核心组件存在
    assert hasattr(mod, "TestRunner")
    assert hasattr(mod, "TestCase")
    assert hasattr(mod, "_get_test_cases")


def test_build_all():
    """验证构建成功"""
    result = subprocess.run(
        ["cmake", "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=120
    )
    assert result.returncode == 0, f"cmake configure failed: {result.stderr[:500]}"

    result = subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "--config", "Release", "-j"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=600
    )
    assert result.returncode == 0, f"cmake build failed: {result.stderr[-500:]}"


def test_all_executables_exist():
    """验证所有可执行文件都已生成"""
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    chapters = [
        "ch00_environment", "ch01_vector_add", "ch02_sync", "ch03_memory",
        "ch04_gemm", "ch05_conv", "ch06_elemwise", "ch07_norm",
        "ch08_attention", "ch09_profiling", "ch10_arena", "ch11_quant",
        "ch12_graph", "ch13_mlp", "ch14_llm", "ch15_coopmat",
        "ch16_subgroup", "ch17_flash_attn", "ch18_quant_deep",
        "ch19_moe", "ch20_frontier_quant"
    ]
    missing = []
    for ch in chapters:
        exe = BUILD_DIR / ch / f"{ch}{exe_suffix}"
        if not exe.exists():
            missing.append(str(exe))
    assert not missing, f"缺失可执行文件:\n" + "\n".join(missing)