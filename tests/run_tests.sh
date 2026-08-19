#!/usr/bin/env bash
# ================================================================
# Vulkan 课程 测试运行脚本 (Linux / macOS)
# 用法:  tests/run_tests.sh [选项]      (Windows 请用 tests/run_tests.bat)
#
# 选项:
#   --build         先构建再测试
#   --ch 章节名     只测试指定章节 (如 ch04_gemm)
#   --list          列出所有测试用例
#   --lang en|zh    输出语言 (默认 en)
#   --verbose, -v   详细输出
#   --markdown      生成 Markdown 报告
#   --verify        运行测试向量验证（需 PyTorch）
#   --help          显示帮助
#
# 示例:
#   tests/run_tests.sh              -- 运行所有测试
#   tests/run_tests.sh --build      -- 先构建再运行所有测试
#   tests/run_tests.sh --ch ch04_gemm --verbose   -- 只测试 ch04
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD="${PYTHON:-python3}"

# 检查 Python 是否可用
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo "错误: 未找到 Python，请安装 Python 3.8+"
    exit 1
fi

echo "================================================================"
echo "  Vulkan 课程验证框架 (ch00~ch20)"
echo "================================================================"
echo

# 传递所有参数给 Python 脚本
"$PYTHON_CMD" "$SCRIPT_DIR/test_framework.py" "$@"

if [ $? -eq 0 ]; then
    echo
    echo " 测试全部通过"
else
    echo
    echo " 部分测试未通过，请查看上面的详细信息"
    echo
    echo " 提示: 用 --verbose 查看详细输出"
    echo "       tests/run_tests.sh --ch ch04_gemm --verbose"
fi
