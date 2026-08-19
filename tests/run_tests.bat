@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ================================================================
REM Vulkan 课程 测试运行脚本
REM 用法:  tests\run_tests.bat [选项]
REM
REM 选项:
REM   --build         先构建再测试
REM   --ch 章节名     只测试指定章节 (如 ch04_gemm)
REM   --list          列出所有测试用例
REM   --lang en|zh    输出语言 (默认 en)
REM   --verbose, -v   详细输出
REM   --markdown      生成 Markdown 报告
REM   --verify        运行测试向量验证（需 PyTorch）
REM   --help          显示帮助
REM
REM 示例:
REM   tests\run_tests          -- 运行所有测试
REM   tests\run_tests --build  -- 先构建再运行所有测试
REM   tests\run_tests --ch ch04_gemm --verbose  -- 只测试 ch04
REM ================================================================

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set PYTHON_CMD=python

REM 检查 Python 是否可用
where %PYTHON_CMD% >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 未找到 Python，请安装 Python 3.8+
    pause
    exit /b 1
)

echo ================================================================
echo   Vulkan 课程验证框架 (ch00~ch20)
echo ================================================================
echo.

REM 传递所有参数给 Python 脚本
%PYTHON_CMD% "%SCRIPT_DIR%test_framework.py" %*

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 测试全部通过
) else (
    echo.
    echo ❌ 部分测试未通过，请查看上面的详细信息
    echo.
    echo 提示: 用 --verbose 查看详细输出
    echo       tests\run_tests.bat --ch ch04_gemm --verbose
)

echo.
pause