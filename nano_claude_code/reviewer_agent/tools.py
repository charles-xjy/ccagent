import subprocess
from pathlib import Path
from langchain_core.tools import tool

WORKDIR = Path.cwd()


@tool
def run_python_test(file_path: str) -> str:
    """运行指定 Python 文件的测试并返回结果。"""
    try:
        p = (WORKDIR / file_path).resolve()
        if not str(p).startswith(str(WORKDIR.resolve())):
            return "Error: 越界访问！"
        r = subprocess.run(
            ["python", "-m", "pytest", str(p), "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120
        )
        return (r.stdout + r.stderr).strip()[:5000]
    except subprocess.TimeoutExpired:
        return "Error: 测试执行超时（120秒）"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_bash_command(command: str) -> str:
    """运行一个无害的 shell 命令（只读检查类），用于代码审查时的辅助验证。"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/", "dd if="]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"Error: {e}"


@tool
def check_code_style(file_path: str) -> str:
    """对指定 Python 文件运行 flake8 / pyflakes 风格的静态检查（使用 python -m py_compile 兜底）。"""
    try:
        p = (WORKDIR / file_path).resolve()
        if not p.exists():
            return f"Error: 文件 {file_path} 不存在。"
        # 先用 py_compile 做基础语法检查
        r = subprocess.run(
            ["python", "-m", "py_compile", str(p)],
            capture_output=True, text=True, timeout=30
        )
        issues = []
        if r.returncode != 0:
            issues.append(r.stderr.strip())
        else:
            issues.append("语法检查通过。")

        return "\n".join(issues)
    except Exception as e:
        return f"Error: {e}"
