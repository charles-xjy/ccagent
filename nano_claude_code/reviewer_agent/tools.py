import subprocess
import sys
import venv
from pathlib import Path
from langchain_core.tools import tool

WORKDIR = Path.cwd()


@tool
def run_in_sandbox(file_path: str, extra_packages: str = "") -> str:
    """
    在临时 venv 沙箱中运行 Python 文件或 pytest 测试，完成后自动销毁环境。

    自动处理：
    - 创建与宿主机完全隔离的 venv
    - 检测项目 requirements.txt 并自动安装依赖
    - 支持手动指定额外包（extra_packages，空格分隔，如 "requests numpy"）
    - 自动判断是否为测试文件（文件名含 test），选择 pytest 或直接运行

    Args:
        file_path: 要运行的文件路径（相对 WORKDIR）
        extra_packages: 额外 pip 包，空格分隔，可为空
    """
    import tempfile

    p = (WORKDIR / file_path).resolve()
    if not str(p).startswith(str(WORKDIR.resolve())):
        return "Error: 越界访问！"
    if not p.exists():
        return f"Error: 文件 '{file_path}' 不存在"

    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
        venv_dir = Path(tmpdir) / "venv"

        # 创建隔离 venv
        try:
            venv.create(str(venv_dir), with_pip=True, clear=True)
        except Exception as e:
            return f"Error: 创建 venv 失败: {e}"

        is_win = sys.platform.startswith("win")
        bin_dir = venv_dir / ("Scripts" if is_win else "bin")
        pip_exe  = str(bin_dir / "pip")
        py_exe   = str(bin_dir / "python")

        warnings = []

        # 安装项目依赖
        req_file = WORKDIR / "requirements.txt"
        if req_file.exists():
            r = subprocess.run(
                [pip_exe, "install", "-r", str(req_file), "-q"],
                capture_output=True, text=True, timeout=180
            )
            if r.returncode != 0:
                warnings.append(f"[依赖安装警告]\n{r.stderr.strip()[:500]}")

        # 安装额外包
        if extra_packages.strip():
            r = subprocess.run(
                [pip_exe, "install"] + extra_packages.split() + ["-q"],
                capture_output=True, text=True, timeout=120
            )
            if r.returncode != 0:
                warnings.append(f"[额外包警告]\n{r.stderr.strip()[:300]}")

        # 判断运行方式
        name = p.name.lower()
        is_test = name.startswith("test_") or name.endswith("_test.py") or "test" in p.stem.lower()

        if is_test:
            subprocess.run([pip_exe, "install", "pytest", "-q"],
                           capture_output=True, timeout=60)
            cmd = [py_exe, "-m", "pytest", str(p), "-v", "--tb=short"]
        else:
            cmd = [py_exe, str(p)]

        try:
            r = subprocess.run(
                cmd, cwd=str(WORKDIR),
                capture_output=True, text=True, timeout=120
            )
            output = (r.stdout + r.stderr).strip()[:8000]
            status = "✅ 沙箱执行成功" if r.returncode == 0 else f"❌ 退出码 {r.returncode}"
            parts = [status]
            if warnings:
                parts.extend(warnings)
            parts.append(output)
            return "\n\n".join(parts)
        except subprocess.TimeoutExpired:
            return "Error: 沙箱执行超时（120秒）"


@tool
def run_python_test(file_path: str) -> str:
    """
    在宿主机环境直接运行 pytest（快速，无隔离）。
    适合快速验证语法或确认测试结构，不涉及依赖安装。
    如需环境隔离请使用 run_in_sandbox。
    """
    p = (WORKDIR / file_path).resolve()
    if not str(p).startswith(str(WORKDIR.resolve())):
        return "Error: 越界访问！"
    try:
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
    """运行只读辅助 shell 命令（grep、ls、cat 等），用于代码审查时定位文件或查看目录。禁止执行 Python 脚本。"""
    blocked = ["rm -rf", "sudo", "shutdown", "reboot", "> /dev/", "dd if=", "mkfs",
               "python ", "python3 ", "pytest", "unittest"]
    if any(b in command for b in blocked):
        return "Error: 此命令被禁止。执行 Python 代码或测试请使用 run_in_sandbox 工具。"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"Error: {e}"


@tool
def check_code_style(file_path: str) -> str:
    """对指定 Python 文件运行静态语法检查（py_compile 兜底）。"""
    p = (WORKDIR / file_path).resolve()
    if not p.exists():
        return f"Error: 文件 {file_path} 不存在。"
    try:
        r = subprocess.run(
            ["python", "-m", "py_compile", str(p)],
            capture_output=True, text=True, timeout=30
        )
        return "语法检查通过。" if r.returncode == 0 else r.stderr.strip()
    except Exception as e:
        return f"Error: {e}"
