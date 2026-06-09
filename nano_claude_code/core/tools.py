import subprocess
import sys
import venv
from pathlib import Path

from langchain_core.tools import tool

WORKDIR = Path.cwd()


# ── 通用 ──────────────────────────────────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """读取文件内容。在编辑文件前必须先读取，以获取正确的上下文。"""
    try:
        return (WORKDIR / path).resolve().read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"Error: {e}"


# ── Coder：KVM 沙箱工具 ────────────────────────────────────────────────────────

@tool
def bash(command: str) -> str:
    """在 CubeSandbox KVM 沙箱中运行 shell 命令并返回输出。每次调用共享同一个沙箱会话。"""
    try:
        from core.sandbox import get_sandbox
        return get_sandbox().run_command(command)
    except Exception as e:
        return f"Error: {e}"


@tool
def run_python(code: str) -> str:
    """在 CubeSandbox 沙箱中直接执行 Python 代码并返回输出。"""
    try:
        from core.sandbox import get_sandbox
        return get_sandbox().run_code(code)
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """在本地项目目录中创建或覆盖文件。路径相对于 WORKDIR。"""
    try:
        p = (WORKDIR / path).resolve()
        if not str(p).startswith(str(WORKDIR.resolve())):
            return "Error: 越界访问！"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {p}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """通过查找并替换的方式修改本地文件内容。old_text 必须在文件中唯一出现。"""
    try:
        p = (WORKDIR / path).resolve()
        if not str(p).startswith(str(WORKDIR.resolve())):
            return "Error: 越界访问！"
        content = p.read_text(encoding="utf-8")
        occurrence = content.count(old_text)
        if occurrence == 0:
            return "Error: 找不到指定的文本。请先用 read_file 确认文件内容。"
        if occurrence > 1:
            return f"Error: 找到 {occurrence} 处匹配。请提供唯一的上下文片段。"
        p.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Successfully edited {p}"
    except Exception as e:
        return f"Error: {e}"


# ── Researcher：搜索工具 ───────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 8) -> str:
    """使用 DuckDuckGo 搜索网络信息，无需 API Key。返回标题、URL 和摘要。"""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "未找到相关结果。"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    URL: {r.get('href', '')}\n"
                f"    {r.get('body', '')}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"搜索失败: {e}"


# ── Reviewer：代码审查工具 ────────────────────────────────────────────────────

@tool
def run_in_sandbox(file_path: str, extra_packages: str = "") -> str:
    """在临时 venv 沙箱中运行 Python 文件或 pytest 测试，完成后自动销毁环境。"""
    import tempfile

    p = (WORKDIR / file_path).resolve()
    if not str(p).startswith(str(WORKDIR.resolve())):
        return "Error: 越界访问！"
    if not p.exists():
        return f"Error: 文件 '{file_path}' 不存在"

    with tempfile.TemporaryDirectory(prefix="sandbox_") as tmpdir:
        venv_dir = Path(tmpdir) / "venv"
        try:
            venv.create(str(venv_dir), with_pip=True, clear=True)
        except Exception as e:
            return f"Error: 创建 venv 失败: {e}"

        is_win = sys.platform.startswith("win")
        bin_dir = venv_dir / ("Scripts" if is_win else "bin")
        pip_exe = str(bin_dir / "pip")
        py_exe  = str(bin_dir / "python")

        warnings = []

        req_file = WORKDIR / "requirements.txt"
        if req_file.exists():
            r = subprocess.run(
                [pip_exe, "install", "-r", str(req_file), "-q"],
                capture_output=True, text=True, timeout=180
            )
            if r.returncode != 0:
                warnings.append(f"[依赖安装警告]\n{r.stderr.strip()[:500]}")

        if extra_packages.strip():
            r = subprocess.run(
                [pip_exe, "install"] + extra_packages.split() + ["-q"],
                capture_output=True, text=True, timeout=120
            )
            if r.returncode != 0:
                warnings.append(f"[额外包警告]\n{r.stderr.strip()[:300]}")

        name = p.name.lower()
        is_test = name.startswith("test_") or name.endswith("_test.py") or "test" in p.stem.lower()

        if is_test:
            subprocess.run([pip_exe, "install", "pytest", "-q"], capture_output=True, timeout=60)
            cmd = [py_exe, "-m", "pytest", str(p), "-v", "--tb=short"]
        else:
            cmd = [py_exe, str(p)]

        try:
            r = subprocess.run(cmd, cwd=str(WORKDIR), capture_output=True, text=True, timeout=120)
            output = (r.stdout + r.stderr).strip()[:8000]
            status = "✅ 沙箱执行成功" if r.returncode == 0 else f"❌ 退出码 {r.returncode}"
            return "\n\n".join(filter(None, [status, *warnings, output]))
        except subprocess.TimeoutExpired:
            return "Error: 沙箱执行超时（120秒）"


@tool
def run_python_test(file_path: str) -> str:
    """在宿主机环境直接运行 pytest（快速，无隔离）。如需环境隔离请使用 run_in_sandbox。"""
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
    """对指定 Python 文件运行静态语法检查（py_compile）。"""
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


# ── 工具注册表 ─────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict = {}


def build_tool_registry(mcp_tools: list = None) -> dict:
    """构建工具注册表，必须在所有 run_agent() 调用之前执行。"""
    TOOL_REGISTRY.update({
        "read_file":      read_file,
        "write_file":     write_file,
        "edit_file":      edit_file,
        "bash":           bash,
        "run_python":     run_python,
        "web_search":     web_search,
        "run_in_sandbox":    run_in_sandbox,
        "run_python_test":   run_python_test,
        "run_bash_command":  run_bash_command,
        "check_code_style":  check_code_style,
    })

    if mcp_tools:
        for t in mcp_tools:
            TOOL_REGISTRY[t.name] = t

    return TOOL_REGISTRY
