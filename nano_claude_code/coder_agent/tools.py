import subprocess
from pathlib import Path
from langchain_core.tools import tool

WORKDIR = Path.cwd()

@tool
def bash(command: str) -> str:
    """在当前工作目录运行 shell 命令并返回输出。"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"Error: {e}"

@tool
def write_file(path: str, content: str) -> str:
    """在指定路径创建一个文件并写入内容。"""
    try:
        p = (WORKDIR / path).resolve()
        if not str(p).startswith(str(WORKDIR.resolve())):
            return "Error: 越界访问！不准操作工作目录以外的文件。"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {e}"

@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """通过查找并替换的方式修改现有文件的一小部分内容。"""
    try:
        p = (WORKDIR / path).resolve()
        content = p.read_text(encoding="utf-8")
        occurrence = content.count(old_text)
        if occurrence == 0:
            return "Error: 找不到指定的文本。请先 read_file 确认内容。"
        if occurrence > 1:
            return f"Error: 找到 {occurrence} 处匹配。请提供唯一的上下文片段。"
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {e}"
