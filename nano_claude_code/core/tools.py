from pathlib import Path
from langchain_core.tools import tool

WORKDIR = Path.cwd()

@tool
def read_file(path: str) -> str:
    """读取文件内容。在编辑文件前必须先读取，以获取正确的上下文。"""
    try:
        p = (WORKDIR / path).resolve()
        return p.read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"Error: {e}"
