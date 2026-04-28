"""
File Operations Skill
从 skill.md 加载的工具实现
"""
from typing import List
from langchain.tools import tool
from pathlib import Path
import subprocess


WORKDIR = Path.cwd()


@tool
def bash(command: str) -> str:
    """运行 shell 命令"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, 
                           capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file(path: str) -> str:
    """读取文件内容。在编辑文件前必须先读取。"""
    try:
        p = (WORKDIR / path).resolve()
        return p.read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """创建或写入文件"""
    try:
        p = (WORKDIR / path).resolve()
        if not str(p).startswith(str(WORKDIR.resolve())):
            return "Error: 越界访问！"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """替换文件中的文本"""
    try:
        p = (WORKDIR / path).resolve()
        content = p.read_text(encoding="utf-8")
        occurrence = content.count(old_text)
        if occurrence == 0:
            return f"Error: 找不到指定的文本。"
        if occurrence > 1:
            return f"Error: 在文件中找到了 {occurrence} 处匹配。替换请求具有歧义！"
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def todo_manager(items: List[dict]) -> str:
    """任务规划工具"""
    summary = "\n".join([f"[{i['status']}] #{i['id']}: {i['text']}" for i in items])
    return f"任务列表已更新：\n{summary}"


# Skill 导出
tools = [bash, read_file, write_file, edit_file, todo_manager]
tools_by_name = {t.name: t for t in tools}


# System Prompt (from skill.md)
SYSTEM_PROMPT = f"""你是一个位于 {WORKDIR} 的编程助手。

核心操作规则：
1. 任务规划：对于任何包含多个步骤的任务（如先读取再编辑、创建多个文件等），你必须先创建详细计划。
2. 进度更新：在开始执行某个步骤前，将该任务状态更新为 'in_progress'；完成后更新为 'completed'。
3. 精确编辑：'edit_file' 使用的是完全字符串匹配。如果文件中存在多个相同的代码块，你必须在 'old_text' 中包含上下文'锚点'以确保匹配唯一。
4. 先读后改：在调用 'edit_file' 之前，必须先调用 'read_file' 确认文件内容，严禁凭空猜测代码内容。
5. 错误处理：如果遇到'多重匹配'错误，请重新读取文件并提供更长、更唯一的代码片段进行替换。"""