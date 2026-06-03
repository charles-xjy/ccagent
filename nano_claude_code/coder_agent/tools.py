from langchain_core.tools import tool
from nano_claude_code.core.sandbox import get_sandbox


@tool
def bash(command: str) -> str:
    """在 CubeSandbox KVM 沙箱中运行 shell 命令并返回输出。每次调用共享同一个沙箱会话，可保留状态（环境变量、已安装包等）。"""
    try:
        return get_sandbox().run_command(command)
    except Exception as e:
        return f"Error: {e}"


@tool
def run_python(code: str) -> str:
    """在 CubeSandbox 沙箱中直接执行 Python 代码并返回输出。适合快速运行片段，无需写文件。"""
    try:
        return get_sandbox().run_code(code)
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """在沙箱工作目录中创建文件并写入内容。路径相对于沙箱工作目录 /home/user/workspace/。"""
    try:
        get_sandbox().write_file(path, content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """通过查找并替换的方式修改沙箱中已有文件的内容。old_text 必须在文件中唯一出现。"""
    try:
        sb = get_sandbox()
        content = sb.read_file(path)
        occurrence = content.count(old_text)
        if occurrence == 0:
            return "Error: 找不到指定的文本。请先确认沙箱中文件内容。"
        if occurrence > 1:
            return f"Error: 找到 {occurrence} 处匹配。请提供唯一的上下文片段。"
        sb.write_file(path, content.replace(old_text, new_text, 1))
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def read_sandbox_file(path: str) -> str:
    """读取沙箱工作目录中文件的内容。路径相对于 /home/user/workspace/。"""
    try:
        return get_sandbox().read_file(path)[:5000]
    except Exception as e:
        return f"Error: {e}"
