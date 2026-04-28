import os
import re
from pathlib import Path
from typing import TypedDict, List, Callable

from langchain_classic.chat_models import init_chat_model
from langchain_core.utils.uuid import uuid7
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver


# 定义技能结构
class Skill(TypedDict):
    name: str
    description: str
    content: str


def load_skills_from_directory(directory_path: str) -> List[Skill]:
    """
    从指定目录加载技能文件。
    """
    skills = []
    base_path = Path(directory_path)

    # 如果目录不存在，创建一个示例目录或抛出错误
    if not base_path.exists():
        print(f"警告：目录 {directory_path} 不存在。")
        return []

    # 遍历目录下所有的 .md 文件
    for file_path in base_path.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

            if not text:
                continue
            # 提取描述
            match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
            if match:
                metadata_str = match.group(1)
                content = match.group(2).strip()
                # 从元数据中提取 Description
                name_match = re.search(r"name:\s*(.*)", metadata_str)
                name = name_match.group(1).strip() if name_match else file_path.stem
                description_match = re.search(
                    r"description:\s*(.*?)(?=\n\S+:|$)",
                    metadata_str,
                    re.IGNORECASE | re.DOTALL
                )
                description = description_match.group(1).strip()
            skills.append({
                "name": name,
                "description": description,
                "content": content
            })

    return skills


# --- 初始化过程 ---

# 假设你的技能文件放在当前目录下的 'skills_data' 文件夹中
# 你可以在该文件夹下创建 sales_analytics.md, inventory_management.md 等
SKILLS_DIR = "claude_code"
SKILLS: List[Skill] = load_skills_from_directory(SKILLS_DIR)


# print(SKILLS)


# --- 核心工具和中间件 (逻辑保持不变，但使用动态加载的 SKILLS) ---

@tool
def load_skill(skill_name: str) -> str:
    """加载技能的详细内容。"""
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"已加载技能: {skill_name}\n\n{skill['content']}"

    available = ", ".join(s["name"] for s in SKILLS)
    return f"未找到技能 '{skill_name}'。可用技能: {available}"


import subprocess
from langchain.tools import tool
from pathlib import Path

WORKDIR = Path.cwd()


@tool
def bash(command: str) -> str:
    """
    Run a shell command in the current working directory and return its output.
    Use this to create files, list directories, or run scripts.

    Args:
        command: The full shell command to execute (e.g., 'ls -la' or 'touch index.py').
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file(path: str) -> str:
    """
    读取文件内容。在编辑文件前必须先读取，以获取正确的上下文。
    """
    try:
        p = (WORKDIR / path).resolve()
        return p.read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """
    在指定路径创建一个文件并写入内容。
    如果你需要保存代码、笔记或配置文件，请使用此工具。
    它会自动创建不存在的中间目录。

    Args:
        path: 相对路径字符串，包含文件名。
        content: 要写入文件的完整文本内容。

    Returns:
        成功或错误的提示消息。
    """
    try:
        # 1. 路径拼接与归一化：将工作目录与传入路径合并，并解析掉所有的 '..'
        # 这样可以确保我们得到的是一个唯一的、干净的绝对路径
        p = (WORKDIR / path).resolve()

        # 2. 安全闸门：检查最终生成的绝对路径是否仍然在 WORKDIR 范围内
        # 防止 LLM 或恶意用户通过 '../' 尝试修改系统关键文件（路径穿越攻击）
        if not str(p).startswith(str(WORKDIR.resolve())):
            return "Error: 越界访问！你不准去 WORKDIR 以外的地方。"

        # 3. 递归创建目录：parents=True 表示如果父目录不存在则自动创建
        # exist_ok=True 表示如果目录已存在则不报错，直接跳过
        p.parent.mkdir(parents=True, exist_ok=True)

        # 4. 执行写入：write_text 会处理文件打开、写入内容和关闭文件的所有过程
        # 注意：这会覆盖同名的旧文件
        p.write_text(content, encoding="utf-8")

        return f"Successfully wrote to {path}"

    except Exception as e:
        # 捕获权限错误、磁盘满等异常，并反馈给 LLM 让其知晓失败原因
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    通过查找并替换的方式修改现有文件的一小部分内容。
    这是修改已有代码的首选方式，因为它更安全且高效。

    Args:
        path: 文件相对路径。
        old_text: 要被替换的原始代码片段（必须完全匹配）。
        new_text: 替换后的新代码片段。
    """
    try:
        p = (WORKDIR / path).resolve()
        content = p.read_text(encoding="utf-8")

        # 核心逻辑：检查匹配次数
        occurrence = content.count(old_text)

        if occurrence == 0:
            return f"Error: 找不到指定的文本。请检查拼写、空格或换行符是否完全一致。"

        if occurrence > 1:
            return (f"Error: 在文件中找到了 {occurrence} 处匹配。替换请求具有歧义！\n"
                    f"请提供更多的上下文（包含目标行前后的代码）作为 old_text，确保它是唯一的。")

        # 只有在唯一匹配时才执行替换
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def todo_manager(items: List[dict]) -> str:
    """
    对于任何包含多个步骤的任务，你需要先调用任务规划工具来编排任务

        Args:
        items: 任务对象列表。每个对象必须严格包含以下键：
               - 'id': 任务编号 (如 "1")
               - 'text': 任务描述内容
               - 'status': 状态，只能是 "pending", "in_progress", "completed" 之一。
    """
    summary = "\n".join([f"[{i['status']}] #{i['id']}: {i['text']}" for i in items])
    return f"任务列表已更新：\n{summary}"


# tools_by_name的结构如下
# {
#     "bash": <BaseTool对象: 代表bash函数>,
#     "read_file": <BaseTool对象: 代表read_file函数>,
#     "write_file": <BaseTool对象: 代表write_file函数>,
#     "edit_file": <BaseTool对象: 代表edit_file函数>,
#     "todo_manager": <BaseTool对象: 代表todo_manager函数>
# }


class SkillMiddleware(AgentMiddleware):
    def __init__(self, skills: List[Skill]):
        self.skills = skills
        skills_list = [f"- **{s['name']}**: {s['description']}" for s in self.skills]
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        skills_addendum = (
            f"\n\n## 当前可用的业务技能 (Skill Map)\n\n{self.skills_prompt}\n\n"
            "注意：如果你需要开始处理上述领域的工作，请务必先调用 `load_skill` 获取该技能的完整详细信息。"
        )

        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        modified_request = request.override(system_message=SystemMessage(content=new_content))
        return handler(modified_request)


# --- Agent 设置 ---

model = init_chat_model(
    base_url="http://localhost:8001/v1",  # 请确保端口与 vLLM 启动端口一致
    api_key="vllm-no-key",  # vLLM 本地通常不需要 Key
    model="Qwen_agent",
    model_provider="openai",
    temperature=0
)
tools = [load_skill, bash, read_file, write_file, edit_file, todo_manager]

agent = create_agent(
    model,
    system_prompt="你是一个专业的 AI 助手。当你收到任务请求时，请检查可用技能列表，并使用 load_skill 获取详细操作指南后再回答。",
    tools=tools,
    middleware=[SkillMiddleware(SKILLS)],
    checkpointer=InMemorySaver(),
)

# --- 运行示例 ---
if __name__ == "__main__":
    # 模拟输入
    config = {"configurable": {"thread_id": str(uuid7())}}
    user_input = {"messages": [{"role": "user", "content": "帮我审查我的/home/charles/mycode/ccagent/learn_code/algorithms/quick_sort.py是否写对了？"}]}

    # 如果 SKILLS 为空，Agent 会按常规逻辑回答；如果加载成功，它会先调用工具
    if not SKILLS:
        print(f"提示：请在 {SKILLS_DIR} 目录下添加 .md 文件以启用动态技能。")

    # 正常调用
    for chunk in agent.stream(user_input, stream_mode="updates", version="v2", config=config):
        if "data" in chunk:
            # 2. 遍历 data 里的所有节点更新（比如 'agent'）
            # .items()是字典（dict）的一个非常重要的方法。它的作用是让你同时拿到字典的“钥匙”（Key）和“柜子里的东西”（Value）。
            for node_name, node_update in chunk["data"].items():
                # 3. 检查该节点是否更新了 messages
                if "messages" in node_update:
                    # 4. 遍历消息列表
                    for msg in node_update["messages"]:
                        # 5. 只有消息对象才能调用 pretty_print
                        print(
                            f"\n================================= 节点 [{node_name}] 输出 ===============================")
                        msg.pretty_print()
