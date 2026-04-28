"""
=============================================================================================
##########################         1       配置模型        ####################################
=============================================================================================
"""
from typing import List, Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
import operator

model = init_chat_model(
    base_url="http://localhost:8001/v1",
    api_key="vllm-no-key",
    model="Qwen_agent",
    model_provider="openai",
    temperature=0
)

"""
==================================================================================================
############################      2          Skill 加载器        ###############################
==================================================================================================
"""
import sys
import importlib.util
from pathlib import Path

WORKDIR = Path.cwd()
SKILL_DIR = WORKDIR / "skills"


class SkillLoader:
    """从 skill.py 文件动态加载 skill"""
    
    def __init__(self, skill_file: Path):
        self.skill_file = skill_file
        self._module = None
    
    @property
    def module(self):
        """懒加载模块"""
        if self._module is None:
            self._module = self._load_skill_module()
        return self._module
    
    def _load_skill_module(self):
        """动态加载 skill 模块"""
        spec = importlib.util.spec_from_file_location("skill_module", self.skill_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["skill_module"] = module
        spec.loader.exec_module(module)
        return module
    
    @property
    def tools(self):
        """获取 tools 列表"""
        return getattr(self.module, 'tools', [])
    
    @property
    def tools_by_name(self):
        """获取 tools_by_name 字典"""
        return getattr(self.module, 'tools_by_name', {})
    
    @property
    def system_prompt(self):
        """获取 system prompt"""
        return getattr(self.module, 'SYSTEM_PROMPT', '')


# 从 skill 文件加载工具
skill_loader = SkillLoader(SKILL_DIR / "file-operations.py")
tools = skill_loader.tools
tools_by_name = skill_loader.tools_by_name
SYSTEM_PROMPT = skill_loader.system_prompt
model_with_tools = model.bind_tools(tools)


"""
=============================================================================================
##########################         3       定义状态 (from skill)        #######################
=============================================================================================
"""


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_todo: List[dict]


"""
=============================================================================================
##########################         4       定义节点 (from skill)        #######################
=============================================================================================
"""
import json
from langchain_core.messages import SystemMessage, ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END


# System Prompt (from skill)
# SYSTEM_PROMPT = """你是一个位于 {WORKDIR} 的编程助手。

# 核心操作规则：
# 1. 任务规划：对于任何包含多个步骤的任务（如先读取再编辑、创建多个文件等），你必须先创建详细计划。
# 2. 进度更新：在开始执行某个步骤前，将该任务状态更新为 'in_progress'；完成后更新为 'completed'。
# 3. 精确编辑：'edit_file' 使用的是完全字符串匹配。如果文件中存在多个相同的代码块，你必须在 'old_text' 中包含上下文'锚点'以确保匹配唯一。
# 4. 先读后改：在调用 'edit_file' 之前，必须先调用 'read_file' 确认文件内容，严禁凭空猜测代码内容。
# 5. 错误处理：如果遇到'多重匹配'错误，请重新读取文件并提供更长、更唯一的代码片段进行替换。"""


def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
    """LLM 决策节点"""
    todo_status = json.dumps(state.get("current_todo", []), ensure_ascii=False)
    
    # 使用从 skill 加载的 SYSTEM_PROMPT
    system_prompt = SystemMessage(content=SYSTEM_PROMPT)
    task_prompt = SystemMessage(content=f"\n当前任务计划: {todo_status}")
    
    response = model_with_tools.invoke([system_prompt, task_prompt] + state["messages"])
    return {"messages": [response]}


def execute_tools(state: AgentState) -> dict[str, list[BaseMessage]]:
    """工具执行节点"""
    last_message = state["messages"][-1]
    updates = {"messages": []}

    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            print(f"\n\033[33m[正在执行工具: {tool_call['name']}]\033[0m")

            if tool_call["name"] == "todo_manager":
                updates["current_todo"] = tool_call["args"]["items"]

            tool_func = tools_by_name[tool_call["name"]]
            observation = tool_func.invoke(tool_call["args"])

            updates["messages"].append(ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            ))
    return updates


def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


"""
=============================================================================================
##########################         5       构建workflow        ####################################
=============================================================================================
"""
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

if __name__ == "__main__":
    print("\033[32m=============================== LangGraph Agent ================================ \033[0m")
    print("===============================请输入您的需求，输入q,exit退出=================================")

    while True:
        try:
            query = input("\033[36m >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        
        inputs = {"messages": [HumanMessage(content=query)], "current_todo": []}

        for chunk in app.stream(inputs, stream_mode="updates", version="v2"):
            if "data" in chunk:
                for node_name, node_update in chunk["data"].items():
                    if "messages" in node_update:
                        for msg in node_update["messages"]:
                            print(f"\n================================= 节点 [{node_name}] 输出 ===============================")
                            msg.pretty_print()