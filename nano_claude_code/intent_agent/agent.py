from datetime import datetime
from pathlib import Path
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import interrupt

from core.tools import WORKDIR


def _strip_thinking(content: str) -> str:
    """去除模型输出中的思考过程（</think> 之前的内容）。"""
    for tag in ("</think>", "</thinking>"):
        if tag in content:
            return content.split(tag, 1)[1].strip()
    return content


_SYSTEM_PROMPT = (
    "你是一个软件需求分析专家，负责在开发前与用户充分沟通，确认需求细节。\n\n"
    "每轮你可以：\n"
    "1. 提出 2-4 个关键确认问题（如功能、架构、技术选型、输入输出、集成方式等）\n"
    "2. 若已掌握足够信息，在回复末尾加上 [READY] 标记，表示可以生成需求文档了\n\n"
    "规则：问题要简洁，优先问最关键的，避免重复已确认的内容。"
)

_DOC_PROMPT = (
    "根据以上完整的对话内容，生成一份 Markdown 需求文档，格式如下：\n\n"
    "# <项目名>\n\n"
    "## 项目概述\n\n"
    "## 核心功能\n\n"
    "## 技术架构\n\n"
    "## 实现细节\n\n"
    "## 输入输出规范\n\n"
    "## 特殊要求\n\n"
    "内容要具体可操作，直接供开发者参考实现。不要输出 [READY] 标记。"
)


def create_intent_node(model, project_name: str = ""):
    """返回意图分析节点函数，挂载到主图的 START 之后。"""

    async def intent_analysis_node(state: Dict) -> Dict:
        # 恢复旧会话时（已有 AI 消息）跳过意图分析，直接让 manager 继续
        if any(isinstance(m, AIMessage) for m in state.get("messages", [])):
            return {}

        # 取最后一条用户消息作为原始需求
        user_request = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break
        if not user_request:
            return {}

        history = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_request),
        ]

        # ── Q&A 循环：直到用户完成回答或模型输出 [READY] ──────────
        while True:
            response = await model.ainvoke(history)
            history.append(response)

            model_ready = "[READY]" in (response.content or "")
            display = _strip_thinking((response.content or "").replace("[READY]", "")).strip()

            user_input = interrupt(f"❓ 需求确认\n\n{display}")
            user_str = str(user_input).strip()

            if user_str.lower() == "skip":
                return {}  # 跳过意图分析，直接让 manager 处理原始需求

            if user_str.lower() in ("done", "") or model_ready:
                # 用户表示已回答完毕，或模型认为信息足够
                if user_str and user_str.lower() not in ("done", ""):
                    history.append(HumanMessage(content=user_str))
                break

            history.append(HumanMessage(content=user_str))

        # ── 文档生成 + 确认循环 ───────────────────────────────────
        while True:
            doc_response = await model.ainvoke(
                history + [HumanMessage(content=_DOC_PROMPT)]
            )
            final_content = _strip_thinking(doc_response.content or "")

            confirm = interrupt(f"📄 需求文档\n\n{final_content}")
            confirm_str = str(confirm).strip()

            if confirm_str.lower() == "skip":
                return {}  # 跳过文档确认

            if confirm_str.lower() in ("ok", "confirm", "yes", "确认", "好的"):
                break  # 文档已确认

            # 用户有修改意见 → 带入上下文重新生成
            history.append(doc_response)
            history.append(HumanMessage(content=f"修改意见：{confirm_str}\n请按此修订文档。"))

        # ── 写入文件（存入项目文件夹，不存在则创建）────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"requirements_{timestamp}.md"
        if project_name:
            save_dir = Path(WORKDIR) / project_name
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = Path(WORKDIR)
        save_path = save_dir / filename
        save_path.write_text(final_content, encoding="utf-8")
        print(f"\n\033[32m[OK] 需求文档已保存至 {save_path}\033[0m")

        return {"messages": [HumanMessage(
            content=f"[需求分析完成，文档已保存至 {save_path}]\n\n{final_content}"
        )]}

    return intent_analysis_node
