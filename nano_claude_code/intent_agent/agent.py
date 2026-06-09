from datetime import datetime
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool as lc_tool
from langgraph.types import interrupt

from core.tools import WORKDIR


def _strip_thinking(content: str) -> str:
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


def create_intent_tool(model, project_name: str = ""):
    """
    返回 analyze_intent 工具，挂载到 Supervisor 的工具集。
    Supervisor 对复杂任务主动调用，内部通过 interrupt 与用户交互。
    """

    @lc_tool
    async def analyze_intent(user_request: str) -> str:
        """
        当需求不确定时，与用户进行交互式 Q&A，澄清后生成 Markdown 需求文档。

        调用时机（需求不确定）：
        - 用户描述了目标但未说明实现方式（「加个分享功能」「支持多租户」）
        - 新功能涉及数据结构、接口设计或与现有代码的集成方式尚未确定
        - 项目已有代码，但后续扩展方向仍存在多种可能性

        无需调用（需求已明确）：
        - 修 bug、改样式、重构指定模块等目标和边界清晰的任务
        - 用户已提供足够细节（接口、字段、行为均已说明）
        """
        history = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_request),
        ]

        # ── Q&A 循环 ─────────────────────────────────────────────
        while True:
            response = await model.ainvoke(history)
            history.append(response)

            model_ready = "[READY]" in (response.content or "")
            display = _strip_thinking((response.content or "").replace("[READY]", "")).strip()

            user_input = interrupt(f"❓ 需求确认\n\n{display}")
            user_str = str(user_input).strip()

            if user_str.lower() == "skip":
                return f"[需求澄清已跳过]\n原始需求：{user_request}"

            if user_str.lower() in ("done", "") or model_ready:
                if user_str and user_str.lower() not in ("done", ""):
                    history.append(HumanMessage(content=user_str))
                break

            history.append(HumanMessage(content=user_str))

        # ── 文档生成 + 确认循环 ──────────────────────────────────
        while True:
            doc_response = await model.ainvoke(
                history + [HumanMessage(content=_DOC_PROMPT)]
            )
            final_content = _strip_thinking(doc_response.content or "")

            confirm = interrupt(f"📄 需求文档\n\n{final_content}")
            confirm_str = str(confirm).strip()

            if confirm_str.lower() == "skip":
                break

            if confirm_str.lower() in ("ok", "confirm", "yes", "确认", "好的"):
                break

            history.append(doc_response)
            history.append(HumanMessage(content=f"修改意见：{confirm_str}\n请按此修订文档。"))

        # ── 保存文档 ─────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"requirements_{timestamp}.md"
        save_dir = Path(WORKDIR) / project_name if project_name else Path(WORKDIR)
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / filename
        save_path.write_text(final_content, encoding="utf-8")
        print(f"\n\033[32m[OK] 需求文档已保存至 {save_path}\033[0m")

        return f"[需求分析完成，文档已保存至 {save_path}]\n\n{final_content}"

    return analyze_intent
