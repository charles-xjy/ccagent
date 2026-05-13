import json
import uuid
from datetime import datetime
from typing import Dict, List

import redis.asyncio as redis
from langchain_core.messages import HumanMessage

MEMORY_TYPES = ("user", "feedback", "project", "reference")

_TYPE_LABELS = {
    "user":      "用户信息",
    "feedback":  "行为反馈",
    "project":   "项目上下文",
    "reference": "资源引用",
}

_EXTRACT_PROMPT = """你是一个记忆提炼助手。请从以下对话中提炼值得跨会话保留的信息。

只提炼 4 种类型：
- user: 用户偏好、知识背景、工作角色
- feedback: 用户纠正了 AI 行为，或确认了某种非显而易见的做法（含 Why 和 How to apply）
- project: 当前项目的关键决策、目标、技术约束、截止日期
- reference: 外部资源位置（URL、文件路径、服务地址）

只提炼真正有价值、未来对话会用到的信息。没有则返回空列表 []。

以 JSON 数组返回，每项格式：
{"type": "...", "name": "kebab-case-slug", "description": "一句话（判断相关性用）", "content": "具体内容"}

对话：
"""


class LongTermMemory:
    _INDEX = "ltm:index"
    _PREFIX = "ltm:mem:"

    def __init__(self, redis_url: str):
        self._url = redis_url
        self._r: redis.Redis | None = None

    async def __aenter__(self):
        self._r = redis.from_url(self._url, decode_responses=True)
        return self

    async def __aexit__(self, *_):
        if self._r:
            await self._r.aclose()

    async def save(self, type: str, name: str, description: str, content: str) -> str:
        mem_id = uuid.uuid4().hex[:8]
        await self._r.hset(f"{self._PREFIX}{mem_id}", mapping={
            "id": mem_id,
            "type": type,
            "name": name,
            "description": description,
            "content": content,
            "created_at": datetime.now().isoformat(),
        })
        await self._r.sadd(self._INDEX, mem_id)
        return mem_id

    async def load_all(self) -> List[Dict]:
        ids = await self._r.smembers(self._INDEX)
        mems = []
        for mid in ids:
            data = await self._r.hgetall(f"{self._PREFIX}{mid}")
            if data:
                mems.append(data)
        mems.sort(key=lambda x: x.get("created_at", ""))
        return mems

    async def delete(self, mem_id: str):
        await self._r.delete(f"{self._PREFIX}{mem_id}")
        await self._r.srem(self._INDEX, mem_id)

    async def delete_all(self):
        ids = await self._r.smembers(self._INDEX)
        for mid in ids:
            await self._r.delete(f"{self._PREFIX}{mid}")
        if ids:
            await self._r.delete(self._INDEX)

    def format_for_prompt(self, memories: List[Dict]) -> str:
        if not memories:
            return ""
        lines = ["## 长期记忆（跨会话积累）\n"]
        by_type: Dict[str, List] = {}
        for m in memories:
            by_type.setdefault(m["type"], []).append(m)
        for t in MEMORY_TYPES:
            if t not in by_type:
                continue
            lines.append(f"### {_TYPE_LABELS[t]}")
            for m in by_type[t]:
                lines.append(f"**{m['name']}**: {m['content']}")
            lines.append("")
        return "\n".join(lines)

    async def extract_and_save(self, model, messages: List) -> List[Dict]:
        """用 LLM 从对话消息中提炼记忆并持久化到 Redis。"""
        conv_lines = []
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            role = getattr(msg, "type", "?").upper()
            text = str(content).strip()[:400]
            if text:
                conv_lines.append(f"[{role}]: {text}")

        if not conv_lines:
            return []

        prompt = _EXTRACT_PROMPT + "\n".join(conv_lines[-40:])
        try:
            resp = await model.ainvoke([HumanMessage(content=prompt)])
            raw = resp.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.split("```")[0].strip()

            extracted = json.loads(raw)
            if not isinstance(extracted, list):
                print(f"[!] 记忆提炼：LLM 返回非列表格式，跳过。")
                return []
            if not extracted:
                return []  # LLM 明确返回 []，无需提炼

            saved = []
            for item in extracted:
                if all(k in item for k in ("type", "name", "description", "content")):
                    if item["type"] in MEMORY_TYPES:
                        mid = await self.save(
                            item["type"], item["name"],
                            item["description"], item["content"]
                        )
                        saved.append({**item, "id": mid})
            return saved
        except Exception as e:
            print(f"[!] 记忆提炼失败: {e}")
            return []
