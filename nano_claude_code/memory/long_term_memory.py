import json
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import redis.asyncio as redis
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

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

_DEDUP_PROMPT = """你是记忆去重助手。判断新记忆与已有相似记忆的关系，选择最合适的操作。

新记忆:
  类型: {type}  名称: {name}
  内容: {content}

已有相似记忆（相似度从高到低）:
{candidates}

请选择操作（只能选一个）:
- ADD    : 新记忆是全新知识，与已有记忆不重叠，应当新增
- UPDATE : 新记忆是对某条已有记忆的更新或补充，合并后更完整
- NOOP   : 新记忆与某条已有记忆完全相同，或已有记忆信息更准确，无需变更

以 JSON 返回（严格格式，不要 markdown 代码块）:
{{"action": "ADD" | "UPDATE" | "NOOP", "target_id": "<目标记忆ID，仅 UPDATE 时填写>", "merged_content": "<合并后的内容，仅 UPDATE 时填写>"}}
"""

_MODELSCOPE_MODEL_ID = "BAAI/bge-small-zh-v1.5"
_EMBED_DIM = 512
_DEDUP_SIM_THRESHOLD = 0.78
_SEARCH_TOP_K = 8
_INDEX_NAME = "ltm:idx"
_INDEX_PREFIX = "ltm:mem:"
_SET_KEY = "ltm:index"

# HNSW 参数：M 控制图连接数（越大越准但内存更多），EF 控制构建/查询精度
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_RUNTIME = 10


@lru_cache(maxsize=1)
def _get_embed_model():
    from modelscope import snapshot_download
    from sentence_transformers import SentenceTransformer
    model_dir = snapshot_download(_MODELSCOPE_MODEL_ID)
    return SentenceTransformer(model_dir)


def _embed(text: str) -> np.ndarray:
    return _get_embed_model().encode(text, normalize_embeddings=True).astype(np.float32)


def _to_bytes(vec: np.ndarray) -> bytes:
    return vec.tobytes()


def _d(val) -> str:
    """bytes → str，兼容 decode_responses=False 的 Redis 连接。"""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val or ""


def _parse_json(raw: str) -> object:
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```")[0].strip()
    return json.loads(raw)


class LongTermMemory:

    def __init__(self, redis_url: str, memory_store=None):
        self._url = redis_url
        self._r: redis.Redis | None = None
        self._store = memory_store

    async def __aenter__(self):
        # decode_responses=False：存取二进制向量字段
        self._r = redis.from_url(self._url, decode_responses=False)
        await self._ensure_index()
        return self

    async def __aexit__(self, *_):
        if self._r:
            await self._r.aclose()

    # ── HNSW 索引管理 ────────────────────────────────────────────

    async def _ensure_index(self) -> None:
        """索引不存在时自动创建。"""
        try:
            await self._r.ft(_INDEX_NAME).info()
        except Exception:
            schema = (
                TagField("type"),
                TextField("name"),
                TextField("content"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": str(_EMBED_DIM),
                        "DISTANCE_METRIC": "COSINE",
                        "M": str(_HNSW_M),
                        "EF_CONSTRUCTION": str(_HNSW_EF_CONSTRUCTION),
                        "EF_RUNTIME": str(_HNSW_EF_RUNTIME),
                    },
                ),
            )
            idx_def = IndexDefinition(prefix=[_INDEX_PREFIX], index_type=IndexType.HASH)
            await self._r.ft(_INDEX_NAME).create_index(schema, definition=idx_def)
            print("\033[32m[记忆] HNSW 索引已创建。\033[0m")

    # ── 基础读写 ─────────────────────────────────────────────────

    async def save(
        self,
        type: str,
        name: str,
        description: str,
        content: str,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        mem_id = uuid.uuid4().hex[:8]
        created_at = datetime.now().isoformat()
        if embedding is None:
            embedding = _embed(f"{name}: {content}")
        mapping = {
            b"id":          mem_id.encode(),
            b"type":        type.encode(),
            b"name":        name.encode(),
            b"description": description.encode(),
            b"content":     content.encode(),
            b"created_at":  created_at.encode(),
            b"embedding":   _to_bytes(embedding),   # float32 原始字节
        }
        await self._r.hset(f"{_INDEX_PREFIX}{mem_id}", mapping=mapping)
        await self._r.sadd(_SET_KEY, mem_id)
        if self._store is not None:
            try:
                await self._store.save_ltm(mem_id, type, name, description, content, created_at)
            except Exception as e:
                print(f"[!] 长期记忆双写 MySQL 失败: {e}")
        return mem_id

    async def update(
        self,
        mem_id: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        mapping: Dict[bytes, bytes] = {}
        if content is not None:
            mapping[b"content"] = content.encode()
            if embedding is None:
                name_b = await self._r.hget(f"{_INDEX_PREFIX}{mem_id}", b"name")
                embedding = _embed(f"{_d(name_b)}: {content}")
        if description is not None:
            mapping[b"description"] = description.encode()
        if embedding is not None:
            mapping[b"embedding"] = _to_bytes(embedding)
        if mapping:
            await self._r.hset(f"{_INDEX_PREFIX}{mem_id}", mapping=mapping)
        if self._store is not None and content is not None:
            try:
                await self._store.update_ltm(mem_id, content=content)
            except Exception as e:
                print(f"[!] 长期记忆更新 MySQL 失败: {e}")

    async def load_all(self) -> List[Dict]:
        ids_b = await self._r.smembers(_SET_KEY)
        if not ids_b and self._store is not None:
            await self._restore_from_mysql()
            ids_b = await self._r.smembers(_SET_KEY)
        mems = []
        for mid_b in ids_b:
            raw = await self._r.hgetall(f"{_INDEX_PREFIX}{_d(mid_b)}")
            if raw:
                mems.append({
                    _d(k): _d(v) if k != b"embedding" else None
                    for k, v in raw.items()
                })
        mems.sort(key=lambda x: x.get("created_at", ""))
        return mems

    async def _restore_from_mysql(self) -> None:
        try:
            rows = await self._store.load_all_ltm()
            if not rows:
                return
            for row in rows:
                mid = row["id"]
                emb = _embed(f"{row['name']}: {row.get('content', '')}")
                mapping = {
                    b"id":          mid.encode(),
                    b"type":        row["type"].encode(),
                    b"name":        row["name"].encode(),
                    b"description": row.get("description", "").encode(),
                    b"content":     row.get("content", "").encode(),
                    b"created_at":  str(row.get("created_at", "")).encode(),
                    b"embedding":   _to_bytes(emb),
                }
                await self._r.hset(f"{_INDEX_PREFIX}{mid}", mapping=mapping)
                await self._r.sadd(_SET_KEY, mid)
            print(f"\033[33m[记忆] 从 MySQL 恢复了 {len(rows)} 条长期记忆到 Redis。\033[0m")
        except Exception as e:
            print(f"[!] 从 MySQL 恢复长期记忆失败: {e}")

    async def delete(self, mem_id: str):
        await self._r.delete(f"{_INDEX_PREFIX}{mem_id}")
        await self._r.srem(_SET_KEY, mem_id)
        if self._store is not None:
            try:
                await self._store.delete_ltm(mem_id)
            except Exception as e:
                print(f"[!] 长期记忆删除 MySQL 失败: {e}")

    async def delete_all(self):
        ids_b = await self._r.smembers(_SET_KEY)
        for mid_b in ids_b:
            await self._r.delete(f"{_INDEX_PREFIX}{_d(mid_b)}")
        if ids_b:
            await self._r.delete(_SET_KEY)
        if self._store is not None:
            try:
                await self._store.delete_all_ltm()
            except Exception as e:
                print(f"[!] 长期记忆清空 MySQL 失败: {e}")

    # ── HNSW 搜索 ────────────────────────────────────────────────

    async def search(self, query: str, top_k: int = _SEARCH_TOP_K) -> List[Dict]:
        """
        HNSW KNN 取候选 → BM25 re-ranking 返回 top_k。
        COSINE distance = 1 - cosine_similarity（值越小越相似）。
        """
        if not query.strip():
            return (await self.load_all())[:top_k]

        query_vec = _embed(query)
        candidate_k = min(top_k * 3, 30)

        q = (
            Query(f"*=>[KNN {candidate_k} @embedding $vec AS dist]")
            .sort_by("dist")
            .return_fields("id", "type", "name", "description", "content", "created_at", "dist")
            .paging(0, candidate_k)
            .dialect(2)
        )
        try:
            results = await self._r.ft(_INDEX_NAME).search(
                q, query_params={"vec": _to_bytes(query_vec)}
            )
        except Exception as e:
            print(f"[!] HNSW 搜索失败，降级到全量加载: {e}")
            return (await self.load_all())[:top_k]

        candidates = []
        for doc in results.docs:
            dist = float(_d(getattr(doc, "dist", "1.0")))
            candidates.append({
                "id":          _d(doc.id).removeprefix(_INDEX_PREFIX),
                "type":        _d(getattr(doc, "type", b"")),
                "name":        _d(getattr(doc, "name", b"")),
                "description": _d(getattr(doc, "description", b"")),
                "content":     _d(getattr(doc, "content", b"")),
                "created_at":  _d(getattr(doc, "created_at", b"")),
                "_vec_sim":    1.0 - dist,   # 转换为相似度
            })

        if not candidates:
            return []

        # BM25 re-ranking
        corpus = [f"{m['name']} {m['description']} {m['content']}" for m in candidates]
        bm25_scores = BM25Okapi([doc.split() for doc in corpus]).get_scores(query.split()).tolist()

        max_v = max((m["_vec_sim"] for m in candidates), default=1.0) or 1.0
        max_b = max(bm25_scores, default=1.0) or 1.0

        ranked = sorted(
            enumerate(candidates),
            key=lambda t: 0.6 * (t[1]["_vec_sim"] / max_v) + 0.4 * (bm25_scores[t[0]] / max_b),
            reverse=True,
        )
        return [m for _, m in ranked[:top_k]]

    async def _knn_similar(self, embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """去重时用 HNSW 快速找最近邻，只返回超过阈值的结果。"""
        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS dist]")
            .sort_by("dist")
            .return_fields("id", "type", "name", "content", "dist")
            .paging(0, top_k)
            .dialect(2)
        )
        try:
            results = await self._r.ft(_INDEX_NAME).search(
                q, query_params={"vec": _to_bytes(embedding)}
            )
        except Exception:
            return []

        out = []
        for doc in results.docs:
            dist = float(_d(getattr(doc, "dist", "1.0")))
            sim = 1.0 - dist
            if sim >= _DEDUP_SIM_THRESHOLD:
                out.append({
                    "id":      _d(doc.id).removeprefix(_INDEX_PREFIX),
                    "type":    _d(getattr(doc, "type", b"")),
                    "name":    _d(getattr(doc, "name", b"")),
                    "content": _d(getattr(doc, "content", b"")),
                    "sim":     sim,
                })
        return out

    # ── Prompt 注入 ──────────────────────────────────────────────

    async def search_for_prompt(self, query: str, top_k: int = _SEARCH_TOP_K) -> str:
        return self.format_for_prompt(await self.search(query, top_k=top_k))

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

    # ── 去重/合并 Pipeline ───────────────────────────────────────

    async def _dedup_check(
        self, model, type: str, name: str, content: str, new_emb: np.ndarray
    ) -> Dict:
        candidates = await self._knn_similar(new_emb)
        if not candidates:
            return {"action": "ADD"}

        candidates_text = "\n".join(
            f"  [{m['id']}] (相似度 {m['sim']:.2f}) 类型: {m['type']}  名称: {m['name']}\n"
            f"  内容: {m['content']}"
            for m in candidates
        )
        prompt = _DEDUP_PROMPT.format(
            type=type, name=name, content=content, candidates=candidates_text
        )
        try:
            resp = await model.ainvoke([HumanMessage(content=prompt)])
            return _parse_json(resp.content)
        except Exception as e:
            print(f"[!] 去重 LLM 调用失败，默认 ADD: {e}")
            return {"action": "ADD"}

    # ── 提炼并保存（含去重） ──────────────────────────────────────

    async def extract_and_save(self, model, messages: List) -> List[Dict]:
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
            extracted = _parse_json(resp.content)
            if not isinstance(extracted, list) or not extracted:
                return []
        except Exception as e:
            print(f"[!] 记忆提炼失败: {e}")
            return []

        saved = []
        for item in extracted:
            if not all(k in item for k in ("type", "name", "description", "content")):
                continue
            if item["type"] not in MEMORY_TYPES:
                continue

            new_emb = _embed(f"{item['name']}: {item['content']}")
            decision = await self._dedup_check(
                model, item["type"], item["name"], item["content"], new_emb
            )
            action = decision.get("action", "ADD")

            if action == "ADD":
                mid = await self.save(
                    item["type"], item["name"], item["description"], item["content"],
                    embedding=new_emb,
                )
                saved.append({**item, "id": mid, "action": "ADD"})
                print(f"\033[32m[记忆] ADD  [{item['type']}] {item['name']}\033[0m")

            elif action == "UPDATE":
                target_id = decision.get("target_id", "")
                merged = decision.get("merged_content", item["content"])
                if target_id:
                    await self.update(target_id, content=merged, embedding=new_emb)
                    saved.append({**item, "id": target_id, "action": "UPDATE"})
                    print(f"\033[33m[记忆] UPDATE [{item['type']}] {item['name']} → {target_id}\033[0m")
                else:
                    mid = await self.save(
                        item["type"], item["name"], item["description"], item["content"],
                        embedding=new_emb,
                    )
                    saved.append({**item, "id": mid, "action": "ADD"})

            else:  # NOOP
                print(f"\033[36m[记忆] NOOP  [{item['type']}] {item['name']} (重复，跳过)\033[0m")

        return saved
