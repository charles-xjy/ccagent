import logging
from typing import List, Optional

import aiomysql
from langchain_core.load import dumps, loads
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL UNIQUE,
    title VARCHAR(100) DEFAULT NULL,
    messages_json LONGTEXT NOT NULL,
    message_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_thread_id (thread_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

ADD_TITLE_COLUMN_SQL = """
ALTER TABLE conversations ADD COLUMN title VARCHAR(100) DEFAULT NULL;
"""

CREATE_LTM_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS long_term_memories (
    id VARCHAR(8) PRIMARY KEY,
    type VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    description VARCHAR(255),
    content TEXT,
    created_at DATETIME,
    INDEX idx_type (type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


class MemoryStore:
    """MySQL-backed long-term conversation archive.

    Stores serialized LangChain message histories keyed by thread_id.
    Provides archive/load/delete/list operations for conversation persistence
    beyond Redis checkpoint TTL.
    """

    def __init__(self, pool: aiomysql.Pool):
        self.pool = pool

    @classmethod
    async def create(
        cls,
        host: str = "127.0.0.1",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "ccagent",
        **kwargs,
    ) -> "MemoryStore":
        pool = await aiomysql.create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            db=database,
            autocommit=True,
            charset="utf8mb4",
            **kwargs,
        )
        store = cls(pool)
        await store._ensure_tables()
        logger.info("MemoryStore connected to MySQL %s:%d/%s", host, port, database)
        return store

    async def _ensure_tables(self) -> None:
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(CREATE_TABLE_SQL)
                await cur.execute(CREATE_LTM_TABLE_SQL)
                # 检查 title 列是否存在，不存在才加
                await cur.execute(
                    "SELECT COUNT(*) FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'conversations' AND COLUMN_NAME = 'title'"
                )
                row = await cur.fetchone()
                if row and row[0] == 0:
                    await cur.execute(ADD_TITLE_COLUMN_SQL)

    async def archive(self, thread_id: str, messages: List[BaseMessage], title: Optional[str] = None) -> None:
        """Serialize messages to JSON and upsert into MySQL."""
        messages_json = dumps(messages)
        message_count = len(messages)
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO conversations (thread_id, title, messages_json, message_count) "
                    "VALUES (%s, %s, %s, %s) "
                    "AS new_row ON DUPLICATE KEY UPDATE "
                    "title = COALESCE(new_row.title, conversations.title), "
                    "messages_json = new_row.messages_json, "
                    "message_count = new_row.message_count",
                    (thread_id, title, messages_json, message_count),
                )
        logger.info("Archived thread '%s' (%d messages) to MySQL", thread_id, message_count)

    async def load(self, thread_id: str) -> Optional[List[BaseMessage]]:
        """Load and deserialize messages from MySQL. Returns None if not found."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT messages_json FROM conversations WHERE thread_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        messages = loads(row[0])
        logger.info("Loaded thread '%s' (%d messages) from MySQL", thread_id, len(messages))
        return messages

    async def delete(self, thread_id: str) -> bool:
        """Delete a conversation archive. Returns True if a row was deleted."""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM conversations WHERE thread_id = %s",
                    (thread_id,),
                )
                deleted = cur.rowcount > 0
        if deleted:
            logger.info("Deleted thread '%s' from MySQL", thread_id)
        return deleted

    async def list_threads(self) -> List[dict]:
        """List all archived conversations with metadata."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT thread_id, title, message_count, created_at, updated_at "
                    "FROM conversations ORDER BY updated_at DESC"
                )
                rows = await cur.fetchall()
        return rows

    async def save_ltm(self, mem_id: str, type: str, name: str, description: str, content: str, created_at: str) -> None:
        """将一条长期记忆写入 MySQL（双写备份）。"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT IGNORE INTO long_term_memories (id, type, name, description, content, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (mem_id, type, name, description, content, created_at),
                )

    async def update_ltm(self, mem_id: str, content: str) -> None:
        """更新一条长期记忆的内容。"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE long_term_memories SET content = %s WHERE id = %s",
                    (content, mem_id),
                )

    async def delete_ltm(self, mem_id: str) -> None:
        """删除一条长期记忆。"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM long_term_memories WHERE id = %s", (mem_id,))

    async def delete_all_ltm(self) -> None:
        """清空全部长期记忆。"""
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM long_term_memories")

    async def load_all_ltm(self) -> List[dict]:
        """从 MySQL 加载全部长期记忆，用于 Redis 崩溃后恢复。"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM long_term_memories ORDER BY created_at ASC")
                return await cur.fetchall()

    async def close(self) -> None:
        self.pool.close()
        await self.pool.wait_closed()
        logger.info("MemoryStore closed")

    async def __aenter__(self) -> "MemoryStore":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
