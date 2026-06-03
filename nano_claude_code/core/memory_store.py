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
    messages_json LONGTEXT NOT NULL,
    message_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_thread_id (thread_id)
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

    async def archive(self, thread_id: str, messages: List[BaseMessage]) -> None:
        """Serialize messages to JSON and upsert into MySQL."""
        messages_json = dumps(messages)
        message_count = len(messages)
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO conversations (thread_id, messages_json, message_count) "
                    "VALUES (%s, %s, %s) "
                    "AS new_row ON DUPLICATE KEY UPDATE "
                    "messages_json = new_row.messages_json, "
                    "message_count = new_row.message_count",
                    (thread_id, messages_json, message_count),
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
                    "SELECT thread_id, message_count, created_at, updated_at "
                    "FROM conversations ORDER BY updated_at DESC"
                )
                rows = await cur.fetchall()
        return rows

    async def close(self) -> None:
        self.pool.close()
        await self.pool.wait_closed()
        logger.info("MemoryStore closed")

    async def __aenter__(self) -> "MemoryStore":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
