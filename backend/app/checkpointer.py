from __future__ import annotations

import logging
from dataclasses import dataclass

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


@dataclass
class AsyncCheckpointerBundle:
    pool: AsyncConnectionPool
    saver: AsyncPostgresSaver

    async def close(self) -> None:
        logger.info("Closing async Postgres checkpointer pool")
        await self.pool.close()


async def create_postgres_checkpointer(database_url: str) -> AsyncCheckpointerBundle:
    pool = AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
        },
        open=False,
    )

    await pool.open()
    await pool.wait()

    saver = AsyncPostgresSaver(pool)

    # Required the first time so LangGraph creates/migrates checkpoint tables.
    await saver.setup()

    logger.info("Async Postgres checkpointer is ready")
    return AsyncCheckpointerBundle(pool=pool, saver=saver)