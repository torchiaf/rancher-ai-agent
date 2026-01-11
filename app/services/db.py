import os
import psycopg
import asyncio
import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

class DatabaseManager:
    """
    Manages database connections and operations.
    """
    def __init__(self):
        DB_USER = os.environ.get("DB_USER", "postgres")
        DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
        DB_HOST = os.environ.get("DB_HOST", "localhost")
        DB_PORT = os.environ.get("DB_PORT", "5432")
        DB_NAME = os.environ.get("DB_NAME", "postgres")

        self.db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    async def initialize_database(self) -> None:
        """
        Initialize database schema.
        """
        
        try:
            # Initialize LangGraph schema
            async with AsyncPostgresSaver.from_conn_string(self.db_url) as checkpointer:
                await checkpointer.setup()
                logging.debug("PostgreSQL schema initialized")
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}", exc_info=True)
            
    def notify_thread(self, thread_id: str, user_id: str, active: bool) -> None:
        """
        Notify the database about thread status.
        """
        
        async def _fn():
            async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                logging.debug(f"Notifying thread {thread_id} as {'active' if active else 'inactive'} for user {user_id}")

                await conn.execute(
                    """
                    INSERT INTO r_normalization_thread_queue (thread_id, user_id, active, processed, updated_at)
                    VALUES (%s, %s, %s, FALSE, NOW())
                    ON CONFLICT (thread_id, user_id) DO UPDATE SET
                    active = EXCLUDED.active,
                    processed = FALSE,
                    updated_at = NOW()
                    """,
                    (thread_id, user_id, active)
                )

                await conn.commit()
        
        asyncio.create_task(_fn())
        
    def notify_request(self, thread_id: str, request_id: str) -> None:
        """
        Notify the database about request status.
        """
        
        async def _fn():
            async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                logging.debug(f"Notifying request {request_id} for thread {thread_id}")

                await conn.execute(
                    """
                    INSERT INTO r_normalization_request_queue (thread_id, request_id, processed, updated_at)
                    VALUES (%s, %s, FALSE, NOW())
                    ON CONFLICT (thread_id, request_id) DO UPDATE SET
                    processed = FALSE,
                    updated_at = NOW()
                    """,
                    (thread_id, request_id)
                )

                await conn.commit()
        
        asyncio.create_task(_fn())

async def create_database_manager() -> DatabaseManager:
    """
    Factory function to create a DatabaseManager instance.

    Returns:
        An instance of DatabaseManager.
    """
    
    manager = DatabaseManager()
    await manager.initialize_database()
    
    logging.info("DatabaseManager created and initialized")
    return manager