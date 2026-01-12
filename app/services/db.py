import os
from psycopg import AsyncConnection
from datetime import datetime
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
            
    async def activate_chat(self, thread_id: str, user_id: str, active: bool) -> None:
        """
        Activate or deactivate chat for a given thread and user.
        Deactivating other chats for the user when activating a chat.
        """

        try:
            async with await AsyncConnection.connect(self.db_url) as conn:
                async with conn.cursor() as cur:
                    # Generate a default name for inactive chats
                    name = datetime.now().strftime("Chat %Y-%m-%d %H:%M:%S")
                    
                    if active:    
                        # Deactivate other chats for this user, set name only if empty
                        await cur.execute(
                            """
                            UPDATE r_chats
                            SET
                            active = FALSE,
                            name = CASE WHEN name = '' OR name IS NULL THEN %s ELSE name END,
                            updated_at = NOW()
                            WHERE user_id = %s AND chat_id != %s
                            """,
                            (name, user_id, thread_id)
                        )
                        logging.debug(f"Deactivated other chats for user {user_id}")
                    
                    await cur.execute(
                        """
                        INSERT INTO r_chats (chat_id, user_id, active, created_at, updated_at)
                        VALUES (%s, %s, %s, NOW(), NOW())
                        ON CONFLICT (chat_id, user_id) DO UPDATE SET
                        active = EXCLUDED.active,
                        name = CASE WHEN r_chats.name = '' OR r_chats.name IS NULL THEN %s ELSE r_chats.name END,
                        updated_at = NOW()
                        """,
                        (thread_id, user_id, active, name)
                    )

                    conn.commit()
                    logging.debug(f"Chat thread {thread_id} for user {user_id} set to active={active}")
        except Exception as e:
            logging.error(f"Failed to update chat thread status: {e}", exc_info=True)

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