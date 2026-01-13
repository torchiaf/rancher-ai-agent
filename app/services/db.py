import os
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