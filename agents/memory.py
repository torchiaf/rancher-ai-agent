import os

from clients import RedisClient

class MemoryAgent:
    def __init__(self, cache_client_url: str):
        self.cache_client = RedisClient(cache_client_url)
        # TODO create db_client
        # self.db_client = MySql(mysql_client_url)

    async def destroy(self):
        await self.cache_client.disconnect()
        # TODO disconnect db_client
        # await self.db_client.disconnect()
    
    async def store_messages(self, session_id: str, request_id: str, text: str = "", max_history: int = 1000):
        # TODO when available, async store messages on DB instance by {session_id, request_id}
        # await self.db_client.store(session_id=session_id, request_id=request_id ...
        
        await self.cache_client.store(
            session_id=session_id,
            text=text,
            max_history=max_history
        )

    async def fetch_messages(self, session_id: str) -> list[str]:
        return await self.cache_client.fetch(session_id=session_id)
    
    # TODO when available, fetch messages from the DB
    # async def fetch_history(self, session_id: str) -> list[str]:

async def create_memory_agent(cache_client_url: str) -> MemoryAgent:
    """
    Creates a MemoryAgent instance and connects it to the cache and database.

    Returns:
        A connected MemoryAgent instance.
    """
    client = MemoryAgent(cache_client_url)
    # TODO add db_client url

    await client.cache_client.connect()
    # TODO connect db_client
    # await client.db_client.connect()

    return client
