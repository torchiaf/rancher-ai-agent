from clients import RedisClient

"""
MemoryAgent provides an interface for storing and retrieving
conversation history using a cache.

In the future, this can be extended to support database storage as well by implementing a database client.

The database client can be added as an additional parameter to the MemoryAgent constructor,
and the store and fetch methods can be updated to interact with both the cache and the database as needed.

Or,

it can be designed as a separate DB pod that subscribes to the redis stream and stores the messages in the database.
"""
class MemoryAgent:
    def __init__(self, cache_client_url: str):
        self.cache_client = RedisClient(cache_client_url)

    async def destroy(self):
        await self.cache_client.disconnect()
    
    async def store_messages(self, session_id: str, request_id: str, text: str = ""):        
        await self.cache_client.store(
            session_id=session_id,
            request_id=request_id,
            text=text
        )

    async def fetch_messages(self, session_id: str, max_count: int = 10, reverse: bool = False) -> list[str]:
        return await self.cache_client.fetch(session_id, max_count, reverse)

async def create_memory_agent(cache_client_url: str) -> MemoryAgent:
    """
    Creates a MemoryAgent instance and connects it to the cache and database.

    Returns:
        A connected MemoryAgent instance.
    """
    agent = MemoryAgent(cache_client_url)

    await agent.cache_client.connect()

    return agent
