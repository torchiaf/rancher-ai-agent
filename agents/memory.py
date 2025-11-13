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
    
    async def create_session(self, user_id: str):
        return await self.cache_client.create_session(user_id)

    async def fetch_sessions(self, user_id: str) -> list[str]:
        return await self.cache_client.fetch_sessions(user_id)

    async def get_current_session(self, user_id: str) -> str | None:
        sessions = await self.cache_client.fetch_sessions(user_id)

        # Fetch active session
        for session in reversed(sessions):
            if session.get("status") == "active":
                return session.get("session_id")

        return None

    async def store_chunk(self, session_id: str, request_id: str, text: str = "", role: str = "agent"):
        await self.cache_client.store_chunk(
            session_id=session_id,
            request_id=request_id,
            text=text,
            role=role
        )

    async def fetch_messages(self, session_id: str, max_count: int = 10, role_filter: list[str] | None = None) -> list[str]:
        return await self.cache_client.fetch_messages(session_id, max_count, role_filter)

async def create_memory_agent(cache_client_url: str) -> MemoryAgent:
    """
    Creates a MemoryAgent instance and connects it to the cache and database.

    Returns:
        A connected MemoryAgent instance.
    """
    agent = MemoryAgent(cache_client_url)

    await agent.cache_client.connect()

    return agent
