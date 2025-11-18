from clients import RedisClient, MySQLClient

"""
MemoryAgent

 - Provides an interface for storing and retrieving conversation history using a cache.
 - Provides a DB interface for chat information retrieval.
"""
class MemoryAgent:
    def __init__(self, cache_client_url: str, db_host: str, db_port: int, db_user: str, db_password: str, db_database: str):
        self.cache_client = RedisClient(cache_client_url)
        self.db_client = MySQLClient(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_database
        )

    async def destroy(self):
        await self.cache_client.disconnect()

    async def store_chunk(self, chat_id: str, request_id: str, text: str = "", role: str = "agent"):
        await self.cache_client.store_chunk(
            chat_id=chat_id,
            request_id=request_id,
            text=text,
            role=role
        )

    async def fetch_messages(self, chat_id: str, user_id: str, max_count: int = 10, role_filter: list[str] | None = None) -> list[str]:
        return await self.cache_client.fetch_messages(chat_id, user_id, max_count, role_filter)

    async def create_chat(self, user_id: str):
        return await self.cache_client.create_chat(user_id)

    async def get_chat_info(self, chat_id: str, user_id: str) -> dict | None:
        # Try to get chat info from cache first
        cached_chats = await self.cache_client.fetch_chats(user_id)
        if cached_chats and chat_id in [c["chat_id"] for c in cached_chats]:
            return cached_chats[0]

        return await self.db_client.get_chat_info(chat_id, user_id)

    async def check_chat_permissions(self, chat_id: str, user_id: str) -> bool:
        res = await self.get_chat_info(chat_id, user_id)
        if res:
            return True
        return False

async def create_memory_agent(
        cache_client_url: str,
        db_host: str,
        db_port: int,
        db_user: str,
        db_password: str,
        db_database: str
    ) -> MemoryAgent:
    """
    Creates a MemoryAgent instance and connects it to the cache and database.

    Returns:
        A connected MemoryAgent instance.
    """
    agent = MemoryAgent(
        cache_client_url=cache_client_url,
        db_host=db_host,
        db_port=db_port,
        db_user=db_user,
        db_password=db_password,
        db_database=db_database
    )

    await agent.cache_client.connect()
    await agent.db_client.check_connection()

    return agent
