import logging

import redis.asyncio as aioredis

class RedisClient:
    def __init__(self, url: str):
        """
        Initializes the RedisClient.

        Args:
            url: The URL of the Redis server.
        """
        self.url = url
        self.client = None

    async def connect(self):
        """
        Connects to the Redis server.
        """
        try:
            self.client = aioredis.from_url(self.url, decode_responses=True)
            await self.client.ping()
            logging.info(f"Created Redis client for {self.url} (connection deferred)")
        except Exception as e:
            logging.warning(f"Failed to create Redis client for {self.url}: {e}")
            self.client = None
    
    async def disconnect(self):
        """
        Disconnects from the Redis server.
        """

        if self.client:
            try:
                await self.client.close()
                await self.client.connection_pool.disconnect()

                logging.info(f"Disconnected Redis client for {self.url}")
            except Exception:
                pass
            
    async def store(self, session_id: str, text: str = "", max_history: int = 1000):
            if self.client and session_id:
                try:
                    # Fast aggregation reads by session
                    session_key = f"history:{session_id}"
                    
                    await self.client.rpush(session_key, text)
                    await self.client.ltrim(session_key, -max_history, -1)
                except Exception:
                    pass

    async def fetch(self, session_id: str) -> list[str]:
        if self.client and session_id:
            try:
                # Prefer the session-level list for fast reads
                session_key = f"history:{session_id}"

                return await self.client.lrange(session_key, -10, -1)
            except Exception:
                pass

        return []