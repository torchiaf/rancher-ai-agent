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

    async def store(self, session_id: str, request_id: str, text: str = ""):
        """
        Store a text chunk for a specific session_id and request_id.

        Each request has its own list at key `history:{session_id}:{request_id}`.
        """
        if not self.client or not session_id or not request_id:
            return

        try:
            per_request_key = f"history:s-{session_id}:r-{request_id}"
            exists = await self.client.exists(per_request_key)

            if not exists:
                # Record the request key in the per-session keys list
                keys_list = f"history_keys:s-{session_id}"
                await self.client.rpush(keys_list, per_request_key)

                # Set TTL on per-request key (7 days) to avoid unbounded growth.
                await self.client.expire(per_request_key, 7 * 24 * 3600)

            # Append the chunk to the list for this request and trim
            await self.client.rpush(per_request_key, text)
            await self.client.ltrim(per_request_key, -10000, -1)
        except Exception:
            pass

    async def fetch(self, session_id: str, max_count: int) -> list[str]:
        if not (self.client and session_id):
            return []

        logging.debug(f"Fetching messages for session {session_id} with max_count {max_count}")
        try:
            keys = []
            try:
                keys_list = f"history_keys:s-{session_id}"
                all_keys = await self.client.lrange(keys_list, 0, -1)
                
                if not all_keys:
                    logging.debug(f"history keys list missing or empty for session {session_id}")
                    return []

                keys = all_keys[:max_count]
            except Exception:
                keys = []

            # Aggregate all messages from the selected request keys in chronological order
            messages = []

            pipe = self.client.pipeline()
            for key in keys:
                pipe.lrange(key, 0, -1)
            try:
                slices = await pipe.execute()
                for vals in slices:
                    if vals:
                        # merge chunks for a single request into one string
                        merged = "".join(vals)
                        messages.append(merged)
            except Exception:
                # fallback to per-key fetch on pipeline failure
                for key in keys:
                    try:
                        vals = await self.client.lrange(key, 0, -1)
                        if vals:
                            messages.append("".join(vals))
                    except Exception:
                        pass

            logging.debug(f"Fetched messages for session {session_id}: {messages}")
            return messages
        except Exception:
            return []