import logging
import time
import json
import uuid
import asyncio

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
            raise

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

    async def create_chat(self, user_id: str) -> str:
        """
        Create and register a new chat for a user.

        - Ensures a chat hash exists at `chat:c-{chat_id}` with `status=active` and `created_at`.
        - Appends the chat payload (JSON) to the list `chats:u-{user_id}` (creates the list if missing).

        Returns the created chat_id or empty string on failure.
        """
        if not self.client or not user_id:
            return ""

        try:
            chat_id = str(uuid.uuid4())

            chat_key = f"chat:c-{chat_id}"

            # Store chat metadata
            mapping = {
                "active": 1,
                "chat_id": chat_id,
                "user_id": user_id,
                "created_at": int(time.time()),
            }
            await self.client.hset(chat_key, mapping=mapping)

            # Set TTL on the chat hash only (1 day)
            try:
                await self.client.expire(chat_key, 24 * 3600)
            except Exception:
                pass

            # Append the JSON item to chats:u-{user_id}
            user_list = f"chats:u-{user_id}"
            item = json.dumps(mapping)
            await self.client.rpush(user_list, item)

            logging.info(f"Created chat {chat_id} for user {user_id} and appended to {user_list}")

            # Publish the new chat to a channel for external subscribers
            try:
                channel = f"channel:chats:u-{user_id}"
                task = asyncio.create_task(self.client.publish(channel, item))
                def _on_done(t):
                    if exc := t.exception():
                        logging.warning("Redis chat publish failed: %s", exc)
                task.add_done_callback(_on_done)
            except Exception:
                logging.warning("Failed to publish new chat %s to channel %s", chat_id, channel)
                pass

            return chat_id
        except Exception as e:
            logging.warning(f"Failed to create chat for user {user_id}: {e}")
            return ""

    async def fetch_chats(self, user_id: str) -> list[str]:
        """
        Fetch all chat chats for a user from Redis.
        """
        logging.debug(f"Fetching chats for user {user_id}")

        if not (self.client and user_id):
            return []

        try:
            keys_list = f"chats:u-{user_id}"
            raw = await self.client.lrange(keys_list, 0, -1)
            chats = []

            for item in raw:
                try:
                    chats.append(json.loads(item))
                except Exception:
                    pass

            logging.debug(f"Fetched {len(chats)} chats for user {user_id}")
            return chats
        except Exception:
            return []

    async def store_chunk(self, chat_id: str, request_id: str, text: str = "", context: dict = {}, role: str = "agent"):
        """
        Store a text chunk for a specific chat_id and request_id.

        Each request has its own list at key `history:{chat_id}:{request_id}`.
        """
        if not self.client or not chat_id or not request_id:
            return
        try:
            per_request_key = f"history:c-{chat_id}:r-{request_id}"
            exists = await self.client.exists(per_request_key)

            if not exists:
                # Record the request key in the per-chat keys list
                keys_list = f"history_keys:c-{chat_id}"
                await self.client.rpush(keys_list, per_request_key)

                # Set TTL on per-request key (7 days) to avoid unbounded growth.
                await self.client.expire(per_request_key, 7 * 24 * 3600)
                
            # Convert context dict to json string
            try:
                context_str = json.dumps(context)
            except Exception:
                context_str = "{}"

            # Append the chunk to the list for this request and trim
            item = json.dumps({"role": role, "text": text, "context": context_str, "ts": int(time.time())})

            await self.client.rpush(per_request_key, item)
            await self.client.ltrim(per_request_key, -10000, -1)
            
            # Publish the chunk to a channel for external subscribers
            try:
                channel = f"channel:history:c-{chat_id}:r-{request_id}"
                task = asyncio.create_task(self.client.publish(channel, item))
                def _on_done(t):
                    if exc := t.exception():
                        logging.warning("Redis chunk publish failed: %s", exc)
                task.add_done_callback(_on_done)
            except Exception:
                pass
        except Exception:
            pass

    async def fetch_messages(self, chat_id: str | None, user_id: str, max_count: int, role_filter: list[str] | None = None) -> list[str]:
        logging.debug(f"Fetching messages for chat {chat_id} with max_count {max_count} and role_filter={role_filter}")

        if not self.client:
            return []

        if not chat_id:
            # Fetch very latest chat id from all chats
            try:
                user_chats_key = f"chats:u-{user_id}"
                all_chats = await self.client.lrange(user_chats_key, 0, -1)
                if all_chats:
                    latest_chat = max(all_chats, key=lambda x: json.loads(x).get("created_at", 0))
                    chat_id = json.loads(latest_chat).get("chat_id")

                    logging.warning(f"Latest chat for user {user_id} is {chat_id}")
            except Exception:
                return []

        try:
            keys = []
            try:
                keys_list = f"history_keys:c-{chat_id}"
                all_keys = await self.client.lrange(keys_list, 0, -1)

                if not all_keys:
                    logging.debug(f"history keys list missing or empty for chat {chat_id}")
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
                    if not vals:
                        continue

                    messages = self.__format_messages(vals, messages, role_filter)
            except Exception:
                # fallback to per-key fetch on pipeline failure
                for key in keys:
                    try:
                        vals = await self.client.lrange(key, 0, -1)
                        if not vals:
                            continue
                        messages = self.__format_messages(vals, messages, role_filter)
                    except Exception:
                        pass

            return messages
        except Exception:
            return []

    def __format_messages(self, vals: list[str], messages: list[str], role_filter: list[str] | None = None) -> dict:
        parts = []
        for raw in vals:
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"role": "agent", "text": raw, "ts": 0}
            if role_filter is None or obj.get("role") in role_filter:
                parts.append(obj.get("text", ""))
        if parts:
            messages.append("".join(parts))
            
        return messages