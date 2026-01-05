import pymysql
import logging

class MySQLClient:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        """
        Initializes the MySQLClient.

        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    async def check_connection(self):
        """
        Connects to the MySQL server.
        """
        conn = None
        try:
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
            logging.info(f"Created MySQL client for {self.host}:{self.port}")
        except Exception as e:
            logging.warning(f"Failed to create MySQL client for {self.host}:{self.port}: {e}")
            raise
        finally:
            if conn:
                conn.close()

    async def get_chat(self, chat_id: str, user_id: str) -> dict | None:
        """
        Retrieves chat information from the database.

        Args:
            chat_id: The ID of the chat to retrieve.
        """
        logging.debug(f"Retrieving chat info for chat_id={chat_id}, user_id={user_id}")

        if not chat_id:
            return None

        conn = None
        try:
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, chat_id, user_id, active, created_at "
                    "FROM chats "
                    "WHERE chat_id=%s AND user_id=%s",
                    (chat_id, user_id)
                )
                result = cur.fetchone()

                logging.debug(f"Retrieved chat info for {chat_id}: {result}")

                return result
        except Exception as e:
            logging.warning(f"Failed to retrieve chat info for {chat_id}: {e}")
            return None
        finally:
            if conn:
                conn.close()

    async def activate_chat(self, user_id: str, chat_id: str):
        """
        Activate a chat.
        
        - Sets the 'active' field to True for the given chat_id.
        - Sets the 'active' field to False for all other chats of the same user.

        Args:
            chat_id: The ID of the chat to set as active.
        """
        logging.debug(f"Setting chat {chat_id} as active for user {user_id}")

        if not chat_id:
            return

        conn = None
        try:
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chats SET active=0 WHERE user_id=%s",
                    (user_id,)
                )

                cur.execute(
                    "UPDATE chats SET active=1 WHERE chat_id=%s",
                    (chat_id,)
                )

                conn.commit()

                logging.debug(f"Chat {chat_id} set as active for user {user_id}")
        except Exception as e:
            logging.warning(f"Failed to set chat {chat_id} as active: {e}")
        finally:
            if conn:
                conn.close()