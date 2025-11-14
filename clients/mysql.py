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

    async def connect(self):
        """
        Connects to the MySQL server.
        """
        try:
            self.client = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
            logging.info(f"Created MySQL client for {self.host}:{self.port}")
        except Exception as e:
            logging.warning(f"Failed to create MySQL client for {self.host}:{self.port}: {e}")
            self.client = None
            raise
    
    async def disconnect(self):
        """
        Disconnects from the MySQL server.
        """

        if self.client:
            self.client.close()
            logging.info(f"Disconnected MySQL client for {self.host}:{self.port}")

    async def get_session_info(self, session_id: str, user_id: str) -> dict | None:
        """
        Retrieves session information from the database.

        Args:
            session_id: The ID of the session to retrieve.
        """
        if not self.client or not session_id:
            return None
        try:
            with self.client.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(
                    "SELECT id, session_id, user_id, active, created_at "
                    "FROM sessions "
                    "WHERE session_id=%s AND user_id=%s",
                    (session_id, user_id)
                )
                result = cur.fetchone()
                return result
        except Exception as e:
            logging.warning(f"Failed to retrieve session info for {session_id}: {e}")
            return None
