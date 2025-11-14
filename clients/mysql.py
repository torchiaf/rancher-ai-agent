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

    async def get_session_info(self, session_id: str, user_id: str) -> dict | None:
        """
        Retrieves session information from the database.

        Args:
            session_id: The ID of the session to retrieve.
        """
        logging.debug(f"Retrieving session info for session_id={session_id}, user_id={user_id}")

        if not session_id:
            return None

        conn = None
        try:
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, database=self.database)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, session_id, user_id, active, created_at "
                    "FROM sessions "
                    "WHERE session_id=%s AND user_id=%s",
                    (session_id, user_id)
                )
                result = cur.fetchone()
                
                logging.debug(f"Retrieved session info for {session_id}: {result}")
                
                return result
        except Exception as e:
            logging.warning(f"Failed to retrieve session info for {session_id}: {e}")
            return None
        finally:
            if conn:
                conn.close()
