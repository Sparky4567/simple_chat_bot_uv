import sqlite3
from pathlib import Path

class MemoryDB:
    def __init__(self, db_name="memories.db"):
        self.db_path = Path(db_name)
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT NOT NULL,
                    bot_answer TEXT NOT NULL
                )
                """
            )

    def add_memory(self, user_input, bot_answer):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (user_input, bot_answer) VALUES (?, ?)",
                (user_input, bot_answer),
            )

    def fetch_all(self):
        with self._connect() as conn:
            cursor = conn.execute("SELECT id, user_input, bot_answer FROM memories")
            rows = cursor.fetchall()
            return [
                f"[{row[0]}] User: {row[1]} | Bot: {row[2]}"
                for row in rows
            ]

    def fetch_last(self, n=1):
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT id, user_input, bot_answer FROM memories ORDER BY id DESC LIMIT ?",
                (n,),
            )
            rows = cursor.fetchall()
            return [
                f"[{row[0]}] User: {row[1]} | Bot: {row[2]}"
                for row in rows
            ]
