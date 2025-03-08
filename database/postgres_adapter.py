# -------------------------------------------------------------------------------------------------
#  PostgreSQL adapter for probabilistic_trading
# -------------------------------------------------------------------------------------------------

import sys
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras


# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from database.config import DEFAULT_CONFIG
from database.config import PostgresConfig


class PostgresAdapter:
    """Adapter for PostgreSQL database operations."""

    def __init__(self, config: PostgresConfig | None = None):
        self.config = config or DEFAULT_CONFIG.postgres
        self._conn = None
        self._cursor = None

    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self._conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
            )
            # Use RealDictCursor to return results as dictionaries
            self._cursor = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            return True
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return False

    def disconnect(self):
        """Disconnect from PostgreSQL database."""
        if self._cursor:
            self._cursor.close()
        if self._conn:
            self._conn.close()
        self._conn = None
        self._cursor = None

    def is_connected(self) -> bool:
        """Check if connected to database."""
        if self._conn is None:
            return False
        try:
            # Try a simple query
            cur = self._conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            return True
        except:
            return False

    def execute_query(self, query: str, params: tuple | None = None) -> list[dict]:
        """Execute a query and return results."""
        if not self.is_connected():
            self.connect()

        try:
            self._cursor.execute(query, params)
            if self._cursor.description:  # If query returns rows
                return self._cursor.fetchall()
            return []
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
            self._conn.rollback()
            return []

    def execute_transaction(self, queries: list[tuple[str, tuple | None]]) -> bool:
        """Execute multiple queries in a single transaction."""
        if not self.is_connected():
            self.connect()

        try:
            # Start transaction
            self._conn.autocommit = False

            for query, params in queries:
                self._cursor.execute(query, params)

            # Commit transaction
            self._conn.commit()
            self._conn.autocommit = True
            return True
        except Exception as e:
            print(f"Error executing transaction: {e}")
            self._conn.rollback()
            self._conn.autocommit = True
            return False

    def insert(self, table: str, data: dict[str, Any]) -> int | None:
        """Insert data into a table and return the id."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        values = list(data.values())

        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"

        try:
            if not self.is_connected():
                self.connect()

            self._cursor.execute(query, values)
            self._conn.commit()
            result = self._cursor.fetchone()
            if result:
                return result["id"]
            return None
        except Exception as e:
            print(f"Error inserting data: {e}")
            self._conn.rollback()
            return None

    def update(self, table: str, data: dict[str, Any], condition: str, params: tuple) -> bool:
        """Update data in a table."""
        set_clause = ", ".join([f"{col} = %s" for col in data.keys()])
        values = list(data.values()) + list(params)

        query = f"UPDATE {table} SET {set_clause} WHERE {condition}"

        try:
            if not self.is_connected():
                self.connect()

            self._cursor.execute(query, values)
            self._conn.commit()
            return True
        except Exception as e:
            print(f"Error updating data: {e}")
            self._conn.rollback()
            return False

    def delete(self, table: str, condition: str, params: tuple) -> bool:
        """Delete data from a table."""
        query = f"DELETE FROM {table} WHERE {condition}"

        try:
            if not self.is_connected():
                self.connect()

            self._cursor.execute(query, params)
            self._conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting data: {e}")
            self._conn.rollback()
            return False

    def truncate_all_tables(self) -> bool:
        """Truncate all tables in the database."""
        try:
            if not self.is_connected():
                self.connect()

            self._cursor.execute("SELECT truncate_all_tables()")
            self._conn.commit()
            return True
        except Exception as e:
            print(f"Error truncating tables: {e}")
            self._conn.rollback()
            return False

    def get_all_tables(self) -> list[str]:
        """Get list of all tables in the database."""
        try:
            if not self.is_connected():
                self.connect()

            self._cursor.execute("SELECT get_all_tables()")
            result = self._cursor.fetchone()
            return result["get_all_tables"] if result else []
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []
