# -------------------------------------------------------------------------------------------------
#  Redis adapter for probabilistic_trading
# -------------------------------------------------------------------------------------------------

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import redis


# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from database.config import DEFAULT_CONFIG
from database.config import RedisConfig


class RedisAdapter:
    """Adapter for Redis cache operations."""

    def __init__(self, config: RedisConfig | None = None):
        self.config = config or DEFAULT_CONFIG.redis
        self._redis = None

    def connect(self):
        """Connect to Redis."""
        try:
            self._redis = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=False,  # Keep binary data for flexibility
            )
            return self._redis.ping()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            return False

    def disconnect(self):
        """Disconnect from Redis."""
        self._redis = None

    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if self._redis is None:
            return False
        try:
            return self._redis.ping()
        except:
            return False

    def get(self, key: str) -> bytes | None:
        """Get value for key."""
        if not self.is_connected():
            self.connect()

        try:
            return self._redis.get(key)
        except Exception as e:
            print(f"Error getting value: {e}")
            return None

    def set(self, key: str, value: bytes, expiry: int | None = None) -> bool:
        """Set key with value and optional expiry in seconds."""
        if not self.is_connected():
            self.connect()

        try:
            return self._redis.set(key, value, ex=expiry)
        except Exception as e:
            print(f"Error setting value: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key."""
        if not self.is_connected():
            self.connect()

        try:
            return self._redis.delete(key) > 0
        except Exception as e:
            print(f"Error deleting key: {e}")
            return False

    def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern."""
        if not self.is_connected():
            self.connect()

        try:
            keys = self._redis.keys(pattern)
            return [k.decode("utf-8") for k in keys]
        except Exception as e:
            print(f"Error getting keys: {e}")
            return []

    def flush_db(self) -> bool:
        """Flush the current database."""
        if not self.is_connected():
            self.connect()

        try:
            return self._redis.flushdb()
        except Exception as e:
            print(f"Error flushing database: {e}")
            return False

    def set_json(self, key: str, value: Any, expiry: int | None = None) -> bool:
        """Set JSON value for key."""
        try:
            json_value = json.dumps(value).encode("utf-8")
            return self.set(key, json_value, expiry)
        except Exception as e:
            print(f"Error setting JSON: {e}")
            return False

    def get_json(self, key: str) -> Any | None:
        """Get JSON value for key."""
        try:
            value = self.get(key)
            if value:
                return json.loads(value.decode("utf-8"))
            return None
        except Exception as e:
            print(f"Error getting JSON: {e}")
            return None

    def set_pickle(self, key: str, value: Any, expiry: int | None = None) -> bool:
        """Set Python object for key using pickle."""
        try:
            pickle_value = pickle.dumps(value)
            return self.set(key, pickle_value, expiry)
        except Exception as e:
            print(f"Error setting pickle: {e}")
            return False

    def get_pickle(self, key: str) -> Any | None:
        """Get Python object for key using pickle."""
        try:
            value = self.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            print(f"Error getting pickle: {e}")
            return None

    def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        if not self.is_connected():
            self.connect()

        try:
            return self._redis.publish(channel, message)
        except Exception as e:
            print(f"Error publishing message: {e}")
            return 0
