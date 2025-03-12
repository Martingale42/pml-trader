from nautilus_trader.cache.cache import CacheConfig
from nautilus_trader.common.config import DatabaseConfig
from nautilus_trader.common.config import MessageBusConfig


DEFAULT_REDIS_CONFIG = DatabaseConfig(
    type="redis",
    host="localhost",  # Redis host
    port=6379,  # Redis port, default 6379
    username=None,  # Optional username
    password=None,  # Optional password
    ssl=False,  # Whether to use SSL
    timeout=10,  # Connection timeout seconds
)

# Configure the Redis database for caching
DEFAULT_CACHE_CONFIG = CacheConfig(
    database=DEFAULT_REDIS_CONFIG,
    encoding="msgpack",  # Data serialization format ("msgpack" or "json")
    flush_on_start=True,  # Whether to clear database on startup
    use_trader_prefix=True,  # Use "trader-" prefix in Redis keys
    use_instance_id=False,  # Include UUID instance ID in keys
    tick_capacity=40_000,  # Max ticks stored per instrument
    bar_capacity=20_000,  # Max bars stored per bar type
)

# Configure the Redis database for message bus
DEFAULT_MSG_CONFIG = MessageBusConfig(
    database=DEFAULT_REDIS_CONFIG,
    encoding="msgpack",  # "msgpack" (default) or "json"
    timestamps_as_iso8601=False,  # Store timestamps as integers or ISO strings
    buffer_interval_ms=100,  # Message buffering interval
    autotrim_mins=30,  # Auto-trim messages older than this
    use_trader_prefix=True,  # Use trader prefix in stream keys
    use_trader_id=True,  # Include trader ID in stream keys
    use_instance_id=False,  # Include instance ID in stream keys
    streams_prefix="streams",  # Prefix for all stream keys
    stream_per_topic=True,  # Whether to create a separate stream per topic
    external_streams=None,  # Optional list of external streams to publish to
    heartbeat_interval_secs=5,  # Heartbeat interval in seconds
    types_filter=None,  # Optional list of types to exclude from external publishing
)
