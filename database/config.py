# -------------------------------------------------------------------------------------------------
#  Database configuration for probabilistic_trading
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from dataclasses import field


@dataclass
class PostgresConfig:
    """PostgreSQL database configuration."""

    host: str = "localhost"
    port: int = 5433
    database: str = "nautilus_trader"
    username: str = "nautilus"
    password: str = "password"


@dataclass
class RedisConfig:
    """Redis cache configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None


@dataclass
class DatabaseConfig:
    """Main database configuration."""

    type: str = "postgres"
    postgres: PostgresConfig = field(default_factory=lambda: PostgresConfig())
    redis: RedisConfig = field(default_factory=lambda: RedisConfig())
    host: str = "localhost"  # 提供直接訪問選項
    port: int = 5433  # 默認 PostgreSQL 端口
    timeout: int = 2  # 連接超時秒數
    enabled: bool = True
    debug: bool = False


# Default configuration for development
DEFAULT_CONFIG = DatabaseConfig()

# Production configuration
PRODUCTION_CONFIG = DatabaseConfig(
    postgres=PostgresConfig(
        host="localhost",
        port=5433,
        database="nautilus_trader_prod",
        username="nautilus",
        password="password",
    ),
    redis=RedisConfig(
        host="localhost",
        port=6379,
        db=0,
        password="password",
    ),
    debug=False,
)

# Docker configuration (when running inside Docker containers)
DOCKER_CONFIG = DatabaseConfig(
    postgres=PostgresConfig(
        host="postgres",  # 使用服務名稱作為主機名
        port=5432,        # 在 Docker 網絡中使用默認端口
        database="nautilus_trader",
        username="nautilus",
        password="password",
    ),
    redis=RedisConfig(
        host="redis",     # 使用服務名稱作為主機名
        port=6379,
        db=0,
        password=None,
    ),
    debug=False,
)
