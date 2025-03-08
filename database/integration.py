# -------------------------------------------------------------------------------------------------
#  NautilusTrader integration for database/cache
# -------------------------------------------------------------------------------------------------

from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import TradingNodeConfig

from database.config import DatabaseConfig
from database.config import PostgresConfig
from database.config import RedisConfig


def create_cache_config(
    database_type="redis",
    tick_capacity=10_000,
    bar_capacity=10_000,
    postgres_config=None,
    redis_config=None,
):
    """
    Create a NautilusTrader compatible cache configuration.

    Parameters
    ----------
    database_type : str
        The database type to use (redis or postgres)
    tick_capacity : int
        Maximum number of ticks to store per instrument
    bar_capacity : int
        Maximum number of bars to store per bar type
    postgres_config : PostgresConfig, optional
        PostgreSQL configuration
    redis_config : RedisConfig, optional
        Redis configuration

    Returns
    -------
    CacheConfig
        NautilusTrader compatible cache configuration
    """
    if postgres_config is None:
        postgres_config = PostgresConfig()

    if redis_config is None:
        redis_config = RedisConfig()

    db_config = DatabaseConfig(
        type=database_type,
        postgres=postgres_config,
        redis=redis_config,
        host=postgres_config.host if database_type == "postgres" else redis_config.host,
        port=postgres_config.port if database_type == "postgres" else redis_config.port,
    )

    return CacheConfig(
        database=db_config,
        tick_capacity=tick_capacity,
        bar_capacity=bar_capacity,
    )


def create_backtest_config(cache_config=None):
    """
    Create a BacktestEngineConfig with the specified cache configuration.
    """
    if cache_config is None:
        cache_config = create_cache_config()

    return BacktestEngineConfig(
        cache=cache_config,
    )


def create_trading_node_config(cache_config=None):
    """
    Create a TradingNodeConfig with the specified cache configuration.
    """
    if cache_config is None:
        cache_config = create_cache_config()

    return TradingNodeConfig(
        cache=cache_config,
    )
