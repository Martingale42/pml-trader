#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Setup NautilusTrader cache with database backend
# -------------------------------------------------------------------------------------------------

import sys
from pathlib import Path


# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from database.integration import create_cache_config
from database.setup_db import main as setup_db_main


def setup_nautilus_cache():
    """Initialize database and create NautilusTrader compatible cache configuration."""
    # 首先確保數據庫已正確設置
    setup_db_main()

    # 創建 NautilusTrader 兼容的緩存配置
    cache_config = create_cache_config(
        database_type="redis",  # 或者選擇 "redis"
        tick_capacity=50_000,  # 增加容量以適應您的數據
        bar_capacity=20_000,  # 增加容量以適應您的數據
    )

    print("NautilusTrader cache configuration created.")
    print(f"Database type: {cache_config.database.type}")
    print(f"Tick capacity: {cache_config.tick_capacity}")
    print(f"Bar capacity: {cache_config.bar_capacity}")

    return cache_config


if __name__ == "__main__":
    print("Setting up NautilusTrader cache...")
    setup_nautilus_cache()
    print("Setup complete!")
