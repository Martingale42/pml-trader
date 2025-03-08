#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Database connection check script for probabilistic_trading
# -------------------------------------------------------------------------------------------------

import sys
from pathlib import Path


# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from database.postgres_adapter import PostgresAdapter
from database.redis_adapter import RedisAdapter


def main():
    print("Checking database connections...")

    # Check PostgreSQL
    pg_adapter = PostgresAdapter()
    if pg_adapter.connect():
        print("PostgreSQL connection: SUCCESS")
        tables = pg_adapter.get_all_tables()
        print(f"Available tables: {', '.join(tables) if tables else 'None'}")
        
        # Check if there's data in some key tables
        for table in ['order', 'position', 'trade']:
            if table in tables:
                try:
                    # Use double quotes for table names as they might be SQL keywords
                    result = pg_adapter.execute_query(f'SELECT COUNT(*) FROM "{table}"')
                    if result and len(result) > 0:
                        count = result[0]['count']
                        print(f"  - {table} table: {count} rows")
                    else:
                        print(f"  - {table} table: Unable to get count")
                except Exception as e:
                    print(f"  - Error counting {table} table: {e}")
    else:
        print("PostgreSQL connection: FAILED")
    pg_adapter.disconnect()

    # Check Redis
    redis_adapter = RedisAdapter()
    if redis_adapter.connect():
        print("Redis connection: SUCCESS")
        keys = redis_adapter.keys()
        key_count = len(keys)
        print(f"Redis keys: {key_count}")
        print(f"Keys: {', '.join(keys) if keys else 'None'}")
    else:
        print("Redis connection: FAILED")
    redis_adapter.disconnect()


if __name__ == "__main__":
    main()
