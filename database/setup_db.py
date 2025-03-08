# -------------------------------------------------------------------------------------------------
#  Database setup script for probabilistic_trading
# -------------------------------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

import psycopg2
import redis


# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from database.config import DEFAULT_CONFIG


def create_database(config):
    """Create database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host=config.postgres.host,
            port=config.postgres.port,
            user=config.postgres.username,
            password=config.postgres.password,
            database="postgres",  # Connect to default database first
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{config.postgres.database}'")
        exists = cursor.fetchone()

        if not exists:
            print(f"Creating database {config.postgres.database}...")
            cursor.execute(f"CREATE DATABASE {config.postgres.database}")
            print(f"Database {config.postgres.database} created.")
        else:
            print(f"Database {config.postgres.database} already exists.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Error creating database: {e}")
        sys.exit(1)


def create_schema(config):
    """Create tables and functions in the database."""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=config.postgres.host,
            port=config.postgres.port,
            user=config.postgres.username,
            password=config.postgres.password,
            database=config.postgres.database,
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Execute schema SQL files
        schema_dir = Path(__file__).resolve().parent / "schema"

        # Execute tables.sql
        tables_sql_path = schema_dir / "tables.sql"
        with open(tables_sql_path) as f:
            tables_sql = f.read()
            print("Creating tables...")
            cursor.execute(tables_sql)

        # Execute functions.sql
        functions_sql_path = schema_dir / "functions.sql"
        with open(functions_sql_path) as f:
            functions_sql = f.read()
            print("Creating functions...")
            cursor.execute(functions_sql)

        cursor.close()
        conn.close()
        print("Schema setup complete.")

    except Exception as e:
        print(f"Error creating schema: {e}")
        sys.exit(1)


def check_redis_connection(config):
    """Check Redis connection."""
    try:
        r = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            password=config.redis.password,
            socket_timeout=5,
        )
        if r.ping():
            print("Redis connection successful.")
        else:
            print("Redis connection failed.")
            sys.exit(1)
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Setup database for probabilistic_trading")
    parser.add_argument(
        "--tables-only", action="store_true", help="Only create tables without recreating database"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Only check connections without creating anything"
    )

    args = parser.parse_args()

    config = DEFAULT_CONFIG

    if args.check_only:
        try:
            conn = psycopg2.connect(
                host=config.postgres.host,
                port=config.postgres.port,
                user=config.postgres.username,
                password=config.postgres.password,
                database=config.postgres.database,
            )
            conn.close()
            print("PostgreSQL connection successful.")

            check_redis_connection(config)
            print("All database connections successful.")
        except Exception as e:
            print(f"Database connection check failed: {e}")
            sys.exit(1)
        return

    if not args.tables_only:
        create_database(config)

    create_schema(config)
    check_redis_connection(config)

    print("Database setup complete!")


if __name__ == "__main__":
    main()
