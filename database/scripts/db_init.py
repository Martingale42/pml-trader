#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Database initialization script for probabilistic_trading
# -------------------------------------------------------------------------------------------------

import sys
from pathlib import Path


# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from database.setup_db import main as setup_db_main


if __name__ == "__main__":
    print("Initializing database for probabilistic_trading...")
    setup_db_main()
