# examples/demo_hmm/run_demo_backtest.py
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from probabilistic_trading.examples.demo_hmm.demo_hmm_actor import DemoHMMActor
from probabilistic_trading.examples.demo_hmm.demo_hmm_strategy import DemoHMMStrategy
from probabilistic_trading.examples.demo_hmm.demo_hmm_strategy import DemoHMMStrategyConfig


def main():
    # 設定回測變數
    backtest_time_start = "2024-01-01"
    backtest_time_end = "2024-01-31"
    backtest_timeframe = "15-MINUTE"

    # 設置回測引擎
    engine_config = BacktestEngineConfig(
        logging=LoggingConfig(
            log_level="DEBUG",  # Changed to INFO for less verbose output
        ),
    )
    engine = BacktestEngine(config=engine_config)

    # Add venue
    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, USDT)],
        base_currency=USDT,
        default_leverage=Decimal("2"),
    )

    # 加載您的測試數據
    # Load data from your catalog
    catalog = ParquetDataCatalog(Path("data/catalog"))

    # First load instruments from catalog
    instruments = catalog.instruments()
    if not instruments:
        raise RuntimeError("No instruments found in catalog")

    instrument = instruments[0]  # Get the first instrument
    engine.add_instrument(instrument)

    # Now load bar data
    bar_type = f"{instrument.id}-{backtest_timeframe}-LAST-EXTERNAL"
    print(f"Loading bars for {bar_type}")

    bars = catalog.query(
        data_cls=Bar,
        bar_types=[bar_type],
        start=backtest_time_start,
        end=backtest_time_end,
    )

    if not bars:
        raise RuntimeError(f"No bars found for {bar_type}")

    print(f"Loaded {len(bars)} bars")
    engine.add_data(bars)

    # 添加Actor和Strategy
    actor = DemoHMMActor(bar_type=bar_type)
    engine.add_actor(actor)

    strategy = DemoHMMStrategy(
        config=DemoHMMStrategyConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            position_size=Quantity.from_str("1600"),
            prob_threshold=0.75,
        )
    )
    engine.add_strategy(strategy)

    input("Press Enter to start backtest...")

    # 運行回測
    engine.run()

    # Optionally print additional strategy results
    with pd.option_context(
        "display.max_rows",
        None,  # Show only 10 rows
        "display.max_columns",
        None,  # Show only 10 rows
        "display.width",
        None,
    ):
        n_dashes = 50
        print(f"\n{'-' * n_dashes}\nAccount report for venue: {BINANCE}\n{'-' * n_dashes}")
        print(engine.trader.generate_account_report(BINANCE))

        print(f"\n{'-' * n_dashes}\nOrder fills report: {BINANCE}\n{'-' * n_dashes}")
        print(engine.trader.generate_order_fills_report())

        print(f"\n{'-' * n_dashes}\nPositions report: {BINANCE}\n{'-' * n_dashes}")
        print(engine.trader.generate_positions_report())

    # Cleanup resources
    engine.dispose()


if __name__ == "__main__":
    main()
