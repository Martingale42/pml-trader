from datetime import datetime
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model import TraderId
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.catalog import ParquetDataCatalog

from probabilistic_trading.models.hmm.hmm_actor import HMMActor
from probabilistic_trading.models.hmm.hmm_actor import HMMActorConfig
from probabilistic_trading.strategies.hmm_strategy import HMMStrategy
from probabilistic_trading.strategies.hmm_strategy import HMMStrategyConfig


def main():
    # 設定回測變數
    backtest_time_start = "20240101"
    backtest_time_end = "20241031"
    backtest_timerange = f"{backtest_time_start}__{backtest_time_end}"
    backtest_timeframe = "15-MINUTE"
    trader_id = "BACKTESTER-ENGINE-001"
    backtest_results_dir = "./scripts/backtest/results/"

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create output directory
    output_dir = Path(backtest_results_dir + f"{trader_id}_{current_time}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize backtest engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId(trader_id),
        logging=LoggingConfig(
            log_level="INFO",  # Changed to INFO for less verbose output
            log_directory=str(output_dir),
            log_file_name=f"{trader_id}-{current_time}-{backtest_timeframe}-{backtest_timerange}.log",
            log_level_file="DEBUG",
            log_file_format="json",
            log_component_levels={"Portfolio": "DEBUG"},  # Add component-specific logging
        ),
    )
    engine = BacktestEngine(config=engine_config)

    # Add venue
    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1000, USDT)],
        base_currency=USDT,
        default_leverage=Decimal("5.0"),
    )

    # Load data from your catalog
    catalog = ParquetDataCatalog(Path("./data/binance/catalog"))

    # First load instruments from catalog
    instruments = catalog.instruments()
    if not instruments:
        raise RuntimeError("No instruments found in catalog")

    instrument = instruments[7]  # Get the first instrument
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

    # Initialize strategy and actor with your configurations
    strategy_config = HMMStrategyConfig(
        instrument_id=instrument.id,
        bar_type=bar_type,
        prob_threshold=0.95,
        position_size=Decimal("4000"),
    )
    strategy = HMMStrategy(config=strategy_config)

    hmm_actor_config = HMMActorConfig(
        instrument_id=instrument.id,
        bar_type=bar_type,
        n_states=3,
        min_training_bars=5000,
        pca_components=6,
        retrain_interval=1000,  # 7 * 24 hours
        retrain_window_size=8000,
    )
    hmm_actor = HMMActor(config=hmm_actor_config)

    # Add components to engine
    engine.add_strategy(strategy)
    engine.add_actor(hmm_actor)

    input("Press Enter to run backtest...")
    engine.run()

    # Generate reports
    print("\nGenerating reports...")
    print("\nAccount Report:")
    print(engine.trader.generate_account_report(BINANCE))
    print("\nOrder Fills Report:")
    print(engine.trader.generate_order_fills_report())
    print("\nPositions Report:")
    print(engine.trader.generate_positions_report())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Backtest failed: {e!s}")
