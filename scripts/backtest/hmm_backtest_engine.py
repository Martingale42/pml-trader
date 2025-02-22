# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from decimal import Decimal
from pathlib import Path

# %%
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

# %%
from probabilistic_trading.model.hmm.hmm_actor import HMMActor
from probabilistic_trading.model.hmm.hmm_actor import HMMActorConfig
from probabilistic_trading.strategy.hmm_strategy import HMMStrategy
from probabilistic_trading.strategy.hmm_strategy import HMMStrategyConfig


# %%
def main():
    # Initialize backtest engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTEST_TRADER-001"),
        logging=LoggingConfig(
            log_level="INFO",  # Changed to INFO for less verbose output
            log_level_file="DEBUG",
            log_file_format="json",
            log_component_levels={"Portfolio": "INFO"},  # Add component-specific logging
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
        default_leverage=Decimal("20"),
    )

    # Load data from your catalog
    catalog = ParquetDataCatalog(Path("data/catalog"))

    # First load instruments from catalog
    instruments = catalog.instruments()
    if not instruments:
        raise RuntimeError("No instruments found in catalog")

    instrument = instruments[0]  # Get the first instrument
    engine.add_instrument(instrument)

    # Now load bar data
    bar_type = f"{instrument.id}-15-MINUTE-LAST-EXTERNAL"
    print(f"Loading bars for {bar_type}")

    bars = catalog.query(
        data_cls=Bar,
        bar_types=[bar_type],
    )

    if not bars:
        raise RuntimeError(f"No bars found for {bar_type}")

    print(f"Loaded {len(bars)} bars")
    engine.add_data(bars)

    # Initialize strategy and actor with your configurations
    strategy_config = HMMStrategyConfig(
        instrument_id=instrument.id,
        prob_threshold=0.7,
        position_size=Decimal("700"),
    )
    strategy = HMMStrategy(config=strategy_config)

    hmm_actor_config = HMMActorConfig(
        instrument_id=instrument.id,
        n_states=3,
        min_training_bars=288,
        update_interval=3,
        pca_components=5,
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


# %%
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Backtest failed: {e!s}")

# %%
