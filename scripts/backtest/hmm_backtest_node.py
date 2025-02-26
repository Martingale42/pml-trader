from datetime import datetime
from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import BacktestDataConfig
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import BacktestRunConfig
from nautilus_trader.config import BacktestVenueConfig
from nautilus_trader.config import ImportableActorConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model import TraderId
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def main():
    # 設定回測變數
    backtest_time_start = "2024-01-01"
    backtest_time_end = "2024-02-29"
    backtest_timerange = f"{backtest_time_start}__{backtest_time_end}"
    backtest_timeframe = "15-MINUTE"
    current_time = datetime.now().strftime("%Y-%m-%d__%H_%M_%S")

    # 初始化data catalog
    catalog = ParquetDataCatalog(Path("data/catalog"))

    # 載入instruments
    instruments = catalog.instruments()
    if not instruments:
        raise RuntimeError("No instruments found in catalog")
    instrument = instruments[0]

    bar_type = f"{instrument.id}-{backtest_timeframe}-LAST-EXTERNAL"

    # 設定venue配置
    # BINANCE = Venue("BINANCE")
    venue_configs = [
        BacktestVenueConfig(
            name="BINANCE",
            oms_type="NETTING",
            account_type="MARGIN",
            starting_balances=["1_000 USDT"],
            base_currency="USDT",
            default_leverage=Decimal("5.0"),
        )
    ]

    # 設定要回測的資料
    data_configs = [
        BacktestDataConfig(
            catalog_path=str(catalog.path),
            data_cls=Bar,
            instrument_id=instrument.id,
            start_time=backtest_time_start,
            end_time=backtest_time_end,
            bar_spec=f"{backtest_timeframe}-LAST",
        )
    ]

    # 設定Strategy
    importable_strategy = ImportableStrategyConfig(
        strategy_path="probabilistic_trading.strategy.hmm_strategy:HMMStrategy",
        config_path="probabilistic_trading.strategy.hmm_strategy:HMMStrategyConfig",
        config={
            "instrument_id": instrument.id,
            "bar_type": BarType.from_str(bar_type),
            "prob_threshold": 0.75,
            "position_size": Decimal("4000.0"),
        },
    )

    # 設定Actor
    importable_actor = ImportableActorConfig(
        actor_path="probabilistic_trading.model.hmm.hmm_actor:HMMActor",
        config_path="probabilistic_trading.model.hmm.hmm_actor:HMMActorConfig",
        config={
            "instrument_id": instrument.id,
            "bar_type": BarType.from_str(bar_type),
            "n_states": 2,
            "min_training_bars": 672,
            "pca_components": 5,
        },
    )

    # 創建回測配置
    run_config = BacktestRunConfig(
        engine=BacktestEngineConfig(
            trader_id=TraderId("BACKTEST_TRADER-002"),
            logging=LoggingConfig(
                log_level="INFO",  # Changed to INFO for less verbose output
                log_file_name=f"BACKTEST_TRADER-002_{current_time}_{backtest_timeframe}_{backtest_timerange}.log",
                log_level_file="DEBUG",
                log_file_format="json",
                log_component_levels={"Portfolio": "DEBUG"},  # Add component-specific logging
            ),
            strategies=[importable_strategy],
            actors=[importable_actor],
        ),
        data=data_configs,
        venues=venue_configs,
        chunk_size=1344,
    )

    # 創建和執行 BacktestNode
    node = BacktestNode([run_config])

    input("Press Enter to run backtest...")
    results = node.run()

    # 取得回測結果並印出報告
    result = results[0]  # 第一組回測結果
    print("\nGenerating reports...")
    print("\nOrder Fills Report:")
    print(result.order_fills_report())
    print("\nPositions Report:")
    print(result.positions_report())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Backtest failed: {e!s}")
