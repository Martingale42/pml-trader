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
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def main():
    # 設定回測變數
    backtest_time_start = "20240101"
    backtest_time_end = "20240131"
    backtest_timerange = f"{backtest_time_start}_{backtest_time_end}"
    backtest_timeframe = "15-MINUTE"
    trader_id = "BACKTESTER-NODE-002"
    backtest_results_dir = "./backtest_results/"

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create output directory
    output_dir = Path(backtest_results_dir + f"{trader_id}_{current_time}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 初始化data catalog
    catalog = ParquetDataCatalog(Path("data/binance/catalog"))

    # 載入instruments
    instruments = catalog.instruments()
    if not instruments:
        raise RuntimeError("No instruments found in catalog")
    instrument = instruments[0]

    bar_type = f"{instrument.id}-{backtest_timeframe}-LAST-EXTERNAL"

    # 設定venue配置
    BINANCE = Venue("BINANCE")
    venue_configs = [
        BacktestVenueConfig(
            name="BINANCE",
            oms_type="NETTING",
            account_type="MARGIN",
            starting_balances=["75 USDT"],
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
            "prob_threshold": 0.85,
            "position_size": Decimal("200.0"),
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
            "retrain_interval": 168,
            "retrain_window_size": 672,
            "incremental_training": True,
        },
    )

    # 創建回測配置
    run_config = BacktestRunConfig(
        engine=BacktestEngineConfig(
            trader_id=TraderId(trader_id),
            logging=LoggingConfig(
                log_level="INFO",  # Changed to INFO for less verbose output
                log_directory=str(output_dir),
                log_file_name=f"{trader_id}-{current_time}-{backtest_timeframe}-{backtest_timerange}.log",
                log_level_file="DEBUG",
                log_file_format="json",
                log_component_levels={"Portfolio": "DEBUG"},  # Add component-specific logging
            ),
            strategies=[importable_strategy],
            actors=[importable_actor],
        ),
        data=data_configs,
        venues=venue_configs,
        chunk_size=672,
    )

    # 創建和執行 BacktestNode
    node = BacktestNode([run_config])

    input("Press Enter to run backtest...")
    results = node.run()

    # 取得回測結果並印出報告
    result = results[0]  # 第一組回測結果
    print("\nBacktest results:")
    print(result)

    # Generate reports
    print("\nGenerating reports...")
    engine = node.get_engines()[0]  # 第一組回測引擎
    account_report = engine.trader.generate_account_report(BINANCE)
    account_report.to_csv(f"{output_dir}/account_report_{current_time}.csv")
    print("\nAccount Report:")
    print(account_report)

    orders_report = engine.trader.generate_orders_report()
    orders_report.to_csv(f"{output_dir}/orders_report_{current_time}.csv")
    print("\nOrder Report:")
    print(orders_report)

    order_fills_report = engine.trader.generate_order_fills_report()
    order_fills_report.to_csv(f"{output_dir}/order_fills_report_{current_time}.csv")
    print("\nOrder Fills Report:")
    print(order_fills_report)

    fills_report = engine.trader.generate_fills_report()
    fills_report.to_csv(f"{output_dir}/fills_report_{current_time}.csv")
    print("\nFills Report:")
    print(fills_report)

    positions_report = engine.trader.generate_positions_report()
    positions_report.to_csv(f"{output_dir}/positions_report_{current_time}.csv")
    print("\nPositions Report:")
    print(positions_report)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Backtest failed: {e!s}")
