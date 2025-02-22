from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import BacktestDataConfig
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import BacktestRunConfig
from nautilus_trader.config import BacktestVenueConfig
from nautilus_trader.config import ImportableActorConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def main():
    # 初始化data catalog
    catalog = ParquetDataCatalog(Path("data/catalog"))

    # 載入instruments
    instruments = catalog.instruments()
    if not instruments:
        raise RuntimeError("No instruments found in catalog")

    instrument = instruments[0]

    # 設定venue配置
    BINANCE = Venue("BINANCE")
    venue_configs = [
        BacktestVenueConfig(
            name="BINANCE",
            oms_type="NETTING",
            account_type="MARGIN",
            starting_balances=["1_000 USDT"],
            base_currency="USDT",
            # default_leverage=Decimal("20"),
        )
    ]

    # 設定要回測的資料
    data_configs = [
        BacktestDataConfig(
            catalog_path=str(catalog.path),
            data_cls=Bar,
            instrument_id=instrument.id,
        )
    ]

    # 設定Strategy
    importable_strategy = ImportableStrategyConfig(
        strategy_path="probabilistic_trading.strategy.hmm_strategy:HMMStrategy",
        config_path="probabilistic_trading.strategy.hmm_strategy:HMMStrategyConfig",
        config={
            "instrument_id": instrument.id,
            "prob_threshold": 0.7,
            "position_size": Decimal("700"),
        },
    )

    # 設定Actor
    importable_actor = ImportableActorConfig(
        actor_path="probabilistic_trading.model.hmm.hmm_actor:HMMActor",
        config_path="probabilistic_trading.model.hmm.hmm_actor:HMMActorConfig",
        config={
            "instrument_id": instrument.id,
            "n_states": 3,
            "min_training_bars": 288,
            "update_interval": 3,
            "pca_components": 5,
        },
    )

    # 創建回測配置
    run_config = BacktestRunConfig(
        engine=BacktestEngineConfig(strategies=[importable_strategy], actors=[importable_actor]),
        data=data_configs,
        venues=venue_configs,
    )

    # 創建和執行 BacktestNode
    node = BacktestNode([run_config])

    input("Press Enter to run backtest...")
    results = node.run()

    # 取得回測結果並印出報告
    result = results[0]  # 第一組回測結果
    print("\nGenerating reports...")
    print("\nAccount Report:")
    print(result.account_report(BINANCE))
    print("\nOrder Fills Report:")
    print(result.order_fills_report())
    print("\nPositions Report:")
    print(result.positions_report())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Backtest failed: {e!s}")
