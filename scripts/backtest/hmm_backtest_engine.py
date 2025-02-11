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
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from probabilistic_trading.model.hmm.hmm_actor import HMMActor
from probabilistic_trading.model.hmm.hmm_actor import HMMActorConfig
from probabilistic_trading.strategy.hmm_strategy import HMMStrategy
from probabilistic_trading.strategy.hmm_strategy import HMMStrategyConfig


def main():
    # 創建回測引擎
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTEST_TRADER-001"),
        logging=LoggingConfig(
            log_level="DEBUG",  # set DEBUG log level for console to see loaded bars in logs
        ),
    )
    engine = BacktestEngine(config=engine_config)

    # 添加交易場所
    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000, USDT)],
        base_currency=USDT,  # 多幣種賬戶
        default_leverage=Decimal("20"),
    )

    # 設置交易工具
    instrument = TestInstrumentProvider.btcusdt_perp_binance()
    engine.add_instrument(instrument)

    # 從 catalog 加載數據
    catalog = ParquetDataCatalog(
        Path("/Users/ohh/Desktop/Code/NautilusTraders/data/catalog"),
    )

    # 添加數據
    bar_type_5m = f"{instrument.id}-5-MINUTE-LAST-EXTERNAL"
    # 使用 query 方法加載數據
    bars_5m = catalog.query(
        data_cls=Bar,
        bar_types=[bar_type_5m],
    )
    engine.add_data(bars_5m)

    # 初始化並添加策略
    strategy_config = HMMStrategyConfig(
        instrument_id=instrument.id,
        min_prob_threshold=0.7,  # 交易觸發閾值
        position_size=Decimal("300"),  # 倉位大小
    )

    strategy = HMMStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    hmm_actor_config = HMMActorConfig(
        instrument_id=instrument.id,
        n_states=3,
        min_training_bars=864,
        update_interval=3,
        pca_components=5,
    )

    hmm_actor = HMMActor(config=hmm_actor_config)
    engine.add_actor(hmm_actor)

    # 開始回測
    input("Press Enter to run backtest...")

    # 運行回測
    engine.run(streaming=False)

    # 生成報告
    print("Account Report:")
    print(engine.trader.generate_account_report(BINANCE))
    print("\nOrder Fills Report:")
    print(engine.trader.generate_order_fills_report())
    print("\nPositions Report:")
    print(engine.trader.generate_positions_report())

    # 重置引擎
    engine.reset()

    # 關閉引擎
    engine.dispose()


if __name__ == "__main__":
    main()
