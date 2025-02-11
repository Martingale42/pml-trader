import asyncio
import re
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.common.component import LiveClock
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler


# 加載.env文件中的環境變量
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


def parse_instrument_string(instrument_string: str) -> dict[str, str]:
    """解析交易對文件名，提取相關信息"""
    pattern = r"(?P<base_pair>[A-Z]+)_(?P<quote_pair>[A-Z]+)_[A-Z]+-(?P<timeframe>\d+[a-zA-Z]+)-(?P<instrument_type>[a-z]+)(-processed)?\.parquet"
    match = re.match(pattern, instrument_string)
    if match:
        return match.groupdict()
    else:
        raise ValueError(f"文件名格式不正確: {instrument_string}")


def construct_bar_type_string(parsed_data: dict[str, str]) -> str:
    """構建bar類型字符串"""
    base_pair = parsed_data["base_pair"]
    quote_pair = parsed_data["quote_pair"]
    timeframe = parsed_data["timeframe"]
    instrument_type = parsed_data["instrument_type"]

    if timeframe[-1] == "M" or timeframe[-1] == "m":
        timeframe = timeframe.upper().replace("M", "-MINUTE")
    else:
        timeframe = timeframe.upper().replace("H", "-HOUR")

    if instrument_type == "futures":
        return f"{base_pair}{quote_pair}-PERP.BINANCE-{timeframe}-LAST-EXTERNAL"
    else:
        return f"{base_pair}{quote_pair}-SPOT.BINANCE-{timeframe}-LAST-EXTERNAL"


async def create_provider():
    """
    Create a provider to load all instrument data from live exchange.
    """
    clock = LiveClock()

    client = get_cached_binance_http_client(
        clock=clock,
        account_type=BinanceAccountType.USDT_FUTURE,
        is_testnet=False,
    )

    binance_provider = BinanceFuturesInstrumentProvider(
        client=client,
        clock=clock,
        config=InstrumentProviderConfig(load_all=True, log_warnings=False),
    )

    await binance_provider.load_all_async()
    return binance_provider


if __name__ == "__main__":
    btcusdt_perp = pd.read_parquet(
        "/Users/ohh/Desktop/Code/NautilusTraders/data/binance/futures/BTC_USDT_USDT-5m-futures.parquet"
    )
    btcusdt_perp.rename(columns={"date": "timestamp"}, inplace=True)
    btcusdt_perp.set_index("timestamp", inplace=True)
    btcusdt_perp.sort_values("timestamp", inplace=True)
    # print(btcusdt_perp.head())

    output_path = Path("/Users/ohh/Desktop/Code/NautilusTraders/data/binance/futures_processed")
    output_path.mkdir(parents=True, exist_ok=True)
    btcusdt_perp.to_parquet(output_path / "BTC_USDT_USDT-5m-futures-processed.parquet")

    raw_data_filename = "BTC_USDT_USDT-1h-futures-processed.parquet"
    parsed_data_filename = parse_instrument_string(raw_data_filename)
    # print(parsed_data_filename)
    bar_type_string = construct_bar_type_string(parsed_data_filename)
    # print(bar_type_string)

    # Add a trading venue (multiple venues possible)
    BINANCE = Venue("BINANCE")

    # Use actual Binance instrument for backtesting
    provider: BinanceFuturesInstrumentProvider = asyncio.run(create_provider())

    instrument_id = InstrumentId(symbol=Symbol("BTCUSDT-PERP"), venue=BINANCE)
    instrument = provider.find(instrument_id)
    if instrument is None:
        raise RuntimeError(f"Unable to find instrument {instrument_id}")
    else:
        print(instrument)
        wrangler = BarDataWrangler(
            bar_type=BarType.from_str(bar_type_string), instrument=instrument
        )
        btcusdt_perp_bar = wrangler.process(btcusdt_perp)

        # 指定CATALOG_PATH
        CATALOG_PATH = Path("/Users/ohh/Desktop/Code/NautilusTraders/data/catalog")

        # 如果CATALOG_PATH存在，將其刪除並重建
        if CATALOG_PATH.exists():
            import shutil

            shutil.rmtree(CATALOG_PATH)
        CATALOG_PATH.mkdir(parents=True)

        catalog = ParquetDataCatalog(CATALOG_PATH)
        # Write instrument and bars
        catalog.write_data([instrument], basename_template=f"{instrument.id.value}")
        catalog.write_data(btcusdt_perp_bar, basename_template=f"{instrument.id.value}")
        print("Data written to catalog")
