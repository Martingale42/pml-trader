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
    """解析交易對文件名, 提取相關信息"""
    pattern = (
        r"(?P<base_pair>[A-Z]+)_(?P<quote_pair>[A-Z]+)_[A-Z]+-"
        r"(?P<timeframe>\d+[a-zA-Z]+)-(?P<instrument_type>[a-z]+)"
        r"(-processed)?\.parquet"
    )
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
    """創建提供者以從實時交易所加載所有工具數據"""
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


def process_raw_data(
    raw_data_path: str | Path,
    processed_data_path: str | Path,
    catalog_path: str | Path,
) -> None:
    """
    批量處理原始數據並將其保存到目錄中

    Parameters
    ----------
    raw_data_path : str | Path
        原始數據目錄的路徑
    processed_data_path : str | Path
        處理後數據保存的路徑
    catalog_path : str | Path
        數據目錄的路徑
    """
    # 轉換路徑為Path對象
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)
    catalog_path = Path(catalog_path)

    # 驗證路徑
    if not raw_data_path.exists():
        raise ValueError(f"原始數據路徑不存在: {raw_data_path}")

    # 創建必要的目錄
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # 如果catalog_path存在, 刪除並重建
    if catalog_path.exists():
        shutil.rmtree(catalog_path)
    catalog_path.mkdir(parents=True)

    # 創建catalog實例
    catalog = ParquetDataCatalog(catalog_path)

    # 獲取交易所提供者
    provider: BinanceFuturesInstrumentProvider = asyncio.run(create_provider())

    # 添加交易場所
    BINANCE = Venue("BINANCE")

    # 處理所有futures類型的parquet文件
    for file_path in raw_data_path.glob("*-futures.parquet"):
        try:
            print(f"正在處理: {file_path.name}")

            # 讀取原始數據
            raw_data = pd.read_parquet(file_path)
            raw_data.rename(columns={"date": "timestamp"}, inplace=True)
            raw_data.set_index("timestamp", inplace=True)
            raw_data.sort_values("timestamp", inplace=True)

            # 保存處理後的數據
            processed_filename = file_path.name.replace(".parquet", "-processed.parquet")
            processed_file_path = processed_data_path / processed_filename
            raw_data.to_parquet(processed_file_path)

            # 解析文件名並構建bar類型字符串
            parsed_data = parse_instrument_string(processed_filename)
            bar_type_string = construct_bar_type_string(parsed_data)

            # 獲取交易對信息
            base_pair = parsed_data["base_pair"]
            symbol = Symbol(f"{base_pair}USDT-PERP")
            instrument_id = InstrumentId(symbol=symbol, venue=BINANCE)
            instrument = provider.find(instrument_id)

            if instrument is None:
                print(f"無法找到交易對 {instrument_id}, 跳過處理")
                continue

            # 處理數據
            wrangler = BarDataWrangler(
                bar_type=BarType.from_str(bar_type_string), instrument=instrument
            )
            processed_bars = wrangler.process(raw_data)

            # 寫入catalog
            catalog.write_data([instrument], basename_template=f"{instrument.id.value}")
            catalog.write_data(processed_bars, basename_template=f"{bar_type_string}")

            print(f"成功處理並保存: {file_path.name}")

        except Exception as e:
            print(f"處理 {file_path.name} 時發生錯誤: {e!s}")
            continue

    print("\n所有數據處理完成！")


if __name__ == "__main__":
    # 示例使用
    RAW_DATA_PATH = "/Users/ohh/Desktop/Code/NautilusTraders/data/binance/futures"
    PROCESSED_DATA_PATH = "/Users/ohh/Desktop/Code/NautilusTraders/data/binance/futures_processed"
    CATALOG_PATH = "/Users/ohh/Desktop/Code/NautilusTraders/data/catalog"

    process_raw_data(
        raw_data_path=RAW_DATA_PATH,
        processed_data_path=PROCESSED_DATA_PATH,
        catalog_path=CATALOG_PATH,
    )
