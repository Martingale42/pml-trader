import asyncio  # noqa
import os
import shutil
from datetime import datetime  # noqa
from datetime import timedelta  # noqa
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from nautilus_trader.adapters.databento.constants import PUBLISHERS_FILEPATH  # noqa
from nautilus_trader.adapters.databento.data_utils import databento_data
from nautilus_trader.adapters.databento.data_utils import init_databento_client
from nautilus_trader.adapters.databento.data_utils import load_catalog  # noqa
from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol  # noqa
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler


# 加載.env文件中的環境變量
env_path = Path(__file__).parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

# 設置Databento API密鑰
API_KEY = os.environ.get("DATABENTO_API_KEY")
if not API_KEY:
    raise ValueError("未找到DATABENTO_API_KEY環境變量, 請確保設置正確的API密鑰")


def process_databento_data(
    symbol: str,
    venue: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
    raw_data_path: str | Path = "data/databento/raw",
    processed_data_path: str | Path = "data/databento/processed",
    catalog_path: str | Path = "data/databento/catalog",
) -> None:
    """
    從Databento下載股票數據, 處理並保存到ParquetDataCatalog中

    Parameters
    ----------
    symbol : str
        股票代碼, 例如 "PLTR"
    venue : str
        交易所代碼, 例如 "XNAS"(納斯達克)
    start_date : str
        開始日期, 格式為 "YYYY-MM-DD"
    end_date : str
        結束日期, 格式為 "YYYY-MM-DD"
    timeframe : str, optional
        時間框架, 默認為 "1m" (1分鐘)
    raw_data_path : str | Path, optional
        原始數據保存路徑
    processed_data_path : str | Path, optional
        處理後數據保存路徑
    catalog_path : str | Path, optional
        數據目錄保存路徑
    """
    # 轉換路徑為Path對象
    raw_data_path = Path(raw_data_path)
    processed_data_path = Path(processed_data_path)
    catalog_path = Path(catalog_path)

    # 創建必要的目錄
    raw_data_path.mkdir(parents=True, exist_ok=True)
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # 如果catalog_path存在, 刪除並重建
    if catalog_path.exists():
        shutil.rmtree(catalog_path)
    catalog_path.mkdir(parents=True)

    # 初始化Databento客戶端
    init_databento_client(API_KEY)

    # 添加交易場所
    VENUE = Venue(venue)

    # 構建完整的時間範圍字符串
    start_time = f"{start_date}T00:00:00"
    end_time = f"{end_date}T23:59:59"

    # 決定使用的schema
    if timeframe == "1m":
        schema = "ohlcv-1m"
    elif timeframe == "1h":
        schema = "ohlcv-1h"
    elif timeframe == "1d":
        schema = "ohlcv-1d"
    else:
        raise ValueError(f"不支持的時間框架: {timeframe}")

    print(f"開始下載 {symbol} 從 {start_time} 到 {end_time} 的 {schema} 數據...")

    try:
        # 從Databento下載數據
        result = databento_data(
            symbols=[symbol],
            start_time=start_time,
            end_time=end_time,
            schema=schema,
            file_prefix=f"{symbol}_{venue}_{timeframe}",
            # symbol,  # 保存在以symbol命名的子目錄中
            dataset=f"{venue}.ITCH",  # 使用相應的數據集
            to_catalog=False,  # 不直接保存到目錄, 我們將手動處理
            base_path=raw_data_path,
        )

        # 獲取下載的文件路徑
        databento_file = result["databento_data_file"]
        print(f"數據成功下載到: {databento_file}")

        # 讀取數據並進行處理
        loader = DatabentoDataLoader()
        bars = loader.from_dbn_file(
            path=databento_file, as_legacy_cython=False  # 獲取PyO3對象以便稍後處理
        )

        # 構建要保存的原始數據
        raw_data = pd.DataFrame()
        for bar in bars:
            data = {
                "timestamp": pd.Timestamp(bar.ts_event, unit="ns"),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            raw_data = pd.concat([raw_data, pd.DataFrame([data])], ignore_index=True)

        # 保存處理後的數據
        processed_filename = f"{symbol}_{venue}_{timeframe}-processed.parquet"
        processed_file_path = processed_data_path / processed_filename

        raw_data.set_index("timestamp", inplace=True)
        raw_data.sort_index(inplace=True)
        raw_data.to_parquet(processed_file_path)
        print(f"處理後的數據保存到: {processed_file_path}")

        # 創建數據目錄實例
        catalog = ParquetDataCatalog(catalog_path)

        # 構建條形圖類型字符串
        # 將timeframe轉換為Nautilus格式
        if timeframe == "1m":
            bar_type_str = f"{symbol}.{venue}-1-MINUTE-LAST-EXTERNAL"
        elif timeframe == "1h":
            bar_type_str = f"{symbol}.{venue}-1-HOUR-LAST-EXTERNAL"
        elif timeframe == "1d":
            bar_type_str = f"{symbol}.{venue}-1-DAY-LAST-EXTERNAL"

        # 構造Instrument對象(這裡使用從Databento中獲取的工具定義)
        if result.get("nautilus_definition"):
            instrument = result["nautilus_definition"][0]
        else:
            # 如果沒有獲取到工具定義, 則建立一個基本的InstrumentId
            instrument_id = InstrumentId.from_str(f"{symbol}.{venue}")
            print(f"警告:未找到工具定義, 使用基本ID: {instrument_id}")
            # 在這種情況下, 我們將只能寫入資料, 但沒有完整的工具元數據

        # 使用BarDataWrangler處理數據
        bar_type = BarType.from_str(bar_type_str)
        wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
        processed_bars = wrangler.process(raw_data)

        # 寫入工具和條形圖數據到目錄
        catalog.write_data([instrument], basename_template=f"{instrument.id.value}")
        catalog.write_data(processed_bars, basename_template=bar_type_str)

        print(f"\n成功處理並保存數據到目錄: {catalog_path}")
        print(f"已處理 {len(processed_bars)} 個條形圖")

        # 顯示數據範圍
        if processed_bars:
            first_bar = processed_bars[0]
            last_bar = processed_bars[-1]
            print(f"數據範圍: {first_bar.ts_init} 到 {last_bar.ts_init}")

    except Exception as e:
        print(f"處理數據時發生錯誤: {e}")


if __name__ == "__main__":
    # 示例使用
    process_databento_data(
        symbol="PLTR",
        venue="XNAS",
        start_date="2025-01-01",
        end_date="2025-01-31",
        timeframe="1m",
        raw_data_path="data/databento/raw",
        processed_data_path="data/databento/processed",
        catalog_path="data/databento/catalog",
    )
