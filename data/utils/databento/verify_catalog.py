from pathlib import Path

import pandas as pd
from nautilus_trader.model.data import BarType
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def verify_catalog(catalog_path: str = "data/databento/catalog"):
    """Verify data in the catalog and print summary statistics."""
    catalog = ParquetDataCatalog(Path(catalog_path))

    # 列出目錄中的所有工具
    instruments = catalog.instruments()
    print(f"Found {len(instruments)} instruments in catalog:")

    for instrument in instruments:
        print(f"\nInstrument: {instrument.id}")
        print(f"Symbol: {instrument.symbol}")
        print(f"Venue: {instrument.venue}")

        # 創建此工具的bar類型
        bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")

        # 查詢bars
        bars = catalog.bars(bar_types=[bar_type])

        if not bars:
            print(f"No bars found for {instrument.id}")
            continue

        # 轉換為DataFrame以便分析
        df = pd.DataFrame(
            [
                {
                    "timestamp": bar.ts_init,
                    "open": float(bar.open.as_double()),
                    "high": float(bar.high.as_double()),
                    "low": float(bar.low.as_double()),
                    "close": float(bar.close.as_double()),
                    "volume": float(bar.volume),
                }
                for bar in bars
            ]
        )

        # 添加適當的datetime欄位
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ns")
        df.set_index("datetime", inplace=True)

        # 打印摘要
        print(f"Total bars: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Sample data:\n{df.head()}")

        # 顯示基本統計數據
        print("\nBasic statistics:")
        print(df[["open", "high", "low", "close", "volume"]].describe())


if __name__ == "__main__":
    verify_catalog()
    print("\nVerification complete!")
