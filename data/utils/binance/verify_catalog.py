from pathlib import Path

from nautilus_trader.model.data import Bar
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def check_catalog(catalog_path: str = "data/binance/catalog"):
    catalog = ParquetDataCatalog(Path(catalog_path))

    print("Checking catalog contents...")
    instruments = catalog.instruments()
    print(f"Found {len(instruments)} instruments")

    for instrument in instruments:
        print(f"\nInstrument: {instrument.id}")
        bar_type = f"{instrument.id}-5-MINUTE-LAST-EXTERNAL"
        bars = catalog.query(data_cls=Bar, bar_types=[bar_type])
        print(f"Found {len(bars)} bars")
        if bars:
            print(f"First bar: {bars[0]}")
            print(f"Last bar: {bars[-1]}")


if __name__ == "__main__":
    check_catalog()
