(probabilistic_trader) ohh@Arbitrage probabilistic_trader % ls -R .
README.md               docs                    pyproject.toml          scripts
data                    probabilistic_trading   research                uv.lock

./data:
binance catalog

./data/binance:
futures                 futures_processed

./data/binance/futures:
ADA_USDT_USDT-15m-futures.parquet       DOT_USDT_USDT-15m-futures.parquet       SOL_USDT_USDT-15m-futures.parquet
ADA_USDT_USDT-1h-futures.parquet        DOT_USDT_USDT-1h-futures.parquet        SOL_USDT_USDT-1h-futures.parquet
ADA_USDT_USDT-1m-futures.parquet        DOT_USDT_USDT-1m-futures.parquet        SOL_USDT_USDT-1m-futures.parquet
ADA_USDT_USDT-4h-futures.parquet        DOT_USDT_USDT-4h-futures.parquet        SOL_USDT_USDT-4h-futures.parquet
ADA_USDT_USDT-5m-futures.parquet        DOT_USDT_USDT-5m-futures.parquet        SOL_USDT_USDT-5m-futures.parquet
ADA_USDT_USDT-8h-funding_rate.parquet   DOT_USDT_USDT-8h-funding_rate.parquet   SOL_USDT_USDT-8h-funding_rate.parquet
ADA_USDT_USDT-8h-mark.parquet           DOT_USDT_USDT-8h-mark.parquet           SOL_USDT_USDT-8h-mark.parquet
AVAX_USDT_USDT-15m-futures.parquet      ETH_USDT_USDT-15m-futures.parquet       SUI_USDT_USDT-15m-futures.parquet
AVAX_USDT_USDT-1h-futures.parquet       ETH_USDT_USDT-1h-futures.parquet        SUI_USDT_USDT-1h-futures.parquet
AVAX_USDT_USDT-1m-futures.parquet       ETH_USDT_USDT-1m-futures.parquet        SUI_USDT_USDT-1m-futures.parquet
AVAX_USDT_USDT-4h-futures.parquet       ETH_USDT_USDT-4h-futures.parquet        SUI_USDT_USDT-4h-futures.parquet
AVAX_USDT_USDT-5m-futures.parquet       ETH_USDT_USDT-5m-futures.parquet        SUI_USDT_USDT-5m-futures.parquet
AVAX_USDT_USDT-8h-funding_rate.parquet  ETH_USDT_USDT-8h-funding_rate.parquet   SUI_USDT_USDT-8h-funding_rate.parquet
AVAX_USDT_USDT-8h-mark.parquet          ETH_USDT_USDT-8h-mark.parquet           SUI_USDT_USDT-8h-mark.parquet
BTC_USDT_USDT-15m-futures.parquet       LINK_USDT_USDT-15m-futures.parquet      UNI_USDT_USDT-15m-futures.parquet
BTC_USDT_USDT-1h-futures.parquet        LINK_USDT_USDT-1h-futures.parquet       UNI_USDT_USDT-1h-futures.parquet
BTC_USDT_USDT-1m-futures.parquet        LINK_USDT_USDT-1m-futures.parquet       UNI_USDT_USDT-1m-futures.parquet
BTC_USDT_USDT-4h-futures.parquet        LINK_USDT_USDT-4h-futures.parquet       UNI_USDT_USDT-4h-futures.parquet
BTC_USDT_USDT-5m-futures.parquet        LINK_USDT_USDT-5m-futures.parquet       UNI_USDT_USDT-5m-futures.parquet
BTC_USDT_USDT-8h-funding_rate.parquet   LINK_USDT_USDT-8h-funding_rate.parquet  UNI_USDT_USDT-8h-funding_rate.parquet
BTC_USDT_USDT-8h-mark.parquet           LINK_USDT_USDT-8h-mark.parquet          UNI_USDT_USDT-8h-mark.parquet
DOGE_USDT_USDT-15m-futures.parquet      LTC_USDT_USDT-15m-futures.parquet       XRP_USDT_USDT-15m-futures.parquet
DOGE_USDT_USDT-1h-futures.parquet       LTC_USDT_USDT-1h-futures.parquet        XRP_USDT_USDT-1h-futures.parquet
DOGE_USDT_USDT-1m-futures.parquet       LTC_USDT_USDT-1m-futures.parquet        XRP_USDT_USDT-1m-futures.parquet
DOGE_USDT_USDT-4h-futures.parquet       LTC_USDT_USDT-4h-futures.parquet        XRP_USDT_USDT-4h-futures.parquet
DOGE_USDT_USDT-5m-futures.parquet       LTC_USDT_USDT-5m-futures.parquet        XRP_USDT_USDT-5m-futures.parquet
DOGE_USDT_USDT-8h-funding_rate.parquet  LTC_USDT_USDT-8h-funding_rate.parquet   XRP_USDT_USDT-8h-funding_rate.parquet
DOGE_USDT_USDT-8h-mark.parquet          LTC_USDT_USDT-8h-mark.parquet           XRP_USDT_USDT-8h-mark.parquet

./data/binance/futures_processed:
ADA_USDT_USDT-15m-futures-processed.parquet     LINK_USDT_USDT-15m-futures-processed.parquet
ADA_USDT_USDT-1h-futures-processed.parquet      LINK_USDT_USDT-1h-futures-processed.parquet
ADA_USDT_USDT-1m-futures-processed.parquet      LINK_USDT_USDT-1m-futures-processed.parquet
ADA_USDT_USDT-4h-futures-processed.parquet      LINK_USDT_USDT-4h-futures-processed.parquet
ADA_USDT_USDT-5m-futures-processed.parquet      LINK_USDT_USDT-5m-futures-processed.parquet
AVAX_USDT_USDT-15m-futures-processed.parquet    LTC_USDT_USDT-15m-futures-processed.parquet
AVAX_USDT_USDT-1h-futures-processed.parquet     LTC_USDT_USDT-1h-futures-processed.parquet
AVAX_USDT_USDT-1m-futures-processed.parquet     LTC_USDT_USDT-1m-futures-processed.parquet
AVAX_USDT_USDT-4h-futures-processed.parquet     LTC_USDT_USDT-4h-futures-processed.parquet
AVAX_USDT_USDT-5m-futures-processed.parquet     LTC_USDT_USDT-5m-futures-processed.parquet
BTC_USDT_USDT-15m-futures-processed.parquet     SOL_USDT_USDT-15m-futures-processed.parquet
BTC_USDT_USDT-1h-futures-processed.parquet      SOL_USDT_USDT-1h-futures-processed.parquet
BTC_USDT_USDT-1m-futures-processed.parquet      SOL_USDT_USDT-1m-futures-processed.parquet
BTC_USDT_USDT-4h-futures-processed.parquet      SOL_USDT_USDT-4h-futures-processed.parquet
BTC_USDT_USDT-5m-futures-processed.parquet      SOL_USDT_USDT-5m-futures-processed.parquet
DOGE_USDT_USDT-15m-futures-processed.parquet    SUI_USDT_USDT-15m-futures-processed.parquet
DOGE_USDT_USDT-1h-futures-processed.parquet     SUI_USDT_USDT-1h-futures-processed.parquet
DOGE_USDT_USDT-1m-futures-processed.parquet     SUI_USDT_USDT-1m-futures-processed.parquet
DOGE_USDT_USDT-4h-futures-processed.parquet     SUI_USDT_USDT-4h-futures-processed.parquet
DOGE_USDT_USDT-5m-futures-processed.parquet     SUI_USDT_USDT-5m-futures-processed.parquet
DOT_USDT_USDT-15m-futures-processed.parquet     UNI_USDT_USDT-15m-futures-processed.parquet
DOT_USDT_USDT-1h-futures-processed.parquet      UNI_USDT_USDT-1h-futures-processed.parquet
DOT_USDT_USDT-1m-futures-processed.parquet      UNI_USDT_USDT-1m-futures-processed.parquet
DOT_USDT_USDT-4h-futures-processed.parquet      UNI_USDT_USDT-4h-futures-processed.parquet
DOT_USDT_USDT-5m-futures-processed.parquet      UNI_USDT_USDT-5m-futures-processed.parquet
ETH_USDT_USDT-15m-futures-processed.parquet     XRP_USDT_USDT-15m-futures-processed.parquet
ETH_USDT_USDT-1h-futures-processed.parquet      XRP_USDT_USDT-1h-futures-processed.parquet
ETH_USDT_USDT-1m-futures-processed.parquet      XRP_USDT_USDT-1m-futures-processed.parquet
ETH_USDT_USDT-4h-futures-processed.parquet      XRP_USDT_USDT-4h-futures-processed.parquet
ETH_USDT_USDT-5m-futures-processed.parquet      XRP_USDT_USDT-5m-futures-processed.parquet

./data/catalog:
data

./data/catalog/data:
bar                     crypto_perpetual

./data/catalog/data/bar:
ADAUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL       LINKUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL
ADAUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL     LINKUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL
ADAUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL    LINKUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL
ADAUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL       LINKUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL
ADAUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL     LTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL
AVAXUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL      LTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL
AVAXUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL    LTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL
AVAXUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL   LTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL
AVAXUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL      LTCUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL
AVAXUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL    SOLUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL
BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL       SOLUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL
BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL     SOLUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL
BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL    SOLUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL
BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL       SOLUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL
BTCUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL     SUIUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL
DOGEUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL      SUIUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL
DOGEUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL    SUIUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL
DOGEUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL   SUIUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL
DOGEUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL    SUIUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL
DOTUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL       UNIUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL
DOTUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL     UNIUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL
DOTUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL    UNIUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL
DOTUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL       UNIUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL
DOTUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL     UNIUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL
ETHUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL       XRPUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL
ETHUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL     XRPUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL
ETHUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL    XRPUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL
ETHUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL       XRPUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL
ETHUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL     XRPUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL
LINKUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL

./data/catalog/data/bar/ADAUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
ADAUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ADAUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
ADAUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ADAUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
ADAUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ADAUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
ADAUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ADAUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
ADAUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/AVAXUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
AVAXUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/AVAXUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
AVAXUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/AVAXUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
AVAXUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/AVAXUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
AVAXUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/AVAXUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
AVAXUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
BTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/BTCUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
BTCUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOGEUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
DOGEUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOGEUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
DOGEUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOGEUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
DOGEUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOGEUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
DOGEUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOTUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
DOTUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOTUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
DOTUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOTUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
DOTUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOTUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
DOTUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/DOTUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
DOTUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ETHUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
ETHUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ETHUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
ETHUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ETHUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
ETHUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ETHUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
ETHUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/ETHUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
ETHUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LINKUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
LINKUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LINKUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
LINKUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LINKUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
LINKUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LINKUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
LINKUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LINKUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
LINKUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
LTCUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
LTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
LTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
LTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/LTCUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
LTCUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SOLUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
SOLUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SOLUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
SOLUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SOLUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
SOLUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SOLUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
SOLUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SOLUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
SOLUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SUIUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
SUIUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SUIUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
SUIUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SUIUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
SUIUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SUIUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
SUIUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/SUIUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
SUIUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/UNIUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
UNIUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/UNIUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
UNIUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/UNIUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
UNIUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/UNIUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
UNIUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/UNIUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
UNIUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/XRPUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL:
XRPUSDT-PERP.BINANCE-1-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/XRPUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL:
XRPUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/XRPUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL:
XRPUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/bar/XRPUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL:
XRPUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL.parquet

./data/catalog/data/bar/XRPUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL:
XRPUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL.parquet

./data/catalog/data/crypto_perpetual:
ADAUSDT-PERP.BINANCE    DOGEUSDT-PERP.BINANCE   LINKUSDT-PERP.BINANCE   SUIUSDT-PERP.BINANCE
AVAXUSDT-PERP.BINANCE   DOTUSDT-PERP.BINANCE    LTCUSDT-PERP.BINANCE    UNIUSDT-PERP.BINANCE
BTCUSDT-PERP.BINANCE    ETHUSDT-PERP.BINANCE    SOLUSDT-PERP.BINANCE    XRPUSDT-PERP.BINANCE

./data/catalog/data/crypto_perpetual/ADAUSDT-PERP.BINANCE:
ADAUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/AVAXUSDT-PERP.BINANCE:
AVAXUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/BTCUSDT-PERP.BINANCE:
BTCUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/DOGEUSDT-PERP.BINANCE:
DOGEUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/DOTUSDT-PERP.BINANCE:
DOTUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/ETHUSDT-PERP.BINANCE:
ETHUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/LINKUSDT-PERP.BINANCE:
LINKUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/LTCUSDT-PERP.BINANCE:
LTCUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/SOLUSDT-PERP.BINANCE:
SOLUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/SUIUSDT-PERP.BINANCE:
SUIUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/UNIUSDT-PERP.BINANCE:
UNIUSDT-PERP.BINANCE.parquet

./data/catalog/data/crypto_perpetual/XRPUSDT-PERP.BINANCE:
XRPUSDT-PERP.BINANCE.parquet

./docs:

./probabilistic_trading:
__init__.py     __pycache__     model           strategy

./probabilistic_trading/__pycache__:
__init__.cpython-312.pyc

./probabilistic_trading/model:
__init__.py     __pycache__     hmm             kalman

./probabilistic_trading/model/__pycache__:
__init__.cpython-312.pyc

./probabilistic_trading/model/hmm:
__init__.py     __pycache__     hmm.md          hmm_actor.py    hmm_data.py     hmm_model.py

./probabilistic_trading/model/hmm/__pycache__:
__init__.cpython-312.pyc        hmm_data.cpython-312.pyc
hmm_actor.cpython-312.pyc       hmm_model.cpython-312.pyc

./probabilistic_trading/model/kalman:
__init__.py     __pycache__     kalman_actor.py kalman_data.py  kalman_model.py

./probabilistic_trading/model/kalman/__pycache__:
__init__.cpython-312.pyc        kalman_data.cpython-312.pyc
kalman_actor.cpython-312.pyc    kalman_model.cpython-312.pyc

./probabilistic_trading/strategy:
__init__.py             __pycache__             hmm_strategy.py         hmmkalman_strategy.py   kalman_strategy.py

./probabilistic_trading/strategy/__pycache__:
__init__.cpython-312.pyc                hmmkalman_strategy.cpython-312.pyc
hmm_strategy.cpython-312.pyc            kalman_strategy.cpython-312.pyc

./research:
feature_engineering.ipynb       hmm_dev.ipynb

./scripts:
backtest        live            sandbox         utils

./scripts/backtest:
hmm_backtest_engine.py          hmmkalman_backtest_engine.py

./scripts/live:

./scripts/sandbox:

./scripts/utils:
batch_raw_to_catalog.py raw_to_catalog.py