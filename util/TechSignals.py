import pandas as pd
import ta


def tech_signals(ohlc):
    # technical signals
    ohlc["mid_price"] = (ohlc["high"] + ohlc["low"]) / 2

    bb = ta.volatility.BollingerBands(ohlc["close"], fillna=True)
    ohlc["lbband"] = bb.bollinger_lband()
    ohlc["hbband"] = bb.bollinger_hband()

    # compute the current volatility of the stock
    # 10 as window to be consistent with 10 time
    vatr = ta.volatility.AverageTrueRange(
        ohlc["high"], ohlc["low"], ohlc["close"], window=10
    )
    ohlc["vvolat"] = vatr.average_true_range()

    ohlc["mavg_12"] = ta.volatility.bollinger_mavg(
        ohlc["close"], window=12, fillna=True
    )
    ohlc["mavg_26"] = ta.volatility.bollinger_mavg(
        ohlc["close"], window=26, fillna=True
    )

    ema_12 = ta.trend.EMAIndicator(ohlc["close"], window=12, fillna=True)
    ohlc["ema_12"] = ema_12.ema_indicator()

    ema_26 = ta.trend.EMAIndicator(ohlc["close"], window=26, fillna=True)
    ohlc["ema_26"] = ema_26.ema_indicator()

    macd = ta.trend.MACD(ohlc["close"], fillna=True)
    ohlc["macd"] = macd.macd()

    rsi = ta.momentum.RSIIndicator(ohlc["close"], fillna=True)
    ohlc["rsi"] = rsi.rsi()

    ohlc["momentum"] = ta.momentum.roc(ohlc["close"], fillna=True)
    
    # drop useless columns
    ohlc.drop(
        columns={"open", "high", "low", "close", "volume"}, inplace=True
    )
    return ohlc