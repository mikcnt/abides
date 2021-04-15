import pandas as pd
import ta
import torch
import numpy as np
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler


def extract_signals(ohlc):
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


def normalize(df):
    df_copy = df.copy()
    mid_price_cols = [
        "lbband",
        "hbband",
        "mavg_12",
        "mavg_26",
        "ema_12",
        "ema_26",
    ]
    minmax_cols = ["macd", "rsi", "momentum"]

    for col in mid_price_cols:
        df_copy[col] = df_copy[col] / df_copy["mid_price"]

    for col in minmax_cols:
        non_normalized = df_copy.pop(col).to_numpy().reshape(-1, 1)
        scaler = MinMaxScaler()
        df_copy[col] = scaler.fit_transform(non_normalized)

    df_copy.drop(columns={"mid_price"}, inplace=True)
    return df_copy


def generate_input(ohlc, time):

    cols = [
        "lbband",
        "hbband",
        "mavg_12",
        "mavg_26",
        "ema_12",
        "ema_26",
        "macd",
        "rsi",
        "momentum",
        "vvolat",
        "count",
    ]

    start_time = time - pd.to_datetime('2Min')
    end_time = time - pd.to_datetime('1Min')
        
    ohlc = ohlc.loc[start_time:end_time]
    signals = extract_signals(ohlc)

    signals = normalize(signals)

    tech_signals = []
    tech_signals.append(torch.tensor(signals[cols].mean()))
    tech_signals.append(torch.tensor(signals[cols].std()))
    tech_signals.append(torch.tensor(signals[cols].skew()))
    tech_signals.append(torch.tensor(signals[cols].kurtosis()))

    rand_noise = torch.randn(100 - len(tech_signals) * len(cols))

    noise = torch.cat((*tech_signals, rand_noise)).reshape(100, 1, 1)

    return torch.unsqueeze(noise, 0).float()


def reshape_output(generated):
    return pd.DataFrame(generated.reshape(4, -1).cpu().detach().numpy()).T.rename(
        columns={0: "volume", 1: "price", 2: "direction", 3: "time_diff"}
    )


def load_model(model, path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    if checkpoint is None:
        raise RuntimeError("Checkpoint empty.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def unnormalize(normalized, mid_price, mid_volumes):
    unnormalized = normalized.copy()
    unnormalized["price"] = normalized["price"] * mid_price

    mid_volumes = np.array(mid_volumes).reshape(1, -1)
    scaler = MinMaxScaler()
    scaler.fit(mid_volumes)

    volume_unnormalized = scaler.inverse_transform(
        np.array(normalized["volume"]).reshape(1, -1)
    )
    unnormalized["volume"] = volume_unnormalized.astype(int).ravel()

    unnormalized.loc[normalized["direction"] <= 0.5, "direction"] = -1
    unnormalized.loc[normalized["direction"] > 0.5, "direction"] = 1

    unnormalized["time_diff"] = np.exp(normalized["time_diff"] * 10)

    return unnormalized
