import pandas as pd
import ta
import torch
import numpy as np
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler


def signals(mid_list):
    mid_pd = pd.DataFrame(mid_list, columns=["mid_price"])

    # TODO: compute technical signals using time, not rows
    bb = ta.volatility.BollingerBands(mid_pd["mid_price"], fillna=True)
    mid_pd["lbband"] = bb.bollinger_lband()
    mid_pd["hbband"] = bb.bollinger_hband()

    mid_pd["mavg_12"] = ta.volatility.bollinger_mavg(
        mid_pd["mid_price"], window=12, fillna=True
    )
    mid_pd["mavg_26"] = ta.volatility.bollinger_mavg(
        mid_pd["mid_price"], window=26, fillna=True
    )

    ema_12 = ta.trend.EMAIndicator(mid_pd["mid_price"], window=12, fillna=True)
    mid_pd["ema_12"] = ema_12.ema_indicator()

    ema_26 = ta.trend.EMAIndicator(mid_pd["mid_price"], window=26, fillna=True)
    mid_pd["ema_26"] = ema_26.ema_indicator()

    macd = ta.trend.MACD(mid_pd["mid_price"], fillna=True)
    mid_pd["macd"] = macd.macd()

    rsi = ta.momentum.RSIIndicator(mid_pd["mid_price"], fillna=True)
    mid_pd["rsi"] = rsi.rsi()

    mid_pd["momentum"] = ta.momentum.roc(mid_pd["mid_price"], fillna=True)

    # TODO: `vvolat` (volatility indicator) and `count` (number of orders of previous 10mins)
    # are just placeholders for the moment
    mid_pd["count"] = np.arange(len(mid_pd)) + 10000
    mid_pd["vvolat"] = np.random.rand(len(mid_pd))

    return mid_pd


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


def generate_input(sngls):

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

    tech_signals = []
    tech_signals.append(torch.tensor(sngls[cols].mean()))
    tech_signals.append(torch.tensor(sngls[cols].std()))
    tech_signals.append(torch.tensor(sngls[cols].skew()))
    tech_signals.append(torch.tensor(sngls[cols].kurtosis()))

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
