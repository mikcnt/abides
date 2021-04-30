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
        columns={"open", "high", "low", "close"}, inplace=True
    )
    return ohlc


def normalize(df, average_norders, average_volume):
    df_copy = df.copy()
    mid_price_cols = [
        "open",
        "high",
        "low",
        "close",
    ]

    for col in mid_price_cols:
        df_copy[col] = df_copy[col] / df_copy["mid_price"]

    df_copy["norders"] = df_copy["norders"] / average_norders
    df_copy["volume"] = df_copy["volume"] / average_volume

    return df_copy


def generate_input(ohlc, time):
    # columns fed to the GAN generator
    cols = [
        "open",
        "high",
        "low",
        "close",
        "norders",
        "volume",
    ]

    # get 30 minutes of ohlc before current time
    time = time.floor("30S") - pd.Timedelta(seconds=30)
    ohlc_30_minutes = ohlc.loc[time - pd.Timedelta(seconds=1800):time - pd.Timedelta(seconds=30)]

    # normalize ohlc and leave signals columns
    ohlc_30_minutes = normalize(ohlc_30_minutes)[cols]
    
    # convert to torch and concatenate noise
    ohlc_30_minutes = torch.tensor(ohlc_30_minutes.values, dtype=torch.float32)
    gen_noise = torch.rand((ohlc_30_minutes.shape[0], len(cols)), dtype=torch.float32)
    gen_input = torch.cat((ohlc_30_minutes, gen_noise), dim=1).T

    # unsqueeze to create batch dimension
    return torch.unsqueeze(gen_input, 0)


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


def unnormalize(normalized, mid_price, mid_volumes, avg_volume=None):
    unnormalized = normalized.copy()

    # TODO: use avg_volume and mid_price here not, using the volume in the current way?
    unnormalized["price"] = normalized["price"] * mid_price
#    unnormalized["volume"] = unnormalized["volume"] * avg_volume

    scaler = MinMaxScaler()
    scaler.fit(mid_volumes)
#
    volume_unnormalized = scaler.inverse_transform(
        np.array(normalized["volume"]).reshape(1, -1)
    )
    unnormalized["volume"] = volume_unnormalized.astype(int).ravel()

    unnormalized.loc[normalized["direction"] <= 0.5, "direction"] = -1
    unnormalized.loc[normalized["direction"] > 0.5, "direction"] = 1

    # no unnormalization back, right?
    # TODO: why?
    # unnormalized["time_diff"] = np.exp(normalized["time_diff"] * 10)

    return unnormalized
