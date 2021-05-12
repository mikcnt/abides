import pandas as pd
import ta
import torch
import numpy as np
import pickle
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox

def extract_signals(ohlc):
    # technical signals
    ohlc["mid_price"] = (ohlc["high"] + ohlc["low"]) / 2

    bb = ta.volatility.BollingerBands(ohlc["close"], fillna=True)
    ohlc["lbband"] = bb.bollinger_lband()
    ohlc["hbband"] = bb.bollinger_hband()

    ohlc["mid_price_kurtosis"] = ohlc["mid_price"].rolling(10).kurt()

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


def normalize(df, norm_volume=8592, norm_orders=234):
    df_copy = df.copy()
    mid_price_cols = [
        "lbband",
        "hbband",
        "mavg_12",
        "mavg_26",
        "ema_12",
        "ema_26",
        "mid_price_kurtosis"
    ]
    minmax_cols = ["macd", "rsi", "momentum"]

    for col in mid_price_cols:
        df_copy[col] = df_copy[col] / df_copy["mid_price"]

    df_copy["volume"] = df_copy["volume"] / norm_volume
    df_copy["count"] = df_copy["count"] / norm_orders

    for col in minmax_cols:
        non_normalized = df_copy.pop(col).to_numpy().reshape(-1, 1)
        scaler = MinMaxScaler()
        df_copy[col] = scaler.fit_transform(non_normalized)

    df_copy.drop(columns={"mid_price"}, inplace=True)
    return df_copy


def generate_input(latest_trades, scalers, lambdas, cur_time):
    for col in ["time_diff", "size", "price", "sell1", "vsell1", "buy1", "vbuy1"]:
        if col in ['size', 'time_diff', 'vbuy1', 'vsell1']:
            # clip values if equal to 0
            min_value = latest_trades[col].min()
            latest_trades[col] = (
                    latest_trades[col].clip(lower=latest_trades[latest_trades != min_value][col].min())
                    if min_value == 0
                    else latest_trades[col]
                )
            # normalize with boxcox
            lambda_col = lambdas[col]
            latest_trades[col] = boxcox_(latest_trades[col], lmbda=lambda_col)
        scaler_col = scalers[col]
        # if col == 'size':
        # normalize with minmax
        latest_trades[[col]] = scaler_col.transform(latest_trades[[col]])


    # -------------- TIME SLOT --------------
    # (from 0 to 23 according to hour of the day of trading)
    latest_trades["time_slot"] = 0
    time_slots = pd.date_range("09:30:00", "16:30:00", periods=25)
    # this is something like: ['09:30', '09:47', '10:05', '10:22', ...]
    time_slots = [
        f"{str(t.hour).zfill(2)}:{str(t.minute).zfill(2)}" for t in time_slots
    ]
    
    # set date as index to use `between_time`
    # TODO: Brutto da cambaire
    latest_trades["time_slot"] = cur_time
    latest_trades.index = latest_trades["time_slot"]
    for i in range(len(time_slots) - 1):
        if len(latest_trades.between_time(time_slots[i], time_slots[i + 1])) > 0:       
            latest_trades["time_slot"] = i  
            break
    
    trade_features = ["size", "price", "direction", "time_diff"]
    orderbook_features = ["sell1", "vsell1", "buy1", "vbuy1", "time_slot"]
    trades = torch.tensor(
            latest_trades[trade_features].values, dtype=torch.float32,
        )
    orderbook = torch.tensor(
            latest_trades[orderbook_features].values, dtype=torch.float32,
    
    )
    gan_input = torch.cat((trades, orderbook), dim=-1).unsqueeze(0)
    # gan_input = torch.rand_like(gan_input)
    return gan_input


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


def unnormalize(normalized, scalers, lambdas):
    # print(normalized)
    unnormalized = normalized.copy()
    for col in ["time_diff", "size", "price"]:
        col_df = "volume" if col == "size" else col
        unnormalized[col_df] = scalers[col].inverse_transform(unnormalized[[col_df]]) 
        if col != 'price':
            unnormalized[col_df] = reverse_boxcox(unnormalized[col_df], lambdas[col])
            # unnormalized[col_df] = (lambdas[col] * unnormalized[col_df] + 1) ** (1 / lambdas[col])
    # print(unnormalized)
    unnormalized.loc[normalized["direction"] <= 0, "direction"] = -1
    unnormalized.loc[normalized["direction"] > 0, "direction"] = 1

    # unnormalized["direction"] = np.random.randint(0, 2)
    # unnormalized.loc[normalized["direction"] == 0, "direction"] = -1

    return unnormalized


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def boxcox_(x, lmbda):
    if lmbda != 0:
        return (x ** lmbda - 1) / lmbda
    return np.log(x)

def reverse_boxcox(x, lmbda):
    if lmbda != 0:
        return (lmbda * x + 1) ** (1 / lmbda)
    return np.exp(x)