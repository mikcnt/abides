import pandas as pd
import numpy as np
import torch
import time

from joblib import dump, load
from .utils import *
from .model import Generator
from agent.TradingAgent import TradingAgent

SCALER_DIR = "../scalers/"
SCALER_NAMES = ["vsell1", "vbuy1", "time_diff", "size", "price", "sell1", "buy1"] 

class GanAgent(TradingAgent):
    def __init__(
        self,
        id,
        name,
        type,
        symbol,
        starting_cash,
        min_size,
        max_size,
        mkt_open,
        mkt_close,
        generator_path,
        wake_up_freq="30s",
        subscribe=False,
        log_orders=False,
        random_state=None,
        verbose=False,
        trader_agent=None,
        volume_perc=0.005
    ):

        super().__init__(
            id,
            name,
            type,
            starting_cash=starting_cash,
            log_orders=log_orders,
            random_state=random_state,
        )
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.wake_up_freq = wake_up_freq
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.mid_list = []
        self.volumes = []
        self.orderbook_symbol = None
        self.tech_signals = []
        self.log_orders = log_orders
        self.generator = load_model(Generator(), generator_path)
        self.generator.eval()
        self.state = "AWAITING_WAKEUP"
        self.agent_state = "WAITING"
        self.time_passed = 0
        self.mkt_open = mkt_open
        self.mkt_close = mkt_close

        self.load_scalers()

        ## TODO: check try to have deterministic behavior/simulations
        #torch.use_deterministic_algorithms(True)
        #torch.manual_seed(self.random_state.randint(low=0,  high=2 ** 32))

        # self.sleep_time = int(np.random.exponential(scale=100))
        self.verbose = verbose
        
        self.last_call_gan = self.mkt_open
        # TODO: the ganstartup time is compute also on orderbook class in the same way, be aware on how to change this
        self.ganstartup_time = self.mkt_open # + pd.Timedelta("30m")
        self.volume_perc = volume_perc

        self.trader_agent = trader_agent
    

    def load_scalers(self):
        """ Load min max scalers --- in future also boxcox """
        self.scalers = {}
        self.lambdas = {}
        for sn in SCALER_NAMES:
            if sn in ['size', 'time_diff', 'vbuy1', 'vsell1']:
                self.lambdas[sn] = load_pickle(SCALER_DIR + sn + "_lambda.pickle")
            self.scalers[sn] = load(SCALER_DIR + sn + ".gz") 


    def save_last_ohlc(self, prefix_file):
        """ save a log of the last ohlc to debug purpose and logging """
        ohlc, orders = self.orderbook_symbol.get_ohlc(self.currentTime)
        ohlc.to_pickle(prefix_file + "ohlc_" + self.mkt_close.strftime("%Y-%m-%d") + "_" + self.symbol + ".bz2")

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if self.subscribe and not self.subscription_requested:
            super().requestDataSubscription(self.symbol, levels=1, freq=10e9)
            self.subscription_requested = True
            self.state = "AWAITING_MARKET_DATA"
        elif can_trade and not self.subscribe:
            self.getCurrentSpread(self.symbol)
            self.state = "AWAITING_SPREAD"

    def receiveOrderbook(self, orderbooks):
        assert self.symbol in orderbooks, "The symbol of the GAN agent is not inside the Exchange orderbook"
        self.orderbook_symbol = orderbooks[self.symbol]
    
    def receiveMessage(self, currentTime, msg):
        """ Momentum agent actions are determined after obtaining the best bid and ask in the LOB """
        super().receiveMessage(currentTime, msg)
        self.placeOrders()

    def __get_unormalized_par(self, ohlc):
        """ get the mid price and volume to compute the unnormalization of generated orders """
        last_time = self.currentTime.floor("Min") - pd.Timedelta('1m')
        mid_price = (ohlc.loc[last_time]["open"] + ohlc.loc[last_time]["close"]) / 2

        # TODO: change here maybe
        volumes = ohlc.loc[self.mkt_open:self.ganstartup_time]["volume"] * self.volume_perc

        return mid_price, volumes.values.reshape(-1,1)

    def placeOrders(self):
        """ Get new orders from the GAN model and give them to the related TraderGan to put them in execution """
        # Wait for the first 30 minutes so that there is enough data
        if self.currentTime < self.ganstartup_time:
            self.setWakeup(self.ganstartup_time)
            return

        # If it is passed more than 1 minute between the last call and this one,
        # we can generate trades with the GAN
        if self.currentTime >= self.ganstartup_time:
            # Preprocess the OHLC s.t. the GAN can use that
            # (i.e., generate signals, take last 2 minutes, normalize)
            ohlc, latest_trades = self.orderbook_symbol.get_ohlc(self.currentTime)
            gan_input = generate_input(latest_trades, self.scalers, self.lambdas, self.currentTime)
            # Generate trades with the GAN (generator)
            trades = self.generator(gan_input).reshape(1, 4)
            # trades to pandas
            trades = pd.DataFrame(trades.detach().numpy()).rename(
                    columns={0: "volume", 1: "price", 2: "direction", 3: "time_diff"})
            trades_print = trades.values.reshape(4)
            # if trades_print[2] > 0:
            print(trades.values)
            trades = unnormalize(trades, self.scalers, self.lambdas)
            # print(np.array(trades))
            # change the `time_diff` from difference between each row to absolute time
            trades["time_diff"] =  trades["time_diff"].clip(0.001)*200
            trades["time_diff"] = self.currentTime + pd.to_timedelta(trades['time_diff'].cumsum().clip(0.0001), unit='S')
            trades["volume"] = trades["volume"].clip(1).astype(int)

            next_wake_up = trades.iloc[0]["time_diff"]
            # Pass trades to the TraderAgent so that it can do the actual trading (even in the future)
            self.trader_agent.add_orders(trades.values)
            # Update last call time for next call
            self.last_call_gan = self.currentTime
            self.setWakeup(next_wake_up)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)
