from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np
import torch
import time
from .utils import *
from .model import Generator


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
        generator_path,
        wake_up_freq="1ms",
        subscribe=False,
        log_orders=False,
        random_state=None,
        verbose=False,
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
        self.tech_signals = []
        self.log_orders = log_orders
        self.generator = load_model(Generator(100), generator_path)
        self.state = "AWAITING_WAKEUP"
        self.agent_state = "WAITING"
        self.time_passed = 0
        self.sleep_time = int(np.random.exponential(scale=100))
        self.verbose = verbose
        
        self.n_traded_orders = 0

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

    def receiveOrderbook(self, ohlc):
        self.ohlc = ohlc
    
    def receiveMessage(self, currentTime, msg):
        """ Momentum agent actions are determined after obtaining the best bid and ask in the LOB """
        super().receiveMessage(currentTime, msg)
        if (
            not self.subscribe
            and self.state == "AWAITING_SPREAD"
            and msg.body["msg"] == "QUERY_SPREAD"
        ):
            bid, volume_bid, ask, volume_ask = self.getKnownBidAsk(self.symbol)
            self.currentTime = currentTime
            self.placeOrders(bid, ask, volume_bid, volume_ask)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = "AWAITING_WAKEUP"
        elif (
            self.subscribe
            and self.state == "AWAITING_MARKET_DATA"
            and msg.body["msg"] == "MARKET_DATA"
        ):
            bids, asks = self.known_bids[self.symbol], self.known_asks[self.symbol]
            if bids and asks:
                self.placeOrders(bids[0][0], asks[0][0], bids[0][1], asks[0][1])
            self.state = "AWAITING_MARKET_DATA"

    # Il placeOrders funziona diversamente a seconda della fase:
    # 1) OBSERVING:
    # L'agente accumula ad ogni unità di tempo (e.g., 1ms) un bid e ask
    # Quando ha accumulato 1024 bid+ask, la GAN genera i trade da mandare
    # e passa in fase di trading.
    # 2) TRADING:
    # La nuova unità di tempo sarà definita da il tempo generato dalla GAN.
    # Ad ogni unità di tempo, l'agente esegue un trade e decide il prossimo
    # wakeup con il tempo generato. Una volta finiti i trade, o passata una
    # finestra temporale (e.g., 30s), l'agente torna in fase di osservazione
    # e reimposta la wake up frequency di default.

    def placeOrders(self, bid, ask, volume_bid, volume_ask):
        """ Momentum Agent actions logic """
        if self.agent_state == "WAITING":
            self.sleep_time -= 1
            if self.sleep_time <= 0:
                self.agent_state = "OBSERVING"
                if self.verbose:
                    print(f"GAN Agent {self.id}: WAITING -> OBSERVING.")
        # If in state OBSERVING
        if self.agent_state == "OBSERVING":
            # print(len(self.mid_list))
            if bid and ask:
                # prima fase: accumula i dati fino ad avere i primi 1024 ordini
                mid_price = (bid + ask) / 2
                mid_volume = (volume_bid + volume_ask) / 2
                if (len(self.mid_list) > 0 and mid_price != self.mid_list[-1]) or len(self.mid_list) == 0:
                    self.mid_list.append(mid_price)
                    self.volumes.append(mid_volume)
                    # print(f"GAN Agent {self.id}: # trades = {len(self.mid_list)}")
                if len(self.mid_list) == 1024:
                    # Genera i segnali da mandare in input alla GAN
                    extracted_signals = signals(self.mid_list)
                    # Normalizza i segnali per avere dati coerenti con il training
                    tech_signals = normalize(extracted_signals)
                    # Genera input per la GAN (segnali tecnici + noise)
                    gan_input = generate_input(tech_signals)
                    # Genera i trade, come output della gan e rendi leggibili
                    self.trades = reshape_output(self.generator(gan_input))
                    # Calcola moving average
                    # self.trades = self.trades.rolling(window=100).mean().dropna().reset_index(drop=True)
                    # self.volumes = pd.Series(self.volumes).rolling(window=100).mean().dropna().tolist()
                    # Normalizza al contrario i dati per poter fare trading
                    average_mid_price = np.array(self.mid_list).mean()
                    self.trades = unnormalize(
                        self.trades, average_mid_price, self.volumes
                    )
                    self.agent_state = "TRADING"
                    if self.verbose:
                        print(f"GAN Agent {self.id}: OBSERVING -> TRADING.")

        if self.agent_state == "TRADING":
            # Interrompi il trading se: a) non ci sono più trade b) sono passati 30s
            if len(self.trades) == 0 or self.time_passed > 30:
                # Riazzera il tempo passato per la fase di trading successiva
                self.time_passed = 0
                # Risetta la wake up freq. di default
                self.setWakeup(self.currentTime + self.getWakeFrequency())
                # Setta lo stato di OBSERVING
                self.agent_state = "OBSERVING"
                # self.mid_list = []
                if self.verbose:
                    print(f"GAN Agent {self.id}: TRADING -> OBSERVING.")
                return
            # TRADING
            # Altrimenti l'agente fa trading

            # Estrai dalla prima riga dell'ouput della GAN il trade
            # (più la direzione e il tempo da aspettare per il prossimo trade)
            volume, price, direction, time_diff = self.trades.iloc[0]
            is_buy_order = direction == 1
            # L'agente fa il trade
            self.placeLimitOrder(
                self.symbol,
                quantity=volume,
                is_buy_order=is_buy_order,
                limit_price=price,
            )
            with open('n_orders.txt', 'a') as f:
                f.write(f'{self.id} 1\n')
            self.n_traded_orders += 1
            # Elimina la prima riga => prossimo trade, prossima riga
            self.trades.drop(0, inplace=True)
            self.trades.reset_index(drop=True, inplace=True)
            # Aggiorna il tempo che passerà con il prossimo ordine
            self.time_passed += time_diff
            # Stabilisci quando l'agente si dovrà risvegliare
            time_diff = pd.Timedelta(time_diff, unit="s")
            self.setWakeup(self.currentTime + time_diff)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)
