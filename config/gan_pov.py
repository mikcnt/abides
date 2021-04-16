# GAN (Reference Market Simulation Configuration):
# - 1     Exchange Agent
# - 1     POV Market Maker Agent
# - 100   Value Agents
# - 25    Momentum Agents
# - 5000  Noise Agents
# - 1     (Optional) POV Execution agent

import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent
from model.LatencyModel import LatencyModel

from agent.GanAgent.GanAgent import GanAgent
from agent.GanAgent.TraderAgent import TraderAgent
########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for GAN config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='14:00:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('--real-ohlc',
                    default=None,
                    type=str,
                    help='The ohlc of a symbol to emulate.'
                    )
parser.add_argument('--gan-model-file',
                    required=True,
                    type=str,
                    help='The file of the gan model.'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    action='store_true',
                    help='Flag to allow the execution agent to trade.')
parser.add_argument('-p',
                    '--execution-pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level for execution agent')
# market maker config
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-window-size',
                    type=util.validate_window_size,
                    default='adaptive'
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-num-ticks',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-skew-beta',
                    type=float,
                    default=0
                    )
parser.add_argument('--mm-level-spacing',
                    type=float,
                    default=5
                    )
parser.add_argument('--mm-spread-alpha',
                    type=float,
                    default=0.75
                    )
parser.add_argument('--mm-backstop-quantity',
                    type=float,
                    default=50000)

parser.add_argument('--fund-vol',
                    type=float,
                    default=1e-8,
                    help='Volatility of fundamental time series.'
                    )

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

exchange_log_orders = True
log_orders = None
book_freq = 0

## GAN input
real_ohlc = args.real_ohlc
gan_model_file= args.gan_model_file
print("Real ohlc in use:", real_ohlc)
print("GAN model in use:", gan_model_file)
## GAN input

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

r_bar = 1337186
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11

# Oracle
symbols = {symbol: {'r_bar': r_bar,
                    'kappa': 1.67e-16,
                    'sigma_s': 0,
                    'fund_vol': args.fund_vol,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# 1) Exchange Agent

#  How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 25000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=book_freq,
                             real_ohlc=real_ohlc,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1


# 7a) Trader agents

num_trader_agents = 1

trader_agents = [TraderAgent(
                    id=j,
                    name="TRADER_AGENT_{}".format(j),
                    type="TraderAgent",
                    symbol=symbol,
                    starting_cash=starting_cash,
                    log_orders=log_orders,
                    random_state=np.random.RandomState(
                        seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
                    ),
                    ) for j in range(agent_count, agent_count + num_trader_agents)]

agents.extend(trader_agents)

agent_count += num_trader_agents
agent_types.extend("TraderAgent")

# 7b) Gan Agents
num_gan_agents = num_trader_agents

agents.extend(
    [
        GanAgent(
            id= j,
            name="GAN_AGENT_{}".format(j),
            type="GanAgent",
            symbol=symbol,
            starting_cash=starting_cash,
            min_size=1,
            max_size=10,
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            wake_up_freq="1s",
            log_orders=log_orders,
            generator_path=gan_model_file,
            random_state=np.random.RandomState(
                seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")
            ),
            verbose=True,
            trader_agent= trader_agents[j - agent_count]
        )
        for j in range(agent_count, agent_count + num_gan_agents)
    ]
)
gan_agent_indexes = list(range(agent_count, agent_count + num_gan_agents))
agent_count += num_gan_agents
agent_types.extend("GanAgent")


# 6) Execution Agent

trade = True if args.execution_agents else False

#### Participation of Volume Agent parameters

pov_agent_start_time = mkt_open + pd.to_timedelta('01:30:00')
pov_agent_end_time = mkt_open + pd.to_timedelta('02:00:00')
pov_proportion_of_volume = args.execution_pov
pov_quantity = 12e5 #8   #try to use the same ratio of before, with 120 the 1min volume
pov_frequency = '1min'
pov_direction = "BUY"

pov_agent = POVExecutionAgent(id=agent_count,
                              name='POV_EXECUTION_AGENT',
                              type='ExecutionAgent',
                              symbol=symbol,
                              starting_cash=starting_cash,
                              start_time=pov_agent_start_time,
                              end_time=pov_agent_end_time,
                              freq=pov_frequency,
                              lookback_period=pov_frequency,
                              pov=pov_proportion_of_volume,
                              direction=pov_direction,
                              quantity=pov_quantity,
                              trade=trade,
                              log_orders=True,  # needed for plots so conflicts with others
                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))

execution_agents = [pov_agent]
agents.extend(execution_agents)
agent_types.extend("ExecutionAgent")
agent_count += 1


########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("GAN Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50  # 50 nanoseconds

# LATENCY

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))
pairwise = (agent_count, agent_count)

# All agents sit on line from Seattle to NYC
nyc_to_seattle_meters = 3866660
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                        random_state=latency_rstate)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args
                             )
# KERNEL
kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatencyModel=latency_model,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=args.log_dir)


simulation_end_time = dt.datetime.now()

### SAVE THE OHCL
for i_gan in gan_agent_indexes:
    agents[i_gan].save_last_ohlc("log/" + log_dir + "/")

print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
