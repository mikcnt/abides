# ABIDES for dummies
In the next sections, we are going to consider a dummy simulation on the RMSC03 config for a total time of two hours. We are going to use the default configuration RMSC03 for this purpose, and in particular its derivation to achieve only two hours of trading simulation.
## Run simulation
To run the simulation, run the following:

```shell
python -u abides.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_two_hour
```

## Basic plotting
To produce the basic plottings, run the following:

```shell
cd util/plotting && python -u liquidity_telemetry.py ../../log/rmsc03_two_hour/EXCHANGE_AGENT.bz2 ../../log/rmsc03_two_hour/ORDERBOOK_ABM_FULL.bz2 -o rmsc03_two_hour.png -c configs/plot_09.30_11.30.json && cd ../../
```

To assess the results check file `./abides/util/plotting/rmsc03_two_hour.png`.

## Evaluation
To produce the evaluation metrics on the simulation, create a new directory in `./abides/evaluation` called `rmsc03_two_hour` and run the following:

```shell
cd realism && python -u asset_returns_stylized_facts.py -s ../log/rmsc03_two_hour -o ../evaluation/rmsc03_two_hour && cd ..
```

To assess the results check directory `./abides/evaluation/rmsc03_two_hour`.

## Parameters tweaking 🔧
Let's suppose we want to run the simulation on a complete trading day (09:30 to 16:30). In this section we are going to explain the parameters we need to tweak in order to do so.

### RMSC03 config file
Open the script `./abides/config/rmsc03.py` and change the parser argument `--end_time` and set its default to `16:30:00`.

### Telemetry config file
Open the directory `./abides/util/plotting/configs`, create a new config called `plot_09.30_16.30.json` and paste the following into the json dictionary:

```json
{
  "xmin": "09:32:00",
  "xmax": "16:30:00",
  "linewidth": 0.7,
  "no_bids_color": "blue",
  "no_asks_color": "red",
  "transacted_volume_binwidth": 30,
  "shade_start_time": "01:00:00",
  "shade_end_time": "01:30:00"
}
```

### New commands
**Simulation**:

```shell
python -u abides.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_all_day
```

**Basic plotting**:

```shell
cd util/plotting && python -u liquidity_telemetry.py ../../log/rmsc03_all_day/EXCHANGE_AGENT.bz2 ../../log/rmsc03_all_day/ORDERBOOK_ABM_FULL.bz2 -o rmsc03_all_day.png -c configs/plot_09.30_16.30.json && cd ../../
```

**Evaluation**:

```shell
cd realism && python -u asset_returns_stylized_facts.py -s ../log/rmsc03_all_day -o ../evaluation/rmsc03_all_day && cd ..
```

# Known issues ⚠️
<details>
  <summary>ValueError: low >= high</summary>

  This error is due to some kind of overflow in the ValueAgent, when generating a random integer with NumPy. Open file `./abides/agent/ValueAgent.py` and replace `self.depth_spread*spread` with `np.ceil(self.depth_spread*spread)` (line 202):
  ```python
  adjust_int = np.random.randint(0, self.depth_spread*spread)
  ```
</details>
