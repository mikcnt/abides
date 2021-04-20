
# MULTIPLE RUN

for seed in $(seq 110 112); do
  python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s ${seed} -l gan_pov_demo_no_${seed}_20210201 --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" &
  for pov in  0.01 0.05 0.1 0.5; do 
      python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s ${seed} -l gan_pov_demo_yes_${seed}_pov_${pov}_20210201 -e -p ${pov} --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" &
  done
done
wait


# PLOT TELEMETRY
for seed in $(seq 110 112); do
  cd util/plotting && python -u liquidity_telemetry.py ../../log/gan_pov_demo_no_${seed}_20210201/EXCHANGE_AGENT.bz2 ../../log/gan_pov_demo_no_${seed}_20210201/ORDERBOOK_ABM_FULL.bz2 -o gan_pov_demo_no_${seed}_20210201.png -c configs/plot_09.30_11.30.json && cd ../../ 
  for pov in  0.01 0.5 0.1 0.05; do 
      cd util/plotting && python -u liquidity_telemetry.py ../../log/gan_pov_demo_yes_${seed}_pov_${pov}_20210201/EXCHANGE_AGENT.bz2 ../../log/gan_pov_demo_yes_${seed}_pov_${pov}_20210201/ORDERBOOK_ABM_FULL.bz2 -o gan_pov_demo_yes_${seed}_pov_${pov}_20210201.png -c configs/plot_09.30_11.30.json && cd ../../ 
    done
done
wait

# PLOT STYLED
mkdir evaluation/
for seed in $(seq 110 112); do
  mkdir evaluation/gan_pov_demo_no_${seed}
  python3 -u asset_returns_stylized_facts.py -s ../log/gan_pov_demo_no_${seed}_20210201 -o ../evaluation/gan_pov_demo_no_${seed}/
  for pov in  0.01 0.5 0.1 0.05; do 
      mkdir evaluation/gan_pov_demo_no_${seed}_pov_${pov}
        python3 -u asset_returns_stylized_facts.py -s ../log/gan_pov_demo_yes_${seed}_pov_${pov}_20210201 -o ../evaluation/gan_pov_demo_no_${seed}_pov_${pov}/
    done
done
wait


# AGGREGATE REALISM / REACT PLOTS
# REMOVE CACHE TO BE SURE; AND PREVIOUS LOG DATA
# By deault this uses all the data inside /log/ not just the most recent ones!!!
# So we have to remove all the OLD data from /log/ before using it.
# Also, it try to use the cached results --> /realism/cache/  to be sure remove also them (the OLD ones)!
# maybe not remove, but backup in another folder!
cd realism && python -u impact_multiday_pov.py plot_configs/plot_configs/multiday/gan_pov_demo_multiday.json -n 4 && cd ..
