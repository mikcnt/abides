

for seed in $(seq 105 108); do
  python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s ${seed} -l gan_pov_demo_no_${seed}_20210201 --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" &
  for pov in  0.01 0.05; do
      python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s ${seed} -l gan_pov_demo_yes_${seed}_pov_${pov}_20210201 -e -p ${pov} --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" &
  done
done
wait


cd realism && python -u impact_multiday_pov.py plot_configs/plot_configs/multiday/gan_pov_demo_multiday.json -n 16 && cd ..
