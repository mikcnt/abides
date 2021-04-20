
# CREATE THE REALISM METRICS
mkdir -p evaluation/
mkdir -p evaluation/test_telemetry_gan
cd realism && python3 -u asset_returns_stylized_facts.py -s ../log/${out_dir} -o ../evaluation/test_telemetry_gan/ && cd ..
exit

# TEST THE GAN USING CONFIG: gan_pov and APPL AND MODEL 200.pth.  SAVING ALL TO log/test_gan_liquidity_telemetry 
out_dir=test_gan_liquidity_telemetry
python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s 300 -l ${out_dir} --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" 

# IF we want to test the shock with a POV agent

# NOTE: POV 1 %
#pov=0.01
#out_dir=test_gan_liquidity_telemetry_001
#python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s 300 -l ${out_dir} -e -p ${pov} --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" 

# NOTE: POV 5 %
#pov=0.05
#out_dir=test_gan_liquidity_telemetry_005
#python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s 300 -l ${out_dir} -e -p ${pov} --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" 

# NOTE: POV 10 %
#pov=0.1
#out_dir=test_gan_liquidity_telemetry_01
#python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s 300 -l ${out_dir} -e -p ${pov} --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" 

# NOTE: POV 50 %
#pov=0.5
#out_dir=test_gan_liquidity_telemetry_05
#python3 -u abides.py -c gan_pov -t ABM -d 20210201 -s 300 -l ${out_dir} -e -p ${pov} --gan-model-file "data/200.pth" --real-ohlc "data/ohlc_1min_apple_20210201.csv" 


wait
# CREATE THE LIQUIDITY TELEMETRY
cd util/plotting && python3 -u liquidity_telemetry.py ../../log/${out_dir}/EXCHANGE_AGENT.bz2 ../../log/${out_dir}/ORDERBOOK_ABM_FULL.bz2 -o ${out_dir}.png -c configs/plot_09.30_11.30.json && cd ../../
wait

# CREATE THE REALISM METRICS
mkdir -p evaluation/
mkdir -p evaluation/test_telemetry_gan
cd /realism && python3 -u asset_returns_stylized_facts.py -s ../log/${out_dir} -o ../evaluation/test_telemetry_gan/ && cd ..
