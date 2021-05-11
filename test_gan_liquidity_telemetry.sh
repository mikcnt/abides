
# TEST THE GAN USING CONFIG: gan_pov and APPL AND MODEL 200.pth.  SAVING ALL TO log/test_gan_liquidity_telemetry 


MODEL_PATH="/home/coletta/Scrivania/JP-MORGAN/GAN/test_wgan_1trade/GANs-for-trading/abides/097.pth"
INITIAL_ORDERBOOK="../preprocessed/TSLA_2019-05-02_2019-05-05.csv"
OHLC_INIT = "...."
out_dir=test_gan_liquidity_telemetry
python3 -u abides.py -c gan_pov -t ABM -d 20190502 -s 300 -l ${out_dir} --gan-model-file ${MODEL_PATH} --real-ohlc ${INITIAL_ORDERBOOK} 

wait

# CREATE THE LIQUIDITY TELEMETRY
cd util/plotting && python3 -u liquidity_telemetry.py ../../log/${out_dir}/EXCHANGE_AGENT.bz2 ../../log/${out_dir}/ORDERBOOK_ABM_FULL.bz2 -o ${out_dir}.png -c configs/plot_09.30_11.30.json && cd ../../
wait
exit 

# CREATE THE REALISM METRICS
mkdir -p evaluation/
mkdir -p evaluation/test_telemetry_gan
cd /realism && python3 -u asset_returns_stylized_facts.py -s ../log/${out_dir} -o ../evaluation/test_telemetry_gan/ && cd ..
