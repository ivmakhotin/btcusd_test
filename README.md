# btcusd_test

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py fitting --train_folder_path btcusd-h-30/

python main.py forecasting --data_folder_path btcusd-h-30/data.h5 