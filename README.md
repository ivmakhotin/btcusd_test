# BTC/USD forecast ML model

<!-- ABOUT THE PROJECT -->
## About The Project

A gradient boosting model that predicts relative return ((mid(t + 30s) - mid(t)) / mid(t)) using oreder book and trades data known before t. 

<!-- USAGE EXAMPLES -->
## Usage

Firstly, you need to set up virtual environment and install libraies

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Script works in two modes:

1. Fitting mode

In fitting mode you should specify `train_folder_path` which is path to a folder which contains order book data and trades data `data.h5` and file with relative return `result.h5`. This mode allows to train and save model.

```
python main.py fitting --train_folder_path btcusd-h-30/
```

2. Forecasting mode

In forecasting mode you should specify `data_path` which represents a path to file with data you want to use for forecast. This mode allows to use trained model.

```
python main.py forecasting --data_path btcusd-h-30/data.h5 
```
## Note:

- The model uses GPU by default. You can switch it to CPU in src/config.yaml

- Both fitting and forecasting steps requires ~8GB RAM and each run for a several minutes
