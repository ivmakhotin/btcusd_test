from pathlib import Path

from src.train import evaluate_model, train_model
from src.predict import predict
from src.utils import get_config

if __name__ == '__main__':
    config = get_config()

    evaluate_model(Path("btcusd-h-30")  / config["data_filename"], Path("btcusd-h-30") / config["result_filename"])
    # train_model("btcusd-h-30/data.h5", Path("btcusd-h-30") / config["result_filename"])    
    # predict(Path("btcusd-h-30")  / config["data_filename"], Path(config["model_path"]), Path(config["forecast_result_path"]))
