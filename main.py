from pathlib import Path
import argparse

from src.train import evaluate_model, train_model
from src.predict import predict
from src.utils import get_config


def main():
    parser = argparse.ArgumentParser(description="btcusd forecast")
    parser.add_argument("mode", choices=["fitting", "forecasting"], type=str)
    parser.add_argument("--train_folder_path", type=str, help="Specify a path to a folder with files data.h5 and result.h5. Required in fitting mode.")
    parser.add_argument("--data_folder_path", help="Specify a path to a file with data to forecast. Required in forecasting mode.")
    args = parser.parse_args()

    config = get_config()

    if args.mode == "fitting":
        train_model(Path(args.train_folder_path) / config["data_filename"], Path(args.train_folder_path) / config["result_filename"])
    elif args.mode == "forecasting":
        predict(Path(args.data_folder_path), Path(config["model_path"]), Path(config["forecast_result_path"]))


if __name__ == "__main__":
    main()

    # evaluate_model(Path("btcusd-h-30")  / config["data_filename"], Path("btcusd-h-30") / config["result_filename"])
    # train_model("btcusd-h-30/data.h5", Path("btcusd-h-30") / config["result_filename"])    
    # predict(Path("btcusd-h-30")  / config["data_filename"], Path(config["model_path"]), Path(config["forecast_result_path"]))
