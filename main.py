from pathlib import Path
import argparse

from src.train import train_model
from src.predict import predict
from src.utils import get_config


def main():
    """Console application logic"""

    parser = argparse.ArgumentParser(description="BTC/USD forecast")
    parser.add_argument("mode", choices=["fitting", "forecasting"], type=str)
    parser.add_argument(
        "--train_folder_path",
        type=str,
        help="Specify the path to the folder with files data.h5 and result.h5. Required in fitting mode.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Specify the path to the .h5 file with the data to forecast. Required in forecasting mode.",
    )
    args = parser.parse_args()

    config = get_config()

    if args.mode == "fitting":
        train_model(
            Path(args.train_folder_path) / config["data_filename"],
            Path(args.train_folder_path) / config["result_filename"],
        )
    elif args.mode == "forecasting":
        predict(
            Path(args.data_path),
            Path(config["model_path"]),
            Path(config["forecast_result_path"]),
        )


if __name__ == "__main__":
    main()
