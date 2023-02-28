from pathlib import Path

import h5py
import catboost as cb

from src.load_data import get_features


def load_model(model_path: Path) -> cb.CatBoostRegressor:
    """Reads from disk and returns trained CatBoostRegressor"""

    model = cb.CatBoostRegressor()
    model.load_model(model_path)
    return model


def predict(data_path: Path, model_path: Path, forecast_path: Path) -> None:
    """Calculates predictions with trained model and saves it as .h5 file

    Args:
        data_path: path to a file with order book data and trades data
        model_path: path to a file with trained model
        forecast_path: path to save result as .h5 file
    """

    ts, X = get_features(data_path)
    model = load_model(model_path)
    predict = model.predict(X)
    forecast_file = h5py.File(str(forecast_path), "w")
    group = forecast_file.create_group("Return")
    group.create_dataset(name="TS", shape=ts.shape, dtype=ts.dtype, data=ts)
    group.create_dataset(
        name="Res", shape=predict.shape, dtype=predict.dtype, data=predict
    )
