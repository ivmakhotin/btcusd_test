from pathlib import Path

import h5py
import catboost as cb
from loguru import logger

from src.load_data import get_features


def load_model(model_path: Path) -> cb.CatBoostRegressor:
    """Reads a model from the disk and returns trained CatBoostRegressor"""

    logger.info("Loading model...")
    model = cb.CatBoostRegressor()
    model.load_model(model_path)
    return model


def predict(data_path: Path, model_path: Path, forecast_path: Path) -> None:
    """Calculates predictions with a trained model and saves it as .h5 file

    Args:
        data_path: a path to the file with order book and trades data
        model_path: a path to the file with the trained model
        forecast_path: a path to save the result as .h5 file
    """

    ts, X = get_features(data_path)
    model = load_model(model_path)

    logger.info("Making predicitons...")
    predict = model.predict(X)

    logger.info("Saving predicitons...")
    forecast_file = h5py.File(str(forecast_path), "w")
    group = forecast_file.create_group("Return")
    group.create_dataset(name="TS", shape=ts.shape, dtype=ts.dtype, data=ts)
    group.create_dataset(
        name="Res", shape=predict.shape, dtype=predict.dtype, data=predict
    )
