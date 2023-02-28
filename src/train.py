from pathlib import Path

import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from loguru import logger

from src.load_data import get_features, get_target
from src.utils import get_config


def get_model() -> cb.CatBoostRegressor:
    """
    """

    config = get_config()

    return cb.CatBoostRegressor(
        depth=6,
        verbose=10,
        random_seed=0x1a1a1a1,
        eval_metric='R2',
        iterations=500,
        learning_rate=0.005,
        bagging_temperature=0.1,
        task_type=config['device']
    )

def evaluate_model(data_path: Path, result_path: Path) -> None:
    """
    """

    _, X = get_features(data_path)
    y = get_target(result_path)

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=False)

    model = get_model()
    train = cb.Pool(X_train, y_train)
    val = cb.Pool(X_val, y_val)
    model.fit(train, eval_set=val)

    r2_train = r2_score(y_train, model.predict(X_train))
    r2_val = r2_score(y_val, model.predict(X_val))
    r2_test = r2_score(y_test, model.predict(X_test))
    
    logger.info(f'r2_train = {r2_train:.5f}, r2_val = {r2_val:.5f}, r2_test = {r2_test:.5f}')

def train_model(data_path: Path, result_path: Path) -> None:
    """
    """

    _, X = get_features(data_path)
    y = get_target(result_path)

    config = get_config()

    model = get_model()
    train = cb.Pool(X, y)
    model.fit(train)

    model_path = Path(config["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))