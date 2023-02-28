from pathlib import Path
from typing import Tuple

import numpy as np
import h5py
from loguru import logger

from src.utils import window_lin_reg


def get_features(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Loads file with order book data and trades data and calculating fearures

    Args:
        data_path: path to a file with order book data and trades data

    Returns:
        Numpy arrary with timestamps (n_samples,)
        Numpy arrary with calculated features (n_samples, n_features)
    """

    data_file = h5py.File(str(data_path), "r")

    logger.info("Calculating features..")
    ob_ts = np.array(data_file["OB/TS"])
    trades_ts = np.array(data_file["Trades/TS"])

    trade_idxs = (np.searchsorted(trades_ts, ob_ts, side="right") - 1).astype(int)
    most_recent_amount = np.array(data_file["Trades/Amount"])[trade_idxs]
    most_recent_price = np.array(data_file["Trades/Price"])[trade_idxs]
    mid_price = (
        np.array(data_file["OB/Ask"]).min(axis=1)
        + np.array(data_file["OB/Bid"]).max(axis=1)
    ) / 2

    mid_price_weighted = (
        np.array(data_file["OB/Ask"])[:, 0] * np.array(data_file["OB/AskV"])[:, 0]
        + np.array(data_file["OB/Bid"])[:, 0] * np.array(data_file["OB/BidV"])[:, 0]
    ) / (np.array(data_file["OB/AskV"])[:, 0] + np.array(data_file["OB/BidV"])[:, 0])

    window_sizes = [5, 10, 20, 30]
    relative_preds_by_trends = []
    slopes_features = []
    for w in window_sizes:
        slope, intercept = window_lin_reg(ob_ts, mid_price, w)
        first_w_preds = np.array([np.mean(mid_price)] * (w - 1))
        with np.errstate(invalid="ignore"):
            preds = (ob_ts[(w - 1) :] + 30000) * slope + intercept
        relative_pred_by_trends = (
            np.hstack([first_w_preds, preds]) - mid_price
        ) / mid_price
        relative_preds_by_trends.append(relative_pred_by_trends)
        slopes_features.append(np.hstack([np.zeros(w - 1), slope]))

    volume_spread = (data_file["OB/AskV"][:, 0] - data_file["OB/BidV"][:, 0]) / (
        data_file["OB/AskV"][:, 0] + data_file["OB/BidV"][:, 0]
    )
    normalized_price_spreads = (
        data_file["OB/Ask"][:, :10] - data_file["OB/Bid"][:, :10]
    ) / mid_price.reshape(-1, 1)
    volume_diffs = data_file["OB/AskV"][:, :7] - data_file["OB/BidV"][:, :7]
    volume_ratios = data_file["OB/AskV"][:, :7] / data_file["OB/BidV"][:, :7]

    askV = np.array(data_file["OB/AskV"]).sum(axis=1)
    bidV = np.array(data_file["OB/BidV"]).sum(axis=1)
    askP = np.array(data_file["OB/Ask"]).mean(axis=1)
    bidP = np.array(data_file["OB/Bid"]).mean(axis=1)

    weghted_average_spread = (askV * askP - bidV * bidP) / mid_price
    features = (
        [
            askV,
            bidV,
            volume_spread,
            mid_price_weighted,
            mid_price,
            most_recent_amount,
            most_recent_price,
            most_recent_amount * most_recent_price,
            most_recent_amount / most_recent_price,
            mid_price / most_recent_price,
            normalized_price_spreads,
            weghted_average_spread,
            volume_diffs,
            volume_ratios,
        ]
        + [
            relative_pred_by_trends
            for relative_pred_by_trends in relative_preds_by_trends
        ]
        + [slope for slope in slopes_features]
    )

    features = [f if len(f.shape) > 1 else f.reshape(-1, 1) for f in features]
    return ob_ts, np.hstack(features)


def get_target(return_data_path: Path) -> np.ndarray:
    """Loads file with 30 sec relative return data ((mid(t + 30s) - mid(t)) / mid(t))
    and returns np.ndarray

    Args:
        return_data_path: path to a file with 30 sec relative return data

    Returns:
        Numpy arrary with 30 sec relative return data (n_samples,)
    """

    return_data_file = h5py.File(str(return_data_path), "r")
    return np.array(return_data_file["Return/Res"])
