import logging
import logging.config

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger("mlExample")


def calculate_score(y, pred):
    logger.info("Calculating the score")
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    if not mae:
        logger.warn("No Mae calculated")
    return rmse, mae
