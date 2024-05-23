import logging
import logging.config

from ingest_data import data_ingestion
from train_data import data_training
from logic_score import logic_score

logger = logging.getLogger('mlExample')


def data_prediction():
    """ Fetch the data and Predict the modelling
    """
    logger.info("Fetching the housing data for data prediction")
    # Fetch the data
    data_ingestion.fetch_housing_data()
    # Load the fetched data
    housing = data_ingestion.load_housing_data()

    # train the data
    train_set, test_set, strat_train_set, strat_test_set = (
        data_training.ShuffleSplitstratified(housing)
    )

    logger.debug("Preprossing the fetched data")
    # preprocess_data
    data_ingestion.preprocess_data(
        housing, strat_train_set, strat_test_set, test_set
    )

    # Data Visualiztion for train set
    housing_train = strat_train_set.copy()
    logger.info("Data Visualization for train set")
    data_ingestion.data_visualization(housing_train)

    # Data Visualiztion for test set
    housing_test = strat_train_set.copy()
    logger.info("Data Visualization for test set")
    data_ingestion.data_visualization(housing_test)

    # Feature Extraction for train set
    housing_train, housing_y_train, housing_X_train = \
        data_ingestion.impute_data(strat_train_set)

    housing_X_train = data_ingestion.extract_features(housing_X_train)
    housing_X_train = data_ingestion.create_dummy_data(
        housing_train, housing_X_train
    )
    # Feature Extraction for test set
    housing_test, housing_y_test, housing_X_test = data_ingestion.impute_data(
        strat_test_set
    )
    housing_X_test = data_ingestion.extract_features(housing_X_test)
    housing_X_test = data_ingestion.create_dummy_data(
        housing_test,
        housing_X_test
    )

    # train model for training set
    housing_predictions_lin = data_training.train_data_regression(
        "linear", housing_X_train, housing_y_train
    )

    lin_rmse_train, lin_mae_train = logic_score.logic_score(
        housing_y_train, housing_predictions_lin
    )

    housing_predictions_reg = data_training.train_data_regression(
        "DecessionTree", housing_X_train, housing_y_train
    )
    tree_rmse_train, tree_mae_train = logic_score.logic_score(
        housing_y_train, housing_predictions_reg
    )
    final_model_train_random = data_training.cross_validation(
        'RandomizedSearchCV',
        housing_X_train,
        housing_y_train
    )
    logger.debug("Best Estimator", final_model_train_random)
    final_model_train_grid = data_training.cross_validation('GridSearchCV',
                                                            housing_X_train,
                                                            housing_y_train)
    logger.info("Best Estimator", final_model_train_grid)

    final_predictions_train = final_model_train_grid.predict(housing_X_train)
    final_rmse_train, final_mae_train = logic_score.logic_score(
        housing_y_train, final_predictions_train
    )
    # scoring for train set
    logger.debug("Scoring for train-data: \n",
                 final_rmse_train, "   ", final_mae_train)

    # test using trained models
    final_predictions_test = final_model_train_grid.predict(housing_X_test)
    final_rmse_test, final_mae_test = logic_score.logic_score(
        housing_y_test, final_predictions_test
    )
    # scoring for test set
    logger.info("Scoring for test-data: \n", final_rmse_test, "   ",
                final_mae_test)


if __name__ == '__main__':
    data_prediction()
