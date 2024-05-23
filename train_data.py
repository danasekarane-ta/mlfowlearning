import argparse
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
from const import LINEAR_MODEL, DECESSION_TREE


def stratified_shuffle_split(housing):
    train_set, test_set = train_test_split(housing,
                                           test_size=0.2,
                                           random_state=42)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    print("train set", train_set, test_set, strat_train_set, strat_test_set)
    return train_set, test_set, strat_train_set, strat_test_set


def train_regression_data(model, X, y):
    if model == LINEAR_MODEL:
        reg = LinearRegression()
    elif model == DECESSION_TREE:
        reg = DecisionTreeRegressor()
    reg.fit(X, y)
    pred = reg.predict(X)
    return pred


def train_regression_data1(model, X, y):
    if model == LINEAR_MODEL:
        reg = LinearRegression()
    elif model == DECESSION_TREE:
        reg = DecisionTreeRegressor()
    reg.fit(X, y)
    pred = reg.predict(X)
    return pred


def cross_validation(model, X, y):
    forest_reg = RandomForestRegressor(random_state=42)
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10],
         "max_features": [2, 3, 4]},
    ]
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    if model == 'RandomizedSearchCV':
        search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
    elif model == 'GridSearchCV':
        search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
    search.fit(X, y)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, X.columns), reverse=True)
    final_model = search.best_estimator_
    return final_model