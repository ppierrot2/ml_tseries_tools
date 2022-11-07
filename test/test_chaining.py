import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ml_tseries.cross_validation import PurgedKFold
from ml_tseries.chaining import RegresssorChainTS, ClassifierChainTS


def test_chained_regressor():

    prediction_steps = 3
    n_targets = 2
    date_range = pd.date_range('1/1/2012', '1/1/2019')
    X, Y = make_regression(n_samples=len(date_range), n_features=4, n_informative=2,
                           n_targets=n_targets, random_state=0, shuffle=False)
    X = pd.DataFrame(X, index=date_range,
                     columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    Y = pd.DataFrame(Y, index=date_range)
    base_estimator = RandomForestRegressor()

    chain_estimator = RegresssorChainTS(base_estimator, cv_gen=PurgedKFold(), prediction_steps=prediction_steps)
    chain_estimator.fit(X, Y)

    assert hasattr(chain_estimator, 'estimators_')
    assert len(chain_estimator.estimators_) == prediction_steps
    assert all([len(est) == n_targets for est in chain_estimator.estimators_])

    preds = chain_estimator.predict(X)

    assert preds.shape == (len(X), prediction_steps, n_targets)

