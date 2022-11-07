import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from ml_tseries.calibration import bayesian_tuning
from sklearn.preprocessing import binarize
from sklearn.metrics import f1_score
from hyperopt import hp, tpe
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ml_tseries.cross_validation import (PurgedKFold,
                                         PurgedCombinatorial)


def test_tuning_class_rf():

    date_range = pd.date_range('1/1/2012', '1/1/2019')
    X, y = make_regression(n_samples=len(date_range),
                           n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    X = pd.DataFrame(X, index=date_range)
    sample_weight = y.copy()
    sample_weight = pd.Series(sample_weight, index=date_range)
    y = binarize(y.reshape(-1, 1)).ravel()
    y = pd.Series(y, index=date_range)

    cv_gen = PurgedKFold(n_splits=5)
    param_grid_rfts = {
        'max_depth': hp.choice('max_depth', [int(x) for x in np.linspace(5, 25, 10)]),
        'max_features': hp.choice('max_features', [int(x) for x in np.linspace(1, 4, 2)]),
    }
    static_params = {'n_estimators': 500}
    params = {'max_depth': 10, 'max_features': 4}
    trials = None

    def f1(y_true, y_pred, sample_weight=None):
        return 'f1', f1_score(y_true, y_pred, sample_weight), True

    hyper_opt = bayesian_tuning(X, y,
                                model=RandomForestClassifier(n_jobs=1),
                                param_grid=param_grid_rfts,
                                metric_fun=f1,
                                static_params=static_params,
                                nb_evals=10,
                                optimizer=tpe.suggest,
                                hyperparam_selection=True,
                                cv_gen=cv_gen,
                                points_to_evaluate=[params],
                                sample_weight=sample_weight,
                                trials=trials,
                                )

    assert type(hyper_opt) == list
    assert all([('loss' in dic.keys()) for dic in hyper_opt])
    assert all([('hyperparameters' in dic.keys()) for dic in hyper_opt])
    assert all([dic['hyperparameters'].keys() == {**param_grid_rfts, **static_params}.keys()
                for dic in hyper_opt])

    hyper_opt = bayesian_tuning(X, y,
                                model=RandomForestClassifier(),
                                param_grid=param_grid_rfts,
                                metric_fun=f1_score,
                                static_params=static_params,
                                nb_evals=10,
                                optimizer=tpe.suggest,
                                hyperparam_selection=False,
                                refit=True,
                                cv_gen=cv_gen,
                                sample_weight=sample_weight,
                                )

    assert type(hyper_opt) == list
    assert len(hyper_opt) == 10
    assert all([('loss' in dic.keys()) for dic in hyper_opt])
    assert all([('hyperparameters' in dic.keys()) for dic in hyper_opt])
    assert all([dic['hyperparameters'].keys() == {**param_grid_rfts, **static_params}.keys()
                for dic in hyper_opt])

