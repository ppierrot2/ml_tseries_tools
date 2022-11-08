import numpy as np
import pandas as pd
import datetime as dt
from sklearn.datasets import make_regression
from sklearn.preprocessing import binarize
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from ml_tseries.cross_validation import (combinatorial_cv,
                                         kfold_cv,
                                         tseries_cv,
                                         PurgedKFold,
                                         PurgedWalkForward,
                                         PurgedCombinatorial,
                                         )


def test_cross_validation_kfold():

    X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
                           random_state=0, shuffle=False)

    sample_weight = y.copy()
    model = RandomForestRegressor()
    cv_gen = KFold(n_splits=5)

    score_kfold = kfold_cv(model, X, y, mean_squared_error,
                           cv_gen=cv_gen,
                           sample_weight=None, fit_params=None,
                           verbose=False, n_jobs=-1)

    sample_weight = y.copy()
    y = binarize(y.reshape(-1, 1))
    model = RandomForestClassifier()
    score_strat_kfold = kfold_cv(model, X, y, f1_score,
                                 cv_gen=StratifiedKFold(),
                                 sample_weight=sample_weight,
                                 fit_params=None,
                                 verbose=False, n_jobs=1)

    assert type(score_kfold) == list
    assert type(score_strat_kfold) == list
    assert len(score_kfold) == cv_gen.n_splits
    assert all([type(e) == np.float64 for e in score_kfold])
    assert all([type(e) == np.float64 for e in score_strat_kfold])
    assert all([e >= 0. for e in score_kfold])
    assert all([e >= 0. for e in score_strat_kfold])


def test_cross_validation_tseries():

    date_range = pd.date_range('1/1/2012', '1/1/2019')
    X, y = make_regression(n_samples=len(date_range), n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    X = pd.DataFrame(X, index=date_range)
    sample_weight = y.copy()
    sample_weight = pd.Series(sample_weight, index=date_range)
    y = binarize(y.reshape(-1, 1)).ravel()
    y = pd.Series(y, index=date_range)

    model = RandomForestRegressor()

    score_ts = tseries_cv(model, X,
                          y,
                          metric=mean_squared_error,
                          y_transform=None,
                          fit_params=None,
                          sample_weight=sample_weight,
                          cv_gen=PurgedKFold(),
                          n_jobs=1)

    score_ts_wf = tseries_cv(model, X,
                             y,
                             metric=mean_squared_error,
                             y_transform=None,
                             fit_params=None,
                             sample_weight=sample_weight,
                             cv_gen=PurgedWalkForward(),
                             n_jobs=1)

    model = RandomForestClassifier()
    score_ts_seas = tseries_cv(model, X,
                               y,
                               metric=f1_score,
                               y_transform=None,
                               fit_params=None,
                               sample_weight=sample_weight,
                               cv_gen=PurgedKFold(),
                               n_jobs=1)

    assert type(score_ts) == list
    assert type(score_ts_wf) == list
    assert type(score_ts_seas) == list
    assert len(score_ts) == PurgedKFold().n_splits
    assert len(score_ts_wf) == PurgedWalkForward().n_splits
    assert all([type(e) == np.float64 for e in score_ts])
    assert all([type(e) == np.float64 for e in score_ts_seas])
    assert all([type(e) == np.float64 for e in score_ts_wf])
    assert all([e >= 0. for e in score_ts])
    assert all([e >= 0. for e in score_ts_seas])
    assert all([e >= 0. for e in score_ts_wf])


def test_cross_validation_combinatorial():

    date_range = pd.date_range('1/1/2012', '1/1/2019')
    X, y = make_regression(n_samples=len(date_range), n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    X = pd.DataFrame(X, index=date_range)
    sample_weight = y.copy()
    sample_weight = pd.Series(sample_weight, index=date_range)
    y = binarize(y.reshape(-1, 1)).ravel()
    y = pd.Series(y, index=date_range)

    model = RandomForestRegressor()
    cv_gen = PurgedCombinatorial(embargo=dt.timedelta(days=1),
                                 overlap=dt.timedelta(days=2),
                                 n_splits=5)

    score_combi = combinatorial_cv(model, X, y,
                                   metric=mean_squared_error,
                                   y_transform=None,
                                   fit_params=None,
                                   sample_weight=sample_weight,
                                   print_info=False,
                                   _importances=False,
                                   n_jobs=-1)

    assert type(score_combi) == np.ndarray
    assert score_combi.shape[0] == cv_gen.n_splits

