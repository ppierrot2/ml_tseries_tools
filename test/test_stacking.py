import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import binarize
from sklearn.metrics import f1_score, mean_squared_error
from hyperopt import hp, tpe
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from ml_tseries.calibration import BayesianSearchCV
from ml_tseries.stacking import MetaClassifierTS, MetaRegressorTS, MetaRankingClassifierTS
from ml_tseries.cross_validation import PurgedKFold


def test_meta_classifier():

    date_range = pd.date_range('1/1/2012', '1/1/2019')
    X, y = make_classification(n_samples=len(date_range), n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    X = pd.DataFrame(X, index=date_range, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    sample_weight = y.copy()
    sample_weight = pd.Series(sample_weight, index=date_range)
    y = binarize(y.reshape(-1, 1)).ravel()
    y = pd.Series(y, index=date_range)
    base_estimators = [RandomForestClassifier(),
                       RandomForestClassifier()]
    final_estimator = RandomForestClassifier()

    param_grid = {
        'max_depth': hp.choice('max_depth', [int(x) for x in np.linspace(1, 20, 20)]),
        'min_samples_split': hp.uniform('min_samples_split', 0.01, .99),
        'min_samples_leaf': hp.choice('min_samples_leaf', [int(x) for x in np.linspace(1, 20, 20)]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None])
    }
    cv_gen = PurgedKFold()
    final_estimator = BayesianSearchCV(final_estimator, param_grid,
                                       scoring=f1_score, cv=cv_gen,
                                       optimizer=tpe.suggest, n_iter=10)

    meta_classifier = MetaClassifierTS(estimators=base_estimators,
                                       final_estimator=final_estimator,
                                       cv_gen=PurgedKFold(),
                                       refit=True, n_jobs=1)
    meta_classifier.fit(X, y, sample_weight=sample_weight)
    probs = meta_classifier.predict_proba(X)
    preds = meta_classifier.predict(X)

    assert type(preds) == np.ndarray
    assert type(probs) == np.ndarray
    assert probs.shape == (X.shape[0], 2)
    assert preds.shape == (X.shape[0],)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    assert np.unique(preds).tolist() in [[0, 1], [0], [1]]


def test_metaranking_classifier():

    date_range = pd.date_range('1/1/2012', '1/1/2019')
    X, y = make_classification(n_samples=len(date_range), n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    X = pd.DataFrame(X, index=date_range, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    meta_features = X[['feature_1', 'feature_2']]
    sample_weight = y.copy()
    sample_weight = pd.Series(sample_weight, index=date_range)
    y = binarize(y.reshape(-1, 1)).ravel()
    y = pd.Series(y, index=date_range)
    base_estimators = [RandomForestClassifier(),
                       RandomForestClassifier()]
    final_estimator = RandomForestClassifier()

    param_grid = {
        'max_depth': hp.choice('max_depth', [int(x) for x in np.linspace(1, 20, 20)]),
        'min_samples_split': hp.uniform('min_samples_split', 0.01, .99),
        'min_samples_leaf': hp.choice('min_samples_leaf', [int(x) for x in np.linspace(1, 20, 20)]),
        'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None])
    }
    cv_gen = KFold()
    base_estimators = BayesianSearchCV(final_estimator, param_grid,
                                       scoring=f1_score, cv=cv_gen,
                                       optimizer=tpe.suggest, n_iter=10)

    meta_classifier = MetaRankingClassifierTS(estimators=base_estimators,
                                              final_estimator=final_estimator,
                                              cv_gen=PurgedKFold(),
                                              refit=True, n_jobs=1)
    meta_classifier.fit(X, y, sample_weight=sample_weight, meta_features=meta_features)
    probs = meta_classifier.predict_proba(X, meta_features=meta_features)
    preds = meta_classifier.predict(X, meta_features=meta_features)

    assert type(preds) == np.ndarray
    assert type(probs) == np.ndarray
    assert probs.shape == (X.shape[0], 2)
    assert preds.shape == (X.shape[0],)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    assert np.unique(preds).tolist() in [[0, 1], [0], [1]]


def test_meta_regressor():

    date_range = pd.date_range('1/1/2012', '1/1/2019')
    X, y = make_regression(n_samples=len(date_range), n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    X = pd.DataFrame(X, index=date_range,
                     columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    sample_weight = y.copy()
    sample_weight = pd.Series(sample_weight, index=date_range)
    y = pd.Series(y, index=date_range)
    base_estimators = [RandomForestRegressor(),
                       RandomForestRegressor()]

    param_grid = {
        'alpha': hp.uniform('alpha', 0., 1.),
        'normalize': hp.choice('normalize', [True, False]),
    }
    final_estimator = Lasso()

    cv_gen = KFold()
    final_estimator = BayesianSearchCV(final_estimator, param_grid,
                                       scoring=mean_squared_error, cv=cv_gen,
                                       optimizer=tpe.suggest, n_iter=10)

    meta_regressor = MetaRegressorTS(estimators=base_estimators,
                                     final_estimator=final_estimator,
                                     cv_gen=PurgedKFold(),
                                     refit=True, n_jobs=1)
    meta_regressor.fit(X, y, sample_weight=sample_weight)
    preds = meta_regressor.predict(X)

    assert type(preds) == np.ndarray
    assert preds.shape == (X.shape[0],)
