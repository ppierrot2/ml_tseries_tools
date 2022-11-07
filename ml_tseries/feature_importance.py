import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from sklearn.base import BaseEstimator
from sklearn.base import is_classifier
from sklearn.metrics import log_loss
from typing import Callable

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from .cross_validation import (PurgedKFold,
                               PurgedFoldBase,
                               build_sample_weight,
                               _apply_metric)


def mean_decrease_accuracy(model: BaseEstimator,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv_gen: PurgedFoldBase = PurgedKFold(),
                           clustered_subsets: list = None,
                           n_repeat_mda: int = 1,
                           scoring: Callable[[np.ndarray, np.ndarray], float] = log_loss,
                           sample_weight: np.ndarray = None,
                           fit_params: dict = None,
                           random_state: int = 42,
                           n_jobs=1) -> pd.DataFrame:
    """
    MDA Feature Importance for time series data generalized to any ML model using cross validation
    inspired from https://github.com/hudson-and-thames/mlfinlab/ which is based on
    Advances in Financial Machine Learning, Snippet 8.3, page 116-117.

    You can pass non-time serie data and sklearn cv generators but ensure X and y are pandas dataframes

    To Do : solve parallelisation issue, implement seasonal block permutation mode,

    Parameters
    -----------
    model: sklearn.ClassifierMixin
        Any sklearn model or derived from sklearn.base.BaseEstimator.

    X: pd.DataFrame
        Time indexed dataframe, Train set features.

    y: pd.DataFrame, np.array
        Train set labels.

    cv_gen: PurgedFoldBase
        cv generator object of module cross-validation (PurgedKFold, PurgedWalkForward)

    clustered_subsets: list
        Feature clusters for Clustered Feature Importance (CFI). Default None will not apply CFI.
        Structure of the input must be a list of list/s i.e. a list containing the clusters/subsets of feature
        name/s inside a list. E.g- [['I_0','I_1','R_0','R_1'],['N_1','N_2'],['R_3']]

    n_repeat_mda: int
        Number of time to shuffle a (same group of) feature in each fold, default 1.

    scoring: Callable
        scoring method of sklearn or with signature fun(np.ndarray, np.ndarray)->float.

    sample_weight: pd.Series
        sample weight series used for fitting and scoring, should have same index as X and y

    fit_params: dict
        Fit parameters of the model (sample_weight, ..).

    random_state: int
        Random seed for shuffling the features.

    n_jobs: int
        number of workers for parallel computing

    Returns
    -------
    importance_df: pd.DataFrame
        Mean and standard deviation of feature importance.
    """
    # Generating a numpy random state object for the given random_state
    rs_obj = np.random.RandomState(seed=random_state)
    # Clustered feature subsets will be used for model if
    # clustered_subsets exists else will operate on the single column as MDA
    feature_sets = clustered_subsets if clustered_subsets else [[x] for x in X.columns]

    if not isinstance(X.index, pd.DatetimeIndex):
        log.warning('X dataframe was not provided with DatetimeIndex and will be reindexed')
        X.reset_index(drop=True, inplace=True)

    if not isinstance(y.index, pd.DatetimeIndex):
        y.reset_index(drop=True, inplace=True)

    def __compute_fold_mda(train, test):

        sample_weight_train = build_sample_weight(sample_weight.loc[train])
        sample_weight_score = build_sample_weight(sample_weight.loc[test])
        fit_params['sample_weight'] = sample_weight_train

        model.fit(X=X.loc[train].values,
                  y=y.loc[train].values,
                  **fit_params)

        pred = model.predict(X.loc[test].values)

        # Get overall metrics value on out-of-sample fold
        if scoring.__name__ == 'log_loss':
            prob = model.predict_proba(X.loc[test].values)
            fold_metrics_values = -_apply_metric(scoring, y.loc[test].values, prob,
                                                 sample_weight=sample_weight_score)
        else:
            fold_metrics_values = _apply_metric(scoring, y.loc[test].values, pred,
                                                sample_weight=sample_weight_score)

        features_metrics_values = pd.DataFrame(columns=X.columns)
        # Get feature specific metric on out-of-sample fold
        for j in feature_sets:
            X1_ = X.loc[test].copy()
            for n in range(n_repeat_mda):
                for j_i in j:
                    # Permutation of a single column for MDA or through the whole subset
                    rs_obj.shuffle(X1_[j_i].values)

                pred = model.predict(X1_.values)
                if scoring.__name__ == 'log_loss':
                    prob = model.predict_proba(X1_.values)
                    features_metrics_values.loc[n, j] = -_apply_metric(scoring, y.loc[test].values, prob,
                                                                       sample_weight=sample_weight_score,
                                                                       labels=model.classes_)
                else:
                    features_metrics_values.loc[n, j] = _apply_metric(scoring, y.loc[test].values, pred,
                                                                      sample_weight=sample_weight_score)

        return np.ones((n_repeat_mda, X.shape[1])) * fold_metrics_values, features_metrics_values.values

    fit_params = fit_params or {}
    if sample_weight is None:
        sample_weight = pd.Series(np.ones((X.shape[0],),
                                          dtype=np.float64),
                                  index=X.index)

    folds_errs, features_errs = zip(
                                    *Parallel(n_jobs=n_jobs, backend='threading')(
                                         delayed(__compute_fold_mda)(train, test)
                                         for train, test in cv_gen.split(X)
                                         )
                                     )

    features_errs = np.concatenate(features_errs, axis=0).astype(float)
    folds_errs = np.concatenate(folds_errs, axis=0).astype(float)
    decrease_errs = folds_errs - features_errs

    if is_classifier(model):  # normalization
        if scoring.__name__ == 'log_loss':
            decrease_errs_norm = np.divide(decrease_errs, -features_errs)
        else:
            decrease_errs_norm = np.divide(decrease_errs, (1.0 - features_errs))
    else:
        if scoring.__name__ == 'r2_score':
            decrease_errs_norm = decrease_errs
        else:
            decrease_errs_norm = np.divide(decrease_errs, -features_errs)

    mean_decrease_errs = np.mean(decrease_errs_norm, axis=0)
    std_decrease_errs = np.std(decrease_errs_norm.astype(float), axis=0) * decrease_errs_norm.shape[0] ** -.5
    importance_df = pd.DataFrame(index=X.columns)
    importance_df['mean'] = mean_decrease_errs
    importance_df['std'] = std_decrease_errs

    return importance_df


def single_feature_importance(model: BaseEstimator,
                              X: pd.DataFrame,
                              y: pd.Series,
                              cv_gen: PurgedFoldBase = PurgedKFold(),
                              clustered_subsets: list = None,
                              scoring: Callable[[np.ndarray, np.ndarray], float] = log_loss,
                              sample_weight: np.ndarray = None,
                              fit_params: dict = None,
                              n_jobs=1) -> pd.DataFrame:
    """
    Single importance for time series data generalized to any ML model using cross validation
    inspired from https://github.com/hudson-and-thames/mlfinlab/ which is based on
    Advances in Financial Machine Learning,  Snippet 8.4, page 118.

    Parameters
    -----------
    model: sklearn.Classifier
        Any sklearn model or derived from sklearn.base.BaseEstimator.

    X: pd.DataFrame
        Time indexed dataframe, Train set features.

    y: pd.DataFrame, np.array
        Train set labels.

    cv_gen: PurgedFoldBase
        cv generator object of module cross-validation (PurgedKFold, PurgedWalkForward)

    scoring: Callable
        scoring method of sklearn or with signature fun(np.ndarray, np.ndarray)->float.

    sample_weight: pd.Series
        sample weight series used for fitting and scoring, should have same index as X and y

    fit_params: dict
        Fit parameters of the model (sample_weight, ..).

    n_jobs: int
        number of workers for parallel computing

    Returns
    -------
    importance_df: pd.DataFrame
        Mean and standard deviation of feature importance.
    """
    feature_sets = clustered_subsets if clustered_subsets else [[x] for x in X.columns]

    def __compute_fold_score(train, test, feature_names):

        sample_weight_train = build_sample_weight(sample_weight.loc[train])
        sample_weight_score = build_sample_weight(sample_weight.loc[test])
        fit_params['sample_weight'] = sample_weight_train

        model.fit(X=X.loc[train][feature_names].values,
                  y=y.loc[train].values,
                  **fit_params)

        pred = model.predict(X.loc[test][feature_names].values)

        # Get overall metrics value on out-of-sample fold
        if scoring.__name__ == 'log_loss':
            prob = model.predict_proba(X.loc[test][feature_names].values)
            fold_metrics_values = _apply_metric(scoring, y.loc[test].values, prob,
                                                sample_weight=sample_weight_score)
        else:
            fold_metrics_values = _apply_metric(scoring, y.loc[test].values, pred,
                                                sample_weight=sample_weight_score)
        return fold_metrics_values

    fit_params = fit_params or {}
    if sample_weight is None:
        sample_weight = pd.Series(np.ones((X.shape[0],),
                                          dtype=np.float64),
                                  index=X.index)

    folds_scores = [
                    Parallel(n_jobs=n_jobs, backend='threading')(
                         delayed(__compute_fold_score)(train, test, feature)
                         for train, test in cv_gen.split(X)
                         ) for feature in feature_sets
                   ]

    folds_scores = np.array(folds_scores).astype(float)
    np.nan_to_num(folds_scores, copy=False)
    mean_folds_scores = np.mean(folds_scores, axis=1)
    std_folds_scores = np.std(folds_scores, axis=1) * folds_scores.shape[1] ** -.5
    importance_df = pd.DataFrame(index=X.columns)
    importance_df['mean'] = mean_folds_scores
    importance_df['std'] = std_folds_scores

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame):
    """
    Plot feature importance.
    """
    # Plot mean imp bars with std
    plt.figure(figsize=(10, importance_df.shape[0] / 5))
    importance_df.sort_values('mean', ascending=True, inplace=True)
    importance_df['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=importance_df['std'], error_kw={'ecolor': 'r'})
    return plt.show()
