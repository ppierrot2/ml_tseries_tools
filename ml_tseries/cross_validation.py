import datetime as dt
import itertools
from abc import abstractmethod

import inspect
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from sklearn.model_selection import (KFold, TimeSeriesSplit)
from sklearn.preprocessing import MinMaxScaler
from typing import Iterable, Tuple, List, Callable


__need_proba__metrics = ['log_loss']


def _apply_metric(metric, y_true, y_pred, sample_weight=None, **kw):
    """
    utils method to apply metric method with sample weight

    Parameters
    ----------
    metric: callable
        method with signature fun(y_true, y_pred)->float or
        fun(y_true, y_pred, sample_weight)->float
    y_true: array-like
        1d array of true values
    y_pred: array-like
        1d array of pred values

    Returns
    -------
    float : return value of metric fun
    """
    if 'sample_weight' in inspect.signature(
            metric).parameters.keys():

        return metric(y_true, y_pred, sample_weight=sample_weight, **kw)
    else:
        return metric(y_true, y_pred, **kw)


def build_sample_weight(weight_feature, scaling='sum_to_one'):
    """
    Scaling sample weight values
    :param weight_feature: array-like, raw sample weigh vector
    :param scaling: str, method of scaling
    :return: 1d array, scaled sample weight
    """
    if weight_feature is None:
        pass
    init_val = weight_feature.iloc[0] if isinstance(weight_feature, pd.Series) else weight_feature[0]
    if np.all(weight_feature == init_val):  # no scaling if constant
        return weight_feature

    if scaling == 'minmax':
        scaler = MinMaxScaler()
        weight_feature = np.absolute(weight_feature)
        weights = scaler.fit_transform(weight_feature.reshape(-1, 1))
        return np.ravel(weights)

    elif scaling == 'sum_to_one':
        return (np.absolute(weight_feature) / np.absolute(weight_feature).sum())\
               * len(weight_feature)


def fit_model(model, X, y,
              fit_params=None,
              sample_weight=None,
              y_transform=None):
    """
    util method for fitting model of different nature with some
    pre-processing

    Parameters
    ----------
    model:
        a ML algo implementing fit method

    X: array-like or pd.DataFrame
        X values

    y: array-like or pd.Series
        y values

    fit_params: dict
        dict of parameters passed to model.fit

    sample_weight: pd.Series
        Series of sample weight

    y_transform: callable or None
        optional, method to transform y values
        before fitting if needed

    """
    fit_params = fit_params or {}
    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight.values

    if y_transform is not None:
        y = pd.Series(y_transform(y.values.reshape(-1, 1)).ravel(), index=X.index)

    model.fit(X, y, **fit_params)


class PurgedFoldBase:
    """
    Abstract class for time series cross-validation.

    Time series cross-validation requires time indexed pandas DataFrames or series
    as input for instance methods. It can have non unique datetime index but must be
    ordered.

    overlap: pd.Timedelta
        overlap period

    embargo: pd.Timedelta
        embargo period
    """
    def __init__(self, overlap=dt.timedelta(days=1),
                 embargo=dt.timedelta(days=1)):
        self.overlap = overlap
        self.embargo = embargo

    @abstractmethod
    def split(self, X: pd.DataFrame):

        if not isinstance(X, (pd.DataFrame, X, pd.Series)):
            raise ValueError('X should be a pandas DataFrame/Series.')
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError('X should have DatetimeIndex')

    def purge_data(self,
                   X: pd.DataFrame,
                   test_index) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        '''
        Purge the train set to avoid information leakage between train
        and test sets

        Parameters
        -----------
        X: pd.DataFrame
            time indexed data frame

        test_index: list-like
            list of date times corresponding
            to the points that will be tested

        Returns
        -------
        train: DateTimeIndex
            purged data frame index of train set

        test: DateTimeIndex
            purged data frame index of test set
        '''
        train = X
        # overlap before left to the test period
        idx1 = pd.date_range(test_index[0] - self.overlap, test_index[0])
        # overlap + embargo right to the test period
        idx2 = pd.date_range(test_index[-1], test_index[-1] + self.overlap + self.embargo)
        idx = np.concatenate((idx1, test_index, idx2))
        # the periods that are not in the test period
        train = train[~train.index.isin(idx)].index
        test = X[X.index.isin(test_index)].index
        return train, test


class PurgedKFold(PurgedFoldBase):
    """
    Class for k-folds train/test generation with data purging

    n_splits: int
        number of groups to split the dataset
    n_jobs: int
        number of jobs for parallel computing

    """
    def __init__(self, n_splits=3, n_jobs=1, **kwargs):
        super().__init__(**kwargs)
        self.n_splits = n_splits
        self.n_jobs = n_jobs

    def split(self, X: pd.DataFrame) -> Iterable[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        '''
        Create train/test subsets with cross-validation framework

        Parameters
        ----------
        X: pd.DataFrame
            time indexed data frame

        Returns
        -------
        k-splits: list
            list of length k that contains (train, test) index pairs corresponding to
            the k different splits
        '''
        super().split(X)
        parallel = Parallel(n_jobs=self.n_jobs, max_nbytes=None)
        return parallel(
            delayed(self.purge_data)(X, test_period.index)
            for test_period in np.array_split(X, self.n_splits)
        )


class PurgedWalkForward(PurgedFoldBase):
    """
    Class for train/test folds generation in a walk forward fashion
    with data purging

    n_splits: int
        number of groups to split the dataset
    n_jobs: int
        number of jobs for parallel computing

    """
    def __init__(self, n_splits=3, n_jobs=1, **kwargs):
        super().__init__(**kwargs)
        self.n_splits = n_splits
        self.n_jobs = n_jobs

    def split(self, X: pd.DataFrame) -> Iterable[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        '''
        Create train/test subsets with cross-validation framework

        Parameters
        ----------
        X: pd.DataFrame
            time indexed data frame

        Returns
        -------
        k-splits: list
            list of length k that contains (train, test) index pairs corresponding to
            the k different splits
        '''
        super().split(X)
        parallel = Parallel(n_jobs=self.n_jobs, max_nbytes=None)
        cv_gen = TimeSeriesSplit(n_splits=self.n_splits)
        return parallel(
            delayed(self.purge_data)(X, X.iloc[test_idx].index)
            for _, test_idx in cv_gen.split(X)
        )


class PurgedCombinatorial(PurgedFoldBase):
    """
    Class for train/test folds generation in a walk forward fashion
    with data purging

    n_splits: int
        number of groups to split the dataset
    n_test_splits: int
        number of groups tested
    n_jobs: int
        number of jobs for parallel computing

    Raises
    ------
    ValueError
        when n_test_splits > n_splits
    """
    def __init__(self, n_splits=5, n_test_splits=2, n_jobs=1, **kwargs):

        super().__init__(**kwargs)
        if n_test_splits > n_splits:
            raise ValueError('N must be greater than k')

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.n_jobs = n_jobs

    def split(self, X: pd.DataFrame) -> Iterable[Tuple[pd.DatetimeIndex, List[pd.DatetimeIndex]]]:
        '''
        Create train/test subsets with combinatorial cross-validation framework

        Parameters
        ----------
        X: pd.DataFrame
            time indexed data frame

        Returns
        -------
        splits: list
            list of length C(n_splits,n_test_splits) that contains (train, tests) index pairs corresponding
            to the C(N, n_test_splits) different splits, where 'tests' is a list of n_test_splits time
            series index (the n_test_splits test periods)
        '''
        super().split(X)

        def create_trains_tests(test_periods):
            '''
            Create train/test subsets for further parallelization

            Parameters
            ----------
            test_periods : array
                array of length n_test_splits, contains time series datasets

            Returns
            -------
            train: pd.DataFrame
                time series train set (purged)
            tests: list
                list of length n_test_splits, contains time series test sets, correspond to
                the n_test_splits test periods
            '''
            train = X
            tests = []
            for test_period in test_periods:
                train_idx, test_idx = self.purge_data(train, test_period.index)
                train = train.loc[train_idx]
                tests.append(test_idx)
            return train.index, tests

        parallel = Parallel(n_jobs=self.n_jobs)
        # split data into n_splits groups, then take k parts among N
        return parallel(
            delayed(create_trains_tests)(test_periods)
            for test_periods in itertools.combinations(np.array_split(X, self.n_splits),
                                                       self.n_test_splits)
        )


def kfold_cv(model, X, y,
             metric: Callable[[np.ndarray, np.ndarray], float],
             cv_gen=KFold(),
             sample_weight: np.ndarray = None,
             fit_params: dict = None,
             y_transform: Callable = None,
             verbose: bool = False,
             n_jobs: int = 1) -> List:
    '''
    Perform k-fold cross-validation

    Parameters
    ----------
    model:
        ML model object implementing fit and predict

    X : array-like or pd.DataFrame
        X values

    y : array-like or pd.Series
        y values

    metric: callable
        method for cv scoring, etheir sklearn metric or other
        custom scoring fun matching the signature:
        func(y_true, y_pred)->float or func(y_true, y_pred, sample_weight)->float

    cv_gen: sklearn BaseCrossValidator object
        cv split generator

    sample_weight: np.array
        sample weight vector

    fit_params: dict
        dictionary of parameters passed to model.fit

    y_transform: Callable or None
        not used, just for integration

    verbose: bool
        verbosity, not implemented

    n_jobs :  int
        number of worker for parallelisation

    Returns
    -------
    scores : list
        list of all scores obtained during cross-validation
    '''

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    def _fit(train, test):
        '''
        Process train and test data,
         fit model and compute prediction and score

        Parameters
        ----------
        train: np.array
            train index
        test: np.array
            test index

        Returns
        -------
        score: float
            the value of the score computed on the test set according to
            the metric
        '''
        sample_weight_train = build_sample_weight(sample_weight[train])
        sample_weight_score = build_sample_weight(sample_weight[test])
        fit_params['sample_weight'] = sample_weight_train

        model.fit(X[train, :], y[train], **fit_params)

        if metric.__name__ == 'log_loss':
            y_pred = model.predict_proba(X[test, :])
        else:
            y_pred = model.predict(X[test, :])

        score = _apply_metric(metric, y[test], y_pred,
                              sample_weight=sample_weight_score)
        return score

    fit_params = fit_params or {}
    fit_params = fit_params.copy()
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    parallel = Parallel(n_jobs=n_jobs, max_nbytes=None)
    scores = parallel(
        delayed(_fit)(train, test)
        for train, test in cv_gen.split(X, y)
    )

    return scores


def tseries_cv(model, X: pd.DataFrame,
               y: pd.DataFrame,
               metric: Callable[[np.ndarray, np.ndarray], float],
               cv_gen: PurgedFoldBase = PurgedKFold(),
               sample_weight: np.ndarray = None,
               fit_params: dict = None,
               y_transform: Callable = None,
               verbose: bool = None,
               n_jobs=1) -> np.ndarray:
    """
    fit the time-serie cross-validation and the scores obtained
    during cross-validation depending on the metric used

    Parameters
    ----------
    model: ML model object implementing fit and predict

    X: pd.DataFrame
        X values as time-indexed pandas dataframe

    y: pd.DataFrame or pd.Series
        y values as time-indexed pandas dataframe

    metric: callable
        method for cv scoring, etheir sklearn metric or other
        custom scoring fun matching the signature:
        func(y_true, y_pred)->float or func(y_true, y_pred, sample_weight)->float

    cv_gen: Callable
        train test index generator function to use (can be PurgedKFold,
        PurgedSeasFold or PurgedWalkForward)

    sample_weight: pd.Series
        sample_weight to use when training and scoring the model.
        should have the same index as X and y

    y_transform: callable
        optional, method to transform
        y_values before fitting in case model
        needs different y values as ones for computing
        score, must return array-like of same shape as y

    fit_params: dict
        dictionary of parameters passed to model.fit

    verbose: bool
        verbosity

    n_jobs: int
        number of jobs for parallelisation

    Returns
    -------
    scores: list
        list of all obtained scores during cross-validation
    """
    if not isinstance(cv_gen, (PurgedKFold, PurgedWalkForward)):
        raise ValueError('wrong cv_gen object, only PurgedKFold,'
                         'PurgedWalkForward object are supported')

    assert isinstance(X, pd.DataFrame), 'need to provide time indexed\
         pandas DataFrame as X'
    assert isinstance(y, pd.Series) or isinstance(y, pd.DataFrame), \
        'need to provide time indexed pandas Serie as y'

    def _fit(train, test):
        '''
        intermediate inner fit method for parallelization

        Parameters
        ----------
        train: pandas DatetimeIndex
            datetime indexes of train set
        test: pandas DatetimeIndex
            datetime indexes of train set

        Returns
        -------
        score :
            the value of the score computed on the test set according to
            the metric
        '''
        sample_weight_train = build_sample_weight(sample_weight.loc[train])
        sample_weight_score = build_sample_weight(sample_weight.loc[test])

        fit_model(model, X.loc[train], y.loc[train], fit_params=fit_params,
                  y_transform=y_transform, sample_weight=sample_weight_train)

        if metric.__name__ == 'log_loss':
            y_pred = model.predict_proba(X.loc[test])
        else:
            y_pred = model.predict(X.loc[test])

        score = _apply_metric(metric, y.loc[test].values, y_pred,
                              sample_weight=sample_weight_score)

        return score

    fit_params = fit_params or {}
    if sample_weight is None:
        sample_weight = pd.Series(np.ones((X.shape[0],),
                                          dtype=np.float64),
                                  index=X.index)

    parallel = Parallel(n_jobs=n_jobs, max_nbytes=None)
    scores = parallel(
        delayed(_fit)(train_idx, test_idx)
        for train_idx, test_idx in cv_gen.split(X)
    )

    return scores


def get_path_data(data, n, k):
    '''
    Retrieve the data pathwise

    Parameters
    ----------
    data: np.array
        matrix of C(N,k) lines x k columns, each element is the predicted
        values of every group of test folds

    Returns
    -------
    backtests_paths: np.array
        list of lists C(N,k)*k/N lines x N columns, each list contains a
        backtest path
    '''

    binom = int(scipy.special.binom(n, k))
    phi = int(binom * k / n)
    comb = itertools.combinations(list(range(n)), k)
    comb = itertools.chain(*comb)
    paths = np.zeros((phi, n))
    backtests_paths = np.copy(paths).tolist()

    for i in range(binom):
        for j in range(k):
            sample = next(comb)
            mask = np.where(paths[:, sample] == 0)[0][0]
            paths[mask][sample] = 1
            backtests_paths[mask][sample] = data[i][j]

    return backtests_paths


def fit_combinatorial(model,
                      X: pd.DataFrame,
                      y: pd.DataFrame,
                      metric: Callable[[np.ndarray, np.ndarray], float],
                      cv_gen: PurgedCombinatorial = PurgedCombinatorial(),
                      sample_weight: np.ndarray = None,
                      y_transform: Callable = None,
                      fit_params: dict = None,
                      get_importances: bool = True,
                      verbose: bool = True,
                      n_jobs: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the combinatorial cross-validation.
    Parameters
    ----------
    model:
        ML model object implementing fit and predict

    X: pd.DataFrame
        X values as time-indexed pandas dataframe

    y: pd.DataFrame or pd.Series
        y values as time-indexed pandas dataframe

    metric: callable
        method for cv scoring, etheir sklearn metric or other
        custom scoring fun matching the signature:
        func(y_true, y_pred)->float or func(y_true, y_pred, sample_weight)->float

    cv_gen: PurgedCombinatorial object
        cross validation train/test generator

    sample_weight: pd.Series
        sample weight serie, should have same index as X and y

    y_transform: callable
        optional, method to transform
        y_values before fitting in case model
        needs different y values as ones for computing
        score using metric, must return array-like of
        same shape as y

    fit_params: dict
        dictionary of parameters passed to model.fit

    get_importances:
        whether to store feature_importances_
        during cross_validation or not

    verbose: bool
        whether to print information during training
        or not (it prints the approximate values)

    n_jobs: int
        number of jobs for parallelisation

    Returns
    -------
    scores : list
        the scores for each path
    paths : list
        containts backtest paths, in the right order
    """
    if not isinstance(cv_gen, PurgedCombinatorial):
        raise ValueError('only PurgedCombinatorial object can be passed as cv_gen')

    if verbose == True:
        len_train = int(len(X) * (1 - cv_gen.n_test_splits / cv_gen.n_splits))
        len_val = int(len(X) / cv_gen.n_splits)
        n_paths = int(scipy.special.binom(cv_gen.n_splits, cv_gen.n_test_splits) * cv_gen.n_test_splits / cv_gen.n_splits)

        print(f"Training on {len_train} samples "
              f"\nValidation on {len_val}*{cv_gen.n_test_splits} samples"
              f"\nGenerating {n_paths} paths")

    # intermediate inner function for parallelization
    def _fit(train, test):
        '''
        Process train and test data, fit model and compute predictions and
        scores for each of the test fold cases among folds

        Parameters
        ----------
        train: pandas DatetimeIndex
            datetime indexes of train set
        test: pandas DatetimeIndex
            datetime indexes of train set

        Returns
        -------
        scores: list
                contains the k computed scores on the k different test sets
        preds: list
            contains the k computed predictions on the k different test sets
        imp:
            feature_importances_ for this model
        '''

        sample_weight_train = build_sample_weight(sample_weight.loc[train])

        fit_model(model, X.loc[train], y.loc[train], fit_params=fit_params,
                  y_transform=y_transform, sample_weight=sample_weight_train)

        scores, preds = list(), list()
        for i in range(cv_gen.n_test_splits):
            sample_weight_score = build_sample_weight(sample_weight.loc[test[i]])
            if metric.__name__ == 'log_loss':
                y_pred = model.predict_proba(X.loc[test[i]])
            else:
                y_pred = model.predict(X.loc[test[i]])
            score = _apply_metric(metric, y.loc[test[i]].values, y_pred,
                                  sample_weight=sample_weight_score)

            scores.append(score)
            preds.append(y_pred)

        if get_importances:
            imp = model.feature_importances_
        else:
            imp = None
        return scores, preds, imp

    fit_params = fit_params or {}
    if sample_weight is None:
        sample_weight = pd.Series(np.ones((X.shape[0],),
                                          dtype=np.float64),
                                  index=X.index)

    parallel = Parallel(n_jobs=n_jobs, max_nbytes=None)
    result = parallel(
        delayed(_fit)(train, test)
        for train, test in cv_gen.split(X)
    )

    result = np.array(result)
    scores = result[:, 0]
    scores = get_path_data(scores, cv_gen.n_splits, cv_gen.n_test_splits) 
    predictions = result[:, 1]
    paths = get_path_data(predictions, cv_gen.n_splits, cv_gen.n_test_splits)

    if get_importances == True:
        imp = np.mean(result[:, 2], axis=0)
    else:
        imp = None

    return scores, paths, imp


def combinatorial_cv(model, X: pd.DataFrame, y: pd.DataFrame,
                     metric: Callable[[np.ndarray, np.ndarray], float],
                     cv_gen: PurgedCombinatorial = PurgedCombinatorial(),
                     y_transform: Callable = None,
                     sample_weight: np.ndarray = None,
                     fit_params: dict = None,
                     verbose: bool = False,
                     get_importances: bool = False,
                     n_jobs: int = 1) -> np.ndarray:

    '''
    Perform the combinatorial cross validation with n_kfold and n_testfold.

    The combinatorial framework enable the generation of more backtest paths
    during cross-validation ( :math:`C^k_n.(k/n)` instead of one across
    :math:`C^k_n` possible test folds combination if n is the total number of folds)

    Only relevant for time series data

    Parameters
    ----------
    model:
        ML model object implementing fit and predict

    X: pd.DataFrame
        X values as time-indexed pandas dataframe

    y: pd.DataFrame or pd.Series
        y values as time-indexed pandas dataframe

    metric: str or callable
        method for cv scoring, etheir sklearn metric or other
        custom scoring fun matching the signature:
        func(y_true, y_pred)->float or func(y_true, y_pred, sample_weight)->float

    cv_gen: PurgedCombinatorial object
        cross validation train/test generator

    y_transform: callable or None
        optional, method to transform
        y_values before fitting in case model
        needs different y values as ones for computing
        score using metric, must return array-like of
        same shape as y

    sample_weight: pd.Series
        sample weight serie, should have same index as X and y

    fit_params: dict
        dictionary of parameters passed to model.fit

    get_importances:
        whether to store feature_importances_
        during cross_validation or not

    verbose: bool
        if True, print information during training

    n_jobs: int
        number of jobs for parallelisation

    Returns
    -------
    scores: np.ndarray
        array containing scores
    '''
    assert isinstance(X, pd.DataFrame), 'need to provide time indexed\
         pandas DataFrame as X'
    assert isinstance(y, pd.Series) or isinstance(y, pd.DataFrame), \
        'need to provide time indexed pandas Serie as y'

    scores, paths, _ = fit_combinatorial(model,
                                         X=X,
                                         y=y,
                                         metric=metric,
                                         cv_gen=cv_gen,
                                         y_transform=y_transform,
                                         sample_weight=sample_weight,
                                         fit_params=fit_params,
                                         get_importances=get_importances,
                                         verbose=verbose,
                                         n_jobs=n_jobs,
                                         )
    score = np.array(scores)
    return np.average(score, axis=0)


def cross_val_predictions(model, X, y, cv_gen, sample_weight=None, probas=False, n_jobs=1, **fit_params):
    """

    Parameters
    ----------
    model
    X
    y
    cv_gen
    sample_weight
    probas
    n_jobs
    fit_params

    Returns
    -------

    """
    def __generate_fold_predictions(train, test):

        sample_weight_train = sample_weight.loc[train]
        model.fit(X.loc[train], y.loc[train], sample_weight=sample_weight_train, **fit_params)
        if probas:
            preds = model.predict_proba(X.loc[test])
        else:
            preds = model.predict(X.loc[test])
        preds = pd.DataFrame(preds, index=test)

        return preds

    if sample_weight is None:
        sample_weight = pd.Series(np.ones((X.shape[0],),
                                          dtype=np.float64),
                                  index=X.index)

    executor = Parallel(n_jobs=n_jobs, backend='threading')
    pred_folds = executor(
        delayed(__generate_fold_predictions)(train, test)
        for train, test in cv_gen.split(X)
    )

    preds_df = pd.concat(pred_folds, axis=0)

    return preds_df
