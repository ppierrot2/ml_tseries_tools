"""
The :mod:`ml_tseries.calibration` implements methods for hyperparameters optimization relying on the *Hyperopt*
library (Bayesian optimization or random search). It implements a functional version *bayesian_tuning*
and a sklearn-style class version *BayesianSearchCV*.

These methods support both time-series data (pandas objects with DatetimeIndex) and standard data
(array-like or dataframe) depending on the underlying model provided and cross-validation generator used.
"""

import logging

import numpy as np
import pandas as pd
from hyperopt import Trials, STATUS_OK
from hyperopt import fmin
from hyperopt import tpe
from hyperopt.fmin import generate_trials_to_calculate
from sklearn.base import MetaEstimatorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import is_classifier

from typing import Callable

from .cross_validation import (fit_model,
                               kfold_cv,
                               tseries_cv,
                               combinatorial_cv,
                               PurgedKFold,
                               PurgedWalkForward,
                               PurgedCombinatorial)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


__score__metrics = ["f1_score", "accuracy_score", "auc", "recall_score",
                    "precision_score", "fbeta_score", "r2_score",
                    "explained_variance_score"]

__loss__metrics = ["log_loss", "hinge_loss", "brier_score_loss", "zero_one_loss", "neg_log_loss",
                   "mean_squared_error", "root_mean_square_error", "mean_absolute_error",
                   "mean_poisson_deviance", "mean_gamma_deviance"]


def build_sample_weight(weight_feature, scaling='sum_to_one'):

    if scaling is None or weight_feature is None:
        return weight_feature
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
        weight_feature = np.absolute(weight_feature)
        weights = scaler.fit_transform(weight_feature.reshape(-1, 1))
        return np.ravel(weights)
    elif scaling == 'sum_to_one':
        return (np.absolute(weight_feature) / np.absolute(weight_feature).sum())\
               * len(weight_feature)


def build_loss_fun(metric_fun: Callable):
    """
     Transform a score or loss function into a loss metric
     able to be passed in cv_methods for hyperparameters
     optimization purpose

    Parameters
    ----------
    metric_fun: Callable
        either a metric of module sklearn.metric
        or function with the signature
        func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)
        or func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)

    Returns
    -------
    loss_fun: callable
        a metric method of signature
        loss_fun(y_test, y_pred, sample_weight=None)
    """
    if metric_fun.__name__ in __loss__metrics:
        return metric_fun

    elif metric_fun.__name__ in __score__metrics:
        def loss_metric(y_true, y_pred, sample_weight=None):
            return - metric_fun(y_true, y_pred, sample_weight=sample_weight)
        loss_metric.__name__ = metric_fun.__name__
        return loss_metric

    else:
        def loss_metric(y_test, y_pred, sample_weight=None):
            name, val, is_h_b = metric_fun(y_test, y_pred,
                                           sample_weight=sample_weight)
            return (-1 if is_h_b else 1) * val
        loss_metric.__name__ = metric_fun.__name__
        return loss_metric


def set_params(model, **params):
    if hasattr(model, 'set_params'):
        model.set_params(**params)
    else:
        for param, val in params.items():
            setattr(model, param, val)


def tree_hyperparam_selector(
                            hyperparam_list,
                            max_rank=5,
                            min_diff_max_depth=3):
    """
    Iteratively select a set of hyperparameters based on a minimal
    difference of the 'max_depth' hyperparameter.
    take an hyperopt output list-like object as input and return
    a list of idx corresponding to the idx selected in hyperopt output

    Parameters
    ----------
    hyperparam_list: list of dict
        list of hyperparameter dict, each should contains
        'max_depth'
    max_rank: int
        maximum rank to consider when iterating on the
        hyperopt_output list
    min_diff_max_depth: int
        minimal distance between precedent selected max_depth
        and new max_depth to select the new set of hyperparameters

    Returns
    -------
    bests: list
        list of index in the provided hyperparam_list corresponding to selected hyperparameters sets
    """
    rank = min(max_rank, len(hyperparam_list))
    bests = [0]
    if rank <= 1:
        return bests

    if 'max_depth' not in hyperparam_list[0].keys():
        log.warning('Cannot do hyperparameter selection for this model,')
        return bests
    else:

        def almost_equal(x, y):
            return abs(x - y) < min_diff_max_depth

        k = 1
        while not k >= rank:
            if all([not almost_equal(
                    hyperparam_list[i]['max_depth'],
                    hyperparam_list[k]['max_depth'])
                    for i in bests]):
                bests.append(k)
            k += 1
        return bests


def bayesian_tuning(X, y, model,
                    param_grid,
                    metric_fun,
                    cv_gen=KFold(),
                    folds_weights=None,
                    fit_params=None,
                    static_params=None,
                    trials=Trials(),
                    optimizer=tpe.suggest,
                    nb_evals=50,
                    hyperparam_selection=False,
                    refit=False,
                    sample_weight=None,
                    y_transform=None,
                    random_state=None,
                    n_jobs=1,
                    **kwargs):
    """
    Perform a Bayesian-style optimization of a given ML model
    hyperparameters based on iteratives cross validations and scoring,
    then store trials in an dict. X, y inputs type have to be adapted to cv_gen inputs
    (array or pd.DataFrame for sklearn CV generator or pd.DataFrame with Datetime index for
    Purged CV generators). The method use the library Hyperopt : https://github.com/hyperopt/hyperopt

    Parameters
    ----------
    X: array-like or pd.DataFrame
        X data. It should be a pandas object with DatetimeIndex if cv_gen is a PurgedFoldBase object

    y: array-like or pd.DataFrame or pd.Series
        y data. It should be a pandas object with DatetimeIndex if cv_gen is a PurgedFoldBase object

    model:
        ML model object implementing fit and predict

    param_grid: dict
        Hyperopt type grid search dictionary (see Hyperopt doc :
        https://github.com/hyperopt/hyperopt/wiki/FMin)

    metric_fun: Callable
        either a metric of module sklearn.metric
        or function with the signature
        func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)
        or func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)

    cv_gen: PurgedFoldBase or sklearn BaseCrossValidator object instance
        cross-validation generator for model hyperparameters evaluation
        at each hyperopt fmin iteration. If instance of PurgedFoldBase,
        time-indexed pandas DataFrame and Series object should be provided
        as X and y

    folds_weights : list or array-like
        optional, weights vector to apply to test fold scores. Should have the same lenght as cv_gen.n_splits

    fit_params: dict
        dictionary of parameters passed to model.fit

    y_transform: callable or None
        optional, method to transform
        y_values before fitting in case model
        needs different y values as ones for computing
        score using metric, must return array-like of
        same shape as y

    static_params: dict or None
        model hyperparameter that are passed in tuning loop

    trials: instance of Trials object
        Hyperopt storage object used for hp calibration

    optimizer:
        optimizer algo used by hyperopt

    nb_evals: int
        number of iteration of optimization process

    hyperparam_selection: bool
        weather to keep bests hyperparameters from
        return trials_dict, only relevent for tree-based
        model using 'max_depht' hyprparam

    refit: bool
        weather to train model on all data with best parameters
        once hyperparam optimization finished

    sample_weight: array-like or pd.Series
        raw sample_weight vector, the scaling of values is performed
        just before fitting or scoring on each train / test folds separately.
        It should be a pandas object with DatetimeIndex if cv_gen is a PurgedFoldBase object

    random_state: int or None
        random state of hyperopt fmin func

    n_jobs: int
        number of worker for cross-validation parallel computing (multi-threading backend)

    kwargs: dict
        additional optional arguments passed to hyperopt.fmin method

    Returns
    -------
    trials_dict:
        list of dict containing optimization info at each iteration
    """
    loss_fun = build_loss_fun(metric_fun)

    if isinstance(cv_gen, (PurgedKFold, PurgedWalkForward)):
        cv_method = tseries_cv
    elif isinstance(cv_gen, PurgedCombinatorial):
        cv_method = combinatorial_cv
    else:
        cv_method = kfold_cv

    def weighted_mean(data, weights):
        """function for weights averaging on cv test fold """
        data = data.dropna(axis=1)
        wm = np.average(data.values, axis=0, weights=weights)
        res = {}
        for i in range(len(data.columns)):
            res[data.columns[i]] = wm[i]
        return res

    def objective(hyperparameters):
        """Objective function for hyperopt optimization. Returns
           the cross validation score from a set of hyperparameters."""

        global ITERATION
        ITERATION += 1

        # deal with nested param space
        for param_name in list(hyperparameters):
            if type(hyperparameters[param_name]) == dict:
                # Retrieve each sub-parameter and put it at top level key
                for sub_param in hyperparameters[param_name].keys():
                    if sub_param != param_name:
                        sub_param_val = hyperparameters[param_name].get(sub_param)
                        hyperparameters[sub_param] = sub_param_val
                # put param with nested space at top level key
                hyperparameters[param_name] = \
                    hyperparameters[param_name][param_name]

        # Make sure parameters that need to be integers are integers
        to_int_list = ['num_leaves',
                       'subsample_for_bin',
                       'min_child_samples',
                       'max_depth',
                       'max_bin']
        to_int_list = [e for e in to_int_list if e in hyperparameters.keys()]
        for parameter_name in to_int_list:
            hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

        static = static_params or {}
        all_params = {**hyperparameters, **static}
        set_params(model, **all_params)

        result_score = cv_method(
            model=model,
            X=X,
            y=y,
            metric=loss_fun,
            fit_params=fit_params,
            y_transform=y_transform,
            sample_weight=sample_weight,
            cv_gen=cv_gen,
            n_jobs=n_jobs,
        )

        result_score = pd.DataFrame({'loss': result_score})

        # compute weighted mean on test folds, default weights set to one
        if folds_weights is not None:
            weights = folds_weights
        else:
            weights = np.ones(len(result_score))
        agg_score = weighted_mean(result_score, weights)
        agg_score['hyperparameters'] = all_params
        agg_score['status'] = STATUS_OK
        agg_score['iteration'] = ITERATION

        return agg_score

    global ITERATION
    ITERATION = 0
    # Run optimization
    result = fmin(fn=objective, space=param_grid,
                  algo=optimizer, trials=trials,
                  max_evals=nb_evals, show_progressbar=True,
                  rstate=random_state, **kwargs)

    trials_list = sorted(trials.results, key=lambda x: x['loss'])

    # refactor output result
    to_int_list = ['num_leaves',
                   'subsample_for_bin',
                   'min_child_samples',
                   'max_depth',
                   'max_bin']
    to_int_list = [e for e in to_int_list if e in result.keys()]
    for parameter_name in to_int_list:
        result[parameter_name] = int(result[parameter_name])
    if 'is_unbalance' in result.keys():
        if result['is_unbalance'] == 0:
            result['is_unbalance'] = False
        if result['is_unbalance'] == 1:
            result['is_unbalance'] = True
    if 'boosting_type' in result.keys():
        if result['boosting_type'] == 0:
            result['boosting_type'] = 'gbdt'
        if result['boosting_type'] == 1:
            result['boosting_type'] = 'dart'
        if result['boosting_type'] == 2:
            result['boosting_type'] = 'goss'

    set_params(model, **trials_list[0]['hyperparameters'])
    if refit:
        log.info(f'model trained with following hyperparameters'
                 f"\n{trials_list[0]['hyperparameters']}")
        fit_params = fit_params or {}
        sample_weight = build_sample_weight(sample_weight)
        fit_model(model, X, y,
                  y_transform=y_transform,
                  sample_weight=sample_weight,
                  fit_params=fit_params)

    if hyperparam_selection:
        hyperparam_list = [tr['hyperparameters'] for tr in trials_list]
        return [trials_list[i] for i in tree_hyperparam_selector(
                                            hyperparam_list,
                                            max_rank=5,
                                            min_diff_max_depth=3)]
    else:
        return trials_list


class BayesianSearchCV(MetaEstimatorMixin, BaseEstimator):

    """ Bayesian hyperparameters optimization wrapper class for sklearn pipeline

    Parameters
    ----------
    estimator:
        ML model object implementing fit and predict

    param_distributions: dict
        Hyperopt type grid search dictionary (see Hyperopt doc :
        https://github.com/hyperopt/hyperopt/wiki/FMin)

    scoring: Callable
        either a metric of module sklearn.metric
        or function with the signature
        func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)
        or func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)

    cv_gen: PurgedFoldBase or sklearn BaseCrossValidator object instance
        cross-validation generator for model hyperparameters evaluation
        at each hyperopt fmin iteration. If instance of PurgedFoldBase,
        time-indexed pandas DataFrame and Series object should be provided
        as X and y

    points_to_evaluate: list[dict]
        list of dictionary of hyperparameters to be evaluated

    static_params: dict or None
        model hyperparameter that are passed in tuning loop

    optimizer:
        optimizer algo used by hyperopt

    n_iter: int
        number of iteration of optimization process

    refit: bool
        weather to train model on all data with best parametres
        once hyperparam optimization finished

    y_transform: callable or None
        optional, method to transform
        y_values before fitting in case model
        needs different y values as ones for computing
        score using metric, must return array-like of
        same shape as y

    sample_weight: array-like or pd.Series
        raw sample_weight vector, the scaling of values is performed
        just before fitting or scoring on each train / test folds separately.
        It should be a pandas object with DatetimeIndex if cv is a PurgedFoldBase object

    random_state: int or None
        random state of hyperopt fmin func

    n_jobs: int
        number of worker for cross-validation parallel computing (multi-threading backend)

    kwargs: dict
        additional optional arguments passed to hyperopt.fmin method

    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator which gave highest score
        (or smallest loss if specified) on the left out data.

    best_score_ : float
        Mean cross-validated score of the best_estimator

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    """

    def __init__(self, estimator, param_distributions, scoring, cv=KFold(),
                 static_params=None, points_to_evaluate=None, optimizer=tpe.suggest,
                 n_iter=50, refit=True, y_transform=None, combine_estimators=False,
                 random_state=None, n_jobs=1, **kwargs):

        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.static_params = static_params
        self.points_to_evaluate = points_to_evaluate
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.refit = refit
        self.y_transform = y_transform
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fmin_params = kwargs
        self.combine_estimators = combine_estimators
        if combine_estimators:
            if 'max_depth' not in param_distributions.keys():
                raise NotImplementedError('combine_estimators is only implemented for tree-based model'
                                 '\n with max_depth provided in param_distributions')

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError(f'This {type(self).__name__} instance was initialized '
                                 f'with refit=False. {method_name} is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_params_`` '
                                 'attribute')
        else:
            check_is_fitted(self, attributes='best_estimator_')

    def fit(self, X, y, sample_weight=None, **fit_params):
        """

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
            It should be pandas object with DatetimeIndex if cv is a PurgedFoldBase object

        y : array-like or pd.Series of shape (n_samples, n_output) or (n_samples,)
            Target relative to X for classification or regression;
            It should be pandas object with DatetimeIndex if cv is a PurgedFoldBase object

        **fit_params : dict of str -> object
            Parameters passed to the fit method of the estimator

        Returns
        -------

        """
        if self.points_to_evaluate is not None:
            trials = generate_trials_to_calculate(self.points_to_evaluate)
        else:
            trials = Trials()
        refit = False if self.combine_estimators else self.refit
        bests = bayesian_tuning(X, y, model=self.estimator,
                                param_grid=self.param_distributions,
                                metric_fun=self.scoring,
                                cv_gen=self.cv,
                                fit_params=fit_params,
                                static_params=self.static_params,
                                trials=trials,
                                optimizer=tpe.suggest,
                                nb_evals=self.n_iter,
                                hyperparam_selection=False,
                                refit=refit,
                                sample_weight=sample_weight,
                                y_transform=self.y_transform,
                                random_state=self.random_state,
                                n_jobs=self.n_jobs,
                                **self.fmin_params)
        self.trials_ = bests
        if self.combine_estimators:
            hyperparam_list = [tr['hyperparameters'] for tr in self.trials_]
            selected_bests = [self.trials_[i] for i in tree_hyperparam_selector(
                                hyperparam_list,
                                max_rank=5,
                                min_diff_max_depth=3)]
            self.best_estimator_, self.best_score_,  self.best_params_ = list(), list(), list()
            for tr in selected_bests:
                new_estimator = clone(self.estimator)
                set_params(new_estimator, **tr['hyperparameters'])
                sample_weight = build_sample_weight(sample_weight)
                fit_model(new_estimator, X, y, y_transform=self.y_transform, sample_weight=sample_weight,
                          fit_params=fit_params)
                self.best_estimator_.append(new_estimator)
                self.best_score_.append(tr['loss'])
                self.best_params_.append(tr['hyperparameters'])
        else:
            self.best_estimator_ = self.estimator
            self.best_score_ = bests[0]['loss']
            self.best_params_ = bests[0]['hyperparameters']

    def predict(self, X):
        """
        make prediction using the best_estimator found or combined predictions of selected estimators
        in case combine_model is True

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        predictions : array-like
        """
        self._check_is_fitted('predict')
        if self.combine_estimators:
            if is_classifier(self.estimator):
                probs = self.predict_proba(X)
                return np.apply_along_axis(np.argmax, 1, probs)
            else:
                preds = np.vstack([est.predict(X).ravel() for est in self.best_estimator_])
                return np.mean(preds, axis=0)
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        make proba predictions using the best_estimator found or combined predictions of selected estimators
        in case combine_model is True. Only relevant for classifiers

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        predictions : array-like
        """
        self._check_is_fitted('predict_proba')
        if self.combine_estimators:
            preds = np.mean(np.dstack([est.predict_proba(X) for est in self.best_estimator_]), axis=-1)
            return preds
        else:
            return self.best_estimator_.predict_proba(X)

