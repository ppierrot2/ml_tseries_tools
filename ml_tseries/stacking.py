
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.preprocessing import OneHotEncoder

from .cross_validation import PurgedKFold


class MetaClassifierTS(BaseEstimator, ClassifierMixin):
    """
    Meta-classifier adapted to time series, fitted on base classifier proba prediction
    and optional provided meta-features

    Notes : Meta-classifier tuning can be done by passing a BayesianSearchCV object as final_estimator

    Parameters
    ----------
    estimators : list of estimators
        List of base estimators implementing fit and predict_proba

    final_estimator : estimator
        ml model implementing fit and predict used as meta classifier. Can be GridSearchCV or
        BayesianSearchCV object to enable tuning on this estimator

    cv_gen : sklearn cv generator or instance of PurgedKFold, PurgedWalkForward or PurgedSeasFold
        cv generator (implementing splitting strategy) used to fit each base estimators for final
        estimator training data

    refit : bool
        Clones the base esimators for stacking if True

    passthrough : bool
        Weather to add all original features to final estimator training data

    n_jobs : int
        number of concurrent worker for cv parallel execution

    """
    def __init__(self, estimators, final_estimator, cv_gen=PurgedKFold(),
                 refit=True, passthrough=True, n_jobs=1):

        assert all([is_classifier(clf) for clf in estimators]), 'All estimators should be classifier'
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv_gen = cv_gen
        self.refit = refit
        self.passthrough = passthrough
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """
        Build meta-data with folds base estimator predictions then fit the meta-estimator on these data
        and original features if  passthrough=True

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            pandas dataframe object with DatetimeIndex. Training data

        y : pd.Series
            pandas series object with DatetimeIndex. Target relative to X for classification;

        sample_weight : pd.Series
            pandas series object with DatetimeIndex. Sample weight vector

        """
        if self.refit:
            self.estimators_ = [clone(clf) for clf in self.estimators]
            self.meta_estimator_ = clone(self.final_estimator)
        else:
            self.estimators_ = self.estimators
            self.meta_estimator_ = self.final_estimator

        def __generate_fold_predictions(train, test):

            sample_weight_train = sample_weight.loc[train]
            preds = list()
            for i, estimator in enumerate(self.estimators_):
                estimator.fit(X.loc[train], y.loc[train], sample_weight=sample_weight_train)
                pred = estimator.predict_proba(X.loc[test])
                preds.append(pred)

            preds = pd.DataFrame(np.concatenate(preds, axis=1), index=test)

            return preds

        if sample_weight is None:
            sample_weight = pd.Series(np.ones((X.shape[0],),
                                              dtype=np.float64),
                                      index=X.index)

        executor = Parallel(n_jobs=self.n_jobs, backend='threading')
        pred_folds = executor(
            delayed(__generate_fold_predictions)(train, test)
            for train, test in self.cv_gen.split(X)
        )

        preds_df = pd.concat(pred_folds, axis=0)
        # add additional meta-features
        mX = preds_df
        if self.passthrough is not None:
            mX = pd.concat([X.loc[preds_df.index], mX], axis=1)

        self.meta_estimator_.fit(mX, y.loc[preds_df.index])

        return self

    def predict_proba(self, X):
        """
        Make proba prediction by first computing base estimators prediction on X then predict
        probabilities with meta-estimator predict_proba method

        Parameters
        ----------
        X : pd.DataFrame
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        predictions : array-like of shape (n_samples x 2)
        """
        mX = np.concatenate([est.predict_proba(X) for est in self.estimators_], axis=1)

        if self.passthrough is not None:
            mX = np.concatenate((X.values, mX), axis=1)

        pred_meta = self.meta_estimator_.predict_proba(mX)
        return pred_meta

    def predict(self, X):
        """
        Make label prediction by using predict proba method

        Parameters
        ----------
        X : pd.DataFrame
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        predictions : array-like
        """
        probs = self.predict_proba(X)
        return np.apply_along_axis(np.argmax, 1, probs)


class MetaRankingClassifierTS(BaseEstimator, ClassifierMixin):
    """
    Meta-classifier that dynamically weight base classifier output according to predicted
    performance ranking, Meta-classifier is fitted on base classifier proba prediction
    and provided meta-features (time serie feature on rolling basis for example) and meta-labels
    corresponding to the best base classifier on each data sample.

    Inspired by https://www.monash.edu/business/ebs/our-research/publications/ebs/wp06-2018.pdf

    Parameters
    ----------
    estimators : list of estimators
        List of base estimators implementing fit and predict_proba

    final_estimator : estimator
        ml model implementing fit and predict used as meta classifier. Can be GridSearchCV or
        BayesianSearchCV object to enable tuning on this estimator

    cv_gen : sklearn cv generator or instance of PurgedKFold, PurgedWalkForward or PurgedSeasFold
        cv generator (implementing splitting strategy) used to fit each base estimators for final
        estimator training data

    refit : bool
        Clones the base esimators for stacking if True

    n_jobs : int
        number of concurrent worker for cv parallel execution

    """
    def __init__(self, estimators, final_estimator, cv_gen=PurgedKFold(),
                 refit=True, n_jobs=1):

        assert all([is_classifier(clf) for clf in estimators]), 'All estimators should be classifier'
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv_gen = cv_gen
        self.refit = refit
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None, meta_features=None):
        """
        Build meta-data with folds base estimator predictions then fit the meta-estimator on these data
        and optionally provided meta-features

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            pandas dataframe object with DatetimeIndex. Training data

        y : pd.Series
            pandas series object with DatetimeIndex. Target relative to X for classification;

        sample_weight : pd.Series
            pandas series object with DatetimeIndex. Sample weight vector

        meta_features : pd.DataFrame
            optional meta feature to concatenate with base estimator
             prediction for meta-estimator training data generation

        """
        enc = OneHotEncoder()
        y_enc = enc.fit_transform(y.values.reshape(-1, 1)).toarray()
        y_enc = pd.DataFrame(y_enc, index=X.index)
        if self.refit:
            self.estimators_ = [clone(clf) for clf in self.estimators]
            self.meta_estimator_ = clone(self.final_estimator)
        else:
            self.estimators_ = self.estimators
            self.meta_estimator_ = self.final_estimator

        def __generate_fold_predictions(train, test):

            sample_weight_train = sample_weight.loc[train]
            preds = list()
            scores = list()
            for i, estimator in enumerate(self.estimators_):
                estimator.fit(X.loc[train], y.loc[train], sample_weight=sample_weight_train)
                pred = estimator.predict_proba(X.loc[test])
                score = np.sum(np.multiply(pred, y_enc.loc[test].values), axis=1)
                preds.append(pred)
                scores.append(score)

            scores = np.array(scores).T
            bests = pd.Series(np.argmax(scores, axis=1), index=test)
            preds = pd.DataFrame(np.concatenate(preds, axis=1), index=test)

            return preds, bests

        if sample_weight is None:
            sample_weight = pd.Series(np.ones((X.shape[0],),
                                              dtype=np.float64),
                                      index=X.index)

        executor = Parallel(n_jobs=self.n_jobs, backend='threading')
        pred_folds, best_est_idx = zip(*executor(
            delayed(__generate_fold_predictions)(train, test)
            for train, test in self.cv_gen.split(X)
        ))

        preds_df = pd.concat(pred_folds, axis=0)
        scores_s = pd.concat(best_est_idx)
        # add additional meta-features
        mX = preds_df
        my = scores_s
        if meta_features is not None:
            mX = pd.concat([meta_features.loc[preds_df.index], mX], axis=1)

        self.meta_estimator_.fit(mX, my)

        return self

    def predict_proba(self, X, meta_features=None):
        """
        Make proba prediction by first computing base estimators prediction on X then predict
        probabilities with meta-estimator predict_proba method and provided meta_features

        Parameters
        ----------
        X : pd.DataFrame
            Must fulfill the input assumptions of the underlying estimator.

        meta_features : pd.DataFrame
            optional meta feature. Should be provided if provided for the training phase

        Returns
        -------
        predictions : array-like of shape (n_samples x 2)
        """
        pred_list = []
        for est in self.estimators_:
            pred = est.predict_proba(X)
            pred_list.append(pred)

        mX = np.concatenate(pred_list, axis=1)

        if meta_features is not None:
            mX = np.concatenate((meta_features.values, mX), axis=1)

        pred_meta = self.meta_estimator_.predict_proba(mX)
        return np.tensordot(pred_meta, np.array(pred_list), axes=1)[0]

    def predict(self, X, meta_features=None):
        """
        Make label prediction by using predict proba method

        Parameters
        ----------
        X : pd.DataFrame
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        predictions : array-like
        """
        probs = self.predict_proba(X, meta_features)
        return np.apply_along_axis(np.argmax, 1, probs)


class MetaRegressorTS(BaseEstimator, RegressorMixin):
    """
    Stacking regressor adapted to time series, fitted on base regressor prediction
    output and optional provided meta-features.

    Notes : Meta-regressor tuning can be done by passing a BayesianSearchCV object as final_estimator

    Parameters
    ----------
    estimators : list of estimators
        List of base estimators implementing fit and predict_proba

    final_estimator : estimator
        ml model implementing fit and predict used as meta regressor. Can be GridSearchCV or
        BayesianSearchCV object to enable tuning on this estimator

    cv_gen : sklearn cv generator or instance of PurgedKFold, PurgedWalkForward or PurgedSeasFold
        cv generator (implementing splitting strategy) used to fit each base estimators for final
        estimator training data

    refit : bool
        Clones the base esimators for stacking if True

    passthrough : bool
        Weather to add all original features to final estimator training data

    n_jobs : int
        number of concurrent worker for cv parallel execution

    """

    def __init__(self, estimators, final_estimator, cv_gen=PurgedKFold(),
                 refit=True, passthrough=True, n_jobs=1):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv_gen = cv_gen
        self.refit = refit
        self.passthrough = passthrough
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """
        Build meta-data with folds base estimator predictions then fit the meta-estimator on these data
        and original features if  passthrough=True

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            pandas dataframe object with DatetimeIndex. Training data

        y : pd.Series
            pandas series object with DatetimeIndex. Target relative to X for regression;

        sample_weight : pd.Series
            pandas series object with DatetimeIndex. Sample weight vector

        """
        if self.refit:
            self.estimators_ = [clone(clf) for clf in self.estimators]
            self.meta_estimator_ = clone(self.final_estimator)
        else:
            self.estimators_ = self.estimators
            self.meta_estimator_ = self.final_estimator

        def __generate_fold_predictions(train, test):

            sample_weight_train = sample_weight.loc[train]
            preds = list()
            for i, estimator in enumerate(self.estimators_):
                estimator.fit(X.loc[train], y.loc[train], sample_weight=sample_weight_train)
                pred = estimator.predict(X.loc[test])
                preds.append(pred.ravel())

            preds = pd.DataFrame(np.array(preds).T, index=test)

            return preds

        if sample_weight is None:
            sample_weight = pd.Series(np.ones((X.shape[0],),
                                              dtype=np.float64),
                                      index=X.index)

        executor = Parallel(n_jobs=self.n_jobs, backend='threading')
        pred_folds = executor(
                delayed(__generate_fold_predictions)(train, test)
                for train, test in self.cv_gen.split(X)
            )

        preds_df = pd.concat(pred_folds, axis=0)
        # add additional meta-features
        mX = preds_df
        if self.passthrough is not None:
            mX = pd.concat([X.loc[preds_df.index], mX], axis=1)

        self.meta_estimator_.fit(mX, y.loc[preds_df.index])

        return self

    def predict(self, X):
        """
        Make prediction by first computing base estimators prediction on X then predict
        with meta-estimator

        Parameters
        ----------
        X : pd.DataFrame
            Must fulfill the input assumptions of the underlying estimator.

        Returns
        -------
        predictions : array-like
        """
        mX = np.vstack([est.predict(X).ravel() for est in self.estimators_]).T

        if self.passthrough is not None:
            mX = np.concatenate((X.values, mX), axis=1)

        return self.meta_estimator_.predict(mX)


