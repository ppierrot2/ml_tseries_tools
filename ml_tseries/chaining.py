"""The :mod:`ml_tseries.chaining` module implements a multi-output and multi-step estimator from a base
estimator using chaining : an estimator is iteratively trained with previous estimator prediction
"""

import pandas as pd
import numpy as np
from .cross_validation import PurgedKFold, cross_val_predictions
from sklearn.base import clone


class BaseChainTS:
    """En cours de test"""
    def __init__(self, base_estimator, prediction_steps=1, cv_gen=None, n_jobs=1, random_state=None):
        self.base_estimator = base_estimator
        self.prediction_steps = prediction_steps
        self.n_jobs = n_jobs
        self.random_state = random_state
        if cv_gen is not None:
            self.cv_gen = cv_gen
        else:
            self.cv_gen = PurgedKFold(embargo=pd.Timedelta(0), overlap=pd.Timedelta(0))

    def fit(self, X, Y, sample_weight=None, **fit_params):

        X_aug = X.copy()
        self.estimators_ = list()

        for step in range(self.prediction_steps):

            estimators_ = [clone(self.base_estimator)
                                for _ in range(Y.shape[1])]
            if step > 0:  # condition might be removed if Y is not initially lagged
                Y = Y.iloc[1:]
                X_aug = X_aug.iloc[:-1]
            Y.index = X_aug.index

            for chain_idx, estimator in enumerate(estimators_):
                y = Y.iloc[:, chain_idx]
                estimator.fit(X_aug, y, sample_weight=sample_weight, **fit_params)
                if (step < self.prediction_steps - 1) or (chain_idx < len(estimators_) - 1):
                    cv_preds = cross_val_predictions(self.base_estimator, X_aug, y,
                                                     cv_gen=self.cv_gen, sample_weight=sample_weight,
                                                     n_jobs=self.n_jobs, **fit_params)
                    X_aug = pd.concat([X_aug, cv_preds], axis=1)

            self.estimators_.append(estimators_)
        return self

    def predict(self, X):
        predictions = list()
        X_aug = X.copy()
        for estimators in self.estimators_:
            step_predictions = list()
            for estimator in estimators:
                preds = estimator.predict(X_aug)
                step_predictions.append(preds)
                X_aug = pd.concat([X_aug, pd.Series(preds.ravel(), index=X_aug.index)], axis=1)
            predictions.append(step_predictions)
        predictions = np.array(predictions)
        predictions = np.moveaxis(predictions, -1, 0)
        return predictions


class RegresssorChainTS(BaseChainTS):
    pass


class ClassifierChainTS(BaseChainTS):

    def predict_proba(self, X):
        # ToDo
        raise NotImplementedError