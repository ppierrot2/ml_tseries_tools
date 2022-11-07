import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from ml_tseries.feature_importance import(mean_decrease_accuracy,
                                          plot_feature_importance,
                                          single_feature_importance)
from ml_tseries.cross_validation import (PurgedWalkForward,
                                         PurgedKFold)
from sklearn.metrics import mean_squared_error, log_loss, f1_score, accuracy_score

date_range = pd.date_range('1/1/2012', '1/1/2019')
df = pd.DataFrame(index=date_range)
df['feature1'] = np.random.randint(0, 100, len(date_range))
df['feature2'] = np.random.randint(0, 100, len(date_range))
df['feature3'] = np.random.randint(0, 100, len(date_range))
df['feature4'] = np.random.randint(0, 100, len(date_range))
classification_target = pd.Series(np.random.randint(0,2, len(date_range)), index=date_range)
regression_target = pd.Series(np.random.normal(0, 0.02, len(date_range)), index=date_range)


def test_mda_reg():
    X, y = df, regression_target
    n_repeat_mda = 2
    importance_df = mean_decrease_accuracy(model=RandomForestRegressor(),
                                           X=X,
                                           y=y,
                                           cv_gen=PurgedKFold(),
                                           n_repeat_mda=n_repeat_mda,
                                           scoring=mean_squared_error,
                                           clustered_subsets=None,
                                           sample_weight=y.copy(),
                                           random_state=42,
                                           n_jobs=1)
    assert type(importance_df) == pd.DataFrame
    assert importance_df.shape[0] == X.shape[1]


def test_mda_reg_cluster():
    X, y = df, regression_target
    clustered_subsets = [['feature1', 'feature3'], ['feature2', 'feature4']]
    n_repeat_mda = 2
    importance_df = mean_decrease_accuracy(model=RandomForestRegressor(),
                                           X=X,
                                           y=y,
                                           cv_gen=PurgedKFold(),
                                           n_repeat_mda=n_repeat_mda,
                                           scoring=mean_squared_error,
                                           clustered_subsets=clustered_subsets,
                                           sample_weight=y.copy(),
                                           random_state=42,
                                           n_jobs=1)
    assert type(importance_df) == pd.DataFrame
    assert importance_df.shape[0] == X.shape[1]
    assert importance_df.loc['feature1']['mean'] == importance_df.loc['feature3']['mean']
    assert importance_df.loc['feature1']['std'] == importance_df.loc['feature3']['std']


def test_mda_class():
    X, y = df, classification_target
    n_repeat_mda = 1
    importance_df = mean_decrease_accuracy(model=RandomForestClassifier(),
                                           X=X,
                                           y=y,
                                           cv_gen=PurgedWalkForward(),
                                           n_repeat_mda=n_repeat_mda,
                                           scoring=f1_score,
                                           clustered_subsets=None,
                                           sample_weight=y.copy(),
                                           random_state=42,
                                           n_jobs=1)
    assert type(importance_df) == pd.DataFrame
    assert importance_df.shape[0] == X.shape[1]


def test_mda_plot():
    X, y = df, regression_target
    clustered_subsets = [['feature1', 'feature3'], ['feature2', 'feature4']]
    n_repeat_mda = 1
    importance_df = mean_decrease_accuracy(model=RandomForestRegressor(),
                                           X=X,
                                           y=y,
                                           cv_gen=PurgedKFold(),
                                           n_repeat_mda=n_repeat_mda,
                                           scoring=mean_squared_error,
                                           clustered_subsets=clustered_subsets,
                                           sample_weight=y.copy(),
                                           random_state=42,
                                           n_jobs=1)

    assert type(importance_df) == pd.DataFrame
    assert importance_df.shape[0] == X.shape[1]
    assert importance_df.loc['feature1']['mean'] == importance_df.loc['feature3']['mean']
    assert importance_df.loc['feature1']['std'] == importance_df.loc['feature3']['std']

    plot_feature_importance(importance_df)


def test_sfi():
    X, y = df, classification_target

    importance_df = single_feature_importance(model=RandomForestClassifier(),
                                              X=X,
                                              y=y,
                                              cv_gen=PurgedKFold(n_splits=5),
                                              scoring=log_loss,
                                              sample_weight=regression_target,
                                              n_jobs=1)

    assert type(importance_df) == pd.DataFrame
    assert importance_df.shape[0] == X.shape[1]

    plot_feature_importance(importance_df)