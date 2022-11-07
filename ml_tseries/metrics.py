"""
The :mod:`ml_tseries.metrics` module implements some usefull metrics to be passed as objective function.
*f1_weighted*, *f1_micro*, *f1_macro*, *f1_samples*, *root_mean_squared_error* are some extention of sklearn metrics
that can be passed as *metric_fun* in the bayesian_tuning method of module calibration
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from pandas.api.types import is_list_like
from sklearn.utils import check_array


def f1_weighted(y_true, y_pred, sample_weight=None):
    """
    Calculate metrics for each label, and find their average
    weighted by support (the number of true instances for each label)
    """
    val = f1_score(y_true, y_pred, sample_weight=sample_weight, average='weighted')
    return 'f1_weighted', val, True


def f1_micro(y_true, y_pred, sample_weight=None):
    """
    Calculate metrics globally by counting the total true positives,
     false negatives and false positives.
    """
    val = f1_score(y_true, y_pred, sample_weight=sample_weight, average='micro')
    return 'f1_micro', val, True


def f1_macro(y_true, y_pred, sample_weight=None):
    """
    Calculate metrics for each label, and find their unweighted mean.
    This does not take label imbalance into account.
    """
    val = f1_score(y_true, y_pred, sample_weight=sample_weight, average='macro')
    return 'f1_macro', val, True


def f1_samples(y_true, y_pred, sample_weight=None):
    """
    Calculate metrics for each instance, and find their average
    """
    val = f1_score(y_true, y_pred, sample_weight=sample_weight, average='samples')
    return 'f1_samples', val, True


def root_mean_squared_error(y_true, y_pred, sample_weight=None):
    """
    RMSE, uniform average in case of multi-output
    """
    val = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    return 'rmse', val, False


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    # y_true, y_pred = check_array(y_true.reshape(-1, 1), y_pred)

    # if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100


def smooth_quantile_loss(y_true, y_pred, alpha, delta=1.0, threshold=1.0, var=1.0):
    """
    Quantile loss for XGBoost
    """

    x = y_true - y_pred[:, 0]
    grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
    hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

    grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
                2 * np.random.randint(2, size=len(y_true)) - 1.0) * var
    hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
    return grad, hess


def quantile_loss(y_true, y_pred, alpha, delta):

    x = y_true - y_pred[:, 0]
    grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
    hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
    return grad, hess


def reg_sign(y_true, y_pred):
    """
    Loss function mse/acc that penalizes the distance between the predicted and
    true labels, but also the sign. If acc -> 0, then lost -> inf

    Parameters
    ----------
    y_true : list
        true labels

    y_pred : list
        predicted labels

    Returns
    -------
    reg_sign : float
        mse/accuracy
    """
    if not is_list_like(y_pred):
        y_pred = y_pred.get_label()
    acc = accuracy_score(np.sign(y_true), np.sign(y_pred))
    mse = mean_squared_error(y_true, y_pred)
    return 'reg_sign', mse / (acc + 10e-5), False


def asymmetric_mse(train_data, preds):

    residual = (train_data - preds).astype("float")
    beta = 10.0
    loss = np.where(residual < 0, (residual**2) * beta, residual**2)
    return "asymmetric_mse", np.mean(loss), False


def f1_metric(preds, train_data):

    labels = train_data.get_label()
    return 'f1', f1_score(train_data, preds, average='weighted'), True


def binary_error(preds, train_data):
    """
    custum binary error passed to LGBModel.fit()
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'error', np.mean(labels != (preds > 0.5)), False


def accuracy(preds, train_data):
    """
    custum accuracy passed to LGBModel.fit()
    """
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'accuracy', np.mean(labels == (preds > 0.5)), True


def huber_approx_obj(preds, dtrain):
    """
    implementing huber loss approximation
    """
    d = preds - dtrain
    h = 1 # delta in huber loss formula, can be modified
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def fair_obj(preds, dtrain):
    """implementing fair loss objective
    """
    x = preds - dtrain
    c = 1 # adjustment variable in the formula
    den = abs(x) + c
    grad = c*x / den
    hess = c*c / den ** 2
    return grad, hess


def log_cosh_obj(preds, dtrain):
    """implementing log-cosh objective
    """
    x = preds - dtrain
    grad = np.tanh(x)
    hess = 1 / np.cosh(x)**2
    return grad, hess


def loglikelihood(preds, dtrain):
    """implementing log-likelihood
    """
    labels = dtrain

    preds = 1.0/(1.0 + np.exp(-preds))
    grad = (preds - labels)
    hess = (preds * (1.0 - preds))

    return grad, hess

