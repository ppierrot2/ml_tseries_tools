import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from sklearn.metrics import (confusion_matrix,
                             )
from sklearn.utils.multiclass import unique_labels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


class MissingDataError(Exception):
    """raise exception if missing data"""
    pass


def plot_correlations(data, method='kendall'):
    """
    Parameters
    ----------
    data: pd.DataFrame
    method: ‘pearson’, ‘kendall’, ‘spearman’

    Returns
    -------

    """
    colormap = plt.cm.RdBu
    plt.figure(figsize=(25, 20))
    plt.title('feature correlation', y=1.05, size=16)

    mask = np.zeros_like(data.corr(method=method))
    mask[np.triu_indices_from(mask)] = True

    svm = sns.heatmap(data.corr(), mask=mask, linewidths=0.1, vmax=1.0, vmin=-1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    return plt.show()


def missing_values(dataframe):
    """plot missing values for one dataframe"""
    null_counts = dataframe.isnull().sum()/len(dataframe)
    plt.figure(figsize=(20,6))
    plt.xticks(np.arange(len(null_counts)) + 0.5,
               null_counts.index,
               rotation='vertical')
    plt.ylabel('fraction of rows with missing data')
    plt.bar(np.arange(len(null_counts)), null_counts)
    return plt.show()


def missing_values_multi_data(dataframes_dic):
    """
    plot missing values for multiple dataframe with same columns index
    :param dataframes_dic: dict of dataframes
    :return:
    """
    nb_feature = dataframes_dic[next(iter(dataframes_dic))].values.shape[1]
    left = np.zeros(nb_feature)

    colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85,
                                                len(dataframes_dic.keys())))
    plt.figure(figsize=(15, int(nb_feature/5)))
    for i, val in enumerate(zip(dataframes_dic.keys(), colors)):
        df = dataframes_dic[val[0]]
        null_counts = df.isnull().sum()/len(df)
        plt.barh(np.arange(len(null_counts)),
                 null_counts,
                 left=left,
                 tick_label=str(val[0]),
                 color=val[1])
        left = left + null_counts
        left = left.drop(left.index.difference(null_counts.index))

    plt.yticks(np.arange(len(null_counts))+0.5,
               null_counts.index,
              )
    xt = np.arange(2*len(dataframes_dic.keys()))
    xt_labels = ['0%', '50%', '100%']
    xt_labels.extend([' ' for i in list(xt[2:])])
    plt.xticks(xt, xt_labels)
    plt.xlabel('percentage of missing value per time step')
    plt.legend(dataframes_dic.keys(), loc=1)
    plt.title(' Feature availability per trading time step')
    plt.grid(axis='x')
    return plt.show()


def is_stationnary(series, maxlag=None, level='5%', verbose=False):
    """
    Dickey Fuller test strategy
    H0 : there is a unit root

    Returns True if the series is stationary.
    This function implements a Dickey-Fuller test strategy
    """

    # We do not use model 3 because stationarity in trend has no reality in economics
    if maxlag == None:
        autolag = 'AIC'
    else:
        autolag = None

    if verbose:
        print('Model 2 (constant)')
    # Unit root test for model 2
    adf = adfuller(x=series, maxlag=maxlag, autolag=autolag,
                   regression='c', regresults=True)  # constant
    p_value = adf[1]
    test_statistic = adf[0]
    critical_value = adf[2][level]
    tvalues = adf[3].resols.tvalues  # constant, trend, rho
    threshold = int(level[0]) / 100  # threshold for p-value

    if p_value < threshold:  # We can reject H0

        if np.abs(tvalues[0]) > 1.96:  # constant test, to know if it was the 'right' model
            if verbose:
                print('t-test on the constant\nt-value : {}'.format(tvalues[1]))
            return True, tvalues[1]  # It was the right model so the series is stationnary

        else:
            if verbose:
                print('Model 1 (no constant, no trend)')
            adf = adfuller(x=series,
                           maxlag=maxlag,
                           autolag=autolag,
                           regression='nc',
                           regresults=True)  # no constant, no trend
            p_value = adf[1]
            test_statistic = adf[0]
            critical_value = adf[2][level]
            tvalues = adf[3].resols.tvalues

            return p_value < threshold, "p-value : {}".format(p_value)

    else:  # We fail to reject H0
        if np.abs(test_statistic) > np.abs(critical_value):
            return True, "p-value : {}".format(p_value)
    if verbose:
        print('Model 1 (no constant, no trend)')

    adf = adfuller(x=series, maxlag=maxlag, autolag=autolag,
                   regression='nc', regresults=True)  # no constant, no trend
    p_value = adf[1]
    tvalues = adf[3].resols.tvalues

    return p_value < threshold, "p-value : {}".format(p_value)


def hurst_estimator(x):
    """
    Estimator of the hurst exponent H.

    This is a measure of diffusion spead:
    :math:`< |z(t + \\tau) - z(t)|^2 > ~ \\tau^{2H}`

    Parameters
    ----------
    x : array_like
        input time series

    Returns
    -------
    estimated hurst parameter :
        if value > 0.5 : persistent
        if value < 0.5 : anti-persistent
        if ==0.5 : pure : pure Random walk (GBM)

    Raises
    ------
    ValueError
        when len(data) < 100
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.values
    if x.shape[0] < 100:
        raise ValueError('input data must ba at least of lenght 100')
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), deg=1)
    return poly[0] * 2.0


def stationarity_test(dataframe, maxlag=10, level='5%', verbose=False):

    """do stationarity test and print results of all columns of dataframe"""

    for c in dataframe.columns:
        print('_______________________________')
        try:
            stat = is_stationnary(series=dataframe[c].values,
                                  maxlag=maxlag,
                                  level=level,
                                  verbose=verbose)
            print(c + ': ', stat)
        except MissingDataError:
            print(c, 'missing values')


def plot_confusion_matrix(y_true, y_pred, classes, fig, ax,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def acf(data, lags=10):
    """
    plot acf and pacf for each columns of data
    :param data: (pd.DataFrame) time series dataframe
    :param lags: (int)
    :return:
    """

    for f in data.columns:
        dt = data[f]
        dtd = np.diff(dt, n=1)
        plt.figure(figsize=(20, 3))
        plt.subplot(131)
        plt.plot(dt)
        plt.title(f)
        plt.subplot(132)
        plot_acf(dt, lags=lags, ax=plt.gca())
        plt.title(f)
        plt.subplot(133)
        plot_acf(dtd, lags=lags, ax=plt.gca())
        plt.title(f)

    return plt.show()


def is_extreme_naive(hist_data, new_val, thresh=0.05):
    """naive extreme value detection"""
    qt_sup = hist_data.quantile(1 - thresh, interpolation='higher')
    qt_inf = hist_data.quantile(thresh, interpolation='lower')
    return (new_val > qt_sup) or (new_val < qt_inf)


def critical_val_grubbs(data, alpha=0.05):
    """
    compute critical value for Grubbs test (outlier detection for normally
    distributed data)

    :param data (pd.Series or list-like): the time-series to compute the
                                        critical valiue
    :param alpha (float): the level
    :return: critical value for Grubbs test
    """
    t_dist = scipy.stats.t.ppf(1-alpha/(2 * len(data)), len(data) - 2)
    num = (len(data) - 1) * np.sqrt(np.square(t_dist))
    den = np.sqrt(len(data)) * np.sqrt(len(data) - 2 + np.square(t_dist))
    return num / den


def is_extreme_grubbs(hist_data, new_val, thresh=0.05):
    """
    test if new entry of a serie is outlier

    :param hist_data: (pd.Series)
    :param new_val: (float)
    :param thresh:
    :return:
    """
    def has_outlier(data):
        val_m_avg = data - data.max()
        abs_val_m_avg = val_m_avg.abs()
        suspected_val = (abs_val_m_avg / data.std()).max()
        if suspected_val > critical_val_grubbs(data, alpha=thresh):
            return True
        else:
            return False

    if not has_outlier(hist_data):
        if hist_data.min() <= new_val <= hist_data.max():
            return False
        else:
            return has_outlier(hist_data.append(new_val))
    #else:
     #   completed_data = hist_data.append(new_val)


def kl_div(pk, qk):
    """kl divergence for real-time monitoring"""
    ent = scipy.stats.entropy(pk, qk, base=2)
    abs_diff = np.abs(pk-qk)
    max_deviation = np.max(abs_diff)
    return ent, abs_diff, max_deviation


def knn_distance(point, sample, k):
    """ Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample` """
    norms = np.linalg.norm(sample - point, axis=1)
    return np.sort(norms)[k]


def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert(len(s1.shape) == len(s2.shape) == 2)
    # Check dimensionality of sample is identical
    assert(s1.shape[1] == s2.shape[1])


def kl_estimator(s1, s2, k=1):
    """ KL-Divergence estimator using MC (numpy) k-NN

    Parameters
    ----------
    s1 : array-like of shape (N_1,D)
        Sample drawn from distribution P
    s2 : array-like of shape (N_2,D)
        Sample drawn from distribution Q
    k : int
        Number of neighbours considered (default 1)

    Returns
    -------
    estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    D = np.log(m / (n - 1))
    d = float(s1.shape[1])

    for p1 in s1:
        nu = knn_distance(p1, s2, k-1)  # -1 because 'p1' is not in 's2'
        rho = knn_distance(p1, s1, k)
        D += (d/n)*np.log(nu/rho)
    return D


def get_bsadf(logP, minSL, constant, lags):

    y, x = getYX(logP, constant=constant, lags=lags)
    startPoints, bsadf, allADF = range(0,y.shape[0]+lags-minSL+1), None, []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = getBetas(y_, x_)
        bMean_, bStd_ = bMean_[0,0], bStd_[0,0]**.5
        allADF.append(bMean_/bStd_)
    if allADF[-1] > bsadf:
        bsadf = allADF[-1]
    out = {'Time': logP.index[-1], 'gsadf': bsadf}
    return out


def getYX(series, constant, lags):

    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1:-1, 0]
    y = series_.iloc[-x.shape[0]:].values
    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
        if constant == 'ctt':
            x = np.append(x, trend ** 2, axis=1)
    return y, x


def lagDF(df0, lags):

    df1 = pd.DataFrame()
    if isinstance(lags, int): lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how='outer')
    return df1


def getBetas(y,x):

    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv, xy)
    err = y - np.dot(x, bMean)
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    return bMean, bVar
