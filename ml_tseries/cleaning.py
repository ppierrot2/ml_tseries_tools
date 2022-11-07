import pandas as pd
import numpy as np


def remove_outliers_by_values(data, feature_list, thresholds_list):
    """remove outliers on features of feature_list with resp. thresholds
     in threshold_list
     :return: pd.DataFrame
     """
    for i in range(len(feature_list)):
        data = data[data[feature_list[i]] < thresholds_list[i]]
        data = data[data[feature_list[i]] > -thresholds_list[i]]
    return data


def replace_outliers(df, quantile_thold=0.001, fraction_thold=0.01, handle='mean'):
    """
    handle outlier values. Find values that are above a quantile threshold, then
    consider outliers if they represent less than a certain fraction threshold of the data

    :param df: (pd.DataFrame): data to clean up
    :param quantile_thold: (0 < int < 1): quantile for outlier suspicion
    :param fraction_thold: (0 < int < 1): the fraction threshold
    :param handle: (str): either 'mean', 'quantile', or nan : the way to replace
                    outliers
    :return: cleaned dataframe
    """
    nrows = df.shape[0]
    for col in df.columns:
        idx_up_outliers = []
        idx_down_outliers = []

        # Upper quantile:
        upper_val = df[col].quantile(1 - quantile_thold)
        suspected_ouliers = df[df[col] >= upper_val][col]
        nupper = suspected_ouliers.count()
        # Consider them outliers only if less than 1% registers
        if nupper / nrows < fraction_thold:
            idx_up_outliers += suspected_ouliers.index.tolist()

        # Lower bound compute:
        lower_val = df[col].quantile(quantile_thold)
        suspected_ouliers = df[df[col] < lower_val][col]
        nlower = suspected_ouliers.count()
        if nlower / nrows < fraction_thold:
            idx_down_outliers += suspected_ouliers.index.tolist()

        if handle == 'quantile':
            for e in idx_up_outliers:
                df.at[e, col] = upper_val
            for e in idx_down_outliers:
                df.at[e, col] = lower_val

        elif handle == 'mean':
            idx_outliers = idx_up_outliers + idx_down_outliers
            for e in idx_outliers:
                df.at[e, col] = df[col].mean()
        elif handle == 'nan':
            idx_outliers = idx_up_outliers + idx_down_outliers
            for e in idx_outliers:
                df.at[e, col] = np.nan
    return df


def remove_na_variable(df, threshold=0.15):
    """remove columns that contains a percentage higher than defined
     threshold of nan values
     """
    select = [
        col for col in df.columns if
        df[col].isnull().sum() / len(df) < threshold
    ]
    df = df[select]
    return df


def remove_small_values(df, col, quantile_thresh=0.05, sign='pos'):
    """
    remove small pos or neg values from a specific columns of a pd.DataFrame

    :param df: pd.DataFrame, dataframe on which to remove small values
    :param col: str, name of the column to be considered
    :param relative_value: fraction of the col mean below which to drop entry
    :param sign: weather to do on positive or negative values of col
    :return: cleaned dataframe
    """
    if quantile_thresh == 0:
        return df
    else:
        if sign == 'pos':
            pos_df = df[df[col] >= 0]
            thresh = pos_df[col].quantile(quantile_thresh)
            small_values = pos_df[pos_df[col] <= thresh][col]
            idx_to_remove = small_values.index.tolist()
            if small_values.count()/len(pos_df) <= 0.05: df = df.drop(
                idx_to_remove)

        elif sign == 'neg':
            neg_df = df[df[col] <= 0]
            thresh = neg_df[col].quantile(1 - quantile_thresh)
            small_values = neg_df[neg_df[col] >= thresh][col]
            idx_to_remove = small_values.index.tolist()
            if small_values.count() / len(neg_df) <= 0.05: df = df.drop(
                idx_to_remove)
        else:
            raise ValueError('invalid sign')

    return df
