import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def split_data(df_data):
    df_data = df_data.copy()
    df_data['income_cat'] = pd.cut(
        df_data['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in split.split(df_data, df_data['income_cat']):
        df_train = df_data.loc[train_idx]
        df_test = df_data.loc[test_idx]
        df_train.drop('income_cat', axis=1, inplace=True)
        df_test.drop('income_cat', axis=1, inplace=True)
        return df_train, df_test


def split_train_test(df_data):
    df_train, df_test = split_data(df_data)
    return __split_x_y(df_train), __split_x_y(df_test)


def __split_x_y(df_data):
    return df_data.drop('median_house_value', axis=1), df_data['median_house_value']
