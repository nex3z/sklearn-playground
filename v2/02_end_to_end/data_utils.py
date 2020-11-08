import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from combined_attributes_adder import CombinedAttributesAdder


def read_data(file_path='./housing.csv'):
    df_data = pd.read_csv(file_path)
    return df_data


def train_test_split(df_data=None, seed=42):
    if df_data is None:
        df_data = read_data()

    df_data['income_cat'] = pd.cut(
        df_data['median_income'],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for train_index, test_index in split.split(df_data, df_data['income_cat']):
        df_train = df_data.loc[train_index].drop('income_cat', axis=1)
        df_test = df_data.loc[test_index].drop('income_cat', axis=1)
        return split_xy(df_train), split_xy(df_test)


def split_xy(df_data):
    X = df_data.drop('median_house_value', axis=1)
    y = df_data['median_house_value']
    return X, y


def build_pipeline(df_X):
    num_attribs = list(df_X.columns)
    print(f"num_attribs = {num_attribs}")
    num_attribs.remove('ocean_proximity')
    cat_attribs = ['ocean_proximity']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder(num_attribs)),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs),
    ])

    return full_pipeline
