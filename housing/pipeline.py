import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.preprocessing import StandardScaler


def add_extra_feature(X, columns, add_bedrooms_per_room):
    rooms_idx = columns.index('total_rooms')
    bedrooms_idx = columns.index('total_bedrooms')
    population_idx = columns.index('population')
    household_idx = columns.index('households')

    rooms_per_household = X[:, rooms_idx] / X[:, household_idx]
    population_per_household = X[:, population_idx] / X[:, household_idx]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_idx] / X[:, rooms_idx]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]


def build_add_extra_feature_transformer(columns, add_bedrooms_per_room=True):
    return FunctionTransformer(add_extra_feature, validate=False,
                               kw_args={'columns': columns, 'add_bedrooms_per_room': add_bedrooms_per_room})


def build_numeric_feature_pipeline(columns):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('feature_adder', build_add_extra_feature_transformer(columns)),
        ('std_scaler', StandardScaler()),
    ])
    return numeric_pipeline


def build_preprocess_pipeline(columns):
    numeric_columns = columns.copy()
    numeric_columns.remove('ocean_proximity')
    categorical_columns = ['ocean_proximity']
    full_pipeline = ColumnTransformer([
        ('numeric', build_numeric_feature_pipeline(numeric_columns), numeric_columns),
        ('categorical', OneHotEncoder(), categorical_columns),
    ])
    return full_pipeline
