import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, add_bedrooms_per_room=True):
        self.columns = columns
        self.col_idx = dict((col, idx) for idx, col in enumerate(self.columns))
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, self.col_idx['total_rooms']] / X[:, self.col_idx['households']]
        population_per_household = X[:, self.col_idx['population']] / X[:, self.col_idx['households']]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.col_idx['total_bedrooms']] / X[:, self.col_idx['total_rooms']]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
