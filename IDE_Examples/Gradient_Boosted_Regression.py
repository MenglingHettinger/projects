%%writefile custom_model.py

import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self, missing_value=np.nan, cat_imputer=None, num_imputer='mean'):
        
        self.missing_value = np.nan

        self.cat_imputer = cat_imputer
        self.num_imputer = num_imputer
        super(DataFrameImputer, self).__init__()

    def fit(self, X, y=None):
        fill = []
        for column in X:

            if X[column].dtype == np.dtype('O'):
                if self.cat_imputer == None:
                    fill.append(X[column])
                elif self.cat_imputer == "mode":
                    fill.append(X[column].value_counts().index[0])

                else:
                    raise InvalidImputer("Invalid categorical imputer. %s is invalid.  Please use None or 'mode'." % self.cat_imputer)

            else:
                if self.num_imputer == "mean":
                    fill.append(X[column].mean())
                elif self.num_imputer == "median":
                    fill.append(X[column].median())
                elif self.num_imputer == "mode":
                    fill.append(X[column].value_counts().index[0])
                else:
                    raise InvalidImputer("Invalid numerical imputer. %s is invalid.  Please use 'mode', 'median' or 'mean'." % self.num_imputer)

        self.fill = pd.Series(fill, index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

class MultiColumnLabelEncoder:

    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


class CustomModel(GradientBoostingRegressor):

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, 
        min_samples_leaf=1,  max_depth=3, init=None, random_state=None, 
        max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False):

        super(CustomModel, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            subsample=subsample, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth, init=init, 
            random_state=random_state, max_features=max_features,
            alpha=alpha, verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start)

    def fit(self, X, Y, extras=None):

        categorical_list = []
        for column in X:
            if X[column].dtype == np.dtype('O'):
                categorical_list.append(column)
        X = DataFrameImputer().fit_transform(X)
        X = MultiColumnLabelEncoder(columns = categorical_list).fit_transform(X)

        # Training model
        super(CustomModel, self).fit(X, Y)
        return self

    def predict(self, X):
        categorical_list = []
        for column in X:
            if X[column].dtype == np.dtype('O'):
                categorical_list.append(column)
        X = DataFrameImputer().fit_transform(X)
        X = MultiColumnLabelEncoder(columns = categorical_list).fit_transform(X)

        return super(CustomModel, self).predict(X)


    
