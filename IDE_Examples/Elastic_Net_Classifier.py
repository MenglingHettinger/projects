%%writefile custom_model.py

import sys

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
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


class CustomModel(ElasticNet):

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, 
        normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, 
        warm_start=False, positive=False):

        super(CustomModel, self).__init__(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, normalize=normalize, 
            precompute=precompute, max_iter=max_iter, copy_X=copy_X, tol=tol, warm_start=warm_start, 
            positive=positive)

    def preprocessing_Y(self, X, Y):
        # Remove NA target

        Y = pd.Series(Y)
        Y = Y.dropna()
        index_list = Y.index.values.tolist()
        X = X.ix[index_list]

        # Check to ensure that it is a binary classification problem
        unique_values  = len(set(Y.ravel()))
        if unique_values != 2:
            sys.exit("Supports binary classification only")
        return X, Y


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


    
