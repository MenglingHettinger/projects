%%writefile custom_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, Imputer
import sys

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


class CustomModel(GradientBoostingClassifier):

    def __init__(self, loss='deviance', learning_rate=0.05, n_estimators=100,
            subsample=1.0, min_samples_split=4,
            min_samples_leaf=1,
            max_depth=3, init=None, random_state=None,
            max_features=None, verbose=0,Imputer
            max_leaf_nodes=None, warm_start=False):

        super(CustomModel, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start)

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

        X = MultiColumnLabelEncoder(columns = categorical_list).fit_transform(X)

        # Training model
        super(CustomModel, self).fit(X, Y)
        return self

    def predict(self, X):
        categorical_list = []
        for column in X:
            if X[column].dtype == np.dtype('O'):
                categorical_list.append(column)

        X = MultiColumnLabelEncoder(columns = categorical_list).fit_transform(X)

        return super(CustomModel, self).predict(X)


    
