import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def get_default_params():
    return pd.DataFrame.from_dict({'n_split':1,
                                   'test_size':0.3,
                                   'random_state':0})


class StratifiedShuffleSplit:

    def __init__(self, X, y, params: pd.DataFrame=get_default_params()):
        sss = StratifiedShuffleSplit(n_splits=params['n_splits'],
                                     test_size=params['test_size'],
                                     random_state=params['random_state'])
        sss.get_n_splits(X, y)
        for train_index, test_index in sss.split(X, y):
            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]