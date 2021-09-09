import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2


def get_default_params():
    return pd.DataFrame.from_dict({'k': 5}, orient='index').T


class Select_KBest:
    
    def __init__(self, X, y, params: pd.DataFrame = get_default_params()):
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_plus = scaler.transform(X)
        selector = SelectKBest(chi2, k=params['k'][0])
        selector.fit(X_plus, y)
        X_new = selector.transform(X)
        X_new.shape
        self.columns =  [list(X.columns)[it] for it, x in enumerate(selector.get_support().tolist()) if x]
        