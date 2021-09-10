import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2


def get_default_params():
    return pd.DataFrame.from_dict({'k': 5}, orient='index').T

def get_params_from_file(nom_fichier):
    params_csv = pd.DataFrame()
    df = pd.DataFrame({'k': pd.Series([], dtype=int)})
    params_csv = params_csv.append(pd.read_csv(nom_fichier, index_col='index'))
    for it_i, row in params_csv.iterrows():
        for it, col in row.iteritems():
            col = None if col == 'nan' else col
            df.at[it_i, it] = pd.to_numeric(col, errors='ignore')
    df['k'] = df['k'].astype(int)
    return df

class Select_KBest:
    
    def __init__(self, X, y, params: pd.DataFrame = get_default_params()):
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_plus = scaler.transform(X)
        selector = SelectKBest(chi2, k=params['k'])
        selector.fit(X_plus, y)
        X_new = selector.transform(X)
        X_new.shape
        self.columns = [list(X.columns)[it] for it, x in enumerate(selector.get_support().tolist()) if x]
        