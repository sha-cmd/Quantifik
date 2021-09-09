import numpy as np
import pandas as pd
from src.algorithms.Algorithm import Algorithm


def default_params():
    return pd.DataFrame.from_dict({'n_estimators': 100,
                                   'criterion': 'gini',
                                   'max_depth': None,
                                   'min_samples_split': 2,
                                   'min_samples_leaf': 1,
                                   'min_weight_fraction_leaf': 0.0,
                                   'max_features': 'auto',
                                   'max_leaf_nodes': None,
                                   'min_impurity_decrease': 0.0,
                                   'min_impurity_split': None,
                                   'bootstrap': True,
                                   'oob_score': False,
                                   'n_jobs': None,
                                   'random_state': None,
                                   'verbose': 0,
                                   'warm_start': False,
                                   'class_weight': None,
                                   'ccp_alpha': 0.0,
                                   'max_samples': None}, orient='index').T


class RandomForestClassifier(Algorithm):

    def __init__(self, name, params: pd.DataFrame = default_params()):
        super(RandomForestClassifier, self).__init__(name)
        for it, row in params.iterrows():
            #self.type = row['Type']
            self.n_estimators = row['n_estimators']
            self.criterion = row['criterion']
            self.max_depth = row['max_depth']
            self.min_samples_split = row['min_samples_split']
            self.min_samples_leaf = row['min_samples_leaf']
            self.min_weight_fraction_leaf = row['min_weight_fraction_leaf']
            self.max_features = row['max_features']
            self.max_leaf_nodes = row['max_leaf_nodes']
            self.min_impurity_decrease = row['min_impurity_decrease']
            self.min_impurity_split = row['min_impurity_split']
            self.bootstrap = row['bootstrap']
            self.oob_score = row['oob_score']
            self.n_jobs = row['n_jobs']
            self.random_state = row['random_state']
            self.verbose = row['verbose']
            self.warm_start = row['warm_start']
            self.class_weight = row['class_weight']
            self.ccp_alpha = row['ccp_alpha']
            self.max_samples = row['max_samples']

    def gen_df_params(self):
        default_params().to_csv('RandomForestClassifier_params.csv')

    def __str__(self):
        return self.name