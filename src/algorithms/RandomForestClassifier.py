import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RFC

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


class RandomForestClassifierAlgorithm(Algorithm):

    def __init__(self, X, y, params: pd.DataFrame = default_params()):
        #        super(RandomForestClassifier, self).__init__(name)

        self.clf: RFC = RFC()
        self.n_estimators = params['n_estimators'][0]
        self.criterion = params['criterion'][0]
        self.max_depth = params['max_depth'][0]
        self.min_samples_split = params['min_samples_split'][0]
        self.min_samples_leaf = params['min_samples_leaf'][0]
        self.min_weight_fraction_leaf = params['min_weight_fraction_leaf'][0]
        self.max_features = params['max_features'][0]
        self.max_leaf_nodes = params['max_leaf_nodes'][0]
        self.min_impurity_decrease = params['min_impurity_decrease'][0]
        self.min_impurity_split = params['min_impurity_split'][0]
        self.bootstrap = params['bootstrap'][0]
        self.oob_score = params['oob_score'][0]
        self.n_jobs = params['n_jobs'][0]
        self.random_state = params['random_state'][0]
        self.verbose = params['verbose'][0]
        self.warm_start = params['warm_start'][0]
        self.class_weight = params['class_weight'][0]
        self.ccp_alpha = params['ccp_alpha'][0]
        self.max_samples = params['max_samples'][0]
        self.make_classifier()
        self.clf.fit(X, y)


    def gen_df_params(self):
        default_params().to_csv('RandomForestClassifier_params.csv')

    def make_classifier(self):
        self.clf = RFC(n_estimators=self.n_estimators,
                       criterion=self.criterion,
                       max_depth=self.max_depth,
                       min_samples_split=self.min_samples_split,
                       min_samples_leaf=self.min_samples_leaf,
                       min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                       max_features=self.max_features,
                       max_leaf_nodes=self.max_leaf_nodes,
                       min_impurity_decrease=self.min_impurity_decrease,
                       min_impurity_split=self.min_impurity_split,
                       bootstrap=self.bootstrap,
                       oob_score=self.oob_score,
                       n_jobs=self.n_jobs,
                       random_state=self.random_state,
                       verbose=self.verbose,
                       warm_start=self.warm_start,
                       class_weight=self.class_weight,
                       ccp_alpha=self.ccp_alpha,
                       max_samples=self.max_samples
                       )

    def __str__(self):
        return self.name
