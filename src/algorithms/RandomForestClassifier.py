import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import ast

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.metrics import (precision_score, recall_score,
                             f1_score)
from src.algorithms.Algorithm import Algorithm


def default_df():
    return pd.DataFrame({'n_estimators': pd.Series([], dtype=int),
                         'criterion': pd.Series([], dtype=str),
                         'max_depth': pd.Series([], dtype=int),
                         'min_samples_split': pd.Series([], dtype=int),
                         'min_samples_leaf': pd.Series([], dtype=int),
                         'min_weight_fraction_leaf': pd.Series([], dtype=int),
                         'max_features': pd.Series([], dtype=str),
                         'max_leaf_nodes': pd.Series([], dtype=int),
                         'min_impurity_decrease': pd.Series([], dtype=float),
                         'min_impurity_split': pd.Series([], dtype=int),
                         'bootstrap': pd.Series([], dtype=bool),
                         'oob_score': pd.Series([], dtype=bool),
                         'n_jobs': pd.Series([], dtype=int),
                         'random_state': pd.Series([], dtype=int),
                         'verbose': pd.Series([], dtype=int),
                         'warm_start': pd.Series([], dtype=bool),
                         'class_weight': pd.Series([], dtype=str),
                         'ccp_alpha': pd.Series([], dtype=float),
                         'max_samples': pd.Series([], dtype=int)})


def default_params():
    df = default_df()
    return df.append({'n_estimators': 100,
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
                      'max_samples': None}, ignore_index=True
                     )


class RandomForestClassifierAlgorithm(Algorithm):

    def __init__(self, X, y, params: pd.DataFrame = default_params()):
        self.clf: RFC = RFC()
        self.n_estimators = int(params['n_estimators']) if not pd.isna(params['n_estimators']) else 100
        self.criterion = params['criterion']
        self.max_depth = params['max_depth']if not pd.isna(params['max_depth']) else 2
        self.min_samples_split = float(params['min_samples_split']) if (
                    params['min_samples_split'] <= 1) else \
            (int(params['min_samples_leaf'])) if (params['min_samples_leaf'] > 1) else (int(2))
        self.min_samples_leaf = float(params['min_samples_leaf']) if (
                    params['min_samples_leaf'] < 1) else \
            (int(params['min_samples_leaf'])) if (params['min_samples_leaf'] >= 1) else int(1)
        self.min_weight_fraction_leaf = params['min_weight_fraction_leaf'] if not pd.isna(
            params['min_weight_fraction_leaf']) else 0.0
        self.max_features = params['max_features'] if not pd.isna(params['max_features']) else "auto"
        self.max_leaf_nodes = params['max_leaf_nodes'] if not pd.isna(params['max_leaf_nodes']) else 1000
        self.min_impurity_decrease = params['min_impurity_decrease']
        self.min_impurity_split = params['min_impurity_split'] if not pd.isna(
            params['min_impurity_split']) else 1.0
        self.bootstrap = params['bootstrap'] if not pd.isna(params['bootstrap']) else True
        self.oob_score = params['oob_score'] if not pd.isna(params['oob_score']) else False
        self.n_jobs = int(params['n_jobs']) if not pd.isna(params['n_jobs']) else -1
        self.random_state = params['random_state'] if not pd.isna(params['random_state']) else 0
        self.verbose = int(params['verbose']) if not pd.isna(params['verbose']) else 0
        self.warm_start = params['warm_start'] if not pd.isna(params['warm_start']) else False
        self.class_weight = ast.literal_eval(params['class_weight']) if not (
                    pd.isna(params['class_weight'])) else "balanced"
        self.ccp_alpha = params['ccp_alpha'] if not pd.isna(params['ccp_alpha']) else 0.0
        self.max_samples = params['max_samples'] if not pd.isna(params['max_samples']) else X.shape[0]
        self.make_classifier()
        self.clf.fit(X, y)

    def gen_df_params(self):
        default_params().to_csv('RandomForestClassifier_params.csv', index_label='index')

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
                       #min_impurity_split=self.min_impurity_split,  # Deprecated
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

    def metrics(self, X_test, y_test, dossier, name, format):
        y_pred = self.clf.predict(X_test)
        plot_roc_curve(self.clf, X_test, y_test)
        plt.savefig(dossier + '/' + 'roc_curve_' + name + '.' + format)
        plt.close()
        plot_confusion_matrix(self.clf, X_test, y_test)
        plt.savefig(dossier + '/' + 'conf_matrix_' + name + '.' + format)
        plt.close()
        predictions = self.clf.predict(X_test)
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.savefig(dossier + '/' + 'precrecalldisp_' + name + '.' + format)
        result_df = pd.DataFrame.from_dict({
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred)
        }, orient='index').T
        result_df.to_csv(dossier + '/' + 'performances.csv', index_label='index')
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

    def __str__(self):
        return self.name
