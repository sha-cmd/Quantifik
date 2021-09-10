import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import ast
import scikitplot as skplt

from src.utils.logger import log_init as log
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.metrics import (precision_score, recall_score,
                             f1_score)
from sklearn.inspection import permutation_importance

log = log()


def default_logistic_regression():
    return pd.DataFrame({'penalty': pd.Series([], dtype=str),
                         'dual': pd.Series([], dtype=bool),
                         'tol': pd.Series([], dtype=float),
                         'C': pd.Series([], dtype=float),
                         'fit_intercept': pd.Series([], dtype=bool),
                         'intercept_scaling': pd.Series([], dtype=int),
                         'class_weight': pd.Series([], dtype=str),
                         'random_state': pd.Series([], dtype=int),
                         'solver': pd.Series([], dtype=str),
                         'max_iter': pd.Series([], dtype=int),
                         'multi_class': pd.Series([], dtype=str),
                         'verbose': pd.Series([], dtype=int),
                         'warm_start': pd.Series([], dtype=bool),
                         'n_jobs': pd.Series([], dtype=int),
                         'l1_ratio': pd.Series([], dtype=float)})


def default_logistic_regression_params():
    df = default_logistic_regression()
    return df.append({'penalty': 'l2',
                      'dual': False,
                      'tol': 0.0001,
                      'C': 1.0,
                      'fit_intercept': True,
                      'intercept_scaling': 1,
                      'class_weight': None,
                      'random_state': None,
                      'solver': 'lbfgs',
                      'max_iter': 100,
                      'multi_class': 'auto',
                      'verbose': 0,
                      'warm_start': False,
                      'n_jobs': None,
                      'l1_ratio': None}, ignore_index=True
                     )


def debugg(x, letter):
    print(type(x))
    print(letter, x)


class LogisticRegressionAlgorithm():

    def __init__(self, X, y, params: pd.DataFrame = default_logistic_regression_params()):
        self.penalty = params['penalty']
        self.dual = params['dual']
        self.tol = params['tol']
        self.C = params['C']
        self.fit_intercept = params['fit_intercept']
        self.intercept_scaling = params['intercept_scaling']
        self.class_weight = params['class_weight']
        self.random_state = params['random_state'] if not pd.isna(params['random_state']) else None
        self.solver = params['solver']
        self.max_iter = params['max_iter']
        self.multi_class = params['multi_class']
        self.verbose = params['verbose']
        self.warm_start = params['warm_start']
        self.n_jobs = params['n_jobs'] if not pd.isna(params['n_jobs']) else -1
        self.l1_ratio = params['l1_ratio'] if not pd.isna(params['l1_ratio']) else None
        self.make_classifier()

    def fit_clf(self, X, y):
        self.clf.fit(X, y.values.ravel())

    def gen_logistic_regression_params(self):
        default_logistic_regression_params().to_csv('LogisticRegression_params.csv', index_label='index')

    def make_classifier(self):
        self.clf = LR(penalty=self.penalty,
                      dual=self.dual,
                      tol=self.tol,
                      C=self.C,
                      fit_intercept=self.fit_intercept,
                      intercept_scaling=self.intercept_scaling,
                      class_weight=self.class_weight,
                      random_state=self.random_state,
                      solver=self.solver,
                      max_iter=self.max_iter,
                      multi_class=self.multi_class,
                      verbose=self.verbose,
                      warm_start=self.warm_start,
                      n_jobs=self.n_jobs,
                      l1_ratio=self.l1_ratio
                      )

    def metrics(self, X_test, y_test, dossier, name, fmt):
        log.info('metrics')
        y_pred = self.clf.predict(X_test)
        plot_roc_curve(self.clf, X_test, y_test)
        plt.savefig(dossier + '/' + 'roc_curve_' + name + '.' + fmt)
        plt.close()
        plot_confusion_matrix(self.clf, X_test, y_test)
        plt.savefig(dossier + '/' + 'conf_matrix_' + name + '.' + fmt)
        plt.close()
        predictions = self.clf.predict(X_test)
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        display = PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot()
        plt.savefig(dossier + '/' + 'precrecalldisp_' + name + '.' + fmt)
        result_df = pd.DataFrame.from_dict({
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=1),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred, average='weighted', zero_division=1)
        }, orient='index').T
        result_df.to_csv(dossier + '/' + 'performances.csv', index_label='index')
        self.permutation_importances(X_test, y_test, 'Test set', dossier, name, fmt)
        self.calibration_curve(X_test, y_test, dossier, name, fmt)
        self.learning_curve(X_test, y_test, dossier, name, fmt)
        self.ks_stat(X_test, y_test, dossier, name, fmt)



    def permutation_importances(self, X, y, datasetname, dossier, name, fmt):
        result = permutation_importance(self.clf, X, y, n_repeats=10,
                                        random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=X.columns[sorted_idx])
        ax.set_title(f"Permutation Importances ({datasetname})")
        fig.tight_layout()
        plt.savefig(dossier + '/' + 'permutimport_' + name + datasetname[:5] + '.' + fmt)
        plt.close()

    def calibration_curve(self, X_test, y_test, dossier, name, fmt):
        probas_list = [self.clf.predict_proba(X_test)]
        clf_names = ['Random Forest']

        skplt.metrics.plot_calibration_curve(y_test.values.ravel(),
                                             probas_list=probas_list,
                                             clf_names=clf_names,
                                             n_bins=10)

        plt.savefig(dossier + '/' + 'calibcurve_' + name + '.' + fmt)
        plt.close()

    def learning_curve(self, X, y, dossier, name, fmt):
        skplt.estimators.plot_learning_curve(self.clf, X=X, y=y.values.ravel(), n_jobs=-1)
        plt.savefig(dossier + '/' + 'learncurve_' + name + '.' + fmt)
        plt.close()

    def ks_stat(self, X_test, y_test, dossier, name, fmt):
        probas_list = self.clf.predict_proba(X_test)
        skplt.metrics.plot_ks_statistic(y_true=y_test.values.ravel(), y_probas=probas_list)
        plt.savefig(dossier + '/' + 'ksstat_' + name + '.' + fmt)
        plt.close()

    def __str__(self):
        return self.name
