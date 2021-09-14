import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shap
import time
import ast
import scikitplot as skplt

from src.utils.logger import log_init as log
from sklearn.svm import NuSVC
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.metrics import (precision_score, recall_score,
                             f1_score)
from sklearn.inspection import permutation_importance

log = log()


def default_NuSVC():
    return pd.DataFrame({'nu': pd.Series([], dtype=float),
                         'kernel': pd.Series([], dtype=str),
                         'degree': pd.Series([], dtype=int),
                         'gamma': pd.Series([], dtype=str),
                         'coef0': pd.Series([], dtype=float),
                         'shrinking': pd.Series([], dtype=bool),
                         'probability': pd.Series([], dtype=bool),
                         'tol': pd.Series([], dtype=float),
                         'cache_size': pd.Series([], dtype=int),
                         'class_weight': pd.Series([], dtype=str),
                         'verbose': pd.Series([], dtype=bool),
                         'max_iter': pd.Series([], dtype=int),
                         'decision_function_shape': pd.Series([], dtype=str),
                         'break_ties': pd.Series([], dtype=bool),
                         'random_state': pd.Series([], dtype=int), })


def default_NuSVC_params():
    df = default_NuSVC()
    return df.append({'nu': 0.5,
                      'kernel': 'linear',
                      'degree': 3,
                      'gamma': 'scale',
                      'coef0': 0.0,
                      'shrinking': True,
                      'probability': True,
                      'tol': 0.001,
                      'cache_size': 200,
                      'class_weight': None,
                      'verbose': False,
                      'max_iter': -1,
                      'decision_function_shape': 'ovr',
                      'break_ties': False,
                      'random_state': None,
                      }, ignore_index=True
                     )


def debugg(x, letter):
    print(type(x))
    print(letter, x)


def gen_nusvc_params():
    default_NuSVC_params().to_csv('NuSVC_params.csv', index_label='index')


class NuSVCAlgorithm:

    def __init__(self, X, y, params: pd.DataFrame = default_NuSVC_params()):
        df_init = default_NuSVC_params()
        self.nu = params['nu'] if not pd.isna(params['nu']) else df_init['nu'].values.ravel()
        self.kernel = params['kernel'] if not pd.isna(params['kernel']) else df_init['kernel'].values.ravel()
        self.degree = int(params['degree']) if not pd.isna(params['degree']) else int(df_init['degree'].values.ravel())
        self.gamma = params['gamma'] if not pd.isna(params['gamma']) else df_init['gamma'].values.ravel()
        self.coef0 = params['coef0'] if not pd.isna(params['coef0']) else df_init['coef0'].values.ravel()
        self.shrinking = params['shrinking'] if not pd.isna(params['shrinking']) else df_init[
            'shrinking'].values.ravel()
        self.probability = params['probability'] if not pd.isna(params['probability']) else df_init[
            'probability'].values.ravel()
        self.tol = params['tol'] if not pd.isna(params['tol']) else df_init['tol'].values.ravel()
        self.cache_size = int(params['cache_size']) if not pd.isna(params['cache_size']) else int(df_init[
            'cache_size'].values.ravel())
        self.class_weight = params['class_weight'] if not pd.isna(params['class_weight']) else df_init[
            'class_weight'].values.ravel()[0]
        self.verbose = params['verbose'] if not pd.isna(params['verbose']) else df_init['verbose'].values.ravel()
        self.max_iter = int(params['max_iter']) if not pd.isna(params['max_iter']) else int(df_init['max_iter'].values.ravel())
        self.decision_function_shape = params['decision_function_shape'] if not pd.isna(
            params['decision_function_shape']) else df_init['decision_function_shape'].values.ravel()
        self.break_ties = params['break_ties'] if not pd.isna(params['break_ties']) else df_init[
            'break_ties'].values.ravel()
        self.random_state = int(params['random_state']) if not pd.isna(params['random_state']) else df_init[
            'random_state'].values.ravel()[0]

        self.make_classifier()

    def fit_clf(self, X, y):
        self.clf.fit(X, y.values.ravel())

    def make_classifier(self):
        print(self.nu, 'nu')
        print(self.kernel, 'kernel')
        print(self.degree, 'degree')
        print(self.gamma, 'gamma')
        print(self.coef0, 'coef0')
        print(self.shrinking, 'shrinking')
        print(self.probability, 'probability')
        print(self.tol, 'tol')
        print(self.cache_size, 'cache_size')
        print(self.class_weight, 'class_weight')
        print(self.verbose, 'verbose')
        print(self.max_iter, 'max_iter')
        print(self.decision_function_shape, 'decision_function_shape')
        print(self.break_ties, 'break_ties')
        print(self.random_state, 'random_state')

        self.clf = NuSVC(nu=self.nu,
                        kernel=self.kernel,
                        degree=self.degree,
                        gamma=self.gamma,
                        coef0=self.coef0,
                        shrinking=self.shrinking,
                        probability=self.probability,
                        tol=self.tol,
                        cache_size=self.cache_size,
                        class_weight=self.class_weight,
                        verbose=self.verbose,
                        max_iter=self.max_iter,
                        decision_function_shape=self.decision_function_shape,
                        break_ties=self.break_ties,
                        random_state=self.random_state
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
        predictions = self.clf.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        display = PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot()
        plt.savefig(dossier + '/' + 'precrecalldisp_' + name + '.' + fmt)
        result_df = pd.DataFrame.from_dict({
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=1),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=1),
            'F1-score': f1_score(y_test, y_pred, average='weighted', zero_division=1)
        }, orient='index').T
        result_df.to_csv(dossier + '/' + 'performances.csv', index_label='index')
        print('permutation test')
        self.permutation_importances(X_test, y_test, 'Test set', dossier, name, fmt)
        print('calibration curve')
        self.calibration_curve(X_test, y_test, dossier, name, fmt)
        print('learning curve')
        self.learning_curve(X_test, y_test, dossier, name, fmt)
        print('ks statistics')
        self.ks_stat(X_test, y_test, dossier, name, fmt)
        print('shap')
        shap_values = shap.LinearExplainer(self.clf, X_test.iloc[:1000, :]).shap_values(X_test.iloc[:1000, :])

        self.dependence(shap_values, X_test.iloc[:1000, :], dossier, name, fmt)
        self.summary(shap_values, X_test.iloc[:1000, :], dossier, name, fmt)

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
        clf_names = ['NuSVC']

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

    def dependence(self, shap_values, X, dossier, name, fmt):
        fig = plt.figure(figsize=(14, 7))
        plots_cnt = np.min([9, X.shape[1]])
        cols_cnt = 3
        rows_cnt = 3
        if plots_cnt < 4:
            rows_cnt = 1
        elif plots_cnt < 7:
            rows_cnt = 2
        for i in range(plots_cnt):
            ax = fig.add_subplot(rows_cnt, cols_cnt, i + 1)
            shap.dependence_plot(
                f"rank({i})",
                shap_values,
                X,
                show=False,
                title=f"Importance #{i + 1}",
                ax=ax,
            )

        fig.tight_layout(pad=2.0)
        fig.savefig(dossier + '/' + 'shap_dependence_' + name + '.' + fmt
                    )
        plt.close("all")

    def summary(self, shap_values, X, dossier, name, fmt):
        fig = plt.gcf()

        shap.summary_plot(
            shap_values, X, plot_type="bar", show=False  # , class_names=classes
        )
        fig.tight_layout(pad=2.0)
        plt.savefig(dossier + '/' + 'shap_summary_' + name + '.' + fmt
                    )
        plt.close("all")

    def __str__(self):
        return self.name
