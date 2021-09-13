import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shap
import time
import ast
import scikitplot as skplt

from src.utils.logger import log_init as log
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.metrics import (precision_score, recall_score,
                             f1_score)
from sklearn.inspection import permutation_importance

log = log()


def default_hist_gradient_boosting_classifier():
    return pd.DataFrame({'loss': pd.Series([], dtype=str),
                         'learning_rate': pd.Series([], dtype=float),
                         'max_iter': pd.Series([], dtype=int),
                         'max_leaf_nodes': pd.Series([], dtype=int),
                         'max_depth': pd.Series([], dtype=int),
                         'min_samples_leaf': pd.Series([], dtype=int),
                         'l2_regularization': pd.Series([], dtype=float),
                         'max_bins': pd.Series([], dtype=int),
                         'categorical_features': pd.Series([], dtype=str),
                         'monotonic_cst': pd.Series([], dtype=str),
                         'warm_start': pd.Series([], dtype=bool),
                         'early_stopping': pd.Series([], dtype=str),
                         'warm_start': pd.Series([], dtype=bool),
                         'scoring': pd.Series([], dtype=str),
                         'validation_fraction': pd.Series([], dtype=float),
                         'n_iter_no_change': pd.Series([], dtype=int),
                         'tol': pd.Series([], dtype=float),
                         'verbose': pd.Series([], dtype=int),
                         'random_state': pd.Series([], dtype=int)})


def default_hist_gradient_boosting_params():
    df = default_hist_gradient_boosting_classifier()
    return df.append({'loss': 'auto',
                      'learning_rate': 0.1,
                      'max_iter': 100,
                      'max_leaf_nodes': 31,
                      'max_depth': None,
                      'min_samples_leaf': 20,
                      'l2_regularization': 0.0,
                      'max_bins': 255,
                      'categorical_features': None,
                      'monotonic_cst': None,
                      'warm_start': False,
                      'early_stopping': 'auto',
                      'scoring': 'loss',
                      'validation_fraction': 0.1,
                      'n_iter_no_change': 10,
                      'tol': 1e-07,
                      'verbose': 0,
                      'random_state': None
                      }, ignore_index=True
                     )


def debugg(x, letter):
    print(type(x))
    print(letter, x)


def gen_hist_gradient_boosting_params():
    default_hist_gradient_boosting_params().to_csv('HistGradientBoostingClassifier_params.csv', index_label='index')


class HistGradientBoostingAlgorithm:

    def __init__(self, X, y, params: pd.DataFrame = default_hist_gradient_boosting_params()):
        df_init = default_hist_gradient_boosting_params()
        self.loss = params['loss'] if not pd.isna(params['loss']) else df_init['loss'].values.ravel()
        self.learning_rate = params['learning_rate'] if not pd.isna(params['learning_rate']) else df_init[
            'learning_rate'].values.ravel()
        self.max_iter: int = int(params['max_iter']) if not pd.isna(params['max_iter']) else int(df_init['max_iter'].values.ravel())
        self.max_leaf_nodes = params['max_leaf_nodes'] if not pd.isna(params['max_leaf_nodes']) else df_init[
            'max_leaf_nodes'].values.ravel()
        self.max_depth = params['max_depth'] if not pd.isna(params['max_depth']) else df_init[
            'max_depth'].values.ravel()[0]
        self.min_samples_leaf = params['min_samples_leaf'] if not pd.isna(params['min_samples_leaf']) else df_init[
            'min_samples_leaf'].values.ravel()
        self.l2_regularization = params['l2_regularization'] if not pd.isna(params['l2_regularization']) else df_init[
            'l2_regularization'].values.ravel()
        self.max_bins: int = int(params['max_bins']) if not pd.isna(params['max_bins']) else int(df_init['max_bins'].values.ravel())
        self.categorical_features = params['categorical_features'] if not pd.isna(params['categorical_features']) else \
            df_init['categorical_features'].values.ravel()[0]
        self.monotonic_cst = params['monotonic_cst'] if not pd.isna(params['monotonic_cst']) else df_init[
            'monotonic_cst'].values.ravel()[0]
        self.warm_start = params['warm_start'] if not pd.isna(params['warm_start']) else df_init[
            'warm_start'].values.ravel()
        self.early_stopping = params['early_stopping'] if not pd.isna(params['early_stopping']) else df_init[
            'early_stopping'].values.ravel()
        self.warm_start = params['warm_start'] if not pd.isna(params['warm_start']) else df_init[
            'warm_start'].values.ravel()
        self.scoring = params['scoring'] if not pd.isna(params['scoring']) else df_init['scoring'].values.ravel()
        self.validation_fraction = params['validation_fraction'] if not pd.isna(params['validation_fraction']) else \
            df_init['validation_fraction'].values.ravel()
        self.n_iter_no_change: int = int(params['n_iter_no_change']) if not pd.isna(params['n_iter_no_change']) else int(df_init[
            'n_iter_no_change'].values.ravel())
        self.tol = params['tol'] if not pd.isna(params['tol']) else df_init['tol'].values.ravel()
        self.verbose = params['verbose'] if not pd.isna(params['verbose']) else df_init['verbose'].values.ravel()
        self.random_state = params['random_state'] if not pd.isna(params['random_state']) else df_init[
            'random_state'].values.ravel()[0]

        debugg(df_init['random_state'], 'random state')
        self.make_classifier()

    def fit_clf(self, X, y):
        self.clf.fit(X, y.values.ravel())

    def make_classifier(self):
        print(self.loss, 'loss')
        print(self.learning_rate, 'learning_rate')
        print(self.max_iter, 'max_iter')
        print(self.max_leaf_nodes, 'max_leaf_nodes')
        print(self.max_depth, 'max_depth')
        print(self.min_samples_leaf, 'min_samples_leaf')
        print(self.l2_regularization, 'l2_regularization')
        print(self.max_bins, 'max_bins')
        print(self.categorical_features, 'categorical_features')
        print(self.monotonic_cst, 'monotonic_cst')
        print(self.warm_start, 'warm_start')
        print(self.early_stopping, 'early_stopping')
        print(self.warm_start, 'warm_start')
        print(self.scoring, 'scoring')
        print(self.validation_fraction, 'validation_fraction')
        print(self.n_iter_no_change, 'n_iter_no_change')
        print(self.tol, 'tol')
        print(self.verbose, 'verbose')
        print(self.random_state, 'random_state')

        self.clf = HGBC(loss=self.loss,
                        learning_rate=self.learning_rate,
                        max_iter=self.max_iter,
                        max_leaf_nodes=self.max_leaf_nodes,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        l2_regularization=self.l2_regularization,
                        max_bins=self.max_bins,
                        categorical_features=self.categorical_features,
                        monotonic_cst=self.monotonic_cst,
                        warm_start=self.warm_start,
                        early_stopping=self.early_stopping,
                        scoring=self.scoring,
                        validation_fraction=self.validation_fraction,
                        n_iter_no_change=self.n_iter_no_change,
                        tol=self.tol,
                        verbose=self.verbose,
                        random_state=self.random_state,
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
        # print('permutation test')
        # self.permutation_importances(X_test, y_test, 'Test set', dossier, name, fmt)
        print('calibration curve')
        self.calibration_curve(X_test, y_test, dossier, name, fmt)
        print('learning curve')
        self.learning_curve(X_test, y_test, dossier, name, fmt)
        print('ks statistics')
        self.ks_stat(X_test, y_test, dossier, name, fmt)
        print('shap')
        shap_values = shap.TreeExplainer(self.clf).shap_values(X_test.iloc[:1000, :])

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
        clf_names = ['Hist Gradient Boosting']

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
            shap_values, X, plot_type="bar", show=False#, class_names=classes
        )
        fig.tight_layout(pad=2.0)
        plt.savefig(dossier + '/' + 'shap_summary_' + name + '.' + fmt
                    )
        plt.close("all")

    def __str__(self):
        return self.name
