
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import ast
import scikitplot as skplt

from src.utils.logger import log_init as log
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.metrics import (precision_score, recall_score,
                             f1_score)
from sklearn.inspection import permutation_importance

log = log()


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


def debugg(x, letter):
    print(type(x))
    print(letter, x)


class RandomForestClassifierAlgorithm():

    def __init__(self, X, y, params: pd.DataFrame = default_params()):
        self.clf: RFC = RFC()
        self.n_estimators = int(params['n_estimators']) if not pd.isna(params['n_estimators']) else 100
        self.criterion = params['criterion']
        self.max_depth = params['max_depth']if not pd.isna(params['max_depth']) else 2
        self.min_samples_split = float(params['min_samples_split']) if (
                    params['min_samples_split'] <= 1) else \
            (int(params['min_samples_split'])) if (params['min_samples_split'] > 1) else (int(2))
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
        self.max_samples = params['max_samples'] if not pd.isna(params['max_samples']) else None
        self.make_classifier()
       # debugg(self.max_samples, 'max samples')


    def fit_clf(self, X, y):
        self.clf.fit(X, y.values.ravel())

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
        print('feature importance')
        self.feature_importances(X_test, y_test, dossier, name, fmt)
        print('permutation importance test')
        self.permutation_importances(X_test, y_test, 'Test set', dossier, name, fmt)
        print('calibration curve')
        self.calibration_curve(X_test, y_test, dossier, name, fmt)
        print('learning curve')
        self.learning_curve(X_test, y_test, dossier, name, fmt)
        print('ks statistics')
        self.ks_stat(X_test, y_test, dossier, name, fmt)

    def feature_importances(self, X, y, dossier, name, fmt):
        feature_names = [f'{i}' for i in list(X.columns)]
        start_time = time.time()
        importances = self.clf.feature_importances_
        #debugg(importances.shape, 'Import')
        #debugg(X.columns.shape, 'X Columns')
        np.array(importances)
        imp_df = pd.DataFrame(data=importances).T
        #debugg(imp_df.columns, 'columsn')
        imp_df = imp_df.rename(columns={it: x for it, x in enumerate(X.columns)})
        imp_df.to_csv(dossier + '/' + 'featimp_' + name + '.csv', index_label='index')
        std = np.std([
            tree.feature_importances_ for tree in self.clf.estimators_], axis=0)

        forest_importances = pd.Series(importances, index=feature_names)

        fig, ax = plt.subplots()
        forest_importances = forest_importances.sort_values(ascending=False)
        forest_importances.plot.bar(yerr=std, ax=ax)
        #debugg(forest_importances, 'A')
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.savefig(dossier + '/' + 'featimport_' + name + '.' + fmt)
        plt.close()

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
