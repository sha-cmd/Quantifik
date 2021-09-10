import numpy as np
import os
import pandas as pd

from src.utils.logger import log_init as log
from src.algorithms.RandomForestClassifier import RandomForestClassifierAlgorithm
from src.algorithms.LogisticRegressionAlgorithm import LogisticRegressionAlgorithm
from src.features_selection import Select_KBest, get_params_from_file
from src.model_selection import StratifiedShuffleSplit
from src.algorithms.RandomForestClassifier import default_df

class Algorithm:

    def __init__(self, name):
        self.name = name

    def ml_algo_RandomForest(self, nom):
        params_csv = default_df()

        os.chdir('..')
        nom_fichier = 'RandomForestClassifier_params.csv'
        params_csv = params_csv.append(pd.read_csv(nom_fichier, index_col='index'))
        fichier_sel_k_best = 'Select_KBest_params.csv'
        select_k_best_csv = get_params_from_file(fichier_sel_k_best)
        os.chdir('AutoML')
        df = default_df()

        for it_kbest, row_kbest in select_k_best_csv.iterrows():
            a = Select_KBest(X, y, row_kbest)
            b = StratifiedShuffleSplit(X[a.columns], y)
            for it_i, row in params_csv.iterrows():
                for it, col in row.iteritems():
                    col = None if col == 'nan' else col
                    df.at[it_i, it] = pd.to_numeric(col, errors='ignore')

                c: RandomForestClassifierAlgorithm = RandomForestClassifierAlgorithm(b.X_train.to_numpy(),
                                                                                     b.y_train.iloc[:, 0].ravel(), row)
                c.fit_clf(b.X_train, b.y_train)
                nom = 'RandomForest'
                num = str(int(it_i))

                dossier = nom + '_' + num + 'KBest_' + str(row_kbest['k'])
                if dossier not in os.listdir():
                    os.mkdir(dossier)
                c.metrics(b.X_test, b.y_test, dossier, nom, 'jpg')
                c.permutation_importances(b.X_train, b.y_train, 'Train set', dossier, nom, 'jpg')

    def ml_algo_LogisticRegression(self, nom):
        params_csv = default_df()

        os.chdir('..')
        nom_fichier = 'LogisticRegression_params.csv'
        params_csv = params_csv.append(pd.read_csv(nom_fichier, index_col='index'))
        fichier_sel_k_best = 'Select_KBest_params.csv'
        select_k_best_csv = get_params_from_file(fichier_sel_k_best)
        os.chdir('AutoML')
        df = default_df()

        for it_kbest, row_kbest in select_k_best_csv.iterrows():
            a = Select_KBest(X, y, row_kbest)
            b = StratifiedShuffleSplit(X[a.columns], y)
            for it_i, row in params_csv.iterrows():
                for it, col in row.iteritems():
                    col = None if col == 'nan' else col
                    df.at[it_i, it] = pd.to_numeric(col, errors='ignore')

                c: LogisticRegressionAlgorithm = LogisticRegressionAlgorithm(b.X_train.to_numpy(),
                                                                             b.y_train.iloc[:, 0].ravel(), row)
                c.fit_clf(b.X_train, b.y_train)
                nom = 'LogisticRegression'
                num = str(int(it_i))

                dossier = nom + '_' + num + '_' + 'KBest_' + str(row_kbest['k'])
                if dossier not in os.listdir():
                    os.mkdir(dossier)
                c.metrics(b.X_test, b.y_test, dossier, nom, 'jpg')
                c.permutation_importances(b.X_train, b.y_train, 'Train set', dossier, nom, 'jpg')
