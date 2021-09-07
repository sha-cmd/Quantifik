import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from highcharts import Highchart


def pourcent_of_duplicated(data: pd.Series) -> float:
    """
    Renvoie le pourcentage de valeurs dupliquées dans une Series de pandas.
    :param data: Series de pandas
    :return: float: le pourcentage
    """
    if data.ndim == 1:
        return round((~data.duplicated()).sum() * 100 / len(data), 2)
    else:
        print('le paramètre est-il vraiment une Series de pandas ?')


def pourcent_of_null(data: pd.Series) -> float:
    """
    Renvoie le pourcentage de valeurs nulles dans une Series de pandas.
    :param data: Series de pandas
    :return: float: le pourcentage
    """
    if data.ndim == 1:
        return round((~data.notnull()).sum() * 100 / len(data), 2)
    else:
        print('le paramètre est-il vraiment une Series de pandas ?')


def pourcent_outside_3iqr(data: pd.Series) -> float:
    """
    Renvoie le pourcentage de valeurs au-dehors de l'intervalle de confiance externe.
    :param data: une Series de pandas.
    :return: float: le pourcentage.
    """
    if data.ndim == 1:
        is_a_series_of_str = True in map((lambda x: type(x) == str), data)
        if not is_a_series_of_str:
            iqr = round((data.quantile(0.75) - data.quantile(0.25)) * 3, 2)
            if len(data) > 0:
                pct_outside_confidence_interval = round(len(data.loc[(data > iqr) | (data < 0)]) / len(data) * 100, 2)
            else:
                pct_outside_confidence_interval = 0
            return pct_outside_confidence_interval


def iqr(data: pd.Series) -> float:
    """
    Renvoie le pourcentage de valeurs au-dehors de l'intervalle de confiance externe.
    :param data: une Series de pandas.
    :return: float: le pourcentage.
    """
    if data.ndim == 1:
        is_a_series_of_str = True in map((lambda x: type(x) == str), data)
        if not is_a_series_of_str:
            iqr_value = round((data.quantile(0.75) - data.quantile(0.25)) * 3, 2)
            return iqr_value


def amount_type(X):
    nb_flt = []  # colonnes de flottant
    nb_str = []  # colonnes d'entier
    nb_int = []  # colonnes de string
    for col in X.columns:
        if str(X[col].dtype) in 'float64':
            nb_flt.append(col)
        if (str(X[col].dtype) in 'object') | (str(X[col].dtype) in 'uint8'):
            nb_str.append(col)
        if str(X[col].dtype) in 'int64':
            nb_int.append(col)
    df_col_type = pd.DataFrame({'flottants': pd.Series(nb_flt), 'entiers': pd.Series(nb_int)
                                   , 'str': pd.Series(nb_str)})
    return nb_flt, nb_int, nb_str, df_col_type


def prepare_box_plot(df_att):
    df = df_att[np.abs(df_att - df_att.mean()) < (3 * df_att.std())]
    data = []
    for it, col in enumerate(df.columns):
        liste_val = []
        liste_val.append(df[col].min())
        liste_val.append(df[col].quantile(0.25))
        liste_val.append(df[col].median())
        liste_val.append(df[col].quantile(0.75))
        liste_val.append(df[col].max())
        data.append(liste_val)
    return data


def prepare_outliers_box_plot(df_att):
    df = df_att[np.abs(df_att - df_att.mean()) >= (3 * df_att.std())]
    data = []
    for it, col in enumerate(df.columns):
        values = df[col].loc[df[col].notnull()].tolist()
        for val in values:
            data.append([col, val])
    return data


def boxplot(Xy, word, type_val):
    # Xy = pd.read_csv('../notebooks/Xy.csv', index_col='index')
    X_str = Xy.drop('TARGET', axis=1).copy()
    y_str = Xy['TARGET'].map({1: '1', 0: '0'}).copy()
    y_int = Xy['TARGET']
    nb_flt, nb_int, nb_str, df_col_type = amount_type(X_str)
    # onehotencoder = pd.get_dummies(X_str[nb_str])
    # X_onehotenc = X_str.join(onehotencoder).drop(nb_str, axis=1).copy()
    df_type = df_col_type[df_col_type[type_val].notnull()][type_val]
    df_type_word = df_type.loc[df_type.str.contains(word)]
    data_type_word = Xy[list(df_type_word.values)]  # X_onehotenc[list(df_type_word.values)]#.iloc[:, :4]
    H = Highchart(width=550, height=400)

    options = {
        'chart': {
            'type': 'boxplot'
        },
        'title': {
            'text': 'Highcharts Box Plot Example'
        },
        'legend': {
            'enabled': False
        },
        'xAxis': {
            'categories': list(df_type_word.values),
            'title': {
                'text': 'Experiment No.'
            }
        },

        'yAxis': {
            'title': {
                'text': 'Observations'
            },

        },
    }

    data = prepare_box_plot(data_type_word)
    data_outline = prepare_outliers_box_plot(data_type_word)

    H.set_dict_options(options)
    H.add_data_set(data, 'boxplot', 'Observations', tooltip={
        'headerFormat': '<em>Experiment No {point.key}</em><br/>'})
    H.add_data_set(data_outline, 'scatter', 'Outlier', marker={
        'fillColor': 'white',
        'lineWidth': 1,
        'lineColor': 'Highcharts.getOptions().colors[0]'
    },
                   tooltip={
                       'pointFormat': 'Observation: {point.y}'
                   })

    return H


def generate_val_for_pie3D(y):
    list_values = []
    for col in list(y.value_counts().index.values):
        list_values.append([str(col), int(y.value_counts()[col])])
    return list_values


def piechart3D_cible(list_values, text, subtitle):
    H = Highchart(width=550, height=400)

    options = {
        'chart': {
            'type': 'pie',
            'options3d': {
                'enabled': True,
                'alpha': 45
            }
        },
        'title': {
            'text': text
        },
        'subtitle': {
            'text': subtitle
        },
        'plotOptions': {
            'pie': {
                'innerSize': 100,
                'depth': 45
            }
        },
    }

    data = list_values

    H.set_dict_options(options)
    H.add_data_set(data, 'pie', 'Delivered amount')
    return H


def null_number_plot(X, eda_df, col_type, title, subtitle):
    null_number = X.isnull().sum(axis=1).sum()
    cell_number = X.shape[0] * X.shape[1]
    percent_of_null = X.isnull().sum(axis=1).sum() * 100 / (X.shape[0] * X.shape[1])
    liste_val = []
    for it, row in eda_df.iterrows():
        col = row['columns']
        if row['columns'] in col_type:
            val = {'name': row['columns'],
                   'y': round((X[col].isnull().sum()) * 100 / null_number, 2),
                   'drilldown': row['columns']}
            liste_val.append(val)
            del val

    from highcharts import Highchart
    H = Highchart(width=850, height=400)

    """
    Drilldown chart can be created using add_drilldown_data_set method: 
    add_drilldown_data_set(data, series_type, id, **kwargs):
    id is the drilldown parameter in upperlevel dataset (Ex. drilldown parameters in data)
    drilldown dataset is constructed similar to dataset for other chart
    """

    data = liste_val

    options = {
        'chart': {
            'type': 'column'
        },
        'title': {
            'text': title
        },
        'subtitle': {
            'text': subtitle
        },
        'xAxis': {
            'type': 'category'
        },
        'yAxis': {
            'title': {
                'text': title
            }

        },
        'legend': {
            'enabled': False
        },
        'plotOptions': {
            'series': {
                'borderWidth': 0,
                'dataLabels': {
                    'enabled': True,
                    'format': '{point.y:.2f}%'
                }
            }
        },

        'tooltip': {
            'headerFormat': '<span style="font-size:11px">{series.name}</span><br>',
            'pointFormat': '<span style="color:{point.color}">{point.name}</span>: <b>{point.y:.2f}%</b> of total<br/>'
        },

    }

    H.set_dict_options(options)

    H.add_data_set(data, 'column', "Brands", colorByPoint=True)

    return H


def column_plot(eda, col, title, subtitle, yaxis):
    list_of = []
    for it, row in eda[['columns', col]].iterrows():
        dict_df = {}
        dict_df.update({'name': str(row['columns'])})
        dict_df.update({'y': float(row[col])})
        dict_df.update({'drilldown': row['columns']})
        list_of.append(dict_df)

    """
    Highcharts Demos
    Column with drilldown: http://www.highcharts.com/demo/column-drilldown
    """

    H = Highchart(width=850, height=400)

    """
    Drilldown chart can be created using add_drilldown_data_set method: 
    add_drilldown_data_set(data, series_type, id, **kwargs):
    id is the drilldown parameter in upperlevel dataset (Ex. drilldown parameters in data)
    drilldown dataset is constructed similar to dataset for other chart
    """

    data = list_of
    options = {
        'chart': {
            'type': 'column'
        },
        'title': {
            'text': title
        },
        'subtitle': {
            'text': subtitle
        },
        'xAxis': {
            'type': 'category'
        },
        'yAxis': {
            'title': {
                'text': yaxis
            }

        },
        'legend': {
            'enabled': False
        },
        'plotOptions': {
            'series': {
                'borderWidth': 0,
                'dataLabels': {
                    'enabled': True,
                    'format': '{point.y:.1f}'
                }
            }
        },

        'tooltip': {
            'headerFormat': '<span style="font-size:11px">{series.name}</span><br>',
            'pointFormat': '<span style="color:{point.color}">{point.name}</span>: <b>{point.y:.2f}</b> of total<br/>'
        },

    }

    H.set_dict_options(options)

    H.add_data_set(data, 'column', "Brands", colorByPoint=True)

    return H


def table(data):
    """
    Renvoie un tableau pour présenter les seuils choisis (Lim), le maximum et le minimum de la colonne, et le nombre
    de valeurs au-dessus du seuil, puis le nombre de valeurs négatives.
    :param data: pandas DataFrame
    :return: void
    """

    dataframe = pd.DataFrame({'columns': pd.Series([], dtype=str),
                              'type': pd.Series([], dtype=str),
                              'unique': pd.Series([], dtype=int),
                              'mean': pd.Series([], dtype=float),
                              'std': pd.Series([], dtype=float),
                              'pct_null': pd.Series([], dtype=float),
                              'pct>3iqr': pd.Series([], dtype=float),
                              'non-nulls': pd.Series([], dtype=int),
                              # 'limits': pd.Series([], dtype=float),
                              'max': pd.Series([], dtype=float),
                              'min': pd.Series([], dtype=float),
                              'outliers': pd.Series([], dtype=int),
                              'negatives': pd.Series([], dtype=int),
                              'kurtosis': pd.Series([], dtype=float),
                              'skewness': pd.Series([], dtype=float),
                              'mode': pd.Series([], dtype=float),
                              'median': pd.Series([], dtype=float)
                              })
    # dtypes information
    dtypes_uniques = set()  # Collection of unique elements
    dtypes_listes = []  # list of complete data columns
    memory_usage = 0  # in MB
    nb_values_to_modify: int = 0
    nb_values_not_null: int = 0
    nb_values_is_null: int = 0
    list_de_cols = enumerate(data.columns) if isinstance(data, pd.DataFrame) else [data.name]
    is_a_str_series = isinstance(data, pd.Series)
    for col in list_de_cols:  # Feed the set and the list

        min_col = str(data[col].min()) if not is_a_str_series else str(data.min())

        nb_values_not_null += data[col].count() if not is_a_str_series else data.count()

        dtype = data[col].dtypes if not is_a_str_series else data.dtype
        dtypes_listes.append(str(dtype))

        if str(dtype) not in dtypes_uniques:
            dtypes_uniques.add(dtype)
        max_col = str(data[col].max()) if not is_a_str_series else str(data.max())
        out_col = data[col][data[col] > 3 * iqr(data[col])].count() if not is_a_str_series else data[
            data > 3 * iqr(data)].count()
        neg_col = data[col][data[col] < 0].count() if not is_a_str_series else data[data < 0].count()
        nb_values_to_modify += out_col + neg_col
        try:
            nb_values_is_null += data[col].isnull().value_counts()[True] if not is_a_str_series else \
            data.isnull().value_counts()[True]
        except KeyError as error:
            nb_values_is_null += 0

        dataframe = dataframe.append({'columns': col, 'type': data[col].dtype if not is_a_str_series else data.dtype,
                                      'unique': len(data[col].unique()) if not is_a_str_series else len(data.unique()),
                                      'mean': data[col].mean() if not is_a_str_series else data.mean(),
                                      'std': data[col].std() if not is_a_str_series else data.std(),
                                      'pct_null': pourcent_of_null(
                                          data[col]) if not is_a_str_series else pourcent_of_null(data),
                                      'pct>3iqr': pourcent_outside_3iqr(
                                          data[col]) if not is_a_str_series else pourcent_outside_3iqr(data),
                                      'non-nulls': data[col].count() if not is_a_str_series else data.count(),
                                      #   'limits': rec.get(col),
                                      'max': max_col, 'min': min_col, 'outliers': out_col,
                                      'negatives': neg_col,
                                      'kurtosis': data[col].kurt() if not is_a_str_series else data.kurt(),
                                      'skewness': data[col].skew() if not is_a_str_series else data.skew(),
                                      'mode': data[col].mode()[0] if not is_a_str_series else data.mode(),
                                      'median': data[col].median() if not is_a_str_series else data.median()},
                                     ignore_index=True)
        # Collect informatives on disk usage by observation onto the data column
        memory_usage += int(data[col].memory_usage(index=True, deep=True)) if not is_a_str_series else int(
            data.memory_usage(index=True, deep=True))
    # Blend of set and list to print the information line as usual
    dtypes_string = ''
    for x in dtypes_uniques:
        dtypes_string += '{}({}), '.format(x, dtypes_listes.count(x))
    print('\ndtypes: {}'.format(dtypes_string))
    # Digit format to write mem usage in comprehensive format
    print('\nmemory usage: {:.4} MB\n'.format(memory_usage / (1024 * 1024)))
    print('\nNombre de lignes: {}\n'.format(data.shape[0]) if not is_a_str_series else len(data))
    print('\nNombre de valeurs non-nulles: {}\n'.format(nb_values_not_null))
    print('\nNombre de valeurs nulles: {}\n'.format(nb_values_is_null))
    #   print('\nNombre de valeurs aberrantes et atypiques: {}\n'.format(nb_values_to_modify))
    #    print('\nNombre de valeurs au-dessus du seuil: {}\n'.format(nb_values_to_modify))
    return dataframe


def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better.

    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance

    Returns:
        shows a plot of the 15 most importance features

        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    plt.style.use('dark_background')
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')
    plt.show()

    return df
