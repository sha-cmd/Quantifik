import zipfile
import os
import sqlite3
import pandas as pd

data_path: str = '../data/Projet+Mise+en+prod+-+home-credit-default-risk.zip'
dir_data_extract = '../data'
tables_name_file = '../data/names_of_tables.txt'
database = '../data/projet_4_scoring.db'


def unzip():
    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall(dir_data_extract)
    print('file unzipped')


def select_all_csv_files():
    """
    Permet de créer un fichier avec le nom des tables de la base de données, qui sont
    les noms de fichiers, réduits en lower case, sans le nom de format.
    :return: la liste des noms de fichiers csv du répertoire
    """
    files_list: list = []
    chemin = os.getcwd()
    root_dir = chemin  # path to the root directory to search
    if os.path.isfile(tables_name_file):
        print('Le fichier ' + tables_name_file + ' doit d\'abord être détruit')
        return
    f = open(tables_name_file, 'w')
    for root, dirs, files in os.walk(dir_data_extract, onerror=None):  # walk the root dir
        for filename in files:  # iterate over the files in the current dir
            if filename.endswith('.csv'):
                files_list.append(filename)
                f.write(filename.lower()[:-4] + '\n')
    f.close()
    return files_list


def repair_a_file():
    """
    Repare le fichier qui écrit n'a pas un bon index au départ, et en plus ajoute un
    's' à la fin des colonnes car le mot Table est un mot clé dans le language SQLITE de
    base de données.
    :return:
    """
    df = pd.read_csv('HomeCredit_columns_description.csv', encoding='unicode_escape', engine='python', index_col=0)
    df.columns = [x + 's' for x in df.columns]
    df.to_csv('HomeCredit_columns_description.csv', index=0, index_label='indexid')


def read_table_names():
    """
    Lit les lignes dans le fichier des tables de la base.
    :return: Une liste des tables de la base
    """
    f = open(tables_name_file, 'r')
    file_reader = f.read()
    tables_list: list = file_reader.split('\n')
    return [x for x in tables_list if x != '']


def create_db():
    """
    Deprecated. Use `% sqlite projet_4_scoring.db < script.schema` in a terminal instead
    :return:
    """
    if not os.path.isfile("../data/projet_4_scoring.db"):
        con = sqlite3.connect('../data/projet_4_scoring.db')
        con.close()
        print('db created')
    else:
        print('db does exist')


def open_db_con():
    return sqlite3.connect(database)


def open_file_csv(file_name):
    """
    Load csv file into a data frame. Prends en compte l'erreur du fichier Home Crédit.
    :param file_name: string file name, does contains prefix, as ../data/
    :return: data frame
    """
    if 'HomeCredit' in file_name:  # replace a bug of unicode encoding
        df =  pd.read_csv('../data/HomeCredit_columns_description.csv', encoding='unicode_escape', engine='python')
        df.index.name = 'indexid'
        return df
    else:
        df = pd.read_csv(file_name)
        df.index.name = 'indexid'
        return df


def add_all_files_in_tables(file_names):
    """
    Création de la base de données centralisées contenant tous les documents csv dans
    une base de données relationnelle.
    :param file_names: liste des noms de fichier, ne contenant pas de préfixe du genre ../data/
    :return: boolean
    """
    con = open_db_con()
    for file in file_names:
        print('opération sur le fichier : ' + file)
        table_name = file.lower()[:-4]  # lower case without formet
        print(table_name)
        pd.DataFrame.to_sql(open_file_csv('../data/' + file), name=table_name, con=con, if_exists="replace")
    con.close()

    return True

def main():
    #read_table_names()
    # unzip()
    file_names = select_all_csv_files()
    #create_db()
    add_all_files_in_tables(file_names)
    #repair_a_file()

if __name__ == '__main__':
    main()
