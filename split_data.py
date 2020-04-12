import sys
import logging
import pandas as pd
from clean_data import export_data
from sklearn.model_selection import train_test_split
from classification_model.config import data_modelpath, label

logger = logging.getLogger('split_data')
logger.setLevel(logging.INFO)
console_handle = logging.StreamHandler(sys.stdout)
console_handle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handle.setFormatter(formatter)
logger.addHandler(console_handle)


def load_data(file_path: str) -> pd.DataFrame:
    """
    
    :param file_path: 
    :return: 
    """
    dataframe = pd.read_csv(file_path, sep=';')
    return dataframe


def transform_label(dataframe: pd.DataFrame, target: str) -> pd.DataFrame:
    dataframe[target].apply(lambda x: 1 if x == 'offer completed' else 0)
    logger.info('Se ha transformado la variable dependiente correctamente!')
    return dataframe


def split_data(dataframe: pd.DataFrame, label: str, test_size: float):
    """

    :param dataframe:
    :param label:
    :param test_size:
    :return:
    """
    label = dataframe.pop(label)
    train_data, test_data, train_label, test_label = train_test_split(dataframe, label, test_size=test_size,
                                                                      stratify=label)
    logger.info('Se ha partido el set de datos en un set de entrenamiento y otro de prueba.')
    return train_data, test_data, train_label, test_label


def main():
    data = load_data(file_path=data_modelpath)
    data = transform_label(dataframe=data, target=label)
    train_data, test_data, train_label, test_label = split_data(dataframe=data, label=label, test_size=0.3)
    export_data(train_data, file_path='data/train_data.csv')
    export_data(train_label, file_path='data/train_label.csv')
    export_data(test_data, file_path='data/test_data.csv')
    export_data(test_label, file_path='data/test_label.csv')


if __name__ == '__main__':
    main()
