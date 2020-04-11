import sys
import logging
import numpy as np
import pandas as pd
from classification_model.config import values_reward, portfolio_filepath, profile_filepath,\
                                        transcript_filepath, portfolio_outputpath, profile_outputpath,\
                                        transcript_outputpath, final_filepath


logger = logging.getLogger('clean_starbucks_data')
logger.setLevel(logging.INFO)
console_handle = logging.StreamHandler(sys.stdout)
console_handle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s -%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handle.setFormatter(formatter)
logger.addHandler(console_handle)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a json file
    :param file_path: relative or absolute file path
    :return: pd.DataFrame
    """
    dataframe = pd.read_json(file_path, orient='records', lines=True)
    logger.info('El archivo ha cargado exitosamente!')
    return dataframe


def clean_portfolio_file(dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    :return:
    """
    channels = ['web', 'mobile', 'email', 'social']
    for channel in channels:
        dataframe[f'{channel}'] = dataframe['channels'].apply(lambda x: 1 if channel in str(x) else 0)
    dataframe.drop('channels', axis=1, inplace=True)
    dataframe.rename(columns={'id': 'offer_id'}, inplace=True)
    logger.info('El archivo portfolio ha sido limpiado exitosamente!')
    return dataframe


def clean_profile_file(dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    :param dataframe:
    :return:
    """
    indices = dataframe[dataframe['gender'].isna()].index.tolist()
    dataframe.drop(indices, axis=0, inplace=True)
    dataframe.rename(columns={'id': 'person_id'}, inplace=True)
    logger.info('El archivo profile ha sido limpiado exitosamente!')
    return dataframe


def clean_transcript_file(dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    :type dataframe: pandas.DataFrame
    :param dataframe:
    :return:
    """
    get_offer_id = lambda x: x.get('offer_id') if 'offer_id' in x.keys() else \
        (x.get('offer id') if 'offer id' in x.keys() else np.nan)
    get_amount = lambda x: x.get('amount') if 'amount' in x.keys() else 0
    get_reward = lambda x: x.get('reward') if 'reward' in x.keys() else 0
    dataframe['offer_id'] = dataframe['value'].apply(get_offer_id)
    dataframe['amount'] = dataframe['value'].apply(get_amount)
    dataframe['reward'] = dataframe['value'].apply(get_reward)
    dataframe.drop('value', axis=1, inplace=True)
    dataframe.rename(columns={'person': 'person_id'}, inplace=True)
    logger.info('El archivo transcript ha sido limpiado exitosamente!')
    return dataframe


def clean_final_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """

    :param dataframe:
    :return:
    """
    rewards = []
    for x, y in zip(dataframe['reward_x'].values.tolist(), dataframe['reward_y'].values.tolist()):
        if x in values_reward and y in values_reward:
            rewards.append(x)
        elif x not in values_reward and y in values_reward:
            rewards.append(y)
        elif x in values_reward and y not in values_reward:
            rewards.append(x)
        else:
            rewards.append(np.nan)
    dataframe['reward'] = rewards
    dataframe.drop(['reward_x', 'reward_y'], axis=1, inplace=True)
    logger.info('El conjunto de datos final ha sido limpiado exitosamente!')
    return dataframe


def merge_dataframes(left_dataframe: pd.DataFrame, right_dataframe: pd.DataFrame, on: str) -> pd.DataFrame:
    """
    
    :param on:
    :param left_dataframe:
    :param right_dataframe: 
    :return: 
    """
    dataframe = pd.merge(left=left_dataframe, right=right_dataframe, how='left', on=on)
    logger.info('Se han unido ambos set de datos exitosamente!')
    return dataframe


def export_data(dataframe: pd.DataFrame, file_path: str):
    """

    :param dataframe:
    :param file_path:
    """
    logger.info('Se ha exportado el set de datos exitosamente!')
    return dataframe.to_csv(file_path, sep=';', index=False)


def main():
    portfolio = load_data(file_path=portfolio_filepath)
    portfolio = clean_portfolio_file(dataframe=portfolio)
    profile = load_data(file_path=profile_filepath)
    profile = clean_profile_file(dataframe=profile)
    transcript = load_data(file_path=transcript_filepath)
    transcript = clean_transcript_file(dataframe=transcript)
    final_dataframe = merge_dataframes(left_dataframe=transcript, right_dataframe=profile, on='person_id')
    final_dataframe = merge_dataframes(left_dataframe=final_dataframe, right_dataframe=portfolio, on='offer_id')
    final_dataframe = clean_final_dataframe(dataframe=final_dataframe)
    export_data(dataframe=portfolio, file_path=portfolio_outputpath)
    export_data(dataframe=profile, file_path=profile_outputpath)
    export_data(dataframe=transcript, file_path=transcript_outputpath)
    export_data(dataframe=final_dataframe, file_path=final_filepath)


if __name__ == '__main__':
    main()
