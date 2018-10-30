import os
import json


class Config:
    """
    Read default configurations from setting file
    """
    PATH_PARENT = os.path.dirname(os.getcwd())

    config_file = "{0}/config.json".format(PATH_PARENT)

    with open(config_file) as cf:
        config = json.load(cf)

        APP_NAME = config['APP_NAME']
        VERSION = config['VERSION']
        ENV = int(config['ENV'])

        PATH_SRC = config['PATHS']['SOURCE_CODE']
        PATH_DATA = config['PATHS']['DATA']
        PATH_DATA_INPUT = config['PATHS']['DATA_INPUT']
        PATH_DATA_OUTPUT = config['PATHS']['DATA_OUTPUT']
        PATH_LOG = config['PATHS']['LOG']
        PATH_TESTING = config['PATHS']['TESTING']

        PATH_MALLET = config['SETTINGS']['MALLET_PATH']

        INPUT_NEWSGROUPS = config['INPUT']['NEWSGROUPS']

        OUTPUT_DOCUMENT_DOMINANT_TOPIC = config['OUTPUT']['DOCUMENT_DOMINANT_TOPIC']
        OUTPUT_TOPIC_REPRESENTATIVE_DOCUMENT = config['OUTPUT']['TOPIC_REPRESENTATIVE_DOCUMENT']
        OUTPUT_TOPIC_VOLUME_DISTRIBUTION = config['OUTPUT']['TOPIC_VOLUME_DISTRIBUTION']


class Environment:
    """
    Environment Values
    """
    UAT = 1
    PROD = 9
