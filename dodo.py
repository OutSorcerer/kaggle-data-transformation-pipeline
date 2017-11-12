import os
import sys
import configparser
import doit
from doit.cmd_base import ModuleTaskLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import inspect
from os.path import isfile
import kaggle
import baseline_model


DOIT_CONFIG = {
    'default_tasks': ['train'],
    'verbosity': 2
}


def read_credentials():
    # By default use environment variables to inject credentials by Docker, for example.
    # Alternatively, override environment variables from local .credentials.ini file with the following format:
    #
    # [kaggle]
    # login=your_login
    # password=your_password
    #
    if isfile('.credentials.ini'):
        config = configparser.ConfigParser()
        config.read('.credentials.ini')
        os.environ["KAGGLE_LOGIN"] = config['kaggle']['login']
        os.environ["KAGGLE_PASSWORD"] = config['kaggle']['password']


def task_read_credentials():
    return {
        'actions': [read_credentials]
    }


def download_train():
    login = os.environ["KAGGLE_LOGIN"]
    password = os.environ["KAGGLE_PASSWORD"]
    kaggle.download('statoil-iceberg-classifier-challenge', 'train.json.7z', login, password,
                    'data/train.json.7z')


def task_download_train():
    return {
        'actions': [download_train],
        'uptodate': [True],
        'setup': ['read_credentials'],
        'targets': ['data/train.json.7z']
    }


def download_test():
    login = os.environ["KAGGLE_LOGIN"]
    password = os.environ["KAGGLE_PASSWORD"]
    kaggle.download('statoil-iceberg-classifier-challenge', 'test.json.7z', login, password,
                    'data/test.json.7z')


def task_download_test():
    return {
        'actions': [download_test],
        'uptodate': [True],
        'setup': ['read_credentials'],
        'targets': ['data/test.json.7z']
    }


# requires 7zip
# sudo apt-get install p7zip-full
def task_unzip_train():
    return {
        'actions': ['bash -c "7z e data/train.json.7z -y -odata"'],
        'file_dep': ['data/train.json.7z'],
        'targets': ['data/train.json']
    }


def task_unzip_test():
    return {
        'actions': ['bash -c "7z e data/test.json.7z -y -odata"'],
        'file_dep': ['data/test.json.7z'],
        'targets': ['data/test.json']
    }


# TODO: extract convert_train_to_numpy to a separate file and replace file_dep from baseline_model.py to that file.
def task_convert_train_to_numpy():
    return {
        'actions': [baseline_model.convert_train_to_numpy],
        'file_dep': ['baseline_model.py', 'data/train.json'],
        'targets': ['data/train.npy']
    }


def task_train():
    return {
        'actions': [baseline_model.train],
        'file_dep': ['baseline_model.py', 'data/train.npy']
    }


def train():
    doit.doit_cmd.DoitMain(ModuleTaskLoader(sys.modules[__name__])).run(['train'])


if __name__ == "__main__":
    train()
