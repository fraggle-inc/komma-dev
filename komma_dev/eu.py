# coding: utf8
"""
Functions for downloading and loading Danish EU data.

More info on the dataset here: http://www.statmt.org/europarl/
"""
import os
import os.path
import tarfile
import urllib.request
from pathlib import Path
import shutil


DATA_PATH = Path(__file__).parent.parent / 'data'
DATASETS = {'da': {'url': "http://www.statmt.org/europarl/v7/da-en.tgz", 'filename': "europarl-v7.da-en.da"}}


def load(language: str):
    """Load language with specified ISO 639-1 code. If the dataset does not yet exist, it will be downloaded."""
    if language not in DATASETS:
        raise NotImplementedError('Language not supported: ' + language)

    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)

    filename = DATASETS[language]['filename']
    if not os.path.exists(DATA_PATH / filename):
        url = DATASETS[language]['url']

        _download_tgz(url, filename, DATA_PATH)

    with open(DATA_PATH / filename) as file:
        print("loading:", filename)
        return file.read().splitlines()


def clear():
    """Remove all downloaded datasets."""
    if not os.path.isdir(DATA_PATH):
        return False

    print("removing downloaded data")
    shutil.rmtree(DATA_PATH)
    return True


def _download_tgz(url, file_to_extract, output_folder):
    print("downloading:", url)
    with urllib.request.urlopen(url) as response:
        tar = tarfile.open(fileobj=response, mode='r|gz')
        for item in tar:
            if item.name != file_to_extract:
                continue

            print("extracting:", item.name)
            tar.extract(item, output_folder)
            break
