#!/usr/bin/python

"""
Import Packages / Libraries
"""
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora
from gensim.utils import simple_preprocess as spp
from gensim.models import CoherenceModel as cm

# spacy for lemmatisation
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib as plt
# %matplotlib inline

# Enable logging for gensim
import logging
import warnings

# NLTK Stopwords
from nltk.corpus import stopwords as sw

from my_config import Config as cfg


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    stop_words = sw.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Import Newsgroup Dataset
    newsgroup_file = "{0}/{1}/{2}/{3}".format(
        cfg.PARENT_PATH, cfg.PATH_DATA, cfg.PATH_DATA_INPUT, cfg.INPUT_NEWSGROUPS)
    df = pd.read_json(newsgroup_file)
    print(df.target_names.unique())
    df.head()


if __name__ == '__main__':
    main()
