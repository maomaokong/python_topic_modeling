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
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

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
from my_config import Environment as env


def remove_stopwords(texts, stop_words):
    """
    Define functions for stopwords

    :param texts: Processed texts from main module
    :return: Texts that already removed a stopwords
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    """
    Define functions for bigrams

    :param texts: Processed texts from main module
    :return: Texts that already processed with bigram model
    """
    # Build the bigram models
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases

    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    """
    Define functions for trigram

    :param texts: Processed texts from main module
    :return: Texts that already processed with trigram model
    """
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatisation(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_output = []

    # Initialise spacy 'en' model, keeping only tagger component (for efficiency)
    # $ python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_output


def sent_to_words(sentences):
    """
    Tokenise each sentence that remove punctuations.

    :param sentences: sentences from JSON
    :return: sentences after processed
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def remove_unnecessary_char(raw_texts):
    """
    Clean up unnecessary characters.

    :param raw_texts: Raw text from main module
    :return: Cleaned-up text
    """

    """
    Remove emails

    \S: Matches any non-whitespace character; this is equivalent to the class [^ \t\n\r\f\v].
    *: Greedily matches the expression to its left 0 or more times.
    \s: Matches any whitespace character; this is equivalent to the class [ \t\n\r\f\v].
    ?: ? | Greedily matches the expression to its left 0 or 1 times. But if ? is added to qualifiers (+, *, and ? itself) it will perform matches in a non-greedy manner
    """
    """
    for txt in raw_texts:
        proc_texts = re.sub('\S*@\S*\s?', '', txt)
    """
    proc_texts = [re.sub('\S*@\S*\s?', '', txt) for txt in raw_texts]
    #print()
    #print(proc_texts)

    # Remove new line characters
    proc_texts = [re.sub('\s+', ' ', txt) for txt in proc_texts]
    #print()
    #print(proc_texts)

    # Remove distracting single quotes
    proc_texts = [re.sub("\'", "", txt) for txt in proc_texts]
    #print()
    #print(proc_texts)

    #print()
    #pprint(proc_texts[:1])

    proc_text_words = list(sent_to_words(proc_texts))

    #print()
    #print(proc_text_words[:1])

    return proc_text_words


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    if cfg.ENV == env.UAT:
        print()
        print("# Topic Modeling running in UAT environment! #")
    elif cfg.ENV == env.PROD:
        print()
        print("# Topic Modeling running in PROD environment! #")
        print("## --This will take more longer time to run the application!  ##")

    stop_words = sw.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Import NewsGroups Dataset
    newsgroups_file = "{0}/{1}/{2}/{3}".format(
        cfg.PARENT_PATH, cfg.PATH_DATA, cfg.PATH_DATA_INPUT, cfg.INPUT_NEWSGROUPS)
    df = pd.read_json(newsgroups_file)
    #print()
    #print(df.target_names.unique())

    if cfg.ENV == env.UAT:
        df = df.head(2)  # Temporary only work with first two items

    #print()
    #print(df)

    # Convert to list
    data_words_1 = df.content.values.tolist()
    #print()
    #print(data_words_1)

    data_words_2 = remove_unnecessary_char(data_words_1)
    #print()
    #print(data_words_2)

    # Remove Stop Words
    data_words_3 = remove_stopwords(data_words_2, stop_words)
    #print()
    #print(data_words_3)

    # Form Bigrams
    data_words_4 = make_bigrams(data_words_3)
    #print()
    #print(data_words_4)

    # Do lemmatization keeping only noun, adj, verb and adverb
    data_words_5 = lemmatisation(data_words_4, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #print()
    #print(data_words_5[:1])

    """
    Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of (word_id, word_frequency).

    For example, (0, 1) above implies, word id 0 occurs once in the first document. Likewise, word id 1 occurs twice and so on.
    """
    # Create Dictionary
    id2word = corpora.Dictionary(data_words_5)

    # Create Corpus
    #texts = data_words_5
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_words_5]

    #print()
    #print([[(words2id[wid], freq) for wid, freq in cp] for cp in corpus[:1]])

    """
    Have everything required to train the LDA model.
    
    --'alpha' are hyperparameters that affect sparsity of the topic. According to the Gensim docs, both defaults to 1.0/num_topics prior.
    --'chucksize' is the number of documents to be used in each training chuck.
    --'update_every' determines how often the model parameters should be updated.
    --'passes' is the total number of training passes
    """
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus
                                                , id2word=id2word
                                                , num_topics=20
                                                , random_state=100
                                                , update_every=1
                                                , chunksize=100
                                                , passes=10
                                                , alpha='auto'
                                                , per_word_topics=True
                                                )

    """
    How to interpret the print_topics() output?
    
    Topic 0 is a represented as (weight * keywords):
    0.104*"max" + 0.016*"memory" + 0.015*"disk" + 0.012*"brian" +
    0.009*"switch" + 0.009*"power" + 0.007*"datum" + 0.007*"connect" + 
    0.006*"volt" + 0.006*"performance"
    
    The weight reflect how important a keyword is to that topic.
    """
    # Print the Keyword in the 10 topics
    #print()
    #pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    """
    Model perplexity and topic coherence provide a convenient measure to judge how good a given topic model is.
    """
    # Compute Perplexity, a measure of how good the model os. Lower the better.
    print()
    print("Perplexity: {0}".format(lda_model.log_perplexity(corpus)))

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model
                                         , texts=data_words_5
                                         , dictionary=id2word
                                         , coherence='c_v'
                                         )
    coherence_lda = round(coherence_model_lda.get_coherence(), 4)
    print()
    print("Coherence Score: {0}".format(coherence_lda))

    # Visualise the topics
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds')
    pyLDAvis.show(vis)


if __name__ == '__main__':
    main()
