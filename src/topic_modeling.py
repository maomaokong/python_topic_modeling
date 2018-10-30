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
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim
import logging
import warnings

# NLTK Stopwords
from nltk.corpus import stopwords

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
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


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
    # print()
    # print(proc_texts)

    # Remove new line characters
    proc_texts = [re.sub('\s+', ' ', txt) for txt in proc_texts]
    # print()
    # print(proc_texts)

    # Remove distracting single quotes
    proc_texts = [re.sub("\'", "", txt) for txt in proc_texts]
    # print()
    # print(proc_texts)

    # print()
    # pprint(proc_texts[:1])

    proc_text_words = list(sent_to_words(proc_texts))

    # print()
    # print(proc_text_words[:1])

    return proc_text_words


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics.

    :param dictionary: Gensim dictionary
    :param corpus: Gensim corpus
    :param texts: List of input texts
    :param limit: Maximum number of topics
    :param start:
    :param step:

    :return model_list: List of LDA topic models
    :return coherence_values: Coherence values corresponding to the LDA model with respective number of topics
    """
    model_list = []
    coherence_values = []

    for num in range(start, limit, step):
        # Please update MALLET_PATH in config.json file
        model = gensim.models.wrappers.LdaMallet(cfg.PATH_MALLET
                                                 , corpus=corpus
                                                 , num_topics=num
                                                 , id2word=dictionary
                                                 )
        model_list.append(model)

        coherence_model = CoherenceModel(model=model
                                         , texts=texts
                                         , dictionary=dictionary
                                         , coherence='c_v'
                                         )
        coherence_values.append(round(coherence_model.get_coherence(), 4))

    return model_list, coherence_values


def format_topics_sentences(ldamodel, corpus, texts):
    """
    Determine what topic a given document is about.

    Find the topic number that has the highest percentage contribution in that document.

    :param ldamodel: an optimal LDA Model from main module
    :param corpus: a corpus data from main module
    :param texts: a processed texts data from main module

    :return: Pandas DataFrame
    """
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document.
    for ii, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Percentage Contribution and Keywords for each document.
        for jj, (topic_num, proc_topic) in enumerate(row):
            if jj == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series(
                        [
                            int(topic_num)
                            , round(proc_topic, 4)
                            , topic_keywords
                        ]
                    )
                    , ignore_index=True
                )
            else:
                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return sent_topics_df


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

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Import NewsGroups Dataset
    newsgroups_file = "{0}/{1}/{2}/{3}".format(
        cfg.PATH_PARENT, cfg.PATH_DATA, cfg.PATH_DATA_INPUT, cfg.INPUT_NEWSGROUPS)
    df = pd.read_json(newsgroups_file)
    # print()
    # print(df.target_names.unique())

    if cfg.ENV == env.UAT:
        df = df.head(2)  # Temporary only work with first two items

    # print()
    # print(df)

    # Convert to list
    raw_words = df.content.values.tolist()
    # print()
    # print(raw_words)

    # Remove email, new line characters and single quote
    removed_unnecessary_words = remove_unnecessary_char(raw_words)
    # print()
    # print(removed_unnecessary_words)

    # Remove Stop Words
    removed_stopwords = remove_stopwords(removed_unnecessary_words, stop_words)
    # print()
    # print(removed_stopwords)

    # Form Bigrams
    bigram_words = make_bigrams(removed_stopwords)
    # print()
    # print(bigram_words)

    # Do lemmatization keeping only noun, adj, verb and adverb
    lemmatised_words = lemmatisation(bigram_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # print()
    # print(lemmatised_words[:1])

    """
    Gensim creates a unique id for each word in the document. The produced corpus shown above is a mapping of (word_id, word_frequency).

    For example, (0, 1) above implies, word id 0 occurs once in the first document. Likewise, word id 1 occurs twice and so on.
    """
    # Create Dictionary
    id2word = corpora.Dictionary(lemmatised_words)

    # Create Corpus
    # texts = lemmatised_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in lemmatised_words]

    # print()
    # print([[(words2id[wid], freq) for wid, freq in cp] for cp in corpus[:1]])

    # Take a long time to run
    start = 2
    limit = 40
    step = 6

    model_list, coherence_values = compute_coherence_values(dictionary=id2word
                                                            , corpus=corpus
                                                            , texts=lemmatised_words
                                                            , start=start
                                                            , limit=limit
                                                            , step=step
                                                            )

    # Show graph
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend("Coherence_values", loc='best')

    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Number Topics = {0} has Coherence Value of {1}".format(m, round(cv, 4)))

    #plt.show()

    """
    Finding the dominant topic in each sentence
    """
    # Select the model and print the topics
    # **Problem - how to get the best model??
    optimal_model = model_list[5]
    #model_topics = optimal_model.show_topics(formatted=False)
    #pprint(optimal_model.print_topics(num_words=10))

    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=lemmatised_words)

    # Reset Index and Columns Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib.', 'Keywords', 'Text']

    # Show 'Dominant Topic for each document'
    #print(df_dominant_topic.head(10))
    df_dominant_topic.to_csv(
        "{0}/{1}/{2}/{3}".format(
            cfg.PATH_PARENT
            , cfg.PATH_DATA
            , cfg.PATH_DATA_OUTPUT
            , cfg.OUTPUT_DOCUMENT_DOMINANT_TOPIC
        )
    )

    """
    Find the most representative document for each topic
    """
    # Group top 5 sentences under each topic
    sent_topics_sorted_df_mallet = pd.DataFrame()

    sent_topics_out_df_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for ii, grp in sent_topics_out_df_grpd:
        sent_topics_sorted_df_mallet = pd.concat(
            [
                sent_topics_sorted_df_mallet,
                grp.sort_values(
                    ['Perc_Contribution'],
                    ascending=[0]
                ).head(1)
            ],
            axis=0
        )

    # Reset Index and Format Columns
    sent_topics_sorted_df_mallet.reset_index(drop=True, inplace=True)
    sent_topics_sorted_df_mallet.columns = ['Topic_Num', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show 'Most Representative document for each topic'
    #print(sent_topics_sorted_df_mallet.head())
    sent_topics_sorted_df_mallet.to_csv(
        "{0}/{1}/{2}/{3}".format(
            cfg.PATH_PARENT
            , cfg.PATH_DATA
            , cfg.PATH_DATA_OUTPUT
            , cfg.OUTPUT_TOPIC_REPRESENTATIVE_DOCUMENT
        )
    )

    """
    Topic Distribution across documents
    """
    # Number of documents for each topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

    # Percentage of documents for each topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)

    # Topic Number and Keywords
    topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

    # Concentrate Column Wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

    # Change column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    # Show 'Topic Volume Distribution'
    #print(df_dominant_topics.head())
    df_dominant_topics.to_csv(
        "{0}/{1}/{2}/{3}".format(
            cfg.PATH_PARENT
            , cfg.PATH_DATA
            , cfg.PATH_DATA_OUTPUT
            , cfg.OUTPUT_TOPIC_VOLUME_DISTRIBUTION
        )
    )


if __name__ == '__main__':
    main()
