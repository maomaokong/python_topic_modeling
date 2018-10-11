# Topic Modeling
Topic Modeling is a technique to extract the hidden topics from large volumes of text.
Latent Dirichlet Allocation(LDA) is a popular algorithm for topic modeling with excellent implementations in the Python’s Gensim package.

# Objective
Implementing Topic Modeling using Python with Gensim library

# Prerequisites
Download "nltk" stopwords and "spacy" model

1. Download "nltk" in python console. For information about Natural Language Toolkit (nltk) and stop words, please refer to this [url](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/).
>>> import nltk
>>> nltk.download('stopwords', download_dir={Path})

2. Download "spacy" in terminal
$ pip install spacy --user
$ python3 -m spacy download en

3. Download "gensim" in terminal
$ pip install gensim --user

4. Download "pyLDAvis" in terminal
$ pip install pyLDAvis --user

# Input Data
We will be using the Newsgroups dataset and this newsgroup.json contains about 11k newsgroups post from multiple different topics.

# How to run the script
Use the following command to execute the script under the **src** folder:

$ python ???.py

# Result
The result will generated under **data/output/** folder.

# Setting Configuration
Please adjust the settings from _config.json_ file.

# Language
Python 3.7.0

# Libraries
01. Gensim
02. NumPy

# Others
1. [Code Beautify](https://codebeautify.org/jsonvalidator)

# Reference
1. [Gensim Installation](https://radimrehurek.com/gensim/install.html)
2. [Gensim Tutorial](https://radimrehurek.com/gensim/tutorial.html)