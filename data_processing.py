# Code extracted from https://www.kaggle.com/shahules/cord-tools-and-knowledge-graphs

from typing import *

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize


stopword_list = stopwords.words('english')
lem = WordNetLemmatizer()
stem = PorterStemmer()


def get_stopword_list():
    return stopword_list


def process_df_col(
    df: pd.core.frame.DataFrame,
    col: str,
) -> List[str]:
    df_value_list = df[col].dropna().values.tolist()
    df_corpus_list = ' '.join(df_value_list).split()

    proc_df_corpus_list = [lem.lemmatize(df_word.lower())
                           for df_word in df_corpus_list
                           if df_word not in stopword_list]

    return proc_df_corpus_list


def get_top_ngram(
    corpus: pd.core.series.Series,
    n: int,
):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:10]


def preprocess_news(
    df: pd.core.frame.DataFrame,
):
    corpus = []
    for title in df['title'].dropna()[:5000]:
        word_list = [tokenized_word for tokenized_word
                     in word_tokenize(title)
                     if (tokenized_word not in stopword_list)]

        word_list = [lem.lemmatize(word)
                     for word in word_list
                     if len(word) > 2]

        corpus.append(word_list)
    return corpus
