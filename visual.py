# Code extracted from https://www.kaggle.com/shahules/cord-tools-and-knowledge-graphs

import os
from collections import Counter
from typing import *

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import pyLDAvis.gensim
import gensim

import data_processing as proc


def print_title_len_distribution(
    df: pd.core.frame.DataFrame,
):
    title_len = df['title'].str.len()
    seaborn.distplot(title_len)
    plt.show()


def plot_most_common(
    df: pd.core.frame.DataFrame,
    col: str
):
    proc_title_list = proc.process_df_col(
        df=df,
        col=col,
    )

    stopword_list = proc.get_stopword_list()

    counter = Counter(proc_title_list)
    most_common = counter.most_common()

    common_word_list = []
    word_count_list = []
    for word, count in most_common[:10]:
        if word not in stopword_list:
            common_word_list.append(word)
            word_count_list.append(count)

    plt.figure(figsize=(9, 7))
    seaborn.barplot(
        x=word_count_list,
        y=common_word_list
    )


def plot_top_ngrams(
    df: pd.core.frame.DataFrame,
):
    top_n_bigrams = proc.get_top_ngram(df['title'].dropna(), 2)[:10]
    x, y = map(list, zip(*top_n_bigrams))
    plt.figure(figsize=(9, 7))
    sns.barplot(x=y, y=x)

    plt.show()


def generate_pyLDAvis(
    df: pd.core.frame.DataFrame,
):
    corpus = proc.preprocess_news(df)
    dic = gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]

    lda_model = gensim.models.LdaMulticore(
        bow_corpus,
        num_topics=4,
        id2word=dic,
        passes=10,
        workers=2,
    )

    vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    pyLDAvis.save_html(vis, 'index_hdp.html')
