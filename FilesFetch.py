#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/1/7 16:58
# @Author : A yang7

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups_vectorized

import numpy as np
import string
import os

n_stop = stopwords.words('english')
Stem = SnowballStemmer("english")

# twenty_news_all = fetch_20newsgroups(
#       subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
# twenty_news_train = fetch_20newsgroups(
#    subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
# twenty_news_test = fetch_20newsgroups(
#    subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

""" bunch : Bunch object
        bunch.data: sparse matrix, shape [n_samples, n_features]
        bunch.target: array, shape [n_samples]
        bunch.target_names: list, length [n_classes]
        bunch.DESCR: a description of the dataset.
"""
twenty_news_bunch = fetch_20newsgroups_vectorized(
    subset="all", remove=('headers', 'footers', 'quotes'))


def untokenize(tokens):
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def tokenize(text):        # no punctuation & starts with a letter & between 2-15 characters in length
    tokens = [word.strip(string.punctuation) for word in
              RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return tokens


def remove_stopwords(tokens):
    filtered_tokens = [f.lower() for f in tokens if f and f.lower() not in n_stop]
    return filtered_tokens


def stem(filtered_tokens):      # stemmed & > 2 letters
    return [Stem.stem(token) for token in filtered_tokens if len(token) > 1]


class Token():

    def __init__(self, token):
        self.token = token

    def get_20_news(self, sub):
        twenty_news = fetch_20newsgroups(subset=sub,
                                         remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
        X, y, class_names = [], [], set()
        # print(len(twenty_news_all))
        for i, article in enumerate(twenty_news['data']):
            stopped = remove_stopwords(tokenize(article))
            stemmed = stem(stopped)

            group_index = twenty_news['target'][i]
            group_name = twenty_news['target_names'][group_index]

            if (self.token == "stopped"):
                X.append(stopped)
            else:
                X.append(stemmed)
            y.append(group_index)
            class_names.add(group_name)
        X = np.array([np.array(xi) for xi in X])  # rows: Docs. columns: words
        print(X.shape)
        print(len(y))
        print(sorted(list(class_names)))

        return X, np.array(y), sorted(list(class_names))

    # notice that the dir path of data set "imdb" should be same with this file(.py)
    def get_imdb(self, sub):
        X, y, class_names = [], [], set()
        # for data_set in ['train', 'test']:
        for classIndex, directory in enumerate(['neg', 'pos']):
            dir_name = os.getcwd() + os.sep + "aclImdb" + os.sep + sub + os.sep + directory
            for reviewFile in os.listdir(dir_name):
                    # print(reviewFile)
                with open(dir_name + os.sep + reviewFile, 'r', encoding="utf-8") as f:
                    article = f.read()
                stopped = remove_stopwords(tokenize(article))
                stemmed = stem(stopped)
                # fileName = dir_name + '/' + reviewFile
                group_index = classIndex
                group_name = directory

                if (self.token == "stopped"):
                    X.append(stopped)
                else:
                    X.append(stemmed)

                y.append(group_index)
                class_names.add(group_name)
        X = np.array([np.array(xi) for xi in X])  # rows: Docs. columns: words
        return X, np.array(y), sorted(list(class_names))

    def get_imdb_all(self):
        X, y, class_names = [], [], set()
        for name in ['train', 'test']:
            result = self.get_imdb(name)
            for xi in result[0]:
                X.append(xi)
            for yi in result[1]:
                y.append(yi)
            for class_name in result[2]:
                class_names.add(class_name)

        X = np.array([np.array(xi) for xi in X])  # rows: Docs. columns: words
        return X, np.array(y), sorted(list(class_names))

    def get_tokens(self, corpus="twenty-news"):
        if (corpus == 'twenty-news'):
            return self.get_20_news("all")
        else:
            return self.get_imdb_all()


if __name__ == "__main__":
    print(len(Token("stopped").get_imdb("test")[1]))
    print(len(Token("stopped").get_tokens("imdb")[1]))
