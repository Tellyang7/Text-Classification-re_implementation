#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/1/8 16:28
# @Author : A yang7
from WordVectors import WordVectors
from FilesFetch import Token
from Wrapper import VectorizerWrapper, Transform2WordVectors

# from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

import sys
import time
import numpy as np
import logging
import initLogs


def get_neural_net(n_hidden, neurons):
    neural_net = (neurons,)
    for i in range(2,n_hidden):
        neural_net = neural_net + (neurons,)
    return neural_net


def process_args():
    args = sys.argv
    len_args = len(args) - 1
    rea_args = 5

    if (len_args != rea_args):
        logger.critical("argument error !!")
        sys.exit(0)
    else:
        word_corpus = str(args[1])  # 20-news
        min_df = int(args[2])  # minimum
        model = str(args[3])  # classifier
        token = str(args[4])  # stemmed, stopped
        word_vec = str(args[5])  # pre-trained word-vectors -> path

        if ((model == 'nb') and (word_vec != 'none')):
            logger.error("Run Naive Bayes without word vector....")
            exit(0)

        if (word_vec == "none"):
            word_vec = None

        if (((word_vec == 'word2vec') or (word_vec == 'pwe') or (word_vec == 'tbwe')) and token != 'stopped'):
            logger.error("for general embedding, use stopped only")
            exit(0)

    return word_corpus, min_df, model, token, word_vec


def def_models(min_df, classifier, wv=None):
    vectorizers = [
        ('counts', ("vectorizer", VectorizerWrapper(model=CountVectorizer(analyzer=lambda x: x, min_df=min_df)))),
        ('tf_idf', ("vectorizer", VectorizerWrapper(model=TfidfVectorizer(analyzer=lambda x: x, min_df=min_df))))
    ]
    transformer = ('transformer', Transform2WordVectors(wv_obj=wv))
    classifiers = {}
    # classifiers = {'nb': ("nb", MultinomialNB()), 'linearsvc': ("linearsvc", LinearSVC())}
    if (classifier == 'mlp'):
        mlpClassifiers = []
        for nHidden in [1, 2, 3]:
            for neurons in [50, 100, 200]:
                name = str(nHidden) + '-' + str(neurons)
                mlpClf = (name, MLPClassifier(hidden_layer_sizes=get_neural_net(nHidden, neurons), verbose=False))
                mlpClassifiers.append(mlpClf)
        classifiers['mlp'] = mlpClassifiers
    elif (classifier is "nb"):
        nb = ("nb", MultinomialNB())
        classifiers['nb'] = nb
    elif (classifier is "svm"):
        svm = ("linearsvc", LinearSVC())
        classifiers['svm'] = svm

    modelRuns = []

    for vectorizer in vectorizers:
        if(wv):
            name = 'vectorizer-' + vectorizer[0] + '-embed_' + wv.vec_source + '-' + classifier
        else:
            name = 'vectorizer-' + vectorizer[0] + '-no-embed-' + classifier

        if(classifier == "mlp"):
            for mlpClf in classifiers['mlp']:
                modelRun = (name + '-' + mlpClf[0], Pipeline([vectorizer[1], transformer, mlpClf]))
                modelRuns.append(modelRun)
        else:
            modelRun = (name, Pipeline([vectorizer[1], transformer, classifiers[classifier]]))
            modelRuns.append(modelRun)

    return modelRuns


def main(parameters=None):
    start0 = time.time()
    if(parameters is None):
        corpus, min_df, classifier, token, vec_source = process_args()
    else:
        corpus = parameters[0]
        vec_source = parameters[1]
        token = parameters[2]
        classifier = parameters[3]
        min_df = parameters[4]
    logger.info(
        'Running: WordCorpus: {}, Classifiers: {}, TokenType: {}, min_df: {}, wordVecSource: {}'
        .format(corpus, classifier, token, min_df, vec_source))

    X, y, class_names = Token(token=token).get_tokens(corpus)
    vocabulary_gen = CountVectorizer(analyzer=lambda x: x, min_df=min_df).fit(X)
    # This is only to generate a vocabulary with min_df
    corpus_vocab = sorted(vocabulary_gen.vocabulary_, key=vocabulary_gen.vocabulary_.get)
    logger.info(
        'Total Corpus Size: len(corpusVocab) with frequency > min_df : {}, X.shape: {}, y.shape: {}, # classes: {}'
        .format(len(corpus_vocab), X.shape, y.shape, len(class_names)))
    logger.info('Class Names:{}'.format(class_names))

    if (vec_source):
        wv = WordVectors(corpus=corpus, vocabulary=corpus_vocab, vec_source=vec_source, token=token)
        # nWords_in_this_set X wvLength
    else:
        wv = None

    results = {}
    results['timeForDataFetch'] = time.time() - start0
    logger.info('Time Taken For Data Fetch: {}'.format(results['timeForDataFetch']))

    modelRuns = def_models(min_df, classifier, wv)
    logger.info('Model Runs:\n{}'.format(modelRuns))

    if (corpus == 'twenty-news'):
        fraction = 0.2
        sss = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=0)
        sss.get_n_splits(X, y)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

    elif (corpus == 'acl-imdb'):
        X_train, y_train, class_names = Token(token=token).get_imdb('train')
        X_test, y_test, class_names = Token(token=token).get_imdb('test')

    marker = 'X y vocab: Train => Test:' + str(X_train.shape) + ',' + str(y_train.shape) + '=>' + str(
        X_test.shape) + ',' + str(y_test.shape)
    for name, model in modelRuns:
        results[name] = {}
        results[name][marker] = {}
        logger.info('\n\nCurrent Run: {} => {}'.format(name, marker))
        start = time.time()
        logger.info("Training Begin")
        model.fit(X_train, y_train)
        logger.info("Training End")
        logger.info("Prediction Begin")
        predicted = model.predict(X_test)
        logger.info("Prediction End")

        results[name][marker]['model_vocabulary_size'] = len(model.named_steps['vectorizer'].model.vocabulary_)
        results[name][marker]['confusion_matrix'] = confusion_matrix(y_test, predicted)
        results[name][marker]['timeForThisModel_fit_predict'] = time.time() - start

        logger.info('Run:{}, {}, Confusion Matrix:\n{}'.format(name, marker, results[name][marker]['confusion_matrix']))
        logger.info('Run:{}, {}, Classification Report:\n{}'.format(name, marker,
                                                                    classification_report(y_test, predicted,
                                                                                          target_names=class_names)))
        logger.info('Model Vocab Size:{}'.format(results[name][marker]['model_vocabulary_size']))
        logger.info('Time Taken For This Model Run:{}'.format(results[name][marker]['timeForThisModel_fit_predict']))

    results['overAllTimeTaken'] = time.time() - start0
    logger.info('Overall Time Taken:{}'.format(results['overAllTimeTaken']))
    logger.info("Prediction End")


if __name__ == '__main__':

    initLogs.setup()
    logger = logging.getLogger(__name__)
    np.set_printoptions(linewidth=100)
    parent_path = "/home/hadoop/yqi/Word2Vec/output/"

    base_names =["cbow-w5n10-bak.txt", "cbow-w5n20-bak.txt", "cbow-w10n10-bak.txt"]


    corpus_names = ['twenty-news', 'acl-imdb']
    minimum = 2
    token_type = "stopped"
    # classifiers = ['nb', 'linearsvc', 'mlp']
    c = "svm"

    for base in base_names:
        source = parent_path + base
        for corpus_name in corpus_names:
            parameter = (corpus_name, source, token_type, c, minimum)
            main(parameter)
