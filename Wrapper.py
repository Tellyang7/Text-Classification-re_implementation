#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/1/8 10:54
# @Author : A yang7
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
logger = logging.getLogger(__name__)

np.set_printoptions(threshold=np.nan)


__all__ = ['VectorizerWrapper', 'Transform2WordVectors']


def log_trace(message, args):
    logger.info('In {}. #args:{}'.format(message, len(args)))
    for arg in args:
        logger.info("\t{}".format(type(arg)))


class VectorizerWrapper (TransformerMixin, BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, *args):
        log_trace("VectorizerWrapper: fit.", args)
        self.model.fit(args[0], args[1])
        return self

    def transform(self, *args):
        log_trace("VectorizerWrapper: transform.", args)
        return {'sparseX': self.model.transform(args[0]), 'vocab': self.model.vocabulary_}


class Transform2WordVectors (BaseEstimator, TransformerMixin):

    wv_obj = None

    def __init__(self, wv_obj=None):
        Transform2WordVectors.wv_obj = wv_obj

    def fit(self, *args):
        log_trace('Transform2WordVectors: fit.', args)
        return self

    def transform(self, *args):
        log_trace("Transform2WordVectors: transform.", args)
        sparse_x = args[0]['sparseX']
        if(not Transform2WordVectors.wv_obj):
            return sparse_x
        else:
            vocab = args[0]['vocab']
            sorted_words = sorted(vocab, key=vocab.get)
            logger.info('sortedWords. type:{}, size:{}'.format(type(sorted_words), len(sorted_words)))
            word_vec = Transform2WordVectors.wv_obj.get_vectors(sorted_words)
            logger.info('wordVectors. type:{}, shape:{}'.format(type(word_vec), word_vec.shape))

            reduce_matrix = self.sparse_multiply(sparse_x, word_vec)
            logger.info("reduce_matrix. type:{}, shape:{}".format(type(reduce_matrix), reduce_matrix.shape))

        return reduce_matrix

    @staticmethod
    def sparse_multiply(sparse_x, word_vectors):
        wv_length = len(word_vectors[0])
        reduce_matrix = []

        for row in sparse_x:
            new_row = np.zeros(wv_length)
            for non_zero_location, value in list(zip(row.indices, row.data)):
                new_row = new_row + value * word_vectors[non_zero_location]
            reduce_matrix.append(new_row)
        reduce_matrix = np.array([np.array(xi) for xi in reduce_matrix])
        return reduce_matrix

