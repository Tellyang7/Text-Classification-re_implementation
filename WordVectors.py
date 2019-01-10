#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/1/8 15:04
# @Author : A yang7

from gensim.models import KeyedVectors
import numpy as np
import logging
logger = logging.getLogger(__name__)


class WordVectors():
    corpus = {}
    vocabulary = []
    token = 'stopped'
    vec_source = ''

    def __init__(self, corpus="twenty-news", vocabulary=None, vec_source="900-t20-01-skip-w10n10.txt",
                 token="stopped"):
        self.wvLength = 300
        WordVectors.corpus = corpus
        WordVectors.vocabulary = vocabulary
        WordVectors.vec_source = vec_source
        WordVectors.token = token
        self.get_vectors(vocabulary)

# path: end with .txt
    def get_vectors(self, vocab):
        embed_model = KeyedVectors.load_word2vec_format(self.vec_source)
        zero_v = np.zeros(self.wvLength)
        vectors = []
        n_zero = 0
        for word in vocab:
            if(word in embed_model.index2word):
                vectors.append(embed_model.get_vector(word))
            else:
                vectors.append(zero_v)
                n_zero = n_zero + 1
        logger.info('# of words without embeddings:{}'.format(n_zero))
        vectors = np.array([np.array(xi) for xi in vectors])
        return vectors

