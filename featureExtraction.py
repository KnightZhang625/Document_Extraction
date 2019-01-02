# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   26/Dec/2018
# @Last Modified by:    Jiaxin Zhang
# @Last Modified time:  27/Dec/2018  

import re
import sys
from collections import Counter
from pathlib import Path

import bcolz
import numpy as np
from sklearn.externals import joblib

from preProcess import PreprocessAPI

# np.random.seed(3) comment this if don't use 42th line

class WordToVector(object):
    '''
        returns : the word embedding vectors for querying word
        this class is based on the code written by shengli wang
    '''
    def __init__(self, vec_dict):
        self.word2idx, self.vocab, self.vectors = self._load_vec(vec_dict)

    def _load_vec(self, vec_dict):
        return joblib.load(vec_dict['dict'], 'r'), joblib.load(vec_dict['vocab'], 'r'), np.array([v for v in bcolz.open(vec_dict['vec_dir'], 'r')])
    
    def get(self, word):
        try:
            idx = self.word2idx[word]
            return self.vectors[idx]
        except KeyError:
            return self._process_unk_word(word)
    
    def _process_unk_word(self, word):
        vector = np.zeros(300)
        for char in word:
            idx = self.word2idx.get(char)
            # vector_temp = self.vectors[idx] if idx else np.random.rand(300)    
            vector_temp = self.vectors[idx] if idx else np.zeors(300)    
            vector += vector_temp
        return vector

class FeatureExtraction(object):
    '''
        return : sentence embedding matrix
    '''
    def __init__(self, vec_dict):
        self.embedding_dim = 300
        self.word2vector = WordToVector(vec_dict)
    
    def extract(self, documents, stopword_path=None, user_dict_path=None):
        '''
            args : documents should be list contains multiple sentences
        '''
        preprocess = PreprocessAPI(stopword_path, user_dict_path)                   # tokenize the sentence
        sentence_tokenized = preprocess.preProcess(documents)   
        return self._build_sentence_embedding(sentence_tokenized)
    
    def _build_sentence_embedding(self, sentence_tokenized):
        sentence_matrix = np.zeros([len(sentence_tokenized), self.embedding_dim])    
        for index, sentence in enumerate(sentence_tokenized):
            vocabulary = sentence.split(' ')                                        # split each word
            sentence_matrix[index, ] = self._calculate_each_sentence_embedding(vocabulary)
        return sentence_matrix

    def _calculate_each_sentence_embedding(self, vocabulary):
        single_sentence_matrix = np.zeros([len(vocabulary), self.embedding_dim])
        for index, word in enumerate(vocabulary):
            single_sentence_matrix[index, ] = self.word2vector.get(word)
        return np.mean(single_sentence_matrix, axis=0)                          # use the mean of value of each word as the sentence embedding

if __name__ == '__main__':
    pass
    # # sample use
    # cwd = Path(__file__).absolute().parent
    # vec_dir = cwd / 'vec'
    # vec_files = [file for file in vec_dir.iterdir()]
    
    # vec_dict = {}
    # for file in vec_files:
    #     if re.search(r'.+\.vec', str(file)):
    #         vec_dict['vec_dir'] = file
    #     elif re.search(r'.+\.vocab', str(file)):
    #         vec_dict['vocab'] = file
    #     elif re.search(r'.+\.dict', str(file)):
    #         vec_dict['dict'] = file
    
    # doc = ['我毕业于谢菲尔德大学', '今天天气很好']
    # feature_extraction = FeatureExtraction(vec_dict)

    # # the code below is just for test
    # word2idx = joblib.load(vec_dict['dict'], 'r')
    # print(len(set(word2idx)))
    # vocab = joblib.load(vec_dict['vocab'], 'r')
    # print(len(set(vocab)))
    # print(list(set(vocab).difference(set(word2idx.keys()))))
    # vectors = np.array([v for v in bcolz.open(vec_dict['vec_dir'], 'r')])
    # print(vectors.shape)
