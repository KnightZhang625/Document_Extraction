# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   26/Dec/2018
# @Last Modified by:    
# @Last Modified time: 

import numpy as np
from preProcess import PreprocessAPI
from collections import Counter

class featureExtraction(object):
    '''
        return : sentence embedding matrix
    '''
    embedding_dim = 5           # just for test
    def __init__(self):
        pass
    
    @classmethod
    def extract(cls, documents, stopword_path=None, user_dict_path=None):
        '''
            args : documents should be list contains multiple sentences
        '''
        preprocess = PreprocessAPI(stopword_path, user_dict_path)                   # tokenize the sentence
        sentence_tokenized = preprocess.preProcess(documents)   
        return cls._build_sentence_embedding(sentence_tokenized)
    
    @classmethod
    def _build_sentence_embedding(cls, sentence_tokenized):
        sentence_matrix = np.zeros([len(sentence_tokenized), cls.embedding_dim])    
        for index, sentence in enumerate(sentence_tokenized):
            vocabulary = sentence.split(' ')                                        # split each word
            sentence_matrix[index, ] = cls._calculate_each_sentence_embedding(vocabulary)
        return sentence_matrix
    
    @classmethod
    def _calculate_each_sentence_embedding(cls, vocabulary):
        # the code below just for test
        temp_dict = Counter()
        temp_dict['我'] = np.array([1, 2, 3, 4, 5])
        temp_dict['很好'] = np.array([5, 4, 3, 2, 1])
        # end for test

        single_sentence_matrix = np.zeros([len(vocabulary), cls.embedding_dim])
        for index, word in enumerate(vocabulary):
            single_sentence_matrix[index, ] = temp_dict[word]
        return np.mean(single_sentence_matrix, axis=0)                          # use the mean of value of each word as the sentence embedding

if __name__ == '__main__':
    # the below is just for test
    print(featureExtraction.extract(['我毕业于谢菲尔德大学', '今天天气很好']))
