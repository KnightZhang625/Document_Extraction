# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   26/Dec/2018
# @Last Modified by:    
# @Last Modified time:  

import sys
import time

import networkx as nx
import numpy as np
from numpy.random import random
from sklearn.metrics.pairwise import cosine_similarity

class textRank(object):
    '''
        return : a dict which saves the textRank score for each sentence
    '''
    def __init__(self):
        pass
    
    def calculate_similarity(self, sentence_matrix):
        '''
            sentence_matrix : each row indicates a sentence embedding
        '''
        sentence_count = sentence_matrix.shape[0]                     # acquire the number of sentences
        similiar_matrix = np.zeros([sentence_count, sentence_count])  # initial the similarity matrix
        for i in range(sentence_count):
            for j in range(sentence_count):
                if i != j:
                    similiar_matrix[i][j] = cosine_similarity(sentence_matrix[i].reshape(1, -1), sentence_matrix[j].reshape(1, -1)).squeeze()
        return similiar_matrix

    def get_scores(self, similiar_matrix):
        '''
            [s1 s2 s3 ... s_n]              [s1]            [s1]
            [s1 s2 s3 ... s_n]              [s2]            [s2]
            [s1 s2 s3 ... s_n]              [s3]            [s3]
                    .               *         .       =       .
                    .                         .               .
                    .                         .               .
            [s1 s2 s3 ... s_n]              [s_n]           [s_n]
        '''
        nx_graph = nx.from_numpy_array(similiar_matrix)
        return nx.pagerank(nx_graph)

if __name__ == '__main__':
    # the below is just for test
    textrank = textRank()
    sentence_matrix = np.random.rand(10, 3)
    similiar_matrix = textrank.calculate_similarity(sentence_matrix)
    print(similiar_matrix)
    print(textrank.get_scores(similiar_matrix))
