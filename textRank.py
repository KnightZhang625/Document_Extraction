# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   26/Dec/2018
# @Last Modified by:    
# @Last Modified time:  

import re
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.random import random
from sklearn.metrics.pairwise import cosine_similarity

from featureExtraction import FeatureExtraction


class TextRank(object):
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
    # # the below is just for test
    # textrank = TextRank()
    # sentence_matrix = np.random.rand(10, 3)
    # similiar_matrix = textrank.calculate_similarity(sentence_matrix)
    # print(similiar_matrix)
    # print(textrank.get_scores(similiar_matrix))

    cwd = Path(__file__).absolute().parent
    vec_dir = cwd / 'vec'
    vec_files = [file for file in vec_dir.iterdir()]
    
    vec_dict = {}
    for file in vec_files:
        if re.search(r'.+\.vec', str(file)):
            vec_dict['vec_dir'] = file
        elif re.search(r'.+\.vocab', str(file)):
            vec_dict['vocab'] = file
        elif re.search(r'.+\.dict', str(file)):
            vec_dict['dict'] = file
    
    # doc = ['我毕业于谢菲尔德大学', '今天天气很好']

    doc = u'''今天湖人队打的非常好，科比此役复出贡献35分，5个篮板，10个助攻的数据，詹姆斯也贡献15分，12个篮板，12个助攻的准三双数据。
            NBA今天有十场比赛，其中最引人注目的就是洛杉矶湖人对金州勇士。科比拿到35分，复出新高，詹姆斯也拿到三双数据。湖人现在位列西部第二，勇士位列第一。
            在今天结束的比赛中，圣安东尼奥马刺队在莱昂纳德的带领下，引来挑战的火箭队。此役保罗贡献10个助攻，但是在第三节受伤离场。另一场焦点之战，湖人队大比分战胜勇士队。
            体坛快讯最新报道。科比迎来复出首场比赛，拿到全场最高35分带领全队取得胜利。詹姆斯取得三双。另一场比赛步行者战胜小牛。
    '''
    doc = list(set([sentence for sentence in doc.split('。')]))
    print(doc)
    # sys.exit()
    feature_extraction = FeatureExtraction(vec_dict)
    feature_matrix = feature_extraction.extract(doc)
    
    textRank = TextRank()
    similiar_matrix = textRank.calculate_similarity(feature_matrix)
    results = textRank.get_scores(similiar_matrix)
    results_sort = sorted(results.items(), key=lambda item:item[1], reverse=True)
    extract_info = [doc[index] for index, _ in results_sort[:3]]
    print(extract_info)