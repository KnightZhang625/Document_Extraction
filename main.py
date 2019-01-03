# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   02/Jan/2019
# @Last Modified by:    
# @Last Modified time:  

import os
import re
import sys
import time
from pathlib import Path

from featureExtraction import FeatureExtraction
from sentenceSplit import SentenceParser
from textRank import TextRank

cwd = Path(__file__).absolute().parent
os.sys.path.insert(0, 'C:\\Code\\opinion')

from fanjienlu.dao.hda_accessor import load_hda_news_list, load_hda_abstracts
from fanjienlu.business.abstract_business.multi_doc_abstract import WeekSummary


class multiple_extraction(object):
    
    @classmethod
    def start(cls, document, n=3, user_dir=None, remove_duplicate={'flag':False, 'threshold':None}):
        '''
            args : document : the document to be extracted
                   n        : the key sentences need to be extracted
                   user_dir : use user-set embedding data
        '''
        # 1. load the embedding data
        feature_extration = FeatureExtraction(cls.load_data(user_dir=user_dir))

        # 2. split the sentences
        if remove_duplicate['flag']:
            doc_split = SentenceParser.remove_duplicate_sentences(document, remove_duplicate['threshold'])
        else:
            doc_split = SentenceParser.split_text_to_sentences(document)

        # 3. build the feature matrix
        feature_matrix = feature_extration.extract(doc_split)

        # 4. call the textRank object
        textRank = TextRank()

        # 5. build the matrix for textRank
        time_start = time.time()
        similiar_matrix = textRank.calculate_similarity(feature_matrix)
        time_end = time.time()
        print(time_end - time_start)

        # 6. calculate the score for each sentence
        results = textRank.get_scores(similiar_matrix)

        # 7. sort the results in descending order of score
        results_sort = sorted(results.items(), key=lambda item:item[1], reverse=True)
        
        # 8. index the sentence
        candidate_index = [index for index, _ in results_sort[:n]]
        abstract = []
        for i in candidate_index:
            try:
                sentence = doc_split[i-1] + doc_split[i] + doc_split[i+1]
            except IndexError:
                sentence = doc_split[i]
            abstract.append(sentence)
        return abstract

        ######################## ignore the neighboring sentences #####################
        # extract_info = [doc_split[index] for index, _ in results_sort[:n]]
        # return extract_info
        ################################################################################

    @classmethod
    def load_data(cls, default_dir='vec', user_dir=None):
        data_dir = cwd / default_dir
        if user_dir:
            data_dir = user_dir
        vec_files = [file for file in data_dir.iterdir()]
    
        vec_dict = {}
        for file in vec_files:
            if re.search(r'.+\.vec', str(file)):
                vec_dict['vec_dir'] = file
            elif re.search(r'.+\.vocab', str(file)):
                vec_dict['vocab'] = file
            elif re.search(r'.+\.dict', str(file)):
                vec_dict['dict'] = file
        return vec_dict

def gen_weekly_summry(begindate, enddate, nsent=4, ncluster=5, k=3, n_sample=None):
    '''
        obtain the data from database, written by colleague
    '''
    if n_sample:
        docs = load_hda_news_list(begindate, enddate)[:n_sample]
    else:
        docs = load_hda_news_list(begindate, enddate)
    return docs

if __name__ == '__main__':
    # test interface
    doc = u'''今天湖人队打的非常好，科比此役复出贡献35分，5个篮板，10个助攻的数据，詹姆斯也贡献15分，12个篮板，12个助攻的准三双数据。
            NBA今天有十场比赛，其中最引人注目的就是洛杉矶湖人对金州勇士。科比拿到35分，复出新高，詹姆斯也拿到三双数据。湖人现在位列西部第二，勇士位列第一。
            在今天结束的比赛中，圣安东尼奥马刺队在莱昂纳德的带领下，引来挑战的火箭队。此役保罗贡献10个助攻，但是在第三节受伤离场。另一场焦点之战，湖人队大比分战胜勇士队。
            体坛快讯最新报道。科比迎来复出首场比赛，拿到全场最高35分带领全队取得胜利。詹姆斯取得三双。另一场比赛步行者战胜小牛。
    '''
    # print(SentenceParser.split_text_to_sentences(doc))
    # print(multiple_extraction.start(doc))
    
    # obtain data from database
    from fanjienlu.common import datetime_utils as dt

    begindate=dt.gen_datetime(2018, 5, 14)
    enddate = dt.gen_datetime(2018, 5, 15)
    docs = gen_weekly_summry(begindate, enddate, n_sample=50)
    
    # clustering the document
    from fanjienlu.algrithm.others.Clustering import Cluster, group_cluster, evaluate_clustering
    from fanjienlu.business.term_business import term_process as term

    docs_tokenized = term.tokenize(docs, pos=False, stopword=True)
    _, labels = Cluster.spectral(docs_tokenized, 8)
    docs = group_cluster(docs, labels, 8)
    
    # extract from one cluster
    test_string = '\n'.join(docs[2])
    remove_duplicate = {'flag':True, 'threshold':1}     # remove the duplicate sentences
    result = multiple_extraction.start(test_string, remove_duplicate=remove_duplicate)
    print(result)
