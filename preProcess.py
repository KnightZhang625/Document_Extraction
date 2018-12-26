# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   26/Dec/2018
# @Last Modified by:    
# @Last Modified time: 

import re
import sys

import jieba
import jieba.analyse

class PreprocessAPI(object):
    def __init__(self, stopword_path=None, user_dict_path=None):
        '''
            please set the path as absolute path
        '''
        self.stopword_path = stopword_path
        self.user_dict_path = user_dict_path
    
    def preProcess(self, data_list, cut_all=False, cur_for_search=False, HMM=False, enable=(False, None)):
        '''
            data_list : a list which consists of strings
        '''
        if self.stopword_path is not None:
            try:
                jieba.analyse.set_stop_words(self.stopword_path)
            except FileNotFoundError as e:
                print(e)
                sys.exit()
        if self.user_dict_path is not None:
            try:
                jieba.load_userdict(self.user_dict_path)
            except FileNotFoundError as e:
                print(e)
                sys.exit()
        
        data_preprocessed = []
        for sentence in data_list:
            sentence = self._removePunctuation(sentence)
            sentence = self._tokenize(sentence, cut_all, cur_for_search, HMM, enable)
            data_preprocessed.append(sentence)
        return data_preprocessed        # a list which contains processed strings,
                                        # each token in every strings is seperated by space

    def _tokenize(self, sentence, cut_all, cut_for_search, HMM, enable):
        if enable[0]:
            print('use multiprocessing')
            jieba.enable_parallel(enable[1])
        else:
            jieba.disable_parallel()
        if not cut_for_search:
            sentence_temp = ' '.join(jieba.cut(sentence, cut_all, HMM))
            return sentence_temp
        else:
            sentence_temp = ' '.join(jieba.cut_for_search(sentence, HMM))
            return sentence_temp

    def _removePunctuation(self, sentence):
        r_punctuation = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~。（）：＊※，·… 、？！\n👍]+'
        r_emoji = u'[\uD800-\uDBFF][\uDC00-\uDFFF]'
        sentence_temp = sentence.strip()
        sentence_temp = re.sub(r_punctuation, '', sentence_temp)
        sentence_temp = re.sub(r_emoji, '', sentence_temp)
        sentence_temp = re.sub(' ', '', sentence_temp)
        sentence_temp = sentence_temp.lower()
        return sentence_temp

if __name__ == '__main__':
    # the below is just for test
    string = ['我毕业于谢菲尔德大学', '今天天气真好']           # should be list, even contains only one sentence
    preprocess = PreprocessAPI()
    print(preprocess.preProcess(string))
