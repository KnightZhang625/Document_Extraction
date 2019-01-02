# -*- coding: utf-8 -*-
# @Author: Jiaxin Zhang
# @Date:   02/Jan/2019
# @Last Modified by:    
# @Last Modified time:  

import re
from pathlib import Path

from featureExtraction import FeatureExtraction
from sentenceSplit import SentenceParser
from textRank import TextRank

cwd = Path(__file__).absolute().parent

class multiple_extraction(object):
    
    @classmethod
    def start(cls, document, n=3, user_dir=None):
        '''
            args : document : the document to be extracted
                   n        : the key sentences need to be extracted
                   user_dir : use user-set embedding data
        '''
        # 1. load the embedding data
        feature_extration = FeatureExtraction(cls.load_data(user_dir=user_dir))
        # 2. split the sentences
        doc_split = SentenceParser.split_text_to_sentences(document)
        # 3. build the feature matrix
        feature_matrix = feature_extration.extract(doc_split)
        # 4. call the textRank object
        textRank = TextRank()
        # 5. build the matrix for textRank
        similiar_matrix = textRank.calculate_similarity(feature_matrix)
        # 6. calculate the score for each sentence
        results = textRank.get_scores(similiar_matrix)
        # 7. sort the results in descending order of score
        results_sort = sorted(results.items(), key=lambda item:item[1], reverse=True)
        # 8. index the sentence
        extract_info = [doc_split[index] for index, _ in results_sort[:3]]
        return extract_info

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

if __name__ == '__main__':
    doc = u'''今天湖人队打的非常好，科比此役复出贡献35分，5个篮板，10个助攻的数据，詹姆斯也贡献15分，12个篮板，12个助攻的准三双数据。
            NBA今天有十场比赛，其中最引人注目的就是洛杉矶湖人对金州勇士。科比拿到35分，复出新高，詹姆斯也拿到三双数据。湖人现在位列西部第二，勇士位列第一。
            在今天结束的比赛中，圣安东尼奥马刺队在莱昂纳德的带领下，引来挑战的火箭队。此役保罗贡献10个助攻，但是在第三节受伤离场。另一场焦点之战，湖人队大比分战胜勇士队。
            体坛快讯最新报道。科比迎来复出首场比赛，拿到全场最高35分带领全队取得胜利。詹姆斯取得三双。另一场比赛步行者战胜小牛。
    '''
    # print(SentenceParser.split_text_to_sentences(doc))
    print(multiple_extraction.start(doc))