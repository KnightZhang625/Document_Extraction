# written by Shengli Wang

import numpy as np
import math
import re

class SentenceParser(object):
    @staticmethod
    def split_doc_to_sentences_with_position_weights(doc, user_defined_splitter=None, use_default_splitter=True, splitter_isolate=False):
        """Split the document into sentences, with the position weights
        
        The input document will be splitted into sentences, and the position weights will be calculated by:
        1. the position of the paragraph where the sentence is located in;
        2. the position of the sentence within the paragraph;
        
        Args:
            doc: the input text needs splitting and calculating weights
            user_defined_splitter: the special splitters user defined.
            use_default_splitter: True, default - use the default punctuation marks, together with the user defined splitters if specified.
                                  False - only use the user defined splitters if specified.
            splitter_isolate: False, default - the splitter will follow the ahead sentence content
                              True - the splitter will be a isolated sentence in the results.
        
        Returns:
            A tuple of list: sentence list, weights list        
        """
        
        
        if not doc:
            return [''], [1]
        
        
        def cal_position_weights_for_headers_and_tails(_list, _max, _delta):
            """Calculate the sentence position weights, considering where are the sentences are located in the paragraph.
            If the sentences are in the top header or bottom tail of the paragraph, the weights will be enhanced.
            """
            position_weights = np.ones(len(_list), dtype=float)
            
            # calculate position weights for header parts
            for i in range(math.floor(len(_list)/2)):
                new_weight = _max - i * _delta
                position_weights[i] = max(new_weight, 1.0)
                if new_weight < 1.0:
                    break
            
            # calculate position weights for tail parts
            for i in range(len(_list)-1, math.floor(len(_list)/2), -1):
                new_weight = _max - (len(_list)-1-i) * _delta
                position_weights[i] = max(new_weight, 1.0)
                if new_weight < 1.0:
                    break
            
            return position_weights
            
        
        sentence_weights = list()
        sentence_list = list()
        
        para_weight_max = 1.4
        para_weight_delta = 0.2
        sentence_weight_max = 1.3
        sentence_weight_delta = 0.1
                
        
        para_list = list(filter(lambda i: not not i.split(), doc.split('\n')))
        para_weights = cal_position_weights_for_headers_and_tails(para_list, para_weight_max, para_weight_delta)


        for i, para in enumerate(para_list):
            para_sentence_list = SentenceParser.split_text_to_sentences(para, user_defined_splitter=user_defined_splitter, use_default_splitter=use_default_splitter, splitter_isolate=splitter_isolate)
            para_sentence_weights = cal_position_weights_for_headers_and_tails(para_sentence_list, sentence_weight_max, sentence_weight_delta)
            
            sentence_weights.extend([sw * para_weights[i] for sw in para_sentence_weights])
            sentence_list.extend(para_sentence_list)
            
        return sentence_list, sentence_weights
    
    
    @staticmethod
    def split_text_to_sentences(text, user_defined_splitter=None, use_default_splitter=True, splitter_isolate=False):
        """Fetch the list of sentences that compose the text
        
        1. Split the text by the special default punctuation marks: '。', '？', '！', '；','?', '!', ';', '\n', '\t'
        2. Keep the subsequent punctuation marks of each sentence
        
        Args: 
             text: the input text needs splitting into sentences.
             user_defined_splitter: the special splitters user defined.
             use_default_splitter: True, default - use the default punctuation marks, together with the user defined splitters if specified.
                                   False - only use the user defined splitters if specified.
             splitter_isolate: False, default - the splitter will follow the ahead sentence content
                               True - the splitter will be a isolated sentence in the results.
             
        Returns:
            A list of sentences
        """
        if not text:
            return ['']

        sentences = list()
        #default_splitters = {'。', '？', '！', '；','?', '!', ';', '\n', '\t', '\r\n'} if use_default_splitter else set()
        default_splitters = {'。', '？', '！', '?', '!', '\n', '\t', '\r\n'} if use_default_splitter else set()
        
        # combine the user defined splitter into default marks
        if user_defined_splitter:
            if isinstance(user_defined_splitter, list) or isinstance(user_defined_splitter, set) or isinstance(user_defined_splitter, tuple):
                for sp in user_defined_splitter: # to remove duplicate splitter
                    default_splitters.add(sp)
            else:
                default_splitters.add(user_defined_splitter)
        elif not use_default_splitter:
            return [text]
        
        # sort the marks, to avoid the ambigulities among marks
        splitters = sorted(list(default_splitters), key=lambda m: len(m), reverse=True)
        
        s_start = 0
        for i in range(len(text)):
            for sp in splitters:
                if text[i:i+len(sp)] == sp:
                    if splitter_isolate:
                        cleaned_sentence = SentenceParser.cleanup_sentence(text[s_start:i].strip())
                        if cleaned_sentence:
                            sentences.append(cleaned_sentence)
                            sentences.append(sp)
                    else:
                        cleaned_sentence = SentenceParser.cleanup_sentence(text[s_start:i+len(sp)].strip())
                        if cleaned_sentence:
                            sentences.append(cleaned_sentence)
                    s_start = i+len(sp)
                    break
            i += 1
                    
        return sentences

    
    @staticmethod
    def cleanup_sentence(sentence):
        tail1 = r'[,.，。、:：]\D'
        tail2 = r'[,.，。、:：]?\D'
        switcher = {
            0 : r'^[a-z]+' + tail1,    # s = 'a. 内容'
            1 : r'^\d+' + tail1,     # s = '1. 内容'
            2 : r'^[A-Z]+' + tail1,  # s = 'A. 内容'
            3 : r'^[(（][\d+][)）]' + tail2, # s = '(1). 内容'    s = '(1) 内容'
            4 : r'^[(（][a-z]+[)）]' + tail2, # s = '(a). 内容'   s = '(a) 内容'
            5 : r'^[(（][A-Z]+[)）]' + tail2, # s = '(A). 内容'   s = '(A) 内容'
            6 : r'^[图表]\s' + tail2,     # s = '图 描述内容'
            7 : r'^[图表]\d+' + tail2,        # s = '图1. 描述内容'
            8 : r'^[图表][a-z]+' + tail2,     # s = '表a、 描述内容'
            9 : r'^[图表][A-Z]+' + tail2,     # s = '图A 描述内容'
            10 : r'^[图表][(（]\d+[)）]' + tail2,     # s = '图（1） 描述内容'
            11 : r'^[图表][(（][a-z]+[)）]' + tail2,    # s = '表(a) 描述内容'
            12 : r'^[图表][(（][A-Z]+[)）]' + tail2    # s = '图（A） 描述内容'
        }
        
        for i in range(len(switcher)):
            matched = re.match(switcher[i], sentence)
            if matched:
                return sentence[matched.end()-1:].strip() if i < 6 else str()
            
        return sentence
    
    
    @staticmethod
    def remove_duplicate_sentences(text, threshold=3):
        """Remove the duplicated sentences
        
        If the duplicate count is more than the given threshold, only keep the thresholds.
        
        Args:
            threshold: the given threshold for control max retain sentences for the duplication.
        
        Returns:
            the text that removed the duplicated by the threshold, keeping the original sentences sequences after remove.
        """
        if not threshold or not isinstance(threshold, int) or threshold < 1:
            threshold = 3
            
        sentences = SentenceParser.split_text_to_sentences(text)
        
        tmp_dict = dict()
        for s in sentences:
            tmp_dict[s] = 1 + (0 if not tmp_dict.get(s) else tmp_dict.get(s)) # dict = {sentence : count}, make the sentences unique.
        
        dup_sentences = list(filter(lambda i : tmp_dict[i] >threshold, tmp_dict.keys()))
        #print(dup_sentences)
        dup_counter = dict(zip(dup_sentences, np.zeros(len(dup_sentences), int)))
        
        cleaned_sentences = list()
        for s in sentences:
            if s not in dup_counter.keys():
                cleaned_sentences.append(s)
                continue
            
            if dup_counter.get(s)<threshold:
                cleaned_sentences.append(s)
                dup_counter[s] += 1
                
        return cleaned_sentences

