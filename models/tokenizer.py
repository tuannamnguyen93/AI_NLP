# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import string
from models import PUNCT_REGEX
from conect_db import get_entities,get_synonyms
from nltk.tokenize import regexp_tokenize
# import os
# from data import PROJECT_PATH
# import codecs

class Tokenizer(object):
    '''
    Chappie Tokenizer with Pos Tagging Model
    '''
    def __init__(self, model, separator=' '):

        self.punctuation = set(string.punctuation)

        self.separator = separator
        self.synonyms = dict()
        self.stopwords = dict()

        self.is_remove_punctuation = True
        self.is_strip_punctuation = False

        self.model = model
        self.predict_method = self.model.pipeline.predict_single \
            if self.model.pipeline.__class__.__name__ == 'CRF' else self.model.pipeline.predict

        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.word_dictionary = self.model.word_dictionary
        exec(self.model.features, self.__dict__)

    def tagging(self, sent):
        if not self.model.pipeline:
            raise Exception('Need load model first')

        # words = self.punct_regex.findall(sent)
        words = sent.split()
        tagged = self.predict_method([self.features(self, words, index) for index in range(len(words))])
        return zip(words, tagged)

    def tokenize(self, sent):

        # remove multiple spaces
        sent = re.sub(r' +', ' ', sent).strip()

        sent = sent.strip()#.lower()

        # remove punctuation
        if self.is_remove_punctuation:
            sent = ''.join(ch for ch in sent if ch not in self.punctuation)

        # strip punctuation

        if self.is_strip_punctuation:
            sent = sent.strip(string.punctuation)

        output = self.tagging(sent)
        tokens = list()
        grams = []
        for index in range(len(output)):
            if output[index][-1] == '0':
                tokens.append(output[index][0])
            elif output[index][-1] == '1':
                grams.append(output[index][0])
                grams.append(self.separator)
            else:
                grams.append(output[index][0])
                tokens.append(u''.join(grams))
                grams = []

        tokens = [w for w in tokens if not w in self.stopwords]

        i = 0
        for token in tokens:
            tokens[i] = self.synonyms.get(token, token)
            i += 1

        return tokens


class SimpleTokenizer():
    '''
    Tokenizer based on punct regex and word dictionary
    '''

    def __init__(self, separator=' ', word_dictionary=None, synonyms=None, stopwords=None):

        self.word_dictionary = word_dictionary if word_dictionary else dict()
        self._punctuation = string.punctuation.replace('_', '').replace('.', '').replace('+', '').replace('@', '')
        self.punctuation = set(self._punctuation)

        self.separator = separator
        self.synonyms = synonyms if synonyms else dict()
        self.stopwords = stopwords if stopwords else dict()
        self.punct_regex = re.compile(PUNCT_REGEX, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.is_lowercase = False

    @staticmethod
    def replace_x(st):
        st_ = st.strip('.,:-;!?')
        return st_ if st_ else st

    @staticmethod
    def clean(st):
        st = st.replace(u'ệ', u'ệ').\
            replace(u'ẩ', u'ẩ').\
            replace(u'ợ', u'ợ')
        return st


    def tokenize(self, sent):
        '''
        apply n-grams and word dictionary
        :param string:
        :return:
        '''

        text = self.clean(sent)

        text = text.lower()

        # remove all punctuation
        # text = ''.join(ch for ch in text if ch not in self.punctuation)

        # strip punctuation
        text = text.strip(self._punctuation)

        # remove multiple spaces
        text = re.sub(r' +', ' ', text).strip()

        text = text.strip()

        # words split based on from nltk.tokenize import WordPunctTokenizer
        words = [v[0] for v in self.punct_regex.findall(text)]

        tokens = list()
        n_grams = (8, 7, 6, 5, 4, 3, 2)
        max = len(words)
        i = 0
        while i < max:
            has_gram = False
            token = None
            for n_gram in n_grams:
                token = self.replace_x(self.separator.join(words[i:i + n_gram]))
                if token in self.word_dictionary \
                        or token in self.synonyms:
                    i += n_gram
                    has_gram = True
                    break
            if not has_gram:
                token = self.replace_x(words[i])
                i += 1

            if token in self.synonyms:
                token = self.synonyms.get(token)#.lower()

            if token not in self.punctuation:
                tokens.append(token)

        tokens = [w for w in tokens if not w in self.stopwords]

        return tokens

class MyTokenizer():

    def __init__(self, bot_id):
        self.bot_id = bot_id
        self.synonyms = dict()
        self.stopwords = dict()
        self.punctuation = string.punctuation
        self.is_remove_punctuation = True
        self.dictionary, self.newdict = self.get_dictionary()
        print(self.bot_id)

    def get_synonym_fromdb(self):
        my_list = get_synonyms(self.bot_id)
        my_synonym = []
        for synonym in my_list:
            if synonym[1]== None:
                continue
            else:
                my_synonym.append(synonym)
        my_list =  dict([[x[1].lower(),x[0].lower()] for x in my_synonym])
        return my_list

    def get_dictionary(self):
        dict_list = []
        self.synonyms = self.get_synonym_fromdb()
        # with codecs.open(os.path.join(PROJECT_PATH, 'data', 'commune.txt'), 'r', encoding='utf-8') as fin:
        #     for token in fin.read().split('\n'):
        #         dict_list.append(token.lower())
        #
        # with codecs.open(os.path.join(PROJECT_PATH, 'data', 'district.txt'), 'r', encoding='utf-8') as fin:
        #     for token in fin.read().split('\n'):
        #         dict_list.append(token.lower())
        #
        # with codecs.open(os.path.join(PROJECT_PATH, 'data', 'province.txt'), 'r', encoding='utf-8') as fin:
        #     for token in fin.read().split('\n'):
        #         dict_list.append(token.lower())
        #
        # with codecs.open(os.path.join(PROJECT_PATH, 'data', 'product.txt'), 'r', encoding='utf-8') as fin:
        #     for token in fin.read().split('\n'):
        #         dict_list.append(token.lower())

        datadb = get_entities(self.bot_id)
        # print(datadb)
        for token in datadb:
            dict_list.append(token[1].lower())

        for k,v in self.synonyms.items():
            dict_list.append(k.lower())
        newdict = {}
        for item in dict_list:
            dk = item.replace(" ", "_")
            newdict[item] = dk

        newdict_sorted = sorted(newdict, key=len, reverse=True)
        return newdict_sorted, newdict

    def tokenize(self,sent):
        self.punctuation = self.punctuation.replace('-','')
        if self.is_remove_punctuation:
            sent = ''.join(ch for ch in sent if ch not in self.punctuation)
        sent = (' ' + sent + ' ').lower()
        # print(self.dictionary)
        for item in self.dictionary:
            if ' ' + item + ' ' in sent:
                sent = sent.replace(item, self.newdict[item])
        # print(sent)

        res = regexp_tokenize(sent, pattern='\S+')
        my_res = []
        for token in res:
            if '_' in token:
                token = token.replace('_',' ')
                if token in self.synonyms:
                    token = self.synonyms.get(token)
                my_res.append(token)
            else:
                if token in self.synonyms:
                    token = self.synonyms.get(token)
                my_res.append(token)
        return my_res

# mytk = MyTokenizer(bot_id="2")
# dictionary = dict()
# tokenizer = SimpleTokenizer(separator=' ')
# dict,_ = mytk.get_dictionary()
# for value in dict:
#     dictionary[value] = len(value)
#
# tokenizer.word_dictionary = dictionary
# tokenizer.is_lowercase = False
# print(dictionary)

# mytk = MyTokenizer(bot_id="2")
# from extras import normalize_text
# sent = u'Ở Xuân Thủy Cầu giấy Hà Nội, thì mua thuốc vĩnh phúc phú thọ ở trang phuc linh chỗ nào'
# sent = normalize_text(sent)
# x= mytk.tokenize(sent)
# for a in x:
#     print(a)