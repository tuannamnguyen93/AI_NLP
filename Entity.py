# -*- coding:utf8 -*-
# !/usr/bin/env python
from __future__ import print_function
import os.path
from models.tokenizer import MyTokenizer
from models.postagger import PosTagger
from random import randint
import os
import codecs
from data import PROJECT_PATH
from sklearn.externals import joblib
from models.train import *
from models.conect_db import get_entities
from random import shuffle
from models.extras import normalize_text
import collections
import operator
# PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))


class EntityClassifier():

    def __init__(self, bot_id):
        self.bot_id = bot_id
    #
    def load_word_dictionary(self):
        word_dictionary = dict()
        with codecs.open(os.path.join(PROJECT_PATH, 'data', 'Viet39K.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                token = token.lower()
                token = normalize_text(token)
                word_dictionary[token] = '0'
        return word_dictionary

    def load_data_set_fromdb(self):
        datadb = get_entities(self.bot_id)
        dictionary = self.load_word_dictionary()
        for token in datadb:
            token_lower = token[0].lower()
            if token_lower in dictionary:
                del dictionary[token_lower]
        result = collections.Counter(map(operator.itemgetter(1), datadb))
        result = sorted(result.items(), key = operator.itemgetter(1))
        data = []
        for key, value in result:
            for content, entity in datadb:
                entity = unicode(entity)
                # content = normalize_text(content)
                if entity == key:
                    for i in range(len(dictionary) / (value)):
                        # data.append([content,entity])
                        data.append([entity,content.lower()])
        for key, value in dictionary.items():
            data.append([key, value])
        # data = [x for pair in zip(data, data) for x in pair]  # duplicate elements
        shuffle(data)#
        # order dictionary by len of words
        dataset = [data[x:x + randint(15, 35)] for x in xrange(0, len(data), 100)]
        # result = collections.Counter(map(operator.itemgetter(1), data))
        return dataset
    #
    def datasource(self):
        dataset = self.load_data_set_fromdb()
        word_dictionary = dict()
        return dataset, word_dictionary

    def train_entity_model(self):
        trainer = TrainPosTagger()
        trainer.datasource = self.datasource
        trainer.is_overfitting = True
        trainer.train()
        with open(os.path.join(PROJECT_PATH, 'pretrained_models/NER.model'), 'w') as f:
            joblib.dump(trainer.model, f)

    def predict_entiy(self,sent):

        # with open(os.path.join(PROJECT_PATH, 'pretrained_models/tokenizer.model')) as f:
        #     model = joblib.load(f)
        # tokenizer = Tokenizer(model = model)

        tokenizer = MyTokenizer(self.bot_id)

        with open(os.path.join(PROJECT_PATH, 'pretrained_models/NER.model')) as f:
            model = joblib.load(f)
        tagger = PosTagger(model=model, tokenizer=tokenizer)

        tokens = tagger.predict(sent)
        result = []
        for token, tag in tokens:
            if tag!='0':
                result.append([token, tag])
        return result

    def ner(self, sent):
        datadb = get_entities(self.bot_id)
        labels,words = [],[]
        for index in datadb:
            words.append(index[1].lower())
            labels.append(index[0])
        # words = list(read_file['word'])
        # labels = list(read_file['label'])
        tokenizer = MyTokenizer(self.bot_id)
        sent = tokenizer.tokenize(sent)
        # for w in words:
        #     print(w)
        my_result = []
        for i in range(len(sent)):
            if sent[i] in words:
                index = words.index(sent[i])
                label = labels[index]
                my_result.append([sent[i],label])
            #     sent[i] = sent[i]
            #     sent[i] += '-' + label
            # else:
            #     sent[i] = sent[i]
            #     sent[i] += "-0"
        return my_result


    # def test_entiy(self):
    #     sents = [
    #         u' Long Xuyên xuân thuỷ cầu giấy hà nội thì mua thuốc tràng phục linh chỗ nào',
    #         u'tôi ở việt nam trung quốc quốc thì mua thuốc ở đâu',
    #         u'thành phố nông cống thanh hoá là ở đây',
    #         u'cho em đi maxxhair nhé',
    #         u'ngoài kia là vĩnh tường vĩnh phú phải không',
    #         u'có tràng phục linh với vương bảo không các bạn',
    #         u'tôi đang ở minh quán trấn yên yên bái cho hỏi điểm bán thuốc tràng phục linh'
    #     ]
    #     for sent in sents:
    #         entities = self.predict_entiy(sent)
    #         print('-'*100)
    #         for entity,ent in entities:
    #             print(entity)

# EntityClassifier(bot_id="2").train_entity_model()

# sent = u'Ở Xuân Thủy Cầu giấy Hà Nội, thì mua thuốc VBP vĩnh phúc phú thọ ở trang phuc linh chỗ nào'
#
# A = EntityClassifier(bot_id="2")
# #
# print(A.ner(sent))
