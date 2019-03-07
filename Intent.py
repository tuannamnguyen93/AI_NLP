# -*- coding:utf8 -*-
# !/usr/bin/env python
from __future__ import print_function
import os.path
import os
from data import PROJECT_PATH
from sklearn.externals import joblib
import csv
from models.train import *
from models.classifier import *
from models.conect_db import get_train_data,get_answers
from models.tokenizer import MyTokenizer

class IntentClassifier():

    def __init__(self, bot_id, type_intent):
        self.bot_id = bot_id
        self.type_intent = type_intent

    # def load_data_set_from_csv(self):
    #     dataset = list()
    #     with open(os.path.join(PROJECT_PATH, 'data', 'intents.csv')) as f:
    #         reader = csv.DictReader(f)
    #         for row in reader:
    #             sample = row['sample'].strip().lower().decode('utf-8')
    #             intent_name = row['intent_name'].strip()
    #             dataset.append((sample, intent_name))
    #     return dataset
    #
    # def load_synonyms(self):
    #     dataset = dict()
    #     with open(os.path.join(PROJECT_PATH, 'data', 'synonyms.txt')) as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             tokens = line.split(',')
    #             for token in tokens[1:]:
    #                 dataset[token.strip().decode('utf-8')] = tokens[0].strip().decode('utf-8')
    #     return dataset

    # def load_intents_dictionary(self):
    #     dictionary = dict()
    #     with open(os.path.join(PROJECT_PATH, 'data', 'intents_dictionary.csv')) as f:
    #         reader = csv.DictReader(f)
    #         for row in reader:
    #             key = row['key'].strip().decode('utf-8')
    #             value = row['value'].strip().decode('utf-8')
    #             dictionary[key] = value
    #             dictionary[key.lower()] = value
    #     return dictionary

    def generate_bot_name(self):
        name = 'botid_' + str(self.bot_id) + '_type_' + str(self.type_intent) + '.model'
        return name

    def datasource(self):
        dataset = get_train_data(self.bot_id, self.type_intent)#self.load_data_set()
        word_dictionary = dict()#self.load_intents_dictionary()
        return dataset, word_dictionary

    def trainmodel(self,botId):
        # with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
        #     model = joblib.load(f)
        tokenizer = MyTokenizer(botId)  # Tokenizer(model=model)
        # print(tokenizer.synonyms)
        trainer = TrainClassifier(tokenizer=tokenizer)
        trainer.datasource = self.datasource
        trainer.model.answers = get_answers(botId)
        trainer.classifiers = [
            # MultinomialNB,
            # LinearSVC_proba,
            MLPClassifier,
        ]
        # trainer.tokenizer.synonyms = self.load_synonyms()
        # trainer.is_overfitting = False
        # trainer.train()
        trainer.is_overfitting = True
        model = trainer.train()
        bot_name = self.generate_bot_name()
        with open(os.path.join(PROJECT_PATH, 'pretrained_models/'+str(bot_name)), 'w') as f:
            joblib.dump(model, f)

    def classify_intent(self, query, botId,threshold_confidence):
        try:
            bot_name = self.generate_bot_name()
            with open(os.path.join(PROJECT_PATH, 'pretrained_models/'+str(bot_name))) as f:
                model = joblib.load(f)
            classifier = Classifier(model=model)
            intent_result = classifier.predict(query)
            answers = classifier.anwers
            if intent_result[1] >= threshold_confidence:
                for data in answers:
                    if intent_result[0] == data[0]:
                        intent_result.append(data[1])
            else:
                intent_result[0] = 'not_define'
                intent_result.append(u'Chatbot chưa được học vấn đề này')
        except Exception as err:
            intent_result = ['Error',0,'Error']
            print(err)
        return intent_result



    # def test_intents(self):
    #
    #     documents = [
    #         u'Xin chào',
    #         u'em là ai thế nhỉ',
    #         u'sao kém thế nhỉ',
    #         u'Cho hỏi chỗ mua thuốc',
    #         u'em ơi cho anh hỏi giá thuốc tràng phục linh thế nào nhỉ',
    #         u'giỏi đấy',
    #         u'thuốc vương bảo dùng như thế nào ấy nhỉ',
    #         u'cám ơn em nhé',
    #         u'tạm biệt cậu nhé',
    #
    #     ]
        # with open(os.path.join(PROJECT_PATH, 'data/tokenizer.model')) as f:
        #     model = joblib.load(f)
        # tokenizer = Tokenizer(model=model)

        # with open(os.path.join(PROJECT_PATH, 'data/intents.model')) as f:
        #     model = joblib.load(f)
        #
        # classifier = Classifier(model=model)
        # for document in documents:
        #     labels = classifier.predict(document)
        #     print(labels)



