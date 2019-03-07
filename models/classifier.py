# -*- coding: utf-8 -*-
from __future__ import print_function
import re

class Classifier(object):
    def __init__(self, model, tokenizer=None, separator=' '):

        self.separator = separator
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer and separator:
            self.tokenizer.separator = separator

        self.predict_method = self.model.pipeline.predict_proba
        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.word_dictionary = self.model.word_dictionary
        self.anwers = self.model.answers
        exec(self.model.features, self.__dict__)

    def predict(self, document):
        import numpy as np
        if not self.model.pipeline:
            raise Exception('Need load model first')
        # labels = self.predict_method([self.features(self, document)])
        proba = self.model.pipeline.predict_proba([document])
        classes = self.model.pipeline.classes_
        # print(classes,proba[0])
        index = np.argmax(proba[0])
        return [classes[index], proba[0][index]]
