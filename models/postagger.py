# -*- coding: utf-8 -*-
from __future__ import print_function
import re
from tokenizer import SimpleTokenizer


class PosTagger(object):

    def __init__(self, model, tokenizer=None, separator=' '):

        self.separator = separator

        self.model = model

        if not tokenizer:
            tokenizer = SimpleTokenizer(separator=separator)
            tokenizer.word_dictionary = model.word_dictionary
            tokenizer.is_lowercase = False

        self.tokenizer = tokenizer

        if separator:
            self.tokenizer.separator = separator

        self.predict_method = self.model.pipeline.predict_single \
            if self.model.pipeline.__class__.__name__ == 'CRF' else \
            self.model.pipeline.predict

        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.word_dictionary = self.model.word_dictionary
        exec(self.model.features, self.__dict__)
        self.predict_confidence = self.model.pipeline.predict_marginals_single

    def predict(self, sent):

        if not self.model.pipeline:
            raise Exception('Need load model first')

        if self.tokenizer:
            words = self.tokenizer.tokenize(sent)
        else:
            words = self.punct_regex.findall(sent)

        tagged = self.predict_method(
            [self.features(self, words, index) for index in range(len(words))])

        return zip(words, tagged)
