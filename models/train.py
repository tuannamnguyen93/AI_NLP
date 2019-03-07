# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from sklearn_crfsuite import CRF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# from tokenizer import SimpleTokenizer
from sklearn.neural_network import MLPClassifier
import inspect
import textwrap
import re
import time
# import math
from models import TrainModel

from models import LinearSVC_proba


class Trainer(object):
    '''
    Trainer
    '''

    def __init__(self, tokenizer=None, separator=' '):

        self.separator = separator
        self.model = TrainModel()

        self.model.word_dictionary = dict()
        self.model.answers = list()
        self.model.pipeline = None  # Pipeline()
        self.model.features = textwrap.dedent(inspect.getsource(self.features))
        self.model.use_tfidf = False

        self.is_overfitting = False

        self.tokenizer = tokenizer

        self.feature_extractions = [
            ('count', CountVectorizer(
                ngram_range=(1, 2),
                max_features=self.model.max_features,
                tokenizer=self.tokenizer.tokenize if self.tokenizer else None,
            )),
            # ('dict', DictVectorizer(sparse=False)),
            # ('tfidf',TfidfVectorizer()),
        ]

        self.classifiers = [
            RandomForestClassifier,
            MultinomialNB,
            LinearSVC_proba,
            DecisionTreeClassifier,
            LogisticRegression,
            AdaBoostClassifier,
            SGDClassifier,
            KNeighborsClassifier,
            MLPClassifier,
        ]

        self.taggers = None
        self.dumper = None

    def get_classifier(self, cls):
        cls_ = None
        for c in self.classifiers:
            if c == cls:
                if cls.__name__ == 'LogisticRegesstion':
                    cls_ = LogisticRegression(penalty='l2', dual=False, tol=0.01, max_iter=60, )
                elif cls.__name__ == 'AdaBoostClassifier':
                    cls_ = AdaBoostClassifier(n_estimators=100)
                elif cls.__name__ == 'RandomForestClassifier':
                    cls_ = RandomForestClassifier(n_estimators=300)
                elif cls.__name__ == 'MLPClassifier':
                    cls_ = MLPClassifier(hidden_layer_sizes=(100,), )
                else:
                    cls_ = c()
        return ('classifier', cls_)

    def datasource(self):
        '''
        Set data source to self.dataset and self.word_dictionary
        :return: (dataset, word_dictionary)
        '''
        return list((), {})


    def untag(self, tagged_sentence):
        return [w for w, t in tagged_sentence]

    def crf_transform_to_dataset(self, tagged_sentences):
        Xs, ys = [], []
        for tagged in tagged_sentences:
            X, y = [], []
            for index in range(len(tagged)):
                items = self.features(self.untag(tagged), index)
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    X.append(item)
                    y.append(tagged[index][-1])
            Xs.append(X)
            ys.append(y)
        return Xs, ys

    def classify_transform_to_dataset(self, dataset):
        X, y = [], []
        for document, topic in dataset:
            items = document
            if not isinstance(items, list):
                items = [items]
            for item in items:
                X.append(item)
                y.append(topic)
        return X, y


    def preprocessing(self, text):
        dict = {
            u'òa': u'oà', u'óa': u'oá', u'ỏa': u'oả', u'õa': u'oã', u'ọa': u'oạ', u'òe': u'oè', u'óe': u'oé',
            u'ỏe': u'oẻ', u'õe': u'oẽ', u'ọe': u'oẹ', u'ùy': u'uỳ', u'úy': u'uý', u'ủy': u'uỷ', u'ũy': u'uỹ', u'ụy': u'uỵ'
        }
        for k, v in dict.iteritems():
            text = text.replace(k, v)
        return text
        # pass

    def train(self, test_size=0.25, dumper=None):

        dataset, word_dictionary = self.datasource()

        if self.tokenizer:
            self.model.synonyms = self.tokenizer.synonyms
            self.tokenizer.word_dictionary = word_dictionary
        self.model.word_dictionary = word_dictionary
        self.dataset = dataset

        best_classifier = None
        max_accuracy = 0
        clf = None

        # print('Dataset %s' % len(self.dataset))
        if len(self.dataset) == 0: return

        train_set, test_set = train_test_split(self.dataset, test_size=test_size, random_state=10)

        if not train_set or self.is_overfitting:
            train_set = self.dataset
            test_set = self.dataset

        taggers = list()

        if self.classifiers[0].__name__ == 'CRF':
            from sklearn_crfsuite import metrics
            X_train, y_train = self.crf_transform_to_dataset(train_set)
            X_test, y_test = self.crf_transform_to_dataset(test_set)

            if dumper:
                self.dumper = dumper
                dumper(X_train, self.__class__.__name__.lower() + 'X_train.txt')
                dumper(X_test, self.__class__.__name__.lower() + 'X_test.txt')

            print('Train_set %s' % len(X_train))
            print('Test_set %s' % len(X_test))
            # print(len(X_train), len(y_train))

            clf = CRF()
            clf.fit(X_train, y_train)

            accuracy = clf.score(X_test, y_test)
            max_accuracy = accuracy

            # Print F1 score of each label
            if self.is_overfitting == True:
                y_pred = clf.predict(X_test)
                classes = list(clf.classes_)
                labels = []
                for label in classes:
                    if label[:1] != '_':
                        labels.append(label)
                print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))
            else:
                accuracy = clf.score(X_test, y_test)
                max_accuracy = accuracy

        else:

            for feature_extraction in self.feature_extractions:

                X_train, y_train = self.classify_transform_to_dataset(train_set)
                X_test, y_test = self.classify_transform_to_dataset(test_set)
                for classifier in self.classifiers:
                    steps = list()
                    steps.append(feature_extraction)
                    if self.model.use_tfidf:
                        steps.append(('tfidf', TfidfTransformer()))
                    steps.append(self.get_classifier(classifier))
                    clf = Pipeline(steps)

                    try:
                        clf.fit(X_train, y_train)
                    except Exception as e:
                        print('ERROR', e)
                        continue

                    y_pred = clf.predict(X_test)
                    classes = list(clf.classes_)
                    # print(classes)
                    from sklearn import metrics
                    print(metrics.classification_report(y_test, y_pred,
                                                        target_names=classes, digits=3))
                    accuracy = clf.score(X_test, y_test)

                    # for y1,y2,x in zip(y_test,y_pred,X_test):
                    #     if y1!=y2:
                    #         print('Sentence: ',x)
                    #         print('True label',y1)
                    #         print('Predict label',y2)


                    print('feature extraction %s, classifier %s, accuracy: %s' % \
                          (feature_extraction[0], classifier.__name__, accuracy))

                    if accuracy >= max_accuracy:
                        max_accuracy = accuracy
                        best_classifier = clf

        if not best_classifier:
            best_classifier = clf

        feature_extraction = 'dict' if best_classifier.__class__.__name__ == 'CRF' \
            else best_classifier.steps[0][0]

        classifier_name = best_classifier.__class__.__name__ if best_classifier.__class__.__name__ == 'CRF' \
            else best_classifier.steps[-1][1].__class__.__name__


        print('Best model: feature extraction %s, classifier %s, accuracy: %s' % \
              (feature_extraction, classifier_name, max_accuracy))

        self.model.pipeline = best_classifier

        if feature_extraction == 'count':
            self.model.pipeline.steps[0][1].tokenizer = None

        self.model.build_version = time.time()

        self.dataset = zip(X_train, y_train)
        self.taggers = taggers

        return self.model


class TrainTokenizer(Trainer):
    def __init__(self, tokenizer=None):
        super(TrainTokenizer, self).__init__()
        self.tokenizer = tokenizer
        self.classifiers = [
            CRF
        ]

    def features(self, sent, index=0):
        import string
        word = sent[index]

        features = {
            'word': word,
            'len':len(word),
            # 'word_lower': word.lower(),
            # 'word_upper': word.upper(),
            'is_first': index == 0,
            'is_last': index == len(sent) - 1,
            'word[:1]': word[:1],
            'word[:2]': word[:2],
            # 'word[:3]': word[:3],
            # 'word[:4]': word[:4],
            # 'word[:5]': word[:5],
            # 'word[:6]': word[:6],
            # 'word[-6:]': word[-6:],
            # 'word[-5:]': word[-5:],
            # 'word[-4:]': word[-4:],
            # 'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[-1:]': word[-1:],
            # 'word.is_lower': word.islower(),
            # 'word.is_upper': word.isupper(),
            # 'word.is_title': word.istitle(),
            'word.is_digit': word.isdigit(),
            # 'is_all_caps': word.upper() == word,
            # 'capitals_inside': word[1:].lower() != word[1:],
            'prev_word': '' if index == 0 else sent[index - 1],
            'prev_word2': ' ' if index == 0 or index == 1 else sent[index - 2],
            'next_word': '' if index == len(sent) - 1 else sent[index + 1],
            'next_word2': ' ' if index == len(sent) - 1 or index == len(sent) - 2 else sent[index + 2],
            'is_punctuation': word in string.punctuation

        }

        n_grams = (4, 3, 2)
        size_sent = len(sent)
        for n_gram in n_grams:
            tokens = list()
            for i in range(index, index + n_gram):
                if i < size_sent:
                    tokens.append(sent[i])

            word = ' '.join(tokens)
            gram = self.model.word_dictionary.get(word.lower(), -1) + 1
            feature_name = '%s-gram' % gram
            features.update({
                feature_name: gram > 0,
                '%s.word[0]'% feature_name: word.split(' ')[0],
                # '%s.word'% feature_name : word,
                # '%s.word.is_lower' % feature_name: word.islower(),
                # '%s.word.is_upper' % feature_name: word.isupper(),
                # '%s.word.is_title' % feature_name: word.istitle(),
                # '%s.word.is_digit' % feature_name: word.isdigit(),
                # '%s.is_all_caps' % feature_name: word.upper() == word,
                # '%s.capitals_inside': word[1:].lower() != word[1:],
            })
        return features

class TrainPosTagger(Trainer):
    def __init__(self, tokenizer=None):
        super(TrainPosTagger, self).__init__(tokenizer=tokenizer)
        self.classifiers = [
            CRF
        ]

    def features(self, sent, index=0):
        word = sent[index]
        return {
            'word': word,
            'len': len(word),
            # 'is_first': index == 0,
            # 'is_last': index == len(sent) - 1,
            # 'is_capitalized': word[0].upper() == word[0],
            # 'is_second_capitalized': word[1].upper() == word[1] if len(word) > 1 else False,
            'word[:1]': word[:1],
            'word[:2]': word[:2],
            'word[:3]': word[:3],
            'word[:4]': word[:4],
            'word[:5]': word[:5],
            'word[:6]': word[:6],
            'word[:7]': word[:7],
            'word[:8]': word[:8],
            'word[:9]': word[:9],
            'word[:10]': word[:10],
            'word[:11]': word[:11],
            'word[:12]': word[:12],
            'word[:13]': word[:13],
            'word[:-13]': word[:-13],
            'word[:-12]': word[:-12],
            'word[:-11]': word[:-11],
            'word[:-10]': word[:-10],
            'word[:-9]': word[:-9],
            'word[:-8]': word[:-8],
            'word[:-7]': word[:-7],
            'word[:-6]': word[:-6],
            'word[-5:]': word[-5:],
            'word[-4:]': word[-4:],
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[-1:]': word[-1:],
            # 'word.is_lower': word.islower(),
            # 'word.is_upper': word.isupper(),
            'word.is_digit': word.isdigit(),
            'has_hyphen': '-' in word,
            'has_space': ' ' in word,
            # 'capitals_inside': word[1:].lower() != word[1:],
            # 'capitals_': word[:1].upper() == word[:1],
            # 'prev_word': '' if index == 0 else sent[index - 1],
            # 'prev_word2': ' ' if index == 0 or index == 1 else sent[index - 2],
            # 'next_word': '' if index == len(sent) - 1 else sent[index + 1],
            # 'next_word2': ' ' if index == len(sent) - 1 or index == len(sent) - 2 else sent[index + 2],
            # 'is_punctuation': word in string.punctuation
        }

class TrainClassifier(Trainer):
    def __init__(self, tokenizer=None):
        super(TrainClassifier, self).__init__(tokenizer=tokenizer)
        self.punct_regex = re.compile(self.model.punct_regex, re.UNICODE | re.MULTILINE | re.DOTALL)
        self.classifiers = [
            # RandomForestClassifier,
            # MultinomialNB,
            # LinearSVC_proba,
            # # RidgeClassifier,
            # # DecisionTreeClassifier,
            # LogisticRegression,
            # # AdaBoostClassifier,
            # SGDClassifier,
            # KNeighborsClassifier,
            MLPClassifier,
        ]

    def features(self, sent):
        self.preprocessing(sent)
        return sent
