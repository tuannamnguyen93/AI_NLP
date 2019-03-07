from sklearn.svm import LinearSVC
import numpy as np


PUNCT_REGEX = r'((((\d{1,3}[.,])|)\d{1,4}[.,]\d{3}[.,]\d{3})|(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})|(\w+[\-\+]\w+)|([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|(\w+[.,\/\+]\w+)|(\w+\.)|(\w+|[^\w\s]+))'


class TrainModel:
    def __init__(self):
        self.c = dict() # multi words lexicon
        self.parent_dictionary = dict()
        self.punct_regex = PUNCT_REGEX # word split regex
        self.intent_entities_split_regex = r'(\(\?.*?\))'
        self.pipeline = None # Sklearn Pipeline
        self.features = None # features function
        self.use_tfidf = False
        self.max_features = None
        self.synonyms = dict()
        self.entity_regex_dictionary = None
        self.digit_strings = []
        self.intents = {}
        self.build_version = ''


class LinearSVC_proba(LinearSVC):
    '''
    http://www.erogol.com/predict-probabilities-sklearn-linearsvc/
    '''

    def __platt_func(self,x):
        return 1/(1+np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions)
        probs = list(platt_predictions[:, None])
        return probs
