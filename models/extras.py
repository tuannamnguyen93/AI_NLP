# -*- coding: utf-8 -*-
from __future__ import print_function
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VN_CHARS_LOWER = u'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð'
VN_CHARS_UPPER = u'ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ'
VN_CHARS = VN_CHARS_LOWER + VN_CHARS_UPPER


def normalize_text(text):
    dict = {
        u'òa': u'oà', u'óa': u'oá', u'ỏa': u'oả', u'õa': u'oã', u'ọa': u'oạ', u'òe': u'oè', u'óe': u'oé',
        u'ỏe': u'oẻ', u'õe': u'oẽ', u'ọe': u'oẹ', u'ùy': u'uỳ', u'úy': u'uý', u'ủy': u'uỷ', u'ũy': u'uỹ', u'ụy': u'uỵ'
    }
    for k, v in dict.iteritems():
        text = text.replace(k, v)
    return text

def no_marks(s):
    __INTAB = [ch for ch in VN_CHARS]
    __OUTTAB = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"*2
    __OUTTAB += "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"*2
    __r = re.compile("|".join(__INTAB))
    __replaces_dict = dict(zip(__INTAB, __OUTTAB))
    result = __r.sub(lambda m: __replaces_dict[m.group(0)], s)
    return result


def text_similarity(items):
    f = CountVectorizer().fit_transform(items)
    return cosine_similarity(f[0], f[1])[0][0]


def similarity(items):

    if len(items) < 2: return 0
    # for item in items: print(item)
    f = DictVectorizer().fit_transform(items)
    return cosine_similarity(f[0], f[1])[0][0]


def pre_processing(items, ignore_keys=list(), weighting=dict()):

    # pre-processing
    for item in items:
        for k, v in item.items():
            if not v:
                v = 0
                item[k] = v
            if k in ignore_keys:
                del(item[k])
            elif isinstance(v, list):
                for i in range(0, len(v)):
                    item['%s-%d' % (k, i)] = str(v[i])
                del(item[k])
            elif isinstance(v, float):
                item[k] = int(round(v))
            elif not isinstance(v, int):
                item[k] = str(v)

    # build vocabulary for feature with string value
    vocab = dict()
    for item in items:
        for k, v in item.items():
            if isinstance(v, str) or isinstance(v, unicode):
                # print(k,v)
                if k not in vocab:
                    vocab[k] = dict()
                if v not in vocab[k]:
                    vocab[k][v] = len(vocab[k])
                    if k in weighting:
                        vocab[k][v] = vocab[k][v] * weighting[k]

    # update value of feature with vocabulary
    for item in items:
        for k, v in item.items():
            if k in vocab and v in vocab[k]:
                item[k] = vocab[k][v]

    return vocab




if __name__ == '__main__':
    # test_similarity()
    pass
