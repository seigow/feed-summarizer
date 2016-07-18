# coding:utf-8

import re
from collections import Counter, defaultdict
from itertools import chain
import math
import json
import requests
import MeCab
import treetaggerwrapper
import pickle

ja_tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/ipadic")
en_tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='../TreeTagger')


class FeedlyClient():
    def __init__(self, client_id, token):
        self.token = token
        self.client_id = client_id
        self.endpoint = "http://cloud.feedly.com/v3/"
        self.header = {'Authorization': 'OAuth ' + self.token}

    def get_unreads(self, count=20):
        params = dict(
            streamId="user/" + self.client_id + "/category/global.all",
            count=count, unreadOnly=True)
        request_url = self.endpoint + 'streams/ids'
        res = requests.get(url=request_url, params=params, headers=self.header)
        unreads = json.loads(res.text)["ids"]
        return unreads

    def get_unreads_contents(self, count=20, ranked="oldest"):
        params = dict(
            streamId="user/" + self.client_id + "/category/global.all",
            count=count, ranked=ranked, unreadOnly=True)
        request_url = self.endpoint + 'streams/contents'
        res = requests.get(url=request_url, params=params, headers=self.header)
        unreads = json.loads(res.text)
        return unreads


def get_nouns(sentence):
    nouns = []

    is_ja_sentence = any(ord(char) >= 256 for char in sentence)
    if is_ja_sentence:
        ja_tagger.parse("")  # must parse at first, because of the mecab's bag
        node = ja_tagger.parseToNode(sentence)
        with open("./Japanese.txt") as f:
            stopwords_ja = set(f.read().split('\n'))

        while node:
            if node.feature.split(",")[0] == "名詞" \
                    and re.search("[a-zA-Z0-9ぁ-んァ-ン一-龥]", node.surface)\
                    and node.surface not in stopwords_ja:
                nouns.append(node.surface)
            node = node.next
    else:
        # stopwords_en = nltk.corpus.stopwords.words("english")
        # filtered_sentence = ' '.join(
        #                     set(sentence.split(' ')) - set(stopwords_en))
        tags = en_tagger.tag_text(sentence, notagurl=True,
                                  notagemail=True, notagip=True, notagdns=True)
        for tag in tags:
            splitted_tag = tag.split("\t")
            '''
            treetagger tag set reference:
            https://courses.washington.edu/hypertxt/csar-v02/penntable.html
            '''
            if "NN" in splitted_tag[1] or "NP" in splitted_tag[1]:
                nouns.append(splitted_tag[2])

    return nouns


def calc_TF(document):
    n_term = len(document)
    tf = defaultdict(int)
    for k, v in Counter(document).items():
        tf[k] = v/n_term
    return tf


def calc_IDF(documents):
    df = Counter(chain.from_iterable(documents))
    n_docs = len(documents)
    idf = {k: math.log(n_docs / v) + 1 for k, v in df.items()}
    return idf


if __name__ == '__main__':
    with open('token.json') as f:
        contents = json.loads(f.read())

    feedly = FeedlyClient(contents['id'], contents['token'])
    count = 1000
    unreads = feedly.get_unreads_contents(count)

    titles = [unread["title"] for unread in unreads["items"]]
    titles_nouns = [get_nouns(title) for title in titles]

    with open("./unreads.pkl", 'wb') as f:
        pickle.dump(unreads, f)
