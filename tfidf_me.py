import os
import re
import nltk
from nltk.corpus import stopwords
import random
import collections
from collections import Counter
import json
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
import numpy

class TfIdfClassifier:

    def __init__(self):
        self.classes = ['DR', 'DT', 'L']

    def create_dataset(self):
        rows = []
        index = []

        for cl in self.classes:
            i = 0
            path = os.path.join('.\data\\', cl)
            files = os.listdir(path)

            # There may exist files such as '.DS_Store'
            for file in files:
                if file.find('.txt') >= 0:
                    file_path = os.path.join(path, file)
                    with open(file_path, encoding='utf8') as fo:
                        file_content = fo.read()
                        rows.append({'text': file_content, 'class' : cl})
                        index.append(file_path)
                i += 1

        data_frame = DataFrame(rows, index=index)
        return data_frame


    def train(self):
        data = self.create_dataset()
        data = data.reindex(numpy.random.permutation(data.index))
        pip = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('SGD', SGDClassifier())])
        pip.fit(data['text'].values, data['class'].values)
        return pip

    def test(self, pip):
        i=0
        path = os.path.join('./data/', 'test-results.txt')
        success_count = 0
        failure_count = 0
        input = []
        actual = []
        predicted = []

        with open(path, encoding='utf8') as f:
            file_content = f.read()
            lines = file_content.split("\n")
            for line in lines:
                i += 1
                columns = line.split(",")
                with open(".\\data\\TEST\\" + columns[0].strip(), encoding='utf8') as tf:
                    input.append(tf.read())
                    actual.append(columns[1].strip())

            predicted = pip.predict(input)

        for index in range(len(predicted)):
            if predicted[index] == actual[index]:
                success_count += 1
            else:
                failure_count += 1

        print("success - ", success_count)
        print("failure - ", failure_count)

    def classify(self):
        pip = self.train()
        self.test(pip)


tfidf = TfIdfClassifier()
tfidf.classify()


