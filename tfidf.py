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
        self.success_count = 0
        self.failure_count = 0
        self.confusion_matrix = {'DR': {'DR': 0, 'DT': 0, 'L': 0}, 'DT': {'DR': 0, 'DT': 0, 'L': 0}, 'L': {'DR': 0, 'DT': 0, 'L': 0}}

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
            self.confusion_matrix[actual[index]][predicted[index]] += 1
            if predicted[index] == actual[index]:
                self.success_count += 1
            else:
                self.failure_count += 1

    def print_metrics(self):
        print("correct classification - ", self.success_count)
        print("incorrect classification - ", self.failure_count)
        print("accuracy - ", self.success_count*1.0/(self.success_count+self.failure_count), "\n")
        print("\t\t"+self.classes[0]+"\t\t"+self.classes[1]+"\t\t"+self.classes[2]+"\n")
        for i in range(3):
            line = self.classes[i]
            for j in range(3):
                line += "\t\t" + str(self.confusion_matrix[self.classes[i]][self.classes[j]])
            print(line+"\n")

    def classify(self):
        pip = self.train()
        self.test(pip)
        self.print_metrics()


#tfidf = TfIdfClassifier()
#tfidf.classify()