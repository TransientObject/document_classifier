import os
import re

from collections import Counter

class Intelli_Grep:

    def __init__(self):
        self.classes = ['DR', 'DT', 'L']
        self.success_count = 0
        self.failure_count = 0
        self.confusion_matrix = {'DR': {'DR': 0, 'DT': 0, 'L': 0}, 'DT': {'DR': 0, 'DT': 0, 'L': 0}, 'L': {'DR': 0, 'DT': 0, 'L': 0}}

    def preprocessing(self,doc):
        regex = re.compile('[^a-zA-Z]')

        with open(doc, encoding='utf8') as fo:
            # read in the text document as a string
            str_txt = fo.read()
            # replace non-alphabetic characters with spaces
            str_txt = regex.sub(' ', str_txt)
            # replace uppercase ascii characters with lowercase ascii characters
            str_txt = str_txt.lower()
            # collapse multiple adjacent spaces into a single space.
            str_txt = ' '.join(str_txt.split())

        return str_txt

    def intelli_grep(self,doc):
        str_txt = self.preprocessing(doc)
        num_of_key_strs = {}
        num_of_key_strs['DT'] = str_txt.count('deed of trust')
        num_of_key_strs['DR'] = str_txt.count('deed of reconveyance')
        num_of_key_strs['L'] = str_txt.count("lien")

        return max(num_of_key_strs.keys(), key=(lambda k: num_of_key_strs[k]))

    def ig_test(self):
        test_result = {}

        test_files = os.listdir('./data/'+'TEST')
        for doc in test_files:
                if doc.find('.txt') >=0:
                    doc_path = os.path.join('./data/TEST',doc)
                    test_result[doc] = self.intelli_grep(doc_path)
        return test_result

    def evaluation(self,test_result):
        classes = ['DR', 'DT', 'L']
        confusion_matrix = {'DR':{'DR':0,'DT':0,'L':0}, 'DT':{'DR':0,'DT':0,'L':0}, 'L':{'DR':0,'DT':0,'L':0}}
        with open('./data/test-results.txt') as fo:
            str_txt = fo.read().split()
            for item in str_txt:
                doc, real_cl = item.split(',')
                test_cl = test_result[doc]
                confusion_matrix[real_cl][test_cl] += 1
                if (real_cl == test_cl):
                    self.success_count += 1
                else:
                    self.failure_count += 1
        return confusion_matrix

    def print_metrics(self):
        print("correct classification - ", self.success_count)
        print("incorrect classification - ", self.failure_count)
        print("accuracy - ", self.success_count * 1.0 / (self.success_count + self.failure_count), "\n")
        print("\t\t" + self.classes[0] + "\t\t" + self.classes[1] + "\t\t" + self.classes[2] + "\n")
        for i in range(3):
            line = self.classes[i]
            for j in range(3):
                line += "\t\t" + str(self.confusion_matrix[self.classes[i]][self.classes[j]])
            print(line + "\n")

        for cl in self.classes:
            numerator = self.confusion_matrix[cl][cl] * 1.0
            print("precision of class ", cl, numerator / sum([self.confusion_matrix[j][cl] for j in self.classes]))
            print("recall of class ", cl, numerator / sum([self.confusion_matrix[cl][j] for j in self.classes]), "\n")

    def classify(self):
        self.confusion_matrix = self.evaluation(self.ig_test())
        self.print_metrics()
