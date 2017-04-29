import os
import re
import enchant

from collections import Counter

class NaiveBayes_Improve1:

    def __init__(self):
        self.classes = ['DR', 'DT', 'L']
        self.success_count = 0
        self.failure_count = 0
        self.confusion_matrix = {'DR': {'DR': 0, 'DT': 0, 'L': 0}, 'DT': {'DR': 0, 'DT': 0, 'L': 0}, 'L': {'DR': 0, 'DT': 0, 'L': 0}}

    def preprocessing(self,doc):
        regex = re.compile('[^a-zA-Z]')
        d = enchant.Dict("en_US")
        stop_words = ['i', 'me', 'my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won','wouldn']

        with open(doc, encoding='utf8') as fo:
            # read in the text document as a string
            str_txt = fo.read()
            # replace non-alphabetic characters with spaces
            str_txt = regex.sub(' ', str_txt)
            # replace uppercase ascii characters with lowercase ascii characters
            str_txt = str_txt.lower()
            # collapse multiple adjacent spaces into a single space.
            # get rid of stop words
            # set limit to the minimum length
            # keep only english words
            str_txt = ' '.join(word for word in str_txt.split() if (word not in stop_words and len(word)>=3 and d.check(word)))

        return str_txt

    def nb_train(self):
        classes = ['DR', 'DT', 'L']
        num_of_docs_per_class = {}
        word_counts_per_class = {}
        feature_set_per_class = {}
        total_features = set()

        # select features
        for cl in classes:
            files = os.listdir('./data/'+cl)
            num_of_docs_per_class[cl] = sum(1 for file in files if '.txt' in file)
            word_counts_per_class[cl] = Counter()

            path = os.path.join('./data/', cl)
            # There may exist files such as '.DS_Store'
            for file in files:
                if file.find('.txt') >=0:
                    file_path = os.path.join(path,file)
                    str_txt = self.preprocessing(file_path)
                    wordCount = Counter(str_txt.split())
                    word_counts_per_class[cl] += wordCount

            feature_set_per_class[cl] = sorted(word_counts_per_class[cl],key = lambda x:word_counts_per_class[cl][x], reverse=True)[0:20]
            total_features = total_features.union(set(feature_set_per_class[cl]))

        # Calculate P(c)
        fraction_of_docs_per_class = {}
        total_num_of_docs = sum(num_of_docs_per_class.values())
        for cl in classes:
            fraction_of_docs_per_class[cl] = num_of_docs_per_class[cl]/total_num_of_docs

        # create bag of words model for each document in class c
        boolean_BoW_per_doc_per_class = {}
        frequency_of_features_per_class = {}
        likelihood_of_features_per_class = {}
        for cl in classes:
            files = os.listdir('./data/'+cl)
            boolean_BoW_per_doc_per_class[cl] = {}
            frequency_of_features_per_class[cl] = dict((feature, 0) for feature in total_features)
            likelihood_of_features_per_class[cl] = {}
            path = os.path.join('./data/', cl)
            # There may exist files such as '.DS_Store'
            for file in files:
                if file.find('.txt') >=0:
                    boolean_BoW_per_doc_per_class[cl][file] = set()
                    file_path = os.path.join(path,file)
                    str_txt = self.preprocessing(file_path)
                    for word in str_txt.split():
                        if word in total_features:
                            boolean_BoW_per_doc_per_class[cl][file].add(word)

                    for word in boolean_BoW_per_doc_per_class[cl][file]:
                        if word in total_features:
                            frequency_of_features_per_class[cl][word] += 1

            for feature in total_features:
                if frequency_of_features_per_class[cl][feature] > 0:
                    likelihood_of_features_per_class[cl][feature] = frequency_of_features_per_class[cl][feature]/num_of_docs_per_class[cl]
                else:
                    likelihood_of_features_per_class[cl][feature] = 1.0/num_of_docs_per_class[cl]

        # here be optimized. No need for these many loops. Incremental training of naive bayes

        return fraction_of_docs_per_class, likelihood_of_features_per_class, total_features


    def nb_predict(self,doc):
        fraction_of_docs_per_class, likelihood_of_features_per_class, total_features = self.nb_train()

        multi_variate_bernoulli_nb= self.test_helper(doc, fraction_of_docs_per_class, likelihood_of_features_per_class, total_features)

        return sorted(multi_variate_bernoulli_nb, key = lambda x: multi_variate_bernoulli_nb[x], reverse = True)[0]

    def test_helper(self,doc,fraction_of_docs_per_class, likelihood_of_features_per_class, total_features):
        classes = ['DR', 'DT', 'L']
        boolean_BoW_of_this_doc = set()
        multi_variate_bernoulli_nb = {}

        str_txt = self.preprocessing(doc)
        for word in str_txt.split():
            if word in total_features:
                boolean_BoW_of_this_doc.add(word)

        for cl in classes:
            multi_variate_bernoulli_nb[cl] = fraction_of_docs_per_class[cl]

            for feature in total_features:
                if feature in boolean_BoW_of_this_doc:
                    multi_variate_bernoulli_nb[cl] *= likelihood_of_features_per_class[cl][feature]
                else:
                    multi_variate_bernoulli_nb[cl] *= 1 - likelihood_of_features_per_class[cl][feature]

        #return sorted(multi_variate_bernoulli_nb, key = lambda x: multi_variate_bernoulli_nb[x], reverse = True)[0]
        return multi_variate_bernoulli_nb

    def nb_test(self):
        test_result = {}

        fraction_of_docs_per_class, likelihood_of_features_per_class, total_features = self.nb_train()
        boolean_BoW_of_this_doc = set()

        test_files = os.listdir('./data/'+'TEST')
        for doc in test_files:
                if doc.find('.txt') >=0:
                    doc_path = os.path.join('./data/TEST',doc)

                    multi_variate_bernoulli_nb= self.test_helper(doc_path, fraction_of_docs_per_class, likelihood_of_features_per_class, total_features)
                    test_result[doc] = sorted(multi_variate_bernoulli_nb, key = lambda x: multi_variate_bernoulli_nb[x], reverse = True)[0]
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
            print("precision of class ", cl,
                  numerator / sum([self.confusion_matrix[j][cl] for j in self.classes]))
            print("recall of class ", cl, numerator / sum([self.confusion_matrix[cl][j] for j in self.classes]),
                  "\n")

    def classify(self):
        self.confusion_matrix = self.evaluation(self.nb_test())
        self.print_metrics()

