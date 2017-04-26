import os
import re

from collections import Counter

def preprocessing(doc):
  regex = re.compile('[^a-zA-Z]')

  with open(doc) as fo:
    # read in the text document as a string
    str_txt = fo.read()
    # replace non-alphabetic characters with spaces
    str_txt = regex.sub(' ', str_txt)
    # replace uppercase ascii characters with lowercase ascii characters
    str_txt = str_txt.lower()
    # collapse multiple adjacent spaces into a single space.
    str_txt = ' '.join(str_txt.split())

  return str_txt

def nb_train():
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
        str_txt = preprocessing(file_path)
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
        str_txt = preprocessing(file_path)
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



def nb_predict(doc):
  classes = ['DR', 'DT', 'L']
  fraction_of_docs_per_class, likelihood_of_features_per_class, total_features = nb_train()
  boolean_BoW_of_this_doc = set()
  multi_variate_bernoulli_nb = {}

  str_txt = preprocessing(doc)
  for word in str_txt.split():
    if word in total_features:
      boolean_BoW_of_this_doc.add(word)

  for cl in classes:
    multi_variate_bernoulli_nb[cl] = 1

    for feature in total_features:
      if feature in boolean_BoW_of_this_doc:
        multi_variate_bernoulli_nb[cl] *= likelihood_of_features_per_class[cl][feature]
      else:
        multi_variate_bernoulli_nb[cl] *= 1 - likelihood_of_features_per_class[cl][feature]

  return sorted(multi_variate_bernoulli_nb, key = lambda x: multi_variate_bernoulli_nb[x], reverse = True)[0]

nb_train()

