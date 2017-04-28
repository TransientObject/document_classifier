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

def intelli_grep(doc):
  str_txt = preprocessing(doc)
  num_of_key_strs = {}
  num_of_key_strs['DT'] = str_txt.count('deed of trust')
  num_of_key_strs['DR'] = str_txt.count('deed of reconveyance')
  num_of_key_strs['L'] = str_txt.count("lien")

  return max(num_of_key_strs.keys(), key=(lambda k: num_of_key_strs[k]))






